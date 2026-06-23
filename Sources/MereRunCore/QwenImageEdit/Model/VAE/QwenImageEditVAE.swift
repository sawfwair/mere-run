import Foundation
import MLX
import MLXNN

/// VAE wrapper for Qwen-Image-Edit that handles latent packing/unpacking.
/// Uses the 3D VAE (AutoencoderKL3D) for video-compatible decoding.
public final class QwenImageEditVAE: Module {
    private let vae: AutoencoderKL3D
    public let config: QwenImageEditVAEConfig

    /// Patch size for latent packing (2x2)
    public let patchSize: Int = 2

    public init(config: QwenImageEditVAEConfig) {
        self.config = config

        // Create VAE3DConfig from QwenImageEditVAEConfig
        let vae3dConfig = VAE3DConfig(
            inChannels: config.inChannels,
            outChannels: config.outChannels,
            latentChannels: config.latentChannels,
            blockOutChannels: config.blockOutChannels,
            layersPerBlock: config.layersPerBlock,
            normNumGroups: config.normNumGroups,
            scalingFactor: config.scalingFactor,
            shiftFactor: config.shiftFactor ?? 0.0,
            temporalCompressionRatio: config.temporalCompressionRatio ?? 4,
            midBlockAddAttention: config.midBlockAddAttention ?? true
        )

        self.vae = AutoencoderKL3D(config: vae3dConfig)
        super.init()
    }

    /// Access underlying VAE for weight loading
    public var underlyingVAE: AutoencoderKL3D { vae }

    // MARK: - Encoding

    /// Encode image to latents (unpacked)
    /// - Parameter images: Input images [B, C, H, W] normalized to [-1, 1]
    /// - Returns: Latent representation [B, latent_channels, H/8, W/8]
    public func encode(_ images: MLXArray) -> MLXArray {
        return vae.encodeImage(images)
    }

    // MARK: - Decoding

    /// Decode latents to image
    /// - Parameter latents: Latent representation [B, latent_channels, H/8, W/8]
    /// - Returns: Decoded image [B, C, H, W] in [-1, 1] range
    public func decode(_ latents: MLXArray) -> MLXArray {
        // Use single-image decode (adds/removes temporal dimension)
        return vae.decodeImage(latents)
    }

    /// Decode packed latents to image
    /// - Parameter packedLatents: Packed latents [B, latent_channels * patch^2, H/8/patch, W/8/patch]
    /// - Returns: Decoded image [B, C, H, W]
    public func unpackAndDecode(_ packedLatents: MLXArray) -> MLXArray {
        let latents = unpackLatents(packedLatents)
        return decode(latents)
    }

    // MARK: - Latent Packing/Unpacking

    /// Pack latents into 2x2 patches for transformer efficiency
    /// [B, C, H, W] -> [B, C*4, H/2, W/2]
    public func packLatents(_ latents: MLXArray) -> MLXArray {
        let b = latents.dim(0)
        let c = latents.dim(1)
        let h = latents.dim(2)
        let w = latents.dim(3)

        // Reshape to extract 2x2 patches
        // [B, C, H, W] -> [B, C, H/2, 2, W/2, 2]
        var x = latents.reshaped(b, c, h / patchSize, patchSize, w / patchSize, patchSize)

        // Permute to group patch elements with channels
        // [B, C, H/2, 2, W/2, 2] -> [B, H/2, W/2, C, 2, 2]
        x = x.transposed(0, 2, 4, 1, 3, 5)

        // Flatten patch into channels
        // [B, H/2, W/2, C, 2, 2] -> [B, H/2, W/2, C*4]
        x = x.reshaped(b, h / patchSize, w / patchSize, c * patchSize * patchSize)

        // Back to BCHW format
        // [B, H/2, W/2, C*4] -> [B, C*4, H/2, W/2]
        return x.transposed(0, 3, 1, 2)
    }

    /// Unpack latents from 2x2 patches back to full resolution
    /// [B, C*4, H/2, W/2] -> [B, C, H, W]
    public func unpackLatents(_ packedLatents: MLXArray) -> MLXArray {
        let b = packedLatents.dim(0)
        let packedC = packedLatents.dim(1)
        let packedH = packedLatents.dim(2)
        let packedW = packedLatents.dim(3)

        let c = packedC / (patchSize * patchSize)
        let h = packedH * patchSize
        let w = packedW * patchSize

        // [B, C*4, H/2, W/2] -> [B, H/2, W/2, C*4]
        var x = packedLatents.transposed(0, 2, 3, 1)

        // Reshape to separate patch elements
        // [B, H/2, W/2, C*4] -> [B, H/2, W/2, C, 2, 2]
        x = x.reshaped(b, packedH, packedW, c, patchSize, patchSize)

        // Permute to restore spatial layout
        // [B, H/2, W/2, C, 2, 2] -> [B, C, H/2, 2, W/2, 2]
        x = x.transposed(0, 3, 1, 4, 2, 5)

        // Flatten back to full resolution
        // [B, C, H/2, 2, W/2, 2] -> [B, C, H, W]
        return x.reshaped(b, c, h, w)
    }

    // MARK: - Utility

    /// Calculate packed latent dimensions for a given image size
    public func packedLatentShape(imageHeight: Int, imageWidth: Int) -> (height: Int, width: Int, channels: Int) {
        let latentH = imageHeight / config.vaeScaleFactor
        let latentW = imageWidth / config.vaeScaleFactor
        let packedH = latentH / patchSize
        let packedW = latentW / patchSize
        let packedC = config.latentChannels * patchSize * patchSize
        return (packedH, packedW, packedC)
    }

    /// Calculate image sequence length (number of patches) for transformer
    public func imageSeqLen(imageHeight: Int, imageWidth: Int) -> Int {
        let (h, w, _) = packedLatentShape(imageHeight: imageHeight, imageWidth: imageWidth)
        return h * w
    }
}

// MARK: - Weight Loading

extension QwenImageEditVAE {
    /// Weight mapper for loading 3D VAE weights from safetensors
    public static func weightMapper(key: String, value: MLXArray) -> [(String, MLXArray)] {
        var mappedKey = key

        // AutoencoderKL3D keeps the public Qwen module keys in their source
        // snake_case form. Only the top-level quant conv slots are camelCase
        // because `quant_conv`/`post_quant_conv` would collide with Swift
        // naming conventions.
        mappedKey = mappedKey
            .replacingOccurrences(of: "post_quant_conv", with: "postQuantConv")
            .replacingOccurrences(of: "quant_conv", with: "quantConv")

        // resample.1.* -> resample.0.* (PyTorch has [Upsample, Conv], we only have Conv at index 0)
        if mappedKey.contains(".resample.1.") {
            mappedKey = mappedKey.replacingOccurrences(of: ".resample.1.", with: ".resample.0.")
        }

        // Handle weight transposition based on dimension
        var mappedValue = value

        if mappedKey.contains(".weight") {
            if value.ndim == 5 {
                // 5D conv weight: [O, I, kD, kH, kW] - keep as-is (transposed in conv3d call)
                // No transpose needed - our CausalConv3d stores in PyTorch format
                mappedValue = value
            } else if value.ndim == 4 {
                // 4D conv weight: [O, I, kH, kW] -> [O, kH, kW, I] for MLX Conv2d
                mappedValue = HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
            }
        }

        return [(mappedKey, mappedValue)]
    }
}
