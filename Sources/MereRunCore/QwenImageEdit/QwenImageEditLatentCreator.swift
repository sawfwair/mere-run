import MLX
import MLXRandom

/// Utility for creating and manipulating latents for Qwen-Image-Edit
public struct QwenImageEditLatentCreator {

    /// Create random noise latents for generation
    /// - Parameters:
    ///   - batchSize: Number of samples
    ///   - height: Image height (will be divided by VAE scale factor)
    ///   - width: Image width (will be divided by VAE scale factor)
    ///   - latentChannels: Number of latent channels (typically 16)
    ///   - vaeScaleFactor: VAE downscaling factor (typically 8)
    ///   - seed: Optional seed for reproducibility
    /// - Returns: Random latents [B, C, H/scale, W/scale]
    public static func createNoise(
        batchSize: Int = 1,
        height: Int,
        width: Int,
        latentChannels: Int = 16,
        vaeScaleFactor: Int = 8,
        seed: UInt64? = nil
    ) -> MLXArray {
        let latentH = height / vaeScaleFactor
        let latentW = width / vaeScaleFactor
        let shape = [batchSize, latentChannels, latentH, latentW]

        if let seed = seed {
            let key = MLXRandom.key(seed)
            return MLXRandom.normal(shape, key: key).asType(.bfloat16)
        } else {
            return MLXRandom.normal(shape).asType(.bfloat16)
        }
    }

    /// Create packed noise latents (for transformer input)
    /// - Parameters:
    ///   - batchSize: Number of samples
    ///   - height: Image height
    ///   - width: Image width
    ///   - latentChannels: Number of latent channels
    ///   - vaeScaleFactor: VAE downscaling factor
    ///   - patchSize: Latent packing patch size (typically 2)
    ///   - seed: Optional seed
    /// - Returns: Packed noise [B, C*patch^2, H/scale/patch, W/scale/patch]
    public static func createPackedNoise(
        batchSize: Int = 1,
        height: Int,
        width: Int,
        latentChannels: Int = 16,
        vaeScaleFactor: Int = 8,
        patchSize: Int = 2,
        seed: UInt64? = nil
    ) -> MLXArray {
        let latentH = height / vaeScaleFactor
        let latentW = width / vaeScaleFactor
        let packedH = latentH / patchSize
        let packedW = latentW / patchSize
        let packedC = latentChannels * patchSize * patchSize
        let shape = [batchSize, packedC, packedH, packedW]

        if let seed = seed {
            let key = MLXRandom.key(seed)
            return MLXRandom.normal(shape, key: key).asType(.bfloat16)
        } else {
            return MLXRandom.normal(shape).asType(.bfloat16)
        }
    }

    /// Blend source latents with noise for img2img editing
    /// - Parameters:
    ///   - sourceLatents: Encoded source image latents
    ///   - noise: Random noise
    ///   - sigma: Current noise level (from scheduler)
    /// - Returns: Blended latents
    public static func blendWithNoise(
        sourceLatents: MLXArray,
        noise: MLXArray,
        sigma: MLXArray
    ) -> MLXArray {
        // For flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
        let oneMinusSigma = MLXArray(Float(1.0)) - sigma
        return oneMinusSigma * sourceLatents + sigma * noise
    }

    /// Scale latents by initial sigma for scheduler
    /// - Parameters:
    ///   - latents: Input latents
    ///   - sigma: Initial sigma from scheduler
    /// - Returns: Scaled latents
    public static func scaleByInitialSigma(
        latents: MLXArray,
        sigma: MLXArray
    ) -> MLXArray {
        return latents * sigma
    }
}
