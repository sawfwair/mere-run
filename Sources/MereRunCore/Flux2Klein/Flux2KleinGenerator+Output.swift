import Foundation
import MediaIO
import MLX
import MLXNN

extension Flux2KleinGenerator {
    func decodeAndSave(
        latents: MLXArray,
        vae: AutoencoderKL,
        numRefs: Int,
        seqLen: Int,
        patchedHeight: Int,
        patchedWidth: Int,
        bnMean: MLXArray,
        bnVar: MLXArray,
        outputURL: URL,
        debugLog: ((String) -> Void)?,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        // 6. Extract only the noise latent (generated image) for decoding
        // If multi-reference, we take the last seqLen tokens
        let finalNoiseLatent: MLXArray
        if numRefs > 0 {
            finalNoiseLatent = latents[0..., (numRefs * seqLen)..., 0...]
        } else {
            finalNoiseLatent = latents
        }

        // 6a. Reshape to packed format: [batch, seq, 128] -> [batch, 128, height, width]
        let packedLatents = finalNoiseLatent
            .reshaped([1, patchedHeight, patchedWidth, 128])
            .transposed(0, 3, 1, 2)

        // 7. Apply BatchNorm inverse transform (denormalization)
        // mflux: latents = packed * sqrt(var + eps) + mean
        // mflux Flux2BatchNormStats uses eps = 0.0001 (1e-4), NOT PyTorch default 1e-5
        let denormalizedLatents = Flux2KleinBatchNorm.denormalizePackedLatents(
            packedLatents,
            mean: bnMean,
            variance: bnVar
        )

        if let debugLog {
            let bnStd = Flux2KleinBatchNorm.std(mean: bnMean, variance: bnVar)
            debugLog("=== BatchNorm Denormalization ===")
            debugLog("BN mean range: [\(bnMean.min().item(Float.self)), \(bnMean.max().item(Float.self))]")
            debugLog("BN std range: [\(bnStd.min().item(Float.self)), \(bnStd.max().item(Float.self))]")
            debugLog("Packed latents: mean=\(packedLatents.mean().item(Float.self)), std=\(sqrt((packedLatents * packedLatents).mean().item(Float.self)))")
            debugLog("Denormalized: mean=\(denormalizedLatents.mean().item(Float.self)), std=\(sqrt((denormalizedLatents * denormalizedLatents).mean().item(Float.self)))")
        }

        // 8. Unpatchify latents: [batch, 128, H, W] -> [batch, 32, H*2, W*2]
        let unpatchedLatents = Flux2LatentPacking.unpatchifyPackedLatents(
            denormalizedLatents,
            height: patchedHeight,
            width: patchedWidth
        )

        if let debugLog {
            debugLog("Unpatchified: shape=\(unpatchedLatents.shape), mean=\(unpatchedLatents.mean().item(Float.self)), std=\(sqrt((unpatchedLatents * unpatchedLatents).mean().item(Float.self)))")
        }

        // 9. Decode with VAE
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 0, totalSteps: 1))
        let decoded = vae.decode(unpatchedLatents)
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 1, totalSteps: 1))

        if let debugLog {
            let (decodedImg, _) = decoded
            debugLog("Decoded: shape=\(decodedImg.shape), mean=\(decodedImg.mean().item(Float.self)), std=\(sqrt((decodedImg * decodedImg).mean().item(Float.self))), min=\(decodedImg.min().item(Float.self)), max=\(decodedImg.max().item(Float.self))")
        }

        // 10. Save image with quality enhancement

        let outDir = outputURL.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: outDir.path) else {
            throw Flux2Error.invalidOutputDirectory(outputURL)
        }

        let (decodedImage, _) = decoded
        var image = QwenImageIO.denormalizeFromDecoder(decodedImage)

        // Enhance contrast and detail (subtle boost for better faces)
        // Apply mild contrast curve: 0.5 + (x - 0.5) * 1.1
        let contrastFactor: Float = 1.1
        image = 0.5 + (image - 0.5) * contrastFactor

        image = MLX.clip(image, min: 0, max: 1)
        try QwenImageIO.saveImage(array: image, to: outputURL)

    }
}
