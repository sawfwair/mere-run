import Foundation
import MLX
import MLXRandom
import MLXNN

extension Flux2KleinGeneratoriOS {

    // MARK: - Decode and Save

    /// Load VAE, decode latents, save image, unload VAE
    func decodeAndSave(
        latents: MLXArray,
        patchedHeight: Int,
        patchedWidth: Int,
        bnMean: MLXArray,
        bnVar: MLXArray,
        seed: UInt64,
        vaeDirURL: URL,
        requestedOutputURL: URL,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> URL {
        // Load VAE
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        let vaeConfig = try loadVAEConfig(from: vaeDirURL)
        var vae: AutoencoderKL? = AutoencoderKL(configuration: vaeConfig)

        let vaeWeightsURL = vaeDirURL.appendingPathComponent("diffusion_pytorch_model.safetensors")
        do {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: vaeWeightsURL,
                to: vae!,
                verify: [.noUnusedKeys, .shapeMismatch],
                mapper: { key, value in
                    if key.hasPrefix("bn.") { return [] }
                    if value.ndim == 4 && key.contains("conv") {
                        return [(key, HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value))]
                    }
                    return [(key, value)]
                }
            )
        }
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 1, totalSteps: 1))

        // Decode
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 0, totalSteps: 1))

        // Reshape to packed format
        let packedLatents = latents
            .reshaped([1, patchedHeight, patchedWidth, 128])
            .transposed(0, 3, 1, 2)

        // Apply BatchNorm denormalization
        let bnEps: Float = 1e-4
        let bnStd = MLX.sqrt(bnVar.reshaped([1, -1, 1, 1]) + bnEps)
        let bnMeanReshaped = bnMean.reshaped([1, -1, 1, 1])
        let denormalizedLatents = packedLatents * bnStd + bnMeanReshaped

        // Unpatchify
        let unpatchedLatents = unpatchifyPackedLatents(
            denormalizedLatents,
            height: patchedHeight,
            width: patchedWidth
        )

        // Decode with VAE
        let decoded = vae!.decode(unpatchedLatents)
        MLX.eval(decoded.0)

        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 1, totalSteps: 1))

        // Save image
        let outputURL = requestedOutputURL
        let outDir = requestedOutputURL.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: outDir.path) else {
            throw Flux2Error.invalidOutputDirectory(requestedOutputURL)
        }

        let (decodedImage, _) = decoded
        var image = QwenImageIO.denormalizeFromDecoder(decodedImage)
        image = MLX.clip(image, min: 0, max: 1)
        MLX.eval(image)

        try QwenImageIO.saveImage(array: image, to: outputURL)

        // Ensure VAE and image ops have completed before releasing weights and cached buffers.
        Stream.gpu.synchronize()

        // Unload VAE
        vae = nil
        clearGPUMemory(synchronize: false)

        return outputURL
    }


}
