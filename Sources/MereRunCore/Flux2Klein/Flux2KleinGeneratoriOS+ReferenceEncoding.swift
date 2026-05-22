import Foundation
import MediaIO
import MLX
import MLXRandom
import MLXNN

extension Flux2KleinGeneratoriOS {

    // MARK: - Reference Image Encoding

    /// Load VAE, encode reference images, unload VAE
    func encodeReferenceImages(
        urls: [URL],
        width: Int,
        height: Int,
        patchedHeight: Int,
        patchedWidth: Int,
        bnMean: MLXArray,
        bnVar: MLXArray,
        referenceStrength: Float,
        seed: UInt64,
        vaeDirURL: URL,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> [MLXArray] {
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

        // Encode each reference image
        var referenceLatents: [MLXArray] = []
        for (i, refURL) in urls.enumerated() {
            progressHandler?(GenerationProgress(stage: .encodingReferenceImages, stepIndex: i + 1, totalSteps: urls.count))

            var refLatent = try encodeReferenceImage(
                refURL,
                vae: vae!,
                width: width,
                height: height,
                patchedHeight: patchedHeight,
                patchedWidth: patchedWidth,
                bnMean: bnMean,
                bnVar: bnVar
            )

            // Add noise based on referenceStrength
            if referenceStrength > 0 {
                let refNoise = MLXRandom.normal(refLatent.shape, key: MLXRandom.key(seed &+ UInt64(i + 1))).asType(refLatent.dtype)
                refLatent = (1.0 - referenceStrength) * refLatent + referenceStrength * refNoise
            }

            MLX.eval(refLatent)
            referenceLatents.append(refLatent)
        }

        progressHandler?(GenerationProgress(stage: .encodingReferenceImages, stepIndex: urls.count, totalSteps: urls.count))

        // Ensure VAE work has completed before releasing weights.
        Stream.gpu.synchronize()

        // Unload VAE (don't clear cache - reference latents still needed)
        vae = nil

        // Yield to give system time to reclaim memory
        await Task.yield()

        return referenceLatents
    }

    /// Encode a single reference image to patchified latent space
    private func encodeReferenceImage(
        _ url: URL,
        vae: AutoencoderKL,
        width: Int,
        height: Int,
        patchedHeight: Int,
        patchedWidth: Int,
        bnMean: MLXArray,
        bnVar: MLXArray
    ) throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Flux2Error.referenceImageNotFound(url)
        }

        let image: MediaImage
        do {
            image = try MediaImageIO.decode(url)
        } catch {
            throw Flux2Error.referenceImageDecodeFailed(url)
        }

        // Load and resize image
        let resizedArray = try QwenImageIO.resizedPixelArray(
            from: image,
            width: width,
            height: height,
            addBatchDimension: true,
            dtype: .float32
        )

        // Normalize and encode
        let normalized = QwenImageIO.normalizeForEncoder(resizedArray)
        let encoded = vae.encode(normalized)

        // Extract mean (clean latent)
        let latentChannels = vae.configuration.latentChannels
        let mean = encoded[0..., 0..<latentChannels, 0..., 0...]
        let cleanLatent = (mean - MLXArray(vae.configuration.shiftFactor)) * MLXArray(vae.configuration.scalingFactor)

        // Patchify
        let patchified = patchifyLatents(cleanLatent, height: patchedHeight * 2, width: patchedWidth * 2)

        // Apply BatchNorm normalization
        let bnEps: Float = 1e-4
        let bnStd = MLX.sqrt(bnVar.reshaped([1, -1, 1, 1]) + bnEps)
        let bnMeanReshaped = bnMean.reshaped([1, -1, 1, 1])
        let normalizedPacked = (patchified - bnMeanReshaped) / bnStd

        // Reshape to sequence format
        let seqLatent = normalizedPacked
            .transposed(0, 2, 3, 1)
            .reshaped([1, patchedHeight * patchedWidth, 128])
            .asType(.bfloat16)

        return seqLatent
    }


}
