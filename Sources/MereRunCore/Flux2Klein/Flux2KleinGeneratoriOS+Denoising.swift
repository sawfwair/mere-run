import Foundation
import MLX
import MLXRandom
import MLXNN

extension Flux2KleinGeneratoriOS {

    // MARK: - Denoising

    /// Load transformer, run denoising steps, save latents to disk, unload transformer
    func denoiseAndSave(
        promptEmbeds: MLXArray,
        negativePromptEmbeds: MLXArray?,
        referenceLatents: [MLXArray],
        patchedHeight: Int,
        patchedWidth: Int,
        seqLen: Int,
        steps: Int,
        seed: UInt64,
        guidanceScale: Float,
        transformerDirURL: URL,
        transformerQuantization: ModelWeightsLoader.QuantizationParams?,
        isDistilled: Bool,
        sigmaShift: Float?,
        outputURL: URL,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws {
        let timingEnabled = {
            guard let raw = ProcessInfo.processInfo.environment["MERERUN_FLUX2_TIMING"]?.lowercased() else { return false }
            return raw == "1" || raw == "true" || raw == "yes"
        }()

        let numRefs = referenceLatents.count

        // Load transformer
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))

        // Maximize available memory before loading large transformer weights
        Stream.gpu.synchronize()
        Memory.clearCache()

        let transformerConfig = try loadTransformerConfig(from: transformerDirURL)
        var transformer: Flux2Transformer2DModel? = try loadTransformer(
            config: transformerConfig,
            from: transformerDirURL,
            quantization: transformerQuantization
        )
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 1, totalSteps: 1))

        // Prepare initial latents
        let randomKey = MLXRandom.key(seed)
        let noiseLatent = MLXRandom.normal([1, seqLen, 128], key: randomKey).asType(.bfloat16)

        var latents: MLXArray
        if referenceLatents.isEmpty {
            latents = noiseLatent
        } else {
            let allLatents = referenceLatents + [noiseLatent]
            latents = MLX.concatenated(allLatents, axis: 1)
        }

        // Prepare position IDs
        let txtIds = Flux2PosEmbed.prepareTextIds(seqLen: promptEmbeds.shape[1], numAxes: 4)

        let totalImages = numRefs + 1
        var tCoords = (0..<numRefs).map { 10 + 10 * $0 }
        tCoords.append(0)

        let imgIds = Flux2PosEmbed.prepareMultiImageIds(
            imageCount: totalImages,
            height: patchedHeight,
            width: patchedWidth,
            tCoords: tCoords
        )

        // Setup scheduler (driven by manifest variant; no heuristics).
        let scheduler = Flux2EulerScheduler(
            numInferenceSteps: steps,
            numTrainTimesteps: 1000,
            imageSeqLen: seqLen,
            isDistilled: isDistilled,
            sigmaShift: sigmaShift
        )

        let useCFG = guidanceScale > 1.0 && negativePromptEmbeds != nil
        let negTxtIds: MLXArray? = {
            guard useCFG, let negEmbeds = negativePromptEmbeds else { return nil }
            return Flux2PosEmbed.prepareTextIds(seqLen: negEmbeds.shape[1], numAxes: 4)
        }()

        // Denoising loop
        let denoiseStart = timingEnabled ? CFAbsoluteTimeGetCurrent() : 0
        for step in 0..<steps {
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: step, totalSteps: steps))

            let timestepForModel = scheduler.timestep(at: step) / 1000.0
            let timestepTensor = timestepForModel.expandedDimensions(axis: 0)

            let noisePred: MLXArray
            if useCFG, let negEmbeds = negativePromptEmbeds {
                // Compute unconditional noise and eval immediately
                let uncondNoise = transformer!(
                    hiddenStates: latents,
                    encoderHiddenStates: negEmbeds,
                    timestep: timestepTensor,
                    imgIds: imgIds,
                    txtIds: negTxtIds ?? Flux2PosEmbed.prepareTextIds(seqLen: negEmbeds.shape[1], numAxes: 4)
                )
                MLX.eval(uncondNoise)

                // Clear intermediate computations before second pass
                Memory.clearCache()

                // Compute conditional noise
                let condNoise = transformer!(
                    hiddenStates: latents,
                    encoderHiddenStates: promptEmbeds,
                    timestep: timestepTensor,
                    imgIds: imgIds,
                    txtIds: txtIds
                )
                MLX.eval(condNoise)

                noisePred = uncondNoise + guidanceScale * (condNoise - uncondNoise)
                MLX.eval(noisePred)
            } else {
                noisePred = transformer!(
                    hiddenStates: latents,
                    encoderHiddenStates: promptEmbeds,
                    timestep: timestepTensor,
                    imgIds: imgIds,
                    txtIds: txtIds
                )
                MLX.eval(noisePred)
            }

            // Euler step
            if numRefs > 0 {
                let noiseOnlyPred = noisePred[0..., (numRefs * seqLen)..., 0...]
                let currentNoiseLatent = latents[0..., (numRefs * seqLen)..., 0...]
                let updatedNoiseLatent = scheduler.step(modelOutput: noiseOnlyPred, timestepIndex: step, sample: currentNoiseLatent)
                let refLatentsPart = latents[0..., 0..<(numRefs * seqLen), 0...]
                latents = MLX.concatenated([refLatentsPart, updatedNoiseLatent], axis: 1)
            } else {
                latents = scheduler.step(modelOutput: noisePred, timestepIndex: step, sample: latents)
            }
            MLX.eval(latents)

            // Clear cached intermediates after each step
            Memory.clearCache()
        }

        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: steps, totalSteps: steps))
        if timingEnabled {
            let denoiseEnd = CFAbsoluteTimeGetCurrent()
            let total = denoiseEnd - denoiseStart
            let perStepMs = (total / Double(max(1, steps))) * 1000.0
            let message = String(format: "[Flux2KleinGeneratoriOS] denoise_time_s=%.3f steps=%d step_ms=%.3f\n", total, steps, perStepMs)
            FileHandle.standardError.write(Data(message.utf8))
        }

        // Extract final latent (generated image only)
        let finalLatent: MLXArray
        if numRefs > 0 {
            finalLatent = latents[0..., (numRefs * seqLen)..., 0...]
        } else {
            finalLatent = latents
        }
        MLX.eval(finalLatent)

        // Ensure transformer work has completed before releasing weights.
        Stream.gpu.synchronize()

        // Save latents to disk before unloading transformer
        try MLX.save(array: finalLatent, url: outputURL)

        // Unload transformer
        transformer = nil
    }


}
