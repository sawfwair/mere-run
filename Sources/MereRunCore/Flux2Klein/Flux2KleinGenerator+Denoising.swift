import Foundation
import MLX

extension Flux2KleinGenerator {
    func denoise(
        latents initialLatents: MLXArray,
        transformer: Flux2Transformer2DModel,
        scheduler: Flux2EulerScheduler,
        steps: Int,
        guidanceScale: Float,
        promptEmbeds: MLXArray,
        negativePromptEmbeds: MLXArray?,
        txtIds: MLXArray,
        negTxtIds: MLXArray?,
        imgIds: MLXArray,
        numRefs: Int,
        seqLen: Int,
        useCFG: Bool,
        useBatchedCFG: Bool,
        usesEmbeddedGuidance: Bool,
        timingEnabled: Bool,
        debugLog: ((String) -> Void)?,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> MLXArray {
        var latents = initialLatents
        // 5. Denoising loop
        let denoiseStart = timingEnabled ? CFAbsoluteTimeGetCurrent() : 0
        for step in 0..<steps {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: step, totalSteps: steps))

            // mflux passes raw timestep (sigma * 1000), transformer handles scaling conditionally
            let timestepTensor = scheduler.timestep(at: step).expandedDimensions(axis: 0)

            // Run transformer (with CFG if enabled)
            let noisePred: MLXArray
            if useCFG, let negEmbeds = negativePromptEmbeds {
                if useBatchedCFG {
                    let predictions = transformerForward(
                        transformer,
                        hiddenStates: DiffusionCFGExecution.duplicateBatch(latents),
                        encoderHiddenStates: DiffusionCFGExecution.paired(negEmbeds, promptEmbeds),
                        timestep: DiffusionCFGExecution.duplicateBatch(timestepTensor),
                        imgIds: imgIds,
                        txtIds: txtIds
                    )
                    noisePred = DiffusionCFGExecution.combinePredictions(
                        predictions,
                        guidanceScale: guidanceScale
                    )
                } else {
                    // Unconditional pass (FLUX.2 Klein doesn't use guidance parameter)
                    let uncondNoise = transformerForward(
                        transformer,
                        hiddenStates: latents,
                        encoderHiddenStates: negEmbeds,
                        timestep: timestepTensor,
                        imgIds: imgIds,
                        txtIds: negTxtIds ?? Flux2PosEmbed.prepareTextIds(seqLen: negEmbeds.shape[1], numAxes: 4)
                    )

                    // Conditional pass (FLUX.2 Klein doesn't use guidance parameter)
                    let condNoise = transformerForward(
                        transformer,
                        hiddenStates: latents,
                        encoderHiddenStates: promptEmbeds,
                        timestep: timestepTensor,
                        imgIds: imgIds,
                        txtIds: txtIds
                    )

                    // CFG: output = uncond + guidance_scale * (cond - uncond)
                    let g = MLXArray(guidanceScale)
                    noisePred = uncondNoise + g * (condNoise - uncondNoise)
                }
            } else {
                // No CFG - single pass (FLUX.2 Klein doesn't use guidance parameter)
                noisePred = transformerForward(
                    transformer,
                    hiddenStates: latents,
                    encoderHiddenStates: promptEmbeds,
                    timestep: timestepTensor,
                    imgIds: imgIds,
                    txtIds: txtIds,
                    guidance: usesEmbeddedGuidance ? MLXArray([guidanceScale]) : nil
                )
            }

            if let debugLog {
                // Debug-only: the sigma readback lived outside this guard and
                // forced a GPU sync every denoise step in production runs.
                let sigmaValue = scheduler.sigma(at: step).item(Float.self)
                MLX.eval(noisePred)
                let npMean = noisePred.mean().item(Float.self)
                let npStd = sqrt((noisePred * noisePred).mean().item(Float.self))
                let npMin = noisePred.min().item(Float.self)
                let npMax = noisePred.max().item(Float.self)
                let timestepVal = timestepTensor.item(Float.self)
                debugLog("Step \(step): sigma=\(sigmaValue), timestep=\(timestepVal)")
                debugLog("  noise_pred: mean=\(npMean), std=\(npStd), min=\(npMin), max=\(npMax)")
            }

            // Euler step: only update the noise portion (last seqLen tokens)
            // Reference latents stay fixed - they're conditioning context
            if numRefs > 0 {
                // Extract only noise prediction for generated image (last seqLen tokens)
                let noiseOnlyPred = noisePred[0..., (numRefs * seqLen)..., 0...]
                // Extract current noise latent
                let currentNoiseLatent = latents[0..., (numRefs * seqLen)..., 0...]
                // Update only noise latent
                let updatedNoiseLatent = scheduler.step(modelOutput: noiseOnlyPred, timestepIndex: step, sample: currentNoiseLatent)
                // Recompose: keep reference latents fixed, update noise latent
                let refLatentsPart = latents[0..., 0..<(numRefs * seqLen), 0...]
                latents = MLX.concatenated([refLatentsPart, updatedNoiseLatent], axis: 1)
            } else {
                latents = scheduler.step(modelOutput: noisePred, timestepIndex: step, sample: latents)
            }
            if !Self.compileEnabled {
                MLX.eval(latents)
            }

            if let debugLog {
                let latMean = latents.mean().item(Float.self)
                let latStd = sqrt((latents * latents).mean().item(Float.self))
                debugLog("  latents: mean=\(latMean), std=\(latStd)")
            }
        }

        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: steps, totalSteps: steps))
        if timingEnabled {
            let denoiseEnd = CFAbsoluteTimeGetCurrent()
            let total = denoiseEnd - denoiseStart
            let perStepMs = (total / Double(max(1, steps))) * 1000.0
            let message = String(format: "[Flux2KleinGenerator] denoise_time_s=%.3f steps=%d step_ms=%.3f\n", total, steps, perStepMs)
            FileHandle.standardError.write(Data(message.utf8))
        }

        return latents
    }
}
