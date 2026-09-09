import Foundation
import MLX

extension ZImageTurboGenerator {
    func denoise(
        latents initialLatents: MLXArray,
        transformer: ZImageTransformer2DModel,
        scheduler: ZImageTurboLinearScheduler,
        inferenceConfig: ZImageTurboInferenceConfig,
        startStep: Int,
        promptEmbeds: MLXArray,
        negativePromptEmbeds: MLXArray?,
        cfgScale: Float,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> MLXArray {
        var latents = initialLatents
        let machine = MereRunMachineProfile.current
        let useBatchedCFG = negativePromptEmbeds.map { negativeEmbeds in
            DiffusionCFGExecution.canPair(negativeEmbeds, promptEmbeds)
                && DiffusionCFGExecution.shouldBatch(
                    mode: DiffusionCFGExecutionMode.current(
                        modelEnvironmentKey: "MERERUN_ZIMAGE_BATCHED_CFG"
                    ),
                    width: inferenceConfig.width,
                    height: inferenceConfig.height,
                    physicalMemoryBytes: machine.physicalMemoryBytes,
                    activeMemoryBytes: Memory.activeMemory,
                    cacheMemoryBytes: Memory.cacheMemory,
                    isUnifiedMemory: machine.isAppleSiliconMac,
                    baseReserveBytes: 6 * DiffusionCFGExecution.gibibyte,
                    activationBytesPerPixel: 4_096
                )
        } ?? false

        for stepIndex in startStep..<inferenceConfig.numInferenceSteps {
            try Task.checkCancellation()
            transformer.beginDenoisingStep(
                index: stepIndex,
                count: inferenceConfig.numInferenceSteps
            )
            progressHandler?(GenerationProgress(
                stage: .denoising,
                stepIndex: stepIndex,
                totalSteps: inferenceConfig.numInferenceSteps
            ))

            let tInput = (MLXArray([Float(1.0)]) - scheduler.sigmas[stepIndex].asType(.float32)).asType(.float32)
            let latentsModelInput = latents.asType(.bfloat16)

            let noisePred: MLXArray
            if let negativePromptEmbeds, cfgScale > 1 {
                if useBatchedCFG {
                    let predictions = transformer.forward(
                        latents: DiffusionCFGExecution.duplicateBatch(latentsModelInput),
                        timestep: DiffusionCFGExecution.duplicateBatch(tInput),
                        promptEmbeds: DiffusionCFGExecution.paired(negativePromptEmbeds, promptEmbeds)
                    )
                    noisePred = DiffusionCFGExecution.combinePositiveAnchoredPredictions(
                        -predictions.asType(.float32),
                        guidanceScale: cfgScale
                    )
                } else {
                    let predictedPos = transformer.forward(
                        latents: latentsModelInput,
                        timestep: tInput,
                        promptEmbeds: promptEmbeds
                    )
                    let noisePos = (-predictedPos).asType(.float32)
                    let predictedNeg = transformer.forward(
                        latents: latentsModelInput,
                        timestep: tInput,
                        promptEmbeds: negativePromptEmbeds
                    )
                    let noiseNeg = (-predictedNeg).asType(.float32)
                    noisePred = noisePos + (noisePos - noiseNeg) * MLXArray(cfgScale)
                }
            } else {
                let predictedPos = transformer.forward(
                    latents: latentsModelInput,
                    timestep: tInput,
                    promptEmbeds: promptEmbeds
                )
                noisePred = (-predictedPos).asType(.float32)
            }

            latents = scheduler.step(noise: noisePred, timestep: stepIndex, latents: latents)
            MLX.eval(latents)
        }

        return latents
    }
}
