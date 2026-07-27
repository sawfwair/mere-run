import Foundation
import MLX

public struct ACEStepFlowEditConfiguration: Codable, Hashable, Sendable {
    public var sourceCaption: String
    public var sourceLyrics: String
    public var nMin: Float
    public var nMax: Float
    public var nAverage: Int
    public var retakeSeed: UInt64?

    public init(
        sourceCaption: String,
        sourceLyrics: String = "",
        nMin: Float = 0,
        nMax: Float = 1,
        nAverage: Int = 1,
        retakeSeed: UInt64? = nil
    ) {
        self.sourceCaption = sourceCaption
        self.sourceLyrics = sourceLyrics
        self.nMin = nMin
        self.nMax = nMax
        self.nAverage = nAverage
        self.retakeSeed = retakeSeed
    }

    public func validate() throws {
        guard !sourceCaption.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty else {
            throw ACEStepFlowEditError.emptySourceCaption
        }
        guard 0 <= nMin, nMin <= nMax, nMax <= 1 else {
            throw ACEStepFlowEditError.invalidWindow(nMin: nMin, nMax: nMax)
        }
        guard nAverage >= 1 else {
            throw ACEStepFlowEditError.invalidAverageCount(nAverage)
        }
    }
}

public enum ACEStepFlowEditError: LocalizedError {
    case emptySourceCaption
    case invalidWindow(nMin: Float, nMax: Float)
    case invalidAverageCount(Int)

    public var errorDescription: String? {
        switch self {
        case .emptySourceCaption:
            return "Flow edit requires a non-empty source caption."
        case .invalidWindow(let nMin, let nMax):
            return "Flow edit requires 0 <= n-min <= n-max <= 1; "
                + "got \(nMin), \(nMax)."
        case .invalidAverageCount(let count):
            return "Flow edit n-average must be at least 1; got \(count)."
        }
    }
}

public enum ACEStepFlowEdit {
    public static func windowIndices(
        stepCount: Int,
        nMin: Float,
        nMax: Float
    ) -> (minimum: Int, maximum: Int) {
        (
            Int(Float(stepCount) * nMin),
            Int(Float(stepCount) * nMax)
        )
    }

    public static func integrateDelta(
        editedLatents: MLXArray,
        sourceVelocity: MLXArray,
        targetVelocity: MLXArray,
        currentTimestep: Float,
        nextTimestep: Float
    ) -> MLXArray {
        editedLatents
            + MLXArray(nextTimestep - currentTimestep)
                .asType(editedLatents.dtype)
                * (targetVelocity - sourceVelocity)
    }
}

extension ACEStepPipeline {
    public func generateFlowEdit(
        targetCaption: String,
        targetLyrics: String,
        sourceLatents25Hz: MLXArray? = nil,
        sourceAudio48kHz: MLXArray? = nil,
        config: ACEStepInferenceConfig = .init(),
        flowEdit: ACEStepFlowEditConfiguration,
        lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata = .init(),
        referenceTimbreLatents25Hz: [MLXArray]? = nil,
        referenceTimbreAudio48kHz: [MLXArray]? = nil,
        vocalLanguage: String = "en"
    ) throws -> MLXArray {
        try flowEdit.validate()
        guard sourceLatents25Hz != nil || sourceAudio48kHz != nil else {
            throw PipelineError.invalidConditioningInput(
                "flow edit requires sourceLatents25Hz or sourceAudio48kHz."
            )
        }
        let targetFrames = max(
            1,
            Int((Double(config.durationSeconds) * 25).rounded())
        )
        let sourceLatents = try normalizeSourceLatents(
            sourceLatents25Hz,
            sourceAudio48kHz: sourceAudio48kHz,
            targetFrames: targetFrames
        )
        let chunkChannels = chunkChannelsForPromptConditioning()
        let sourceInputs = try preparePromptConditionInputs(
            caption: flowEdit.sourceCaption,
            lyrics: flowEdit.sourceLyrics,
            srcLatents: sourceLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: lmUserMetadata,
            referenceTimbreLatents25Hz: referenceTimbreLatents25Hz,
            referenceTimbreAudio48kHz: referenceTimbreAudio48kHz,
            sourceAudio48kHz: sourceAudio48kHz,
            vocalLanguage: vocalLanguage,
            instruction: ACEStepTask.textToMusic.instruction(),
            task: .textToMusic
        )
        let targetInputs = try preparePromptConditionInputs(
            caption: targetCaption,
            lyrics: targetLyrics,
            srcLatents: sourceLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: lmUserMetadata,
            referenceTimbreLatents25Hz: referenceTimbreLatents25Hz,
            referenceTimbreAudio48kHz: referenceTimbreAudio48kHz,
            sourceAudio48kHz: sourceAudio48kHz,
            vocalLanguage: vocalLanguage,
            instruction: ACEStepTask.textToMusic.instruction(),
            task: .textToMusic
        )
        let contextSource = defaultSourceLatents(
            targetFrames: targetFrames,
            batchSize: sourceLatents.dim(0)
        ).asType(sourceLatents.dtype)
        let attentionMask = MLXArray.ones(
            [sourceLatents.dim(0), targetFrames],
            dtype: .int32
        )
        let sourceCondition = prepareFlowEditCondition(
            sourceInputs,
            contextSource: contextSource,
            attentionMask: attentionMask
        )
        let targetCondition = prepareFlowEditCondition(
            targetInputs,
            contextSource: contextSource,
            attentionMask: attentionMask
        )
        let targetLatents = flowEditSamplingLoop(
            sourceLatents: sourceLatents,
            sourceCondition: sourceCondition,
            targetCondition: targetCondition,
            config: config,
            flowEdit: flowEdit
        )
        if config.useTiledVaeDecode {
            return vae.tiledDecode(
                targetLatents,
                chunkSize: config.vaeChunkSize,
                overlap: config.vaeOverlap
            )
        }
        return vae.decode(targetLatents)
    }

    private func prepareFlowEditCondition(
        _ inputs: ACEStepConditionInputs,
        contextSource: MLXArray,
        attentionMask: MLXArray
    ) -> (
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray,
        contextLatents: MLXArray
    ) {
        prepareCondition(
            textHiddenStates: inputs.textHiddenStates,
            textAttentionMask: inputs.textAttentionMask,
            lyricHiddenStates: inputs.lyricHiddenStates,
            lyricAttentionMask: inputs.lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked:
                inputs.referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: inputs.referAudioOrderMask,
            hiddenStates: contextSource,
            attentionMask: attentionMask,
            silenceLatent: inputs.silenceLatent,
            srcLatents: contextSource,
            chunkMasks: inputs.chunkMasks,
            isCovers: MLXArray.zeros(inputs.isCovers.shape, dtype: .int32)
        )
    }

    private func flowEditSamplingLoop(
        sourceLatents: MLXArray,
        sourceCondition: (
            encoderHiddenStates: MLXArray,
            encoderAttentionMask: MLXArray,
            contextLatents: MLXArray
        ),
        targetCondition: (
            encoderHiddenStates: MLXArray,
            encoderAttentionMask: MLXArray,
            contextLatents: MLXArray
        ),
        config: ACEStepInferenceConfig,
        flowEdit: ACEStepFlowEditConfiguration
    ) -> MLXArray {
        var timesteps = inferenceTimesteps(config)
        if timesteps.last != 0 {
            timesteps.append(0)
        }
        let stepCount = max(0, timesteps.count - 1)
        let window = ACEStepFlowEdit.windowIndices(
            stepCount: stepCount,
            nMin: flowEdit.nMin,
            nMax: flowEdit.nMax
        )
        let usesGuidance = !checkpointVariant.isTurbo
            && config.guidanceScale > 1
        let baseNoiseSeed = flowEdit.retakeSeed
            ?? config.retakeSeed
            ?? config.seed
            ?? UInt64.random(in: UInt64.min...UInt64.max)
        var drawIndex: UInt64 = 0
        var editedLatents = sourceLatents
        var targetTrajectory: MLXArray?
        var sourceMomentum: MLXArray?
        var targetMomentum: MLXArray?
        var previousSourceVelocity: MLXArray?
        var previousTargetVelocity: MLXArray?

        func prediction(
            latents: MLXArray,
            timestep: Float,
            condition: (
                encoderHiddenStates: MLXArray,
                encoderAttentionMask: MLXArray,
                contextLatents: MLXArray
            ),
            momentum: inout MLXArray?
        ) -> (velocity: MLXArray, guidanceDifference: MLXArray?) {
            let modelLatents = usesGuidance
                ? MLX.concatenated([latents, latents], axis: 0)
                : latents
            let modelTimestep = MLXArray(
                Array(repeating: timestep, count: modelLatents.dim(0))
            ).asType(.float32)
            let hiddenStates: MLXArray
            let attentionMask: MLXArray
            let context: MLXArray
            if usesGuidance {
                hiddenStates = MLX.concatenated([
                    condition.encoderHiddenStates,
                    MLX.broadcast(
                        nullConditionEmbedding.asType(
                            condition.encoderHiddenStates.dtype
                        ),
                        to: condition.encoderHiddenStates.shape
                    ),
                ], axis: 0)
                attentionMask = MLX.concatenated([
                    condition.encoderAttentionMask,
                    condition.encoderAttentionMask,
                ], axis: 0)
                context = MLX.concatenated([
                    condition.contextLatents,
                    condition.contextLatents,
                ], axis: 0)
            } else {
                hiddenStates = condition.encoderHiddenStates
                attentionMask = condition.encoderAttentionMask
                context = condition.contextLatents
            }
            let raw = decoder(
                hiddenStates: modelLatents,
                timestep: modelTimestep,
                timestepR: modelTimestep,
                encoderHiddenStates: hiddenStates,
                encoderAttentionMask: attentionMask,
                contextLatents: context
            )
            guard usesGuidance else {
                return (raw, nil)
            }
            let branches = MLX.split(raw, parts: 2, axis: 0)
            let conditional = branches[0]
            let unconditional = branches[1]
            guard config.cfgIntervalStart <= timestep,
                  timestep <= config.cfgIntervalEnd
            else {
                return (conditional, conditional - unconditional)
            }
            switch config.guidanceMode {
            case .apg:
                return (
                    ACEStepGuidance.apg(
                        conditional: conditional,
                        unconditional: unconditional,
                        scale: config.guidanceScale,
                        runningAverage: &momentum
                    ),
                    conditional - unconditional
                )
            case .adg, .cfg:
                return (
                    ACEStepGuidance.cfg(
                        conditional: conditional,
                        unconditional: unconditional,
                        scale: config.guidanceScale
                    ),
                    conditional - unconditional
                )
            }
        }

        func stabilize(
            _ velocity: MLXArray,
            latents: MLXArray,
            previous: MLXArray?
        ) -> MLXArray {
            var result = velocity
            if config.velocityNormThreshold > 0 {
                let velocityNorm = MLX.sqrt(
                    (result * result).sum(axes: [1, 2], keepDims: true)
                )
                let latentNorm = MLX.sqrt(
                    (latents * latents).sum(axes: [1, 2], keepDims: true)
                ) + MLXArray(Float(1e-10))
                let factor = MLX.minimum(
                    MLXArray.ones(velocityNorm.shape, dtype: velocityNorm.dtype),
                    MLXArray(config.velocityNormThreshold)
                        * latentNorm / (velocityNorm + MLXArray(Float(1e-10)))
                )
                result = result * factor
            }
            if config.velocityEMAFactor > 0, let previous {
                result = MLXArray(Float(1 - config.velocityEMAFactor)) * result
                    + MLXArray(config.velocityEMAFactor) * previous
            }
            return result
        }

        for step in 0..<stepCount {
            guard step >= window.minimum else {
                continue
            }
            let current = timesteps[step]
            let next = timesteps[step + 1]
            if step < window.maximum {
                let sourceMomentumBefore = sourceMomentum
                let targetMomentumBefore = targetMomentum
                var sourceVelocitySum = MLXArray.zeros(
                    sourceLatents.shape,
                    dtype: sourceLatents.dtype
                )
                var targetVelocitySum = MLXArray.zeros(
                    sourceLatents.shape,
                    dtype: sourceLatents.dtype
                )
                var sourceDifferenceSum: MLXArray?
                var targetDifferenceSum: MLXArray?

                for _ in 0..<flowEdit.nAverage {
                    sourceMomentum = sourceMomentumBefore
                    targetMomentum = targetMomentumBefore
                    let forwardNoise = MLXRandom.normal(
                        sourceLatents.shape,
                        key: MLXRandom.key(baseNoiseSeed &+ drawIndex)
                    ).asType(sourceLatents.dtype)
                    drawIndex &+= 1
                    let noisedSource = MLXArray(Float(1 - current))
                        * sourceLatents
                        + MLXArray(current) * forwardNoise
                    let noisedTarget = editedLatents
                        + noisedSource - sourceLatents
                    let source = prediction(
                        latents: noisedSource,
                        timestep: current,
                        condition: sourceCondition,
                        momentum: &sourceMomentum
                    )
                    let target = prediction(
                        latents: noisedTarget,
                        timestep: current,
                        condition: targetCondition,
                        momentum: &targetMomentum
                    )
                    sourceVelocitySum = sourceVelocitySum
                        + stabilize(
                            source.velocity,
                            latents: noisedSource,
                            previous: nil
                        )
                    targetVelocitySum = targetVelocitySum
                        + stabilize(
                            target.velocity,
                            latents: noisedTarget,
                            previous: nil
                        )
                    if let difference = source.guidanceDifference {
                        sourceDifferenceSum = sourceDifferenceSum.map {
                            $0 + difference
                        } ?? difference
                    }
                    if let difference = target.guidanceDifference {
                        targetDifferenceSum = targetDifferenceSum.map {
                            $0 + difference
                        } ?? difference
                    }
                }
                let divisor = MLXArray(Float(flowEdit.nAverage))
                if config.guidanceMode == .apg, usesGuidance {
                    sourceMomentum = advancedFlowEditMomentum(
                        previous: sourceMomentumBefore,
                        differenceSum: sourceDifferenceSum,
                        divisor: divisor
                    )
                    targetMomentum = advancedFlowEditMomentum(
                        previous: targetMomentumBefore,
                        differenceSum: targetDifferenceSum,
                        divisor: divisor
                    )
                }
                let sourceVelocity = stabilize(
                    sourceVelocitySum / divisor,
                    latents: sourceLatents,
                    previous: previousSourceVelocity
                )
                let targetVelocity = stabilize(
                    targetVelocitySum / divisor,
                    latents: editedLatents,
                    previous: previousTargetVelocity
                )
                previousSourceVelocity = sourceVelocity
                previousTargetVelocity = targetVelocity
                editedLatents = ACEStepFlowEdit.integrateDelta(
                    editedLatents: editedLatents,
                    sourceVelocity: sourceVelocity,
                    targetVelocity: targetVelocity,
                    currentTimestep: current,
                    nextTimestep: next
                )
            } else {
                if targetTrajectory == nil {
                    let forwardNoise = MLXRandom.normal(
                        sourceLatents.shape,
                        key: MLXRandom.key(baseNoiseSeed &+ drawIndex)
                    ).asType(sourceLatents.dtype)
                    drawIndex &+= 1
                    let noisedSource = MLXArray(Float(1 - current))
                        * sourceLatents
                        + MLXArray(current) * forwardNoise
                    targetTrajectory = editedLatents
                        + noisedSource - sourceLatents
                }
                var trajectory = targetTrajectory!
                var target = prediction(
                    latents: trajectory,
                    timestep: current,
                    condition: targetCondition,
                    momentum: &targetMomentum
                ).velocity
                target = stabilize(
                    target,
                    latents: trajectory,
                    previous: previousTargetVelocity
                )
                previousTargetVelocity = target
                trajectory = trajectory
                    + MLXArray(next - current) * target
                targetTrajectory = trajectory
            }
            MLX.eval(editedLatents, targetTrajectory ?? editedLatents)
        }
        return targetTrajectory ?? editedLatents
    }

    private func advancedFlowEditMomentum(
        previous: MLXArray?,
        differenceSum: MLXArray?,
        divisor: MLXArray
    ) -> MLXArray? {
        guard let differenceSum else {
            return previous
        }
        var momentum = previous
        let averageDifference = differenceSum / divisor
        _ = ACEStepGuidance.apg(
            conditional: averageDifference,
            unconditional: MLXArray.zeros(
                averageDifference.shape,
                dtype: averageDifference.dtype
            ),
            scale: 1,
            runningAverage: &momentum
        )
        return momentum
    }
}
