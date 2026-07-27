import Foundation
import MLX

// Owns LM fallback, condition assembly, and denoising helpers for
// ACEStepPipeline. Public entrypoints stay in the main pipeline file.

extension ACEStepPipeline {
    static func blendRetakeNoise(
        primary: MLXArray,
        retake: MLXArray,
        variance: Float
    ) -> MLXArray {
        let clamped = min(max(variance, 0), 1)
        if clamped == 0 {
            return primary
        }
        if clamped == 1 {
            return retake
        }
        let radians = clamped * .pi / 2
        return MLXArray(cos(radians)).asType(primary.dtype) * primary
            + MLXArray(sin(radians)).asType(primary.dtype) * retake
    }

    func prepareInitialNoise(
        shape: [Int],
        config: ACEStepInferenceConfig
    ) -> MLXArray {
        let primary: MLXArray
        if let seed = config.seed {
            primary = MLXRandom.normal(shape, key: MLXRandom.key(seed))
                .asType(.float32)
        } else {
            primary = MLXRandom.normal(shape).asType(.float32)
        }
        guard config.retakeVariance > 0 else {
            return primary
        }
        let retake: MLXArray
        if let retakeSeed = config.retakeSeed {
            retake = MLXRandom.normal(shape, key: MLXRandom.key(retakeSeed))
                .asType(.float32)
        } else {
            retake = MLXRandom.normal(shape).asType(.float32)
        }
        return Self.blendRetakeNoise(
            primary: primary,
            retake: retake,
            variance: config.retakeVariance
        )
    }

    func decodeAndApplyRepaintSplice(
        latents: MLXArray,
        conditionInputs: ACEStepConditionInputs,
        config: ACEStepInferenceConfig
    ) throws -> MLXArray {
        let decoded: MLXArray
        if config.useTiledVaeDecode {
            decoded = vae.tiledDecode(
                latents,
                chunkSize: config.vaeChunkSize,
                overlap: config.vaeOverlap
            )
        } else {
            decoded = vae.decode(latents)
        }

        guard let repaint = conditionInputs.repaintConfiguration,
              let sourceAudio = conditionInputs.sourceAudio48kHz
        else {
            return decoded
        }
        let normalizedSource = try normalizeReferenceTimbreAudio(sourceAudio)
        return ACEStepRepaint.spliceWaveform(
            generatedAudio: decoded,
            sourceAudio: normalizedSource,
            startSeconds: repaint.startSeconds,
            endSeconds: repaint.endSeconds,
            crossfadeSeconds: repaint.waveformCrossfadeSeconds
        )
    }

    func generateAudioCodesWithFallback(
        lm: ACEStep5HzLM,
        caption: String,
        lyrics: String,
        instruction: String,
        lmSystemInstruction: String,
        targetCodes: Int,
        targetDurationSeconds: Float,
        lmConfig: ACEStep5HzLMGenerationConfig,
        lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata
    ) throws -> ACEStep5HzLMResult {
        let primary = runConstrainedLM(
            lm: lm,
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            lmSystemInstruction: lmSystemInstruction,
            lmConfig: lmConfig,
            targetDurationSeconds: targetDurationSeconds,
            userMetadata: lmUserMetadata
        )
        if primary.audioCodeValues.count >= targetCodes {
            return primary
        }

        var retryConfig = lmConfig
        retryConfig.maxNewTokens = max(retryConfig.maxNewTokens, targetCodes * 8)

        let normalizedDuration = max(10, Int(ceil(Double(targetDurationSeconds))))
        let retryMetadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: lmUserMetadata.bpm ?? "120",
            caption: lmUserMetadata.caption,
            duration: lmUserMetadata.duration ?? String(normalizedDuration),
            keyscale: lmUserMetadata.keyscale ?? "C major",
            language: lmUserMetadata.language,
            timesignature: lmUserMetadata.timesignature ?? "4"
        )

        let retry = runConstrainedLM(
            lm: lm,
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            lmSystemInstruction: lmSystemInstruction,
            lmConfig: retryConfig,
            targetDurationSeconds: targetDurationSeconds,
            userMetadata: retryMetadata
        )
        return retry.audioCodeValues.count >= primary.audioCodeValues.count ? retry : primary
    }

    func runConstrainedLM(
        lm: ACEStep5HzLM,
        caption: String,
        lyrics: String,
        instruction: String,
        lmSystemInstruction: String,
        lmConfig: ACEStep5HzLMGenerationConfig,
        targetDurationSeconds: Float,
        userMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata
    ) -> ACEStep5HzLMResult {
        let sampler = lm.makeConstrainedSampler(
            enabled: true,
            skipCaption: true,
            skipLanguage: true,
            targetDurationSeconds: targetDurationSeconds,
            userMetadata: userMetadata
        )
        return lm.generateConstrained(
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            systemInstruction: lmSystemInstruction,
            config: lmConfig,
            sampler: sampler
        )
    }

    func prepareNonCoverConditionIfNeeded(
        conditionInputs: ACEStepConditionInputs
    ) -> (encoderHiddenStates: MLXArray, encoderAttentionMask: MLXArray, contextLatents: MLXArray)? {
        guard
            let nonCoverTextHiddenStates = conditionInputs.nonCoverTextHiddenStates,
            let nonCoverTextAttentionMask = conditionInputs.nonCoverTextAttentionMask
        else {
            return nil
        }

        let B = conditionInputs.srcLatents.dim(0)
        let T = conditionInputs.srcLatents.dim(1)

        let nonCoverSrcLatents = defaultSourceLatents(targetFrames: T, batchSize: B)
            .asType(conditionInputs.srcLatents.dtype)
        let nonCoverAttentionMask = conditionInputs.attentionMask ?? MLXArray.ones([B, T], dtype: .int32)
        let nonCoverIsCovers = MLXArray.zeros(conditionInputs.isCovers.shape, dtype: conditionInputs.isCovers.dtype)

        return prepareCondition(
            textHiddenStates: nonCoverTextHiddenStates,
            textAttentionMask: nonCoverTextAttentionMask,
            lyricHiddenStates: conditionInputs.lyricHiddenStates,
            lyricAttentionMask: conditionInputs.lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked: conditionInputs.referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: conditionInputs.referAudioOrderMask,
            hiddenStates: nonCoverSrcLatents,
            attentionMask: nonCoverAttentionMask,
            silenceLatent: conditionInputs.silenceLatent,
            srcLatents: nonCoverSrcLatents,
            chunkMasks: conditionInputs.chunkMasks,
            isCovers: nonCoverIsCovers
        )
    }

    func denoiseTurbo(
        noise: MLXArray,
        timesteps: [Float],
        inferMethod: ACEStepInferenceMethod,
        samplerMode: ACEStepSamplerMode = .euler,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray,
        contextLatents: MLXArray,
        sourceLatentsForCoverNoise: MLXArray? = nil,
        nonCoverEncoderHiddenStates: MLXArray? = nil,
        nonCoverEncoderAttentionMask: MLXArray? = nil,
        nonCoverContextLatents: MLXArray? = nil,
        audioCoverStrength: Float = 1.0,
        coverNoiseStrength: Float = 0.0,
        dcwEnabled: Bool = true,
        dcwMode: ACEStepDCWMode = .double,
        dcwScaler: Float = 0.05,
        dcwHighScaler: Float = 0.02,
        guidanceScale: Float = 1,
        guidanceMode: ACEStepGuidanceMode = .apg,
        nullConditionEmbedding: MLXArray? = nil,
        cfgIntervalStart: Float = 0,
        cfgIntervalEnd: Float = 1,
        velocityNormThreshold: Float = 0,
        velocityEMAFactor: Float = 0,
        repaintMask: MLXArray? = nil,
        cleanSourceLatents: MLXArray? = nil,
        repaintInjectionRatio: Float = 0.5,
        repaintCrossfadeFrames: Int = 10
    ) -> MLXArray {
        precondition(!timesteps.isEmpty, "ACE-Step timesteps must not be empty.")

        let coverNoise = Self.prepareCoverNoiseSchedule(
            noise: noise,
            sourceLatents: sourceLatentsForCoverNoise,
            timesteps: timesteps,
            coverNoiseStrength: coverNoiseStrength
        )
        var xt = coverNoise.latents
        let activeTimesteps = coverNoise.timesteps
        let usesGuidance = guidanceScale > 1 && nullConditionEmbedding != nil
        let hasNonCoverCondition =
            nonCoverEncoderHiddenStates != nil
            && nonCoverEncoderAttentionMask != nil
            && nonCoverContextLatents != nil
        let clampedCoverStrength = min(max(audioCoverStrength, 0.0), 1.0)
        let coverSteps = hasNonCoverCondition ? Int(Float(activeTimesteps.count) * clampedCoverStrength) : activeTimesteps.count
        let dcwActive =
            dcwEnabled
            && (dcwScaler != 0 || (dcwMode == .double && dcwHighScaler != 0))
        let hasRepaint =
            repaintMask != nil
            && cleanSourceLatents != nil
        let repaintInjectionCutoff = Int(
            (Double(min(max(repaintInjectionRatio, 0), 1)) * Double(activeTimesteps.count))
                .rounded(.toNearestOrEven)
        )
        var guidanceRunningAverage: MLXArray?
        var previousVelocity: MLXArray?

        func decoderPrediction(
            latents: MLXArray,
            timestep: Float,
            condition: MLXArray,
            conditionMask: MLXArray,
            context: MLXArray
        ) -> MLXArray {
            let modelLatents = usesGuidance
                ? MLX.concatenated([latents, latents], axis: 0)
                : latents
            let modelTimestep = MLXArray(
                Array(repeating: timestep, count: modelLatents.dim(0))
            ).asType(.float32)
            let modelCondition: MLXArray
            let modelConditionMask: MLXArray
            let modelContext: MLXArray
            if usesGuidance {
                let nullCondition = MLX.broadcast(
                    nullConditionEmbedding!.asType(condition.dtype),
                    to: condition.shape
                )
                modelCondition = MLX.concatenated(
                    [condition, nullCondition],
                    axis: 0
                )
                modelConditionMask = MLX.concatenated(
                    [conditionMask, conditionMask],
                    axis: 0
                )
                modelContext = MLX.concatenated([context, context], axis: 0)
            } else {
                modelCondition = condition
                modelConditionMask = conditionMask
                modelContext = context
            }
            return decoder(
                hiddenStates: modelLatents,
                timestep: modelTimestep,
                timestepR: modelTimestep,
                encoderHiddenStates: modelCondition,
                encoderAttentionMask: modelConditionMask,
                contextLatents: modelContext
            )
        }

        func stabilize(
            _ prediction: MLXArray,
            latents: MLXArray,
            previous: MLXArray?
        ) -> MLXArray {
            var result = prediction
            if velocityNormThreshold > 0 {
                let predictionNorm = MLX.sqrt(
                    (result * result).sum(axes: [1, 2], keepDims: true)
                )
                let latentNorm = MLX.sqrt(
                    (latents * latents).sum(axes: [1, 2], keepDims: true)
                ) + MLXArray(Float(1e-10)).asType(latents.dtype)
                let factor = MLX.minimum(
                    MLXArray.ones(predictionNorm.shape, dtype: predictionNorm.dtype),
                    MLXArray(velocityNormThreshold).asType(result.dtype)
                        * latentNorm / (predictionNorm + MLXArray(Float(1e-10)))
                )
                result = result * factor
            }
            if velocityEMAFactor > 0, let previous {
                result = MLXArray(Float(1 - velocityEMAFactor)).asType(result.dtype)
                    * result
                    + MLXArray(velocityEMAFactor).asType(result.dtype) * previous
            }
            return result
        }

        for i in 0..<activeTimesteps.count {
            let t = activeTimesteps[i]
            let useNonCoverCondition = hasNonCoverCondition && i >= coverSteps
            let currentEncoderHiddenStates = useNonCoverCondition ? nonCoverEncoderHiddenStates! : encoderHiddenStates
            let currentEncoderAttentionMask = useNonCoverCondition ? nonCoverEncoderAttentionMask! : encoderAttentionMask
            let currentContextLatents = useNonCoverCondition ? nonCoverContextLatents! : contextLatents

            var vt = decoderPrediction(
                latents: xt,
                timestep: t,
                condition: currentEncoderHiddenStates,
                conditionMask: currentEncoderAttentionMask,
                context: currentContextLatents
            )
            if usesGuidance {
                let predictions = MLX.split(vt, parts: 2, axis: 0)
                let conditional = predictions[0]
                let unconditional = predictions[1]
                if cfgIntervalStart <= t && t <= cfgIntervalEnd {
                    switch guidanceMode {
                    case .apg:
                        vt = ACEStepGuidance.apg(
                            conditional: conditional,
                            unconditional: unconditional,
                            scale: guidanceScale,
                            runningAverage: &guidanceRunningAverage
                        )
                    case .adg:
                        vt = ACEStepGuidance.adg(
                            latents: xt,
                            conditional: conditional,
                            unconditional: unconditional,
                            sigma: t,
                            scale: guidanceScale
                        )
                    case .cfg:
                        vt = ACEStepGuidance.cfg(
                            conditional: conditional,
                            unconditional: unconditional,
                            scale: guidanceScale
                        )
                    }
                } else {
                    vt = conditional
                }
            }
            vt = stabilize(vt, latents: xt, previous: previousVelocity)

            let xtBeforeStep = xt
            let vtForDenoise = vt
            var velocityForNextStep = vt

            if i == activeTimesteps.count - 1 {
                let tBroadcast = MLXArray(t).asType(vt.dtype).reshaped(1, 1, 1)
                xt = xt - vt * tBroadcast
                MLX.eval(xt)
            } else {
                switch inferMethod {
                case .sde:
                    let tBroadcast = MLXArray(t).asType(vt.dtype).reshaped(1, 1, 1)
                    let predClean = xt - vt * tBroadcast
                    let nextT = activeTimesteps[i + 1]
                    let newNoise = MLXRandom.normal(xt.shape).asType(xt.dtype)
                    xt = MLXArray(nextT).asType(xt.dtype) * newNoise
                        + (MLXArray(Float(1.0 - nextT)).asType(xt.dtype) * predClean)
                    MLX.eval(xt)
                case .ode:
                    let dt = t - activeTimesteps[i + 1]
                    if samplerMode == .heun {
                        let nextT = activeTimesteps[i + 1]
                        let predictedLatents = xt
                            - vt * MLXArray(dt).asType(vt.dtype)
                        var corrector = decoderPrediction(
                            latents: predictedLatents,
                            timestep: nextT,
                            condition: currentEncoderHiddenStates,
                            conditionMask: currentEncoderAttentionMask,
                            context: currentContextLatents
                        )
                        if usesGuidance {
                            let predictions = MLX.split(corrector, parts: 2, axis: 0)
                            let conditional = predictions[0]
                            let unconditional = predictions[1]
                            if cfgIntervalStart <= nextT && nextT <= cfgIntervalEnd {
                                if guidanceMode == .adg, nextT > 0 {
                                    corrector = ACEStepGuidance.adg(
                                        latents: predictedLatents,
                                        conditional: conditional,
                                        unconditional: unconditional,
                                        sigma: nextT,
                                        scale: guidanceScale
                                    )
                                } else {
                                    corrector = ACEStepGuidance.cfg(
                                        conditional: conditional,
                                        unconditional: unconditional,
                                        scale: guidanceScale
                                    )
                                }
                            } else {
                                corrector = conditional
                            }
                        }
                        corrector = stabilize(
                            corrector,
                            latents: predictedLatents,
                            previous: vt
                        )
                        let average = MLXArray(Float(0.5)).asType(vt.dtype)
                            * (vt + corrector)
                        xt = xt - average * MLXArray(dt).asType(vt.dtype)
                        velocityForNextStep = average
                    } else {
                        xt = xt - vt * MLXArray(dt).asType(vt.dtype)
                    }
                    MLX.eval(xt)
                }
            }

            if dcwActive {
                let tBroadcast = MLXArray(t).asType(vtForDenoise.dtype)
                    .reshaped(1, 1, 1)
                let denoised = xtBeforeStep - vtForDenoise * tBroadcast
                xt = ACEStepDCW.applyHaar(
                    xNext: xt,
                    denoised: denoised,
                    tCurr: t,
                    enabled: true,
                    mode: dcwMode,
                    scaler: dcwScaler,
                    highScaler: dcwHighScaler
                )
                MLX.eval(xt)
            }

            previousVelocity = velocityForNextStep
            if hasRepaint, i < repaintInjectionCutoff {
                let nextTimestep = i < activeTimesteps.count - 1
                    ? activeTimesteps[i + 1]
                    : 0
                xt = ACEStepRepaint.injectPreservedSource(
                    generatedLatents: xt,
                    cleanSourceLatents: cleanSourceLatents!,
                    repaintMask: repaintMask!,
                    nextTimestep: nextTimestep,
                    noise: noise
                )
                MLX.eval(xt)
            }

            Memory.clearCache()
        }

        if hasRepaint, repaintCrossfadeFrames > 0 {
            xt = ACEStepRepaint.blendLatentBoundaries(
                generatedLatents: xt,
                cleanSourceLatents: cleanSourceLatents!,
                repaintMask: repaintMask!,
                crossfadeFrames: repaintCrossfadeFrames
            )
            MLX.eval(xt)
        }

        return xt
    }

    static func prepareCoverNoiseSchedule(
        noise: MLXArray,
        sourceLatents: MLXArray?,
        timesteps: [Float],
        coverNoiseStrength: Float
    ) -> (latents: MLXArray, timesteps: [Float]) {
        precondition(!timesteps.isEmpty, "Turbo timesteps must not be empty.")

        let clamped = min(max(coverNoiseStrength, 0.0), 1.0)
        guard clamped > 0, let sourceLatents else {
            return (noise, timesteps)
        }
        precondition(
            sourceLatents.shape == noise.shape,
            "sourceLatentsForCoverNoise must match noise shape \(noise.shape); got \(sourceLatents.shape)."
        )

        let effectiveNoiseLevel = Float(1.0) - clamped
        var nearestIndex = 0
        var nearestDistance = abs(timesteps[0] - effectiveNoiseLevel)
        for index in timesteps.indices.dropFirst() {
            let distance = abs(timesteps[index] - effectiveNoiseLevel)
            if distance < nearestDistance {
                nearestIndex = index
                nearestDistance = distance
            }
        }

        let nearestT = timesteps[nearestIndex]
        let t = MLXArray(nearestT).asType(noise.dtype)
        let oneMinusT = MLXArray(Float(1.0 - nearestT)).asType(noise.dtype)
        let initialLatents = t * noise + oneMinusT * sourceLatents.asType(noise.dtype)
        MLX.eval(initialLatents)

        return (initialLatents, Array(timesteps[nearestIndex...]))
    }

    func prepareCondition(
        textHiddenStates: MLXArray,
        textAttentionMask: MLXArray,
        lyricHiddenStates: MLXArray,
        lyricAttentionMask: MLXArray,
        referAudioAcousticHiddenStatesPacked: MLXArray,
        referAudioOrderMask: MLXArray,
        hiddenStates: MLXArray,
        attentionMask: MLXArray,
        silenceLatent: MLXArray?,
        srcLatents: MLXArray,
        chunkMasks: MLXArray,
        isCovers: MLXArray,
        precomputedLmHints25Hz: MLXArray? = nil,
        audioCodes: MLXArray? = nil
    ) -> (encoderHiddenStates: MLXArray, encoderAttentionMask: MLXArray, contextLatents: MLXArray) {
        let (encoderHiddenStates, encoderAttentionMask) = encoder(
            textHiddenStates: textHiddenStates,
            textAttentionMask: textAttentionMask,
            lyricHiddenStates: lyricHiddenStates,
            lyricAttentionMask: lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked: referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: referAudioOrderMask
        )

        let lmHints25Hz: MLXArray = {
            if let precomputedLmHints25Hz {
                return precomputedLmHints25Hz[0..., 0..<srcLatents.dim(1), 0...]
            }

            if let audioCodes {
                let quantized5Hz = tokenizer.quantizer.getOutputFromIndices(audioCodes, dtype: hiddenStates.dtype)
                let hints = detokenizer(quantized5Hz)
                return hints[0..., 0..<srcLatents.dim(1), 0...]
            }

            let (quantized5Hz, _, _) = tokenizeForLMHints(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                silenceLatent: silenceLatent
            )
            let hints = detokenizer(quantized5Hz)
            return hints[0..., 0..<srcLatents.dim(1), 0...]
        }()

        let covers = (isCovers .> MLXArray(Float(0))).asType(.bool)
        let coversExpanded = broadcast(covers.reshaped(covers.dim(0), 1, 1), to: srcLatents.shape)
        let srcLatentsUpdated = MLX.where(coversExpanded, lmHints25Hz.asType(srcLatents.dtype), srcLatents)

        let contextLatents = MLX.concatenated([srcLatentsUpdated, chunkMasks.asType(srcLatents.dtype)], axis: -1)
        return (encoderHiddenStates, encoderAttentionMask, contextLatents)
    }

    func tokenizeForLMHints(
        hiddenStates: MLXArray,
        attentionMask: MLXArray,
        silenceLatent: MLXArray?
    ) -> (quantized5Hz: MLXArray, indices: MLXArray, pooledMask5Hz: MLXArray) {
        let B = hiddenStates.dim(0)
        let T = hiddenStates.dim(1)
        let D = hiddenStates.dim(2)
        let P = decoderConfig.poolWindowSize

        var x = hiddenStates
        var mask = attentionMask

        let padLen = (P - (T % P)) % P
        if padLen > 0 {
            let pad: MLXArray = {
                if let silenceLatent {
                    if silenceLatent.ndim == 3 {
                        let frames = silenceLatent[0, 0..<padLen, 0...]
                        return broadcast(frames.reshaped(1, padLen, D), to: [B, padLen, D]).asType(x.dtype)
                    }
                    if silenceLatent.ndim == 2 {
                        let frames = silenceLatent[0..<padLen, 0...]
                        return broadcast(frames.reshaped(1, padLen, D), to: [B, padLen, D]).asType(x.dtype)
                    }
                }
                return MLXArray.zeros([B, padLen, D], dtype: x.dtype)
            }()

            x = MLX.concatenated([x, pad], axis: 1)
            mask = padded(mask, widths: [[0, 0], [0, padLen]])
        }

        let T5 = x.dim(1) / P
        let patched = x.reshaped(B, T5, P, D)

        let pooled = mask.asType(.float32).reshaped(B, T5, P).max(axis: -1).asType(.int32)

        let (quantized, indices) = tokenizer(patched)
        return (quantized, indices, pooled)
    }
}
