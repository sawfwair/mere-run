import Foundation
import MLX

// Owns LM fallback, condition assembly, and denoising helpers for
// ACEStepPipeline. Public entrypoints stay in the main pipeline file.

extension ACEStepPipeline {
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
        dcwHighScaler: Float = 0.02
    ) -> MLXArray {
        precondition(!timesteps.isEmpty, "Turbo timesteps must not be empty.")

        let coverNoise = Self.prepareCoverNoiseSchedule(
            noise: noise,
            sourceLatents: sourceLatentsForCoverNoise,
            timesteps: timesteps,
            coverNoiseStrength: coverNoiseStrength
        )
        var xt = coverNoise.latents
        let activeTimesteps = coverNoise.timesteps
        let B = xt.dim(0)
        let hasNonCoverCondition =
            nonCoverEncoderHiddenStates != nil
            && nonCoverEncoderAttentionMask != nil
            && nonCoverContextLatents != nil
        let clampedCoverStrength = min(max(audioCoverStrength, 0.0), 1.0)
        let coverSteps = hasNonCoverCondition ? Int(Float(activeTimesteps.count) * clampedCoverStrength) : activeTimesteps.count
        let dcwActive =
            dcwEnabled
            && (dcwScaler != 0 || (dcwMode == .double && dcwHighScaler != 0))

        for i in 0..<activeTimesteps.count {
            let t = activeTimesteps[i]
            let tBatch = MLXArray((0..<B).map { _ in t }).asType(.float32)
            let useNonCoverCondition = hasNonCoverCondition && i >= coverSteps
            let currentEncoderHiddenStates = useNonCoverCondition ? nonCoverEncoderHiddenStates! : encoderHiddenStates
            let currentEncoderAttentionMask = useNonCoverCondition ? nonCoverEncoderAttentionMask! : encoderAttentionMask
            let currentContextLatents = useNonCoverCondition ? nonCoverContextLatents! : contextLatents

            let vt = decoder(
                hiddenStates: xt,
                timestep: tBatch,
                timestepR: tBatch,
                encoderHiddenStates: currentEncoderHiddenStates,
                encoderAttentionMask: currentEncoderAttentionMask,
                contextLatents: currentContextLatents
            )

            let xtBeforeStep = xt
            let vtForDenoise = vt

            if i == activeTimesteps.count - 1 {
                let tBroadcast = tBatch.asType(vt.dtype).reshaped(B, 1, 1)
                xt = xt - vt * tBroadcast
                MLX.eval(xt)
            } else {
                switch inferMethod {
                case .sde:
                    let tBroadcast = tBatch.asType(vt.dtype).reshaped(B, 1, 1)
                    let predClean = xt - vt * tBroadcast
                    let nextT = activeTimesteps[i + 1]
                    let newNoise = MLXRandom.normal(xt.shape).asType(xt.dtype)
                    xt = MLXArray(nextT).asType(xt.dtype) * newNoise
                        + (MLXArray(Float(1.0 - nextT)).asType(xt.dtype) * predClean)
                    MLX.eval(xt)
                case .ode:
                    let dt = t - activeTimesteps[i + 1]
                    xt = xt - vt * MLXArray(dt).asType(vt.dtype)
                    MLX.eval(xt)
                }
            }

            if dcwActive {
                let tBroadcast = tBatch.asType(vtForDenoise.dtype).reshaped(B, 1, 1)
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

            Memory.clearCache()
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
