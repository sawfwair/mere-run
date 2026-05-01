import Foundation
import MLX

// Owns prompt-conditioned input preparation for ACEStepPipeline.
// This file intentionally stops at building conditioning tensors; the public
// generation entrypoints and denoising loop live elsewhere.

extension ACEStepPipeline {
    func normalizeSourceLatents(_ sourceLatents25Hz: MLXArray?, targetFrames: Int) throws -> MLXArray {
        let B = 1
        guard let sourceLatents25Hz else {
            return MLXArray.zeros([B, targetFrames, decoderConfig.audioAcousticHiddenDim], dtype: .bfloat16)
        }

        guard sourceLatents25Hz.ndim == 3,
              sourceLatents25Hz.dim(0) == B,
              sourceLatents25Hz.dim(2) == decoderConfig.audioAcousticHiddenDim
        else {
            throw PipelineError.invalidConditioningInput(
                "sourceLatents25Hz must be [1, T, \(decoderConfig.audioAcousticHiddenDim)]; got \(sourceLatents25Hz.shape)."
            )
        }

        if sourceLatents25Hz.dim(1) == targetFrames {
            return sourceLatents25Hz.asType(.bfloat16)
        }
        if sourceLatents25Hz.dim(1) > targetFrames {
            return sourceLatents25Hz[0..., 0..<targetFrames, 0...].asType(.bfloat16)
        }

        let pad = MLXArray.zeros(
            [B, targetFrames - sourceLatents25Hz.dim(1), decoderConfig.audioAcousticHiddenDim],
            dtype: sourceLatents25Hz.dtype
        )
        return MLX.concatenated([sourceLatents25Hz, pad], axis: 1).asType(.bfloat16)
    }

    func chunkChannelsForPromptConditioning() -> Int {
        let contextChannels = decoderConfig.inChannels - decoderConfig.audioAcousticHiddenDim
        precondition(contextChannels > 0, "Expected context channels > 0.")

        let chunkChannels = contextChannels - decoderConfig.audioAcousticHiddenDim
        precondition(chunkChannels > 0, "Expected chunk mask channels > 0.")
        return chunkChannels
    }

    func preparePromptConditionInputs(
        caption: String,
        lyrics: String,
        srcLatents: MLXArray,
        chunkChannels: Int,
        lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata,
        referenceTimbreLatents25Hz: [MLXArray]? = nil,
        referenceTimbreAudio48kHz: [MLXArray]? = nil,
        audioCoverStrength: Float = 1.0,
        vocalLanguage: String,
        instruction: String,
        isCover: Bool
    ) throws -> ACEStepConditionInputs {
        let B = srcLatents.dim(0)
        let T = srcLatents.dim(1)
        guard B == 1 else {
            throw PipelineError.invalidConditioningInput("preparePromptConditionInputs currently supports batch size 1; got \(B).")
        }
        precondition(chunkChannels > 0, "Expected chunk mask channels > 0.")

        let textConditioning: (
            textHiddenStates: MLXArray,
            textAttentionMask: MLXArray,
            lyricHiddenStates: MLXArray,
            lyricAttentionMask: MLXArray,
            nonCoverTextHiddenStates: MLXArray?,
            nonCoverTextAttentionMask: MLXArray?
        ) = {
            guard let conditionTextTokenizer, let conditionTextEncoder else {
                return (
                    textHiddenStates: MLXArray.zeros([B, 1, decoderConfig.textHiddenDim], dtype: .bfloat16),
                    textAttentionMask: MLXArray.ones([B, 1], dtype: .int32),
                    lyricHiddenStates: MLXArray.zeros([B, 1, decoderConfig.textHiddenDim], dtype: .bfloat16),
                    lyricAttentionMask: MLXArray.ones([B, 1], dtype: .int32),
                    nonCoverTextHiddenStates: nil,
                    nonCoverTextAttentionMask: nil
                )
            }

            let promptCaption: String = {
                if let candidate = lmUserMetadata.caption?.trimmingCharacters(in: .whitespacesAndNewlines), !candidate.isEmpty {
                    return candidate
                }
                return caption
            }()
            let language = {
                let candidate = lmUserMetadata.language?.trimmingCharacters(in: .whitespacesAndNewlines) ?? vocalLanguage
                return candidate.isEmpty ? Self.defaultVocalLanguage : candidate
            }()
            let metas = makePromptMetasString(lmUserMetadata)
            let captionPrompt = makeCaptionPrompt(
                instruction: formatInstruction(instruction),
                caption: promptCaption,
                metas: metas
            )
            let lyricsPrompt = makeLyricsPrompt(lyrics: lyrics, language: language)

            let captionTokens = tokenizePrompt(captionPrompt, maxTokens: 256, tokenizer: conditionTextTokenizer)
            let lyricTokens = tokenizePrompt(lyricsPrompt, maxTokens: 2048, tokenizer: conditionTextTokenizer)

            let textInputIds = MLXArray(captionTokens.ids, [1, captionTokens.ids.count]).asType(.int32)
            let textAttentionMask = MLXArray(captionTokens.mask, [1, captionTokens.mask.count]).asType(.int32)
            let lyricInputIds = MLXArray(lyricTokens.ids, [1, lyricTokens.ids.count]).asType(.int32)
            let lyricAttentionMask = MLXArray(lyricTokens.mask, [1, lyricTokens.mask.count]).asType(.int32)

            let textHiddenStates = conditionTextEncoder.forward(
                inputIds: textInputIds,
                attentionMask: textAttentionMask
            ).lastHiddenState
            let lyricHiddenStates = conditionTextEncoder.embed(inputIds: lyricInputIds)

            let nonCoverTextConditioning: (hiddenStates: MLXArray?, attentionMask: MLXArray?) = {
                guard audioCoverStrength < 1.0 else {
                    return (nil, nil)
                }

                let nonCoverCaptionPrompt = makeCaptionPrompt(
                    instruction: formatInstruction(Self.defaultDiTInstruction),
                    caption: promptCaption,
                    metas: metas
                )
                let nonCoverCaptionTokens = tokenizePrompt(nonCoverCaptionPrompt, maxTokens: 256, tokenizer: conditionTextTokenizer)
                let nonCoverInputIds = MLXArray(nonCoverCaptionTokens.ids, [1, nonCoverCaptionTokens.ids.count]).asType(.int32)
                let nonCoverAttentionMask = MLXArray(nonCoverCaptionTokens.mask, [1, nonCoverCaptionTokens.mask.count]).asType(.int32)
                let nonCoverHiddenStates = conditionTextEncoder.forward(
                    inputIds: nonCoverInputIds,
                    attentionMask: nonCoverAttentionMask
                ).lastHiddenState
                return (nonCoverHiddenStates, nonCoverAttentionMask)
            }()

            return (
                textHiddenStates: textHiddenStates,
                textAttentionMask: textAttentionMask,
                lyricHiddenStates: lyricHiddenStates,
                lyricAttentionMask: lyricAttentionMask,
                nonCoverTextHiddenStates: nonCoverTextConditioning.hiddenStates,
                nonCoverTextAttentionMask: nonCoverTextConditioning.attentionMask
            )
        }()

        let timbre = try packReferenceTimbreLatents(
            referenceTimbreLatents25Hz: referenceTimbreLatents25Hz,
            referenceTimbreAudio48kHz: referenceTimbreAudio48kHz,
            fallbackLatents25Hz: srcLatents
        )

        return ACEStepConditionInputs(
            textHiddenStates: textConditioning.textHiddenStates,
            textAttentionMask: textConditioning.textAttentionMask,
            lyricHiddenStates: textConditioning.lyricHiddenStates,
            lyricAttentionMask: textConditioning.lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked: timbre.packed,
            referAudioOrderMask: timbre.orderMask,
            srcLatents: srcLatents,
            chunkMasks: MLXArray.ones([B, T, chunkChannels], dtype: .bfloat16),
            isCovers: MLXArray([isCover ? Int32(1) : Int32(0)]).asType(.int32),
            hiddenStates: srcLatents,
            attentionMask: MLXArray.ones([B, T], dtype: .int32),
            silenceLatent: nil,
            nonCoverTextHiddenStates: textConditioning.nonCoverTextHiddenStates,
            nonCoverTextAttentionMask: textConditioning.nonCoverTextAttentionMask
        )
    }

    func packReferenceTimbreLatents(
        referenceTimbreLatents25Hz: [MLXArray]?,
        referenceTimbreAudio48kHz: [MLXArray]?,
        fallbackLatents25Hz: MLXArray
    ) throws -> (packed: MLXArray, orderMask: MLXArray) {
        let targetFrames = min(max(1, fallbackLatents25Hz.dim(1)), decoderConfig.timbreFixFrame)
        let latentDim = decoderConfig.audioAcousticHiddenDim

        let references: [MLXArray]
        if let referenceTimbreLatents25Hz, !referenceTimbreLatents25Hz.isEmpty {
            references = referenceTimbreLatents25Hz
        } else if let referenceTimbreAudio48kHz, !referenceTimbreAudio48kHz.isEmpty {
            references = try referenceTimbreAudio48kHz.map { waveform in
                let normalizedAudio = try normalizeReferenceTimbreAudio(waveform)
                return vae.tiledEncode(normalizedAudio)
            }
        } else {
            references = []
        }

        if references.isEmpty {
            let fallback = try normalizeReferenceTimbreLatents(
                fallbackLatents25Hz,
                targetFrames: targetFrames,
                latentDim: latentDim
            )
            return (fallback, MLXArray([Int32(0)]).asType(.int32))
        }

        let packed = MLX.concatenated(
            try references.map {
                try normalizeReferenceTimbreLatents(
                    $0,
                    targetFrames: targetFrames,
                    latentDim: latentDim
                )
            },
            axis: 0
        )
        let order = MLXArray(Array(repeating: Int32(0), count: references.count)).asType(.int32)
        return (packed, order)
    }

    func normalizeReferenceTimbreAudio(_ audio: MLXArray) throws -> MLXArray {
        let stereo2D: MLXArray
        if audio.ndim == 1 {
            let mono = audio.reshaped(audio.dim(0), 1)
            stereo2D = MLX.concatenated([mono, mono], axis: 1)
        } else if audio.ndim == 2 {
            if audio.dim(1) == decoderConfig.audioAcousticHiddenDim {
                throw PipelineError.invalidConditioningInput(
                    "referenceTimbreAudio48kHz expects waveform tensors, but received latent-shaped [T,\(decoderConfig.audioAcousticHiddenDim)]."
                )
            }
            if audio.dim(1) == vaeConfig.audioChannels {
                stereo2D = audio
            } else if audio.dim(0) == vaeConfig.audioChannels {
                stereo2D = audio.transposed(1, 0)
            } else if audio.dim(1) == 1 {
                stereo2D = MLX.concatenated([audio, audio], axis: 1)
            } else if audio.dim(0) == 1 {
                let mono = audio.transposed(1, 0)
                stereo2D = MLX.concatenated([mono, mono], axis: 1)
            } else {
                throw PipelineError.invalidConditioningInput(
                    "referenceTimbreAudio48kHz 2D tensors must be [S,\(vaeConfig.audioChannels)] or [\(vaeConfig.audioChannels),S]; got \(audio.shape)."
                )
            }
        } else if audio.ndim == 3 {
            guard audio.dim(0) == 1 else {
                throw PipelineError.invalidConditioningInput(
                    "referenceTimbreAudio48kHz batch dimension must be 1 when using 3D tensors; got \(audio.shape)."
                )
            }
            if audio.dim(2) == vaeConfig.audioChannels {
                return audio.asType(.bfloat16)
            }
            if audio.dim(1) == vaeConfig.audioChannels {
                return audio.transposed(0, 2, 1).asType(.bfloat16)
            }
            throw PipelineError.invalidConditioningInput(
                "referenceTimbreAudio48kHz 3D tensors must be [1,S,\(vaeConfig.audioChannels)] or [1,\(vaeConfig.audioChannels),S]; got \(audio.shape)."
            )
        } else {
            throw PipelineError.invalidConditioningInput(
                "referenceTimbreAudio48kHz tensors must be 1D, 2D, or 3D; got \(audio.shape)."
            )
        }

        return stereo2D.reshaped(1, stereo2D.dim(0), vaeConfig.audioChannels).asType(.bfloat16)
    }

    func normalizeReferenceTimbreLatents(
        _ latents25Hz: MLXArray,
        targetFrames: Int,
        latentDim: Int
    ) throws -> MLXArray {
        precondition(targetFrames > 0, "targetFrames must be positive.")

        let normalized: MLXArray
        if latents25Hz.ndim == 3 {
            guard latents25Hz.dim(0) == 1, latents25Hz.dim(2) == latentDim else {
                throw PipelineError.invalidConditioningInput(
                    "Reference timbre latents must be [1,T,\(latentDim)]; got \(latents25Hz.shape)."
                )
            }
            normalized = latents25Hz
        } else if latents25Hz.ndim == 2 {
            guard latents25Hz.dim(1) == latentDim else {
                throw PipelineError.invalidConditioningInput(
                    "Reference timbre latents must be [T,\(latentDim)]; got \(latents25Hz.shape)."
                )
            }
            normalized = latents25Hz.reshaped(1, latents25Hz.dim(0), latentDim)
        } else {
            throw PipelineError.invalidConditioningInput(
                "Reference timbre latents must be [1,T,\(latentDim)] or [T,\(latentDim)]; got \(latents25Hz.shape)."
            )
        }

        let frames = normalized.dim(1)
        if frames == targetFrames {
            return normalized.asType(.bfloat16)
        }
        if frames > targetFrames {
            return normalized[0..., 0..<targetFrames, 0...].asType(.bfloat16)
        }

        let pad = MLXArray.zeros([1, targetFrames - frames, latentDim], dtype: normalized.dtype)
        return MLX.concatenated([normalized, pad], axis: 1).asType(.bfloat16)
    }

    func tokenizePrompt(
        _ text: String,
        maxTokens: Int,
        tokenizer: ACEStep5HzLMTokenizer
    ) -> (ids: [Int32], mask: [Int32]) {
        precondition(maxTokens > 0, "maxTokens must be positive.")

        var ids = tokenizer.encode(text, addSpecialTokens: true)
        if ids.count > maxTokens {
            ids = Array(ids.prefix(maxTokens))
        }
        if ids.isEmpty {
            ids = [tokenizer.eosTokenId ?? tokenizer.padTokenId]
        }

        let idsI32 = ids.map(Int32.init)
        return (idsI32, Array(repeating: 1, count: idsI32.count))
    }

    func formatInstruction(_ instruction: String) -> String {
        instruction.hasSuffix(":") ? instruction : "\(instruction):"
    }

    func makePromptMetasString(_ metadata: ACEStep5HzLMConstrainedSampler.UserMetadata) -> String {
        var rows: [String] = []
        if let value = metadata.bpm {
            rows.append("- bpm: \(value)")
        }
        if let value = metadata.timesignature {
            rows.append("- timesignature: \(value)")
        }
        if let value = metadata.keyscale {
            rows.append("- keyscale: \(value)")
        }
        if let value = metadata.duration {
            rows.append("- duration: \(value)")
        }
        if rows.isEmpty {
            return ""
        }
        return rows.joined(separator: "\n") + "\n"
    }

    func makeCaptionPrompt(instruction: String, caption: String, metas: String) -> String {
        """
        # Instruction
        \(instruction)

        # Caption
        \(caption)

        # Metas
        \(metas)<|endoftext|>
        """
        + "\n"
    }

    func makeLyricsPrompt(lyrics: String, language: String) -> String {
        """
        # Languages
        \(language)

        # Lyric
        \(lyrics)<|endoftext|>
        """
    }
}
