import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs

/// Owns prompt preparation, token generation, and waveform assembly for TTS.
/// This file is intentionally separate from model loading so the speech
/// generation flow can be read end-to-end.
extension Qwen3TTSGenerator {
    func generateVoiceClone(
        request: TTSRequest,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        tokenizer: Qwen3TTSTokenizer,
        speechTokenizer: Qwen3TTSSpeechTokenizer,
        speakerEncoder: Qwen3TTSSpeakerEncoder?,
        config: Qwen3TTSModelConfig,
        progressHandler: (@Sendable (TTSProgress) -> Void)?,
        streamingChunkTokenInterval: Int? = nil,
        onToken: ((Int) -> Void)? = nil,
        onAudioDelta: (([Float]) -> Void)? = nil
    ) throws -> MLXArray {
        guard let reference = request.cloneReference else {
            throw Qwen3TTSError.invalidCloneReference("Missing clone reference. Provide --profile or --ref-audio.")
        }

        let transcript = reference.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else {
            throw Qwen3TTSError.invalidCloneReference("Reference transcript is empty.")
        }

        progressHandler?(TTSProgress(stage: .preprocessingReference, message: "Preprocessing reference audio..."))
        let processed = try Qwen3TTSAudioPreprocessor.loadAndProcess(
            from: reference.audioURL,
            targetSampleRate: config.sampleRate
        )

        progressHandler?(TTSProgress(stage: .encodingReference, message: "Encoding speaker reference..."))
        guard speechTokenizer.hasEncoder else {
            throw Qwen3TTSError.cloneAssetsMissing(["speech tokenizer encoder"])
        }

        let speakerEmbedding = speakerEncoder?.extractEmbedding(audio: processed.samples)
        let referenceCodes = speechTokenizer.encode(samples: processed.samples, sampleRate: processed.sampleRate)
        if let speakerEmbedding {
            MLX.eval(speakerEmbedding)
        }
        MLX.eval(referenceCodes)

        progressHandler?(TTSProgress(stage: .buildingPrompt, message: "Building clone prompt..."))
        progressHandler?(TTSProgress(stage: .tokenizing, message: "Preparing generation inputs..."))

        let cloneLanguage = reference.language ?? request.language
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed, referenceCodesBQT) = try prepareICLGenerationInputs(
            text: request.text,
            refText: transcript,
            referenceCodes: referenceCodes,
            language: cloneLanguage,
            speakerEmbedding: speakerEmbedding,
            tokenizer: tokenizer,
            talker: talker,
            config: config
        )

        return try generateVoiceCloneICL(
            inputEmbeds: inputEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            referenceCodesBQT: referenceCodesBQT,
            targetTokenCount: tokenizer.encode(request.text).count,
            talker: talker,
            speechTokenizer: speechTokenizer,
            config: config,
            temperature: request.temperature,
            progressHandler: progressHandler,
            streamingChunkTokenInterval: streamingChunkTokenInterval,
            onToken: onToken,
            onAudioDelta: onAudioDelta
        )
    }

    func generateVoiceDesign(
        text: String,
        language: String,
        instruct: String,
        speakerHintTokens: [Int]?,
        referencePromptTokens: [Int]?,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        tokenizer: Qwen3TTSTokenizer,
        speechTokenizer: Qwen3TTSSpeechTokenizer,
        config: Qwen3TTSModelConfig,
        temperature: Float,
        progressHandler: (@Sendable (TTSProgress) -> Void)?,
        streamingChunkTokenInterval: Int? = nil,
        onToken: ((Int) -> Void)? = nil,
        onAudioDelta: (([Float]) -> Void)? = nil
    ) throws -> MLXArray {
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(
            text: text,
            language: language,
            speaker: nil,
            instruct: instruct,
            speakerHintTokens: speakerHintTokens,
            referencePromptTokens: referencePromptTokens,
            tokenizer: tokenizer,
            talker: talker,
            config: config
        )

        progressHandler?(TTSProgress(stage: .generating, message: "Generating tokens..."))

        let maxTokens = 4096
        let targetTokenCount = tokenizer.encode(text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))
        let topK = 50
        let topP: Float = 1.0
        let repetitionPenalty: Float = 1.05
        let eosTokenId = config.talkerConfig.codecEosTokenId
        let suppressTokens = Array((config.talkerConfig.vocabSize - 1024)..<config.talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        var generatedCodes: [MLXArray] = []
        var emittedSampleCount = 0

        if Qwen3TTSEnvironment.pipelinedDecodeEnabled {
            let result = try runPipelinedTalkerLoop(
                inputEmbeds: inputEmbeds,
                trailingTextHidden: trailingTextHidden,
                ttsPadEmbed: ttsPadEmbed,
                effectiveMaxTokens: effectiveMaxTokens,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                talker: talker,
                config: config,
                progressHandler: progressHandler,
                streamingChunkTokenInterval: streamingChunkTokenInterval,
                onToken: onToken,
                emitDelta: { codes, emitted in
                    try self.emitStreamingAudioDelta(
                        generatedCodes: codes,
                        referenceCodesBQT: nil,
                        speechTokenizer: speechTokenizer,
                        emittedSampleCount: &emitted,
                        onAudioDelta: onAudioDelta
                    )
                }
            )
            generatedCodes = result.codes
            emittedSampleCount = result.emittedSampleCount
        } else {
            var generatedFirstTokens: [Int] = []
            let cache = talker.makeCache()
            var inputEmbedsVar = inputEmbeds
            var trailingIndex = 0

            for step in 0..<effectiveMaxTokens {
                let (logits, hidden) = talker(inputEmbedsVar, cache: cache)
                let nextToken = sampleToken(
                    logits: logits,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    generatedTokens: generatedFirstTokens,
                    suppressTokens: suppressTokens,
                    eosTokenId: eosTokenId
                )

                let tokenValue = nextToken.item(Int.self)
                if tokenValue == eosTokenId {
                    break
                }

                generatedFirstTokens.append(tokenValue)
                onToken?(tokenValue)

                let codeTokens = try generateCodecTokens(
                    firstToken: nextToken,
                    hidden: hidden,
                    talker: talker,
                    temperature: temperature,
                    topK: topK,
                    topP: topP
                )
                let allCodes = MLX.concatenated(codeTokens, axis: 1)
                generatedCodes.append(allCodes)

                if let streamingChunkTokenInterval, generatedCodes.count % streamingChunkTokenInterval == 0 {
                    try emitStreamingAudioDelta(
                        generatedCodes: generatedCodes,
                        referenceCodesBQT: nil,
                        speechTokenizer: speechTokenizer,
                        emittedSampleCount: &emittedSampleCount,
                        onAudioDelta: onAudioDelta
                    )
                }

                let textEmbed: MLXArray
                if trailingIndex < trailingTextHidden.dim(1) {
                    textEmbed = trailingTextHidden[0..., trailingIndex..<(trailingIndex + 1), 0...]
                    trailingIndex += 1
                } else {
                    textEmbed = ttsPadEmbed
                }

                inputEmbedsVar = textEmbed + combineCodecEmbeddings(codeTokens: codeTokens, talker: talker)
                MLX.eval(inputEmbedsVar)

                if step > 0 && step % 25 == 0 {
                    progressHandler?(TTSProgress(stage: .generating, tokensGenerated: step, message: "Generated \(step) tokens..."))
                    Memory.clearCache()
                }
            }
        }

        guard !generatedCodes.isEmpty else {
            throw Qwen3TTSError.noAudioTokensGenerated
        }

        if streamingChunkTokenInterval != nil {
            try emitStreamingAudioDelta(
                generatedCodes: generatedCodes,
                referenceCodesBQT: nil,
                speechTokenizer: speechTokenizer,
                emittedSampleCount: &emittedSampleCount,
                onAudioDelta: onAudioDelta
            )
        }

        progressHandler?(TTSProgress(stage: .decoding, message: "Decoding audio..."))

        let codes = MLX.stacked(generatedCodes, axis: 1)
        MLX.eval(codes)
        Memory.clearCache()

        let (wav, lengths) = speechTokenizer.decode(codes)
        var audio = wav.squeezed(axis: 0)
        let validLength = lengths[0].item(Int.self)
        if validLength > 0 && validLength < audio.size {
            audio = audio[0..<validLength]
        }

        MLX.eval(audio)
        return audio
    }

    /// One confirmed-or-pending step of the pipelined talker loop: the
    /// sampled first token (still on GPU) plus the codec tokens derived from
    /// it. Confirmation (EOS check, callbacks, streaming) happens one step
    /// later, while the GPU executes the next step's graph.
    private struct PipelinedTalkerStep {
        let token: MLXArray
        let codeTokens: [MLXArray]
        let step: Int
    }

    /// Depth-1 pipelined talker loop shared by the style and clone paths.
    /// The legacy loop synchronizes with the GPU roughly nine times per
    /// emitted frame (the sampled token plus every codec sub-token reads
    /// back with `.item()`); here sampling and the codec sub-loop stay on
    /// GPU, the step is scheduled with `asyncEval`, and the previous step's
    /// token is read back while the current one executes. EOS therefore
    /// costs one speculative frame of GPU work, which is discarded.
    private func runPipelinedTalkerLoop(
        inputEmbeds: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbed: MLXArray,
        effectiveMaxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        config: Qwen3TTSModelConfig,
        progressHandler: (@Sendable (TTSProgress) -> Void)?,
        streamingChunkTokenInterval: Int?,
        onToken: ((Int) -> Void)?,
        emitDelta: (([MLXArray], inout Int) throws -> Void)?
    ) throws -> (codes: [MLXArray], emittedSampleCount: Int) {
        let eosTokenId = config.talkerConfig.codecEosTokenId
        let suppressTokens = Array((config.talkerConfig.vocabSize - 1024)..<config.talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }
        var samplerContext = Qwen3TTSSamplerContext(
            vocabSize: config.talkerConfig.vocabSize,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            eosTokenId: eosTokenId,
            suppressTokens: suppressTokens
        )
        let codecContext = Qwen3TTSSamplerContext(
            vocabSize: 0,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: 1.0,
            eosTokenId: nil,
            suppressTokens: nil
        )

        var generatedCodes: [MLXArray] = []
        var emittedSampleCount = 0
        var pending: PipelinedTalkerStep?
        var inputEmbedsVar = inputEmbeds
        var trailingIndex = 0
        let cache = talker.makeCache()

        func confirm(_ step: PipelinedTalkerStep) throws -> Bool {
            let tokenValue = step.token.item(Int.self)
            if tokenValue == eosTokenId {
                return false
            }
            onToken?(tokenValue)
            generatedCodes.append(MLX.concatenated(step.codeTokens, axis: 1))
            if let streamingChunkTokenInterval, let emitDelta,
               generatedCodes.count % streamingChunkTokenInterval == 0 {
                try emitDelta(generatedCodes, &emittedSampleCount)
            }
            let confirmedCount = generatedCodes.count
            if confirmedCount > 0 && confirmedCount % 25 == 0 {
                progressHandler?(TTSProgress(
                    stage: .generating,
                    tokensGenerated: confirmedCount,
                    message: "Generated \(confirmedCount) tokens..."
                ))
                Memory.clearCache()
            }
            return true
        }

        for step in 0..<effectiveMaxTokens {
            let (logits, hidden) = talker(inputEmbedsVar, cache: cache)
            let tokenArray = sampleTokenArrayTTS(logits: logits, context: samplerContext)
            samplerContext.appendHistory(tokenArray)
            let codeTokens = try generateCodecTokensPipelined(
                firstToken: tokenArray,
                hidden: hidden,
                talker: talker,
                context: codecContext
            )

            let textEmbed: MLXArray
            if trailingIndex < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIndex..<(trailingIndex + 1), 0...]
                trailingIndex += 1
            } else {
                textEmbed = ttsPadEmbed
            }
            inputEmbedsVar = textEmbed + combineCodecEmbeddings(codeTokens: codeTokens, talker: talker)
            asyncEval([inputEmbedsVar, tokenArray])

            if let previous = pending {
                pending = nil
                guard try confirm(previous) else {
                    return (generatedCodes, emittedSampleCount)
                }
            }
            pending = PipelinedTalkerStep(token: tokenArray, codeTokens: codeTokens, step: step)
        }

        if let previous = pending {
            _ = try confirm(previous)
        }
        return (generatedCodes, emittedSampleCount)
    }

    /// All-GPU variant of `generateCodecTokens`: identical sub-loop, but the
    /// per-code sampling returns arrays instead of reading each code back
    /// with `.item()`.
    private func generateCodecTokensPipelined(
        firstToken: MLXArray,
        hidden: MLXArray,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        context: Qwen3TTSSamplerContext
    ) throws -> [MLXArray] {
        var codeTokens: [MLXArray] = [firstToken]
        let codeHidden = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
        let codeCache = talker.codePredictor.makeCache()

        for codeIdx in 0..<(talker.config.numCodeGroups - 1) {
            let codeInput: MLXArray
            if codeIdx == 0 {
                let code0Embed = talker.getInputEmbeddings()(firstToken)
                codeInput = MLX.concatenated([codeHidden, code0Embed], axis: 1)
            } else {
                codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
            }

            let (codeLogits, _, _) = talker.codePredictor(
                codeInput,
                cache: codeCache,
                generationStep: codeIdx
            )

            codeTokens.append(sampleTokenArrayTTS(logits: codeLogits, context: context))
        }

        return codeTokens
    }

    private func generateCodecTokens(
        firstToken: MLXArray,
        hidden: MLXArray,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        temperature: Float,
        topK: Int,
        topP: Float
    ) throws -> [MLXArray] {
        var codeTokens: [MLXArray] = [firstToken]
        let codeHidden = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
        let codeCache = talker.codePredictor.makeCache()

        for codeIdx in 0..<(talker.config.numCodeGroups - 1) {
            let codeInput: MLXArray
            if codeIdx == 0 {
                let code0Embed = talker.getInputEmbeddings()(firstToken)
                codeInput = MLX.concatenated([codeHidden, code0Embed], axis: 1)
            } else {
                codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
            }

            let (codeLogits, _, _) = talker.codePredictor(
                codeInput,
                cache: codeCache,
                generationStep: codeIdx
            )

            let nextCode = sampleToken(
                logits: codeLogits,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: 1.0,
                generatedTokens: nil,
                suppressTokens: nil,
                eosTokenId: nil
            )
            codeTokens.append(nextCode)
        }

        return codeTokens
    }

    private func combineCodecEmbeddings(
        codeTokens: [MLXArray],
        talker: Qwen3TTSTalkerForConditionalGeneration
    ) -> MLXArray {
        var codecEmbed = talker.getInputEmbeddings()(codeTokens[0])
        for (idx, code) in codeTokens.dropFirst().enumerated() {
            codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[idx](code)
        }
        return codecEmbed
    }

    func prepareGenerationInputs(
        text: String,
        language: String,
        speaker: String?,
        instruct: String?,
        speakerHintTokens: [Int]?,
        referencePromptTokens: [Int]?,
        tokenizer: Qwen3TTSTokenizer,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        config: Qwen3TTSModelConfig
    ) -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        let talkerConfig = config.talkerConfig
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(chatText).map { Int32($0) }).reshaped(1, -1)

        let ttsTokens = MLXArray([
            Int32(config.ttsBosTokenId),
            Int32(config.ttsEosTokenId),
            Int32(config.ttsPadTokenId)
        ]).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        var languageId: Int?
        if language.lowercased() != "auto", let map = talkerConfig.codecLanguageId {
            languageId = map[language.lowercased()]
        }

        var codecPrefill: [Int]
        if let languageId {
            codecPrefill = [
                talkerConfig.codecThinkId,
                talkerConfig.codecThinkBosId,
                languageId,
                talkerConfig.codecThinkEosId
            ]
        } else {
            codecPrefill = [
                talkerConfig.codecNoThinkId,
                talkerConfig.codecThinkBosId,
                talkerConfig.codecThinkEosId
            ]
        }

        if let speaker,
           let spkMap = talkerConfig.spkId,
           let spkIds = spkMap[speaker.lowercased()] {
            codecPrefill.append(contentsOf: spkIds)
        }
        if let speakerHintTokens, !speakerHintTokens.isEmpty {
            codecPrefill.append(contentsOf: speakerHintTokens)
        }
        if let referencePromptTokens, !referencePromptTokens.isEmpty {
            codecPrefill.append(contentsOf: referencePromptTokens)
        }

        var codecEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill.map { Int32($0) }).reshaped(1, -1))
        let codecSuffix = talker.getInputEmbeddings()(MLXArray([
            Int32(talkerConfig.codecPadId),
            Int32(talkerConfig.codecBosId)
        ]).reshaped(1, 2))
        codecEmbed = MLX.concatenated([codecEmbed, codecSuffix], axis: 1)

        var instructEmbed: MLXArray?
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds[0..., 0..<3]))
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, codecEmbed.dim(1) - 2, ttsPadEmbed.dim(2)])
        var combinedEmbed = MLX.concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., 0..<(codecEmbed.dim(1) - 1), 0...]

        var inputEmbeds: MLXArray
        if let instructEmbed {
            inputEmbeds = MLX.concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = MLX.concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        let firstTextEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds[0..., 3..<4]))
            + codecEmbed[0..., (codecEmbed.dim(1) - 1)..<codecEmbed.dim(1), 0...]
        inputEmbeds = MLX.concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        let totalTokens = inputIds.dim(1)
        let trailingStart = 4
        let trailingEnd = max(trailingStart, totalTokens - 5)
        let trailingEmbed: MLXArray
        if trailingEnd > trailingStart {
            trailingEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds[0..., trailingStart..<trailingEnd]))
        } else {
            trailingEmbed = MLXArray.zeros([1, 0, talkerConfig.hiddenSize])
        }

        return (inputEmbeds, MLX.concatenated([trailingEmbed, ttsEosEmbed], axis: 1), ttsPadEmbed)
    }

    func prepareICLGenerationInputs(
        text: String,
        refText: String,
        referenceCodes: MLXArray,
        language: String,
        speakerEmbedding: MLXArray?,
        tokenizer: Qwen3TTSTokenizer,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        config: Qwen3TTSModelConfig
    ) throws -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray, referenceCodesBQT: MLXArray) {
        let talkerConfig = config.talkerConfig

        guard referenceCodes.ndim == 3 else {
            throw Qwen3TTSError.invalidCloneReference("Reference codec tensor must be rank-3.")
        }

        let refCodesBQT: MLXArray
        if referenceCodes.dim(1) == talkerConfig.numCodeGroups {
            refCodesBQT = referenceCodes.asType(.int32)
        } else {
            refCodesBQT = referenceCodes.transposed(0, 2, 1).asType(.int32)
        }
        guard refCodesBQT.dim(2) > 0 else {
            throw Qwen3TTSError.invalidCloneReference("Reference audio produced empty codec frames.")
        }

        let refChat = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(refChat).map { Int32($0) }).reshaped(1, -1)
        let refTextEnd = max(3, refIds.dim(1) - 2)
        let refTextIds: MLXArray = refTextEnd > 3 ? refIds[0..., 3..<refTextEnd] : MLXArray.zeros([1, 0], dtype: .int32)

        let targetChat = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(targetChat).map { Int32($0) }).reshaped(1, -1)
        let textIdsEnd = max(3, targetIds.dim(1) - 5)
        let textIds: MLXArray = textIdsEnd > 3 ? targetIds[0..., 3..<textIdsEnd] : MLXArray.zeros([1, 0], dtype: .int32)

        let ttsTokens = MLXArray([
            Int32(config.ttsBosTokenId),
            Int32(config.ttsEosTokenId),
            Int32(config.ttsPadTokenId)
        ]).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        let combinedTextIds = MLX.concatenated([refTextIds, textIds], axis: 1)
        var textEmbed = talker.textProjection(talker.getTextEmbeddings()(combinedTextIds))
        textEmbed = MLX.concatenated([textEmbed, ttsEosEmbed], axis: 1)

        let firstCodebookCodes = refCodesBQT[0..., 0, 0...]
        var refCodecEmbed = talker.getInputEmbeddings()(firstCodebookCodes)
        let availableGroups = min(talkerConfig.numCodeGroups, refCodesBQT.dim(1))
        if availableGroups > 1 {
            for groupIdx in 1..<availableGroups {
                refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[groupIdx - 1](refCodesBQT[0..., groupIdx, 0...])
            }
        }

        let codecBosEmbed = talker.getInputEmbeddings()(MLXArray([Int32(talkerConfig.codecBosId)]).reshaped(1, 1))
        let codecEmbedICL = MLX.concatenated([codecBosEmbed, refCodecEmbed], axis: 1)
        let codecPadEmbed = talker.getInputEmbeddings()(MLXArray([Int32(talkerConfig.codecPadId)]).reshaped(1, 1))
        let textWithCodecPad = textEmbed + broadcast(codecPadEmbed, to: [1, textEmbed.dim(1), codecPadEmbed.dim(2)])
        let codecWithTextPad = codecEmbedICL + broadcast(ttsPadEmbed, to: [1, codecEmbedICL.dim(1), ttsPadEmbed.dim(2)])
        let iclInputEmbed = MLX.concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let trailingTextHidden = ttsPadEmbed

        var languageId: Int?
        if language.lowercased() != "auto", let map = talkerConfig.codecLanguageId {
            languageId = map[language.lowercased()]
        }

        var codecPrefill: [Int]
        if let languageId {
            codecPrefill = [talkerConfig.codecThinkId, talkerConfig.codecThinkBosId, languageId, talkerConfig.codecThinkEosId]
        } else {
            codecPrefill = [talkerConfig.codecNoThinkId, talkerConfig.codecThinkBosId, talkerConfig.codecThinkEosId]
        }

        var codecPrefixEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill.map { Int32($0) }).reshaped(1, -1))
        let codecPrefixSuffix = talker.getInputEmbeddings()(MLXArray([
            Int32(talkerConfig.codecPadId),
            Int32(talkerConfig.codecBosId)
        ]).reshaped(1, 2))

        if let speakerEmbedding {
            let speakerEmbed: MLXArray
            if speakerEmbedding.ndim == 1 {
                speakerEmbed = speakerEmbedding.reshaped(1, 1, speakerEmbedding.dim(0))
            } else if speakerEmbedding.ndim == 2 {
                speakerEmbed = speakerEmbedding.reshaped(speakerEmbedding.dim(0), 1, speakerEmbedding.dim(1))
            } else {
                speakerEmbed = speakerEmbedding
            }

            if speakerEmbed.dim(2) == talkerConfig.hiddenSize {
                codecPrefixEmbed = MLX.concatenated([codecPrefixEmbed, speakerEmbed, codecPrefixSuffix], axis: 1)
            } else {
                codecPrefixEmbed = MLX.concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
            }
        } else {
            codecPrefixEmbed = MLX.concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(targetIds[0..., 0..<3]))
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, codecPrefixEmbed.dim(1) - 2, ttsPadEmbed.dim(2)])
        var combinedPrefix = MLX.concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0..<(codecPrefixEmbed.dim(1) - 1), 0...]

        let inputEmbeds = MLX.concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)
        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodesBQT)
    }

    func generateVoiceCloneICL(
        inputEmbeds: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbed: MLXArray,
        referenceCodesBQT: MLXArray,
        targetTokenCount: Int,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        speechTokenizer: Qwen3TTSSpeechTokenizer,
        config: Qwen3TTSModelConfig,
        temperature: Float,
        progressHandler: (@Sendable (TTSProgress) -> Void)?,
        streamingChunkTokenInterval: Int? = nil,
        onToken: ((Int) -> Void)? = nil,
        onAudioDelta: (([Float]) -> Void)? = nil
    ) throws -> MLXArray {
        progressHandler?(TTSProgress(stage: .generating, message: "Generating tokens..."))

        let maxTokens = 4096
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))
        let topK = 50
        let topP: Float = 1.0
        let repetitionPenalty: Float = 1.5
        let eosTokenId = config.talkerConfig.codecEosTokenId
        let suppressTokens = Array((config.talkerConfig.vocabSize - 1024)..<config.talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        var generatedCodes: [MLXArray] = []
        var emittedSampleCount = 0

        if Qwen3TTSEnvironment.pipelinedDecodeEnabled {
            let result = try runPipelinedTalkerLoop(
                inputEmbeds: inputEmbeds,
                trailingTextHidden: trailingTextHidden,
                ttsPadEmbed: ttsPadEmbed,
                effectiveMaxTokens: effectiveMaxTokens,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                talker: talker,
                config: config,
                progressHandler: progressHandler,
                streamingChunkTokenInterval: streamingChunkTokenInterval,
                onToken: onToken,
                emitDelta: { codes, emitted in
                    try self.emitStreamingAudioDelta(
                        generatedCodes: codes,
                        referenceCodesBQT: referenceCodesBQT,
                        speechTokenizer: speechTokenizer,
                        emittedSampleCount: &emitted,
                        onAudioDelta: onAudioDelta
                    )
                }
            )
            generatedCodes = result.codes
            emittedSampleCount = result.emittedSampleCount
        } else {
            var generatedFirstTokens: [Int] = []
            let cache = talker.makeCache()
            var inputEmbedsVar = inputEmbeds
            var trailingIndex = 0

            for step in 0..<effectiveMaxTokens {
                let (logits, hidden) = talker(inputEmbedsVar, cache: cache)
                let nextToken = sampleToken(
                    logits: logits,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    generatedTokens: generatedFirstTokens,
                    suppressTokens: suppressTokens,
                    eosTokenId: eosTokenId
                )

                let tokenValue = nextToken.item(Int.self)
                if tokenValue == eosTokenId {
                    break
                }

                generatedFirstTokens.append(tokenValue)
                onToken?(tokenValue)

                let codeTokens = try generateCodecTokens(
                    firstToken: nextToken,
                    hidden: hidden,
                    talker: talker,
                    temperature: temperature,
                    topK: topK,
                    topP: topP
                )
                let allCodes = MLX.concatenated(codeTokens, axis: 1)
                generatedCodes.append(allCodes)

                if let streamingChunkTokenInterval, generatedCodes.count % streamingChunkTokenInterval == 0 {
                    try emitStreamingAudioDelta(
                        generatedCodes: generatedCodes,
                        referenceCodesBQT: referenceCodesBQT,
                        speechTokenizer: speechTokenizer,
                        emittedSampleCount: &emittedSampleCount,
                        onAudioDelta: onAudioDelta
                    )
                }

                let textEmbed: MLXArray
                if trailingIndex < trailingTextHidden.dim(1) {
                    textEmbed = trailingTextHidden[0..., trailingIndex..<(trailingIndex + 1), 0...]
                    trailingIndex += 1
                } else {
                    textEmbed = ttsPadEmbed
                }

                inputEmbedsVar = textEmbed + combineCodecEmbeddings(codeTokens: codeTokens, talker: talker)
                MLX.eval(inputEmbedsVar)

                if step > 0 && step % 25 == 0 {
                    progressHandler?(TTSProgress(stage: .generating, tokensGenerated: step, message: "Generated \(step) tokens..."))
                    Memory.clearCache()
                }
            }
        }

        guard !generatedCodes.isEmpty else {
            throw Qwen3TTSError.noAudioTokensGenerated
        }

        if streamingChunkTokenInterval != nil {
            try emitStreamingAudioDelta(
                generatedCodes: generatedCodes,
                referenceCodesBQT: referenceCodesBQT,
                speechTokenizer: speechTokenizer,
                emittedSampleCount: &emittedSampleCount,
                onAudioDelta: onAudioDelta
            )
        }

        progressHandler?(TTSProgress(stage: .decoding, message: "Decoding audio..."))

        let generated = MLX.stacked(generatedCodes, axis: 1)
        let referenceCodesBTQ = referenceCodesBQT.transposed(0, 2, 1)
        let fullCodes = MLX.concatenated([referenceCodesBTQ, generated], axis: 1)
        MLX.eval(fullCodes)
        Memory.clearCache()

        let (wav, lengths) = speechTokenizer.decode(fullCodes)
        var audio = wav.squeezed(axis: 0)
        let validLength = lengths[0].item(Int.self)
        if validLength > 0 && validLength < audio.size {
            audio = audio[0..<validLength]
        }

        let refLength = referenceCodesBQT.dim(2)
        let totalLength = fullCodes.dim(1)
        let cut = Int(Double(refLength) / Double(max(totalLength, 1)) * Double(audio.size))
        if cut > 0 && cut < audio.size {
            audio = audio[cut..<audio.size]
        }

        MLX.eval(audio)
        return audio
    }

    func emitStreamingAudioDelta(
        generatedCodes: [MLXArray],
        referenceCodesBQT: MLXArray?,
        speechTokenizer: Qwen3TTSSpeechTokenizer,
        emittedSampleCount: inout Int,
        onAudioDelta: (([Float]) -> Void)?
    ) throws {
        guard !generatedCodes.isEmpty, let onAudioDelta else { return }

        let generated = MLX.stacked(generatedCodes, axis: 1)
        let fullCodes: MLXArray
        if let referenceCodesBQT {
            let referenceCodesBTQ = referenceCodesBQT.transposed(0, 2, 1)
            fullCodes = MLX.concatenated([referenceCodesBTQ, generated], axis: 1)
        } else {
            fullCodes = generated
        }

        let (wav, lengths) = speechTokenizer.decode(fullCodes)
        var audio = wav.squeezed(axis: 0)
        let validLength = lengths[0].item(Int.self)
        if validLength > 0 && validLength < audio.size {
            audio = audio[0..<validLength]
        }

        if let referenceCodesBQT {
            let refLength = referenceCodesBQT.dim(2)
            let totalLength = fullCodes.dim(1)
            let cut = Int(Double(refLength) / Double(max(totalLength, 1)) * Double(audio.size))
            if cut > 0 && cut < audio.size {
                audio = audio[cut..<audio.size]
            }
        }

        MLX.eval(audio)
        let fullSamples = audio.asType(.float32).reshaped(-1).asArray(Float.self)
        let delta = streamingAudioNewTail(fullSamples: fullSamples, emittedSampleCount: emittedSampleCount)
        emittedSampleCount = delta.updatedSampleCount
        if !delta.samples.isEmpty {
            onAudioDelta(delta.samples)
        }
    }

    func sampleToken(
        logits: MLXArray,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        generatedTokens: [Int]?,
        suppressTokens: [Int]?,
        eosTokenId: Int?
    ) -> MLXArray {
        let lastLogits = logits[0..., (logits.dim(1) - 1), 0...]
        var scores = lastLogits.squeezed(axis: 0)
        if scores.dtype == .bfloat16 {
            scores = scores.asType(.float32)
        }

        if let suppressTokens, !suppressTokens.isEmpty {
            let indices = MLXArray(suppressTokens.map { Int32($0) })
            scores[indices] = MLXArray(Array(repeating: -Float.infinity, count: suppressTokens.count))
        }

        if let generatedTokens, !generatedTokens.isEmpty, repetitionPenalty != 1.0 {
            let indices = MLXArray(Array(Set(generatedTokens)).map { Int32($0) })
            let selected = scores[indices]
            let penalized = MLX.where(
                selected .< 0,
                selected * repetitionPenalty,
                selected / repetitionPenalty
            )
            scores[indices] = penalized
        }

        if temperature <= 0 {
            return MLXArray(Int32(argMax(scores, axis: -1).item(Int.self))).reshaped(1, 1)
        }

        scores = scores / temperature
        var eosLogit: MLXArray?
        if let eosTokenId, eosTokenId < scores.dim(0) {
            eosLogit = scores[eosTokenId]
        }

        if topK > 0 && topK < scores.dim(0) {
            let sortedIndices = argSort(scores, axis: -1)
            let sortedScores = scores.take(sortedIndices, axis: -1)
            let threshold = sortedScores[scores.dim(0) - topK]
            scores = MLX.where(scores .< threshold, MLXArray(-Float.infinity), scores)
        }

        if topP < 1.0 {
            let probs = softmax(scores, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = probs.take(sortedIndices, axis: -1)
            let cumulative = cumsum(sortedProbs, axis: -1)
            let cutoffMask = cumulative .> (1.0 - topP)
            let shifted = MLX.concatenated([
                MLX.zeros([1], dtype: .bool),
                cutoffMask[0..<(cutoffMask.dim(0) - 1)]
            ], axis: -1)
            let cutoffIndex = argMax(shifted.asType(.int32), axis: -1).item(Int.self)
            let threshold = sortedProbs[cutoffIndex]
            scores = MLX.where(probs .< threshold, MLXArray(-Float.infinity), scores)
        }

        if let eosTokenId, let eosLogit {
            scores[MLXArray([Int32(eosTokenId)])] = eosLogit.reshaped(1)
        }

        return MLXArray(Int32(categorical(scores).item(Int.self))).reshaped(1, 1)
    }
}
