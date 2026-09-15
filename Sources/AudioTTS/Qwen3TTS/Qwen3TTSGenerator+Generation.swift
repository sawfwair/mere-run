import AudioQwen3TTSModel
import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs

/// Owns prompt preparation, token generation, and waveform assembly for TTS.
/// This file is intentionally separate from model loading so the speech
/// generation flow can be read end-to-end.
/// Non-Sendable values captured by the pipelined decode's emit closure. The
/// closure runs synchronously on the generator actor (runPipelinedTalkerLoop
/// invokes it inline), so no concurrent access exists; the box states that to
/// the Linux toolchain's region-isolation analysis, which otherwise rejects
/// the captures with "sending ... risks causing data races".
private struct Qwen3TTSEmitCaptures: @unchecked Sendable {
    let referenceCodesBQT: MLXArray?
    let speechTokenizer: Qwen3TTSSpeechTokenizer
    let onAudioDelta: (([Float]) -> Void)?
}

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
        try Task.checkCancellation()
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

        try Task.checkCancellation()
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
                emitDelta: { [captures = Qwen3TTSEmitCaptures(
                    referenceCodesBQT: nil,
                    speechTokenizer: speechTokenizer,
                    onAudioDelta: onAudioDelta
                )] codes, emitted in
                    try self.emitStreamingAudioDelta(
                        generatedCodes: codes,
                        referenceCodesBQT: captures.referenceCodesBQT,
                        speechTokenizer: captures.speechTokenizer,
                        emittedSampleCount: &emitted,
                        onAudioDelta: captures.onAudioDelta
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
                try Task.checkCancellation()
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

        try Task.checkCancellation()
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
                emitDelta: { [captures = Qwen3TTSEmitCaptures(
                    referenceCodesBQT: referenceCodesBQT,
                    speechTokenizer: speechTokenizer,
                    onAudioDelta: onAudioDelta
                )] codes, emitted in
                    try self.emitStreamingAudioDelta(
                        generatedCodes: codes,
                        referenceCodesBQT: captures.referenceCodesBQT,
                        speechTokenizer: captures.speechTokenizer,
                        emittedSampleCount: &emitted,
                        onAudioDelta: captures.onAudioDelta
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
                try Task.checkCancellation()
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

        try Task.checkCancellation()
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

}
