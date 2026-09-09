import AudioQwen3TTSModel
import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs

extension Qwen3TTSGenerator {
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
    func runPipelinedTalkerLoop(
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

    func generateCodecTokens(
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

    func combineCodecEmbeddings(
        codeTokens: [MLXArray],
        talker: Qwen3TTSTalkerForConditionalGeneration
    ) -> MLXArray {
        var codecEmbed = talker.getInputEmbeddings()(codeTokens[0])
        for (idx, code) in codeTokens.dropFirst().enumerated() {
            codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[idx](code)
        }
        return codecEmbed
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
