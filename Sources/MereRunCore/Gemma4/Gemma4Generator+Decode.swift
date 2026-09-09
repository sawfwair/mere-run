import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func decodeTokens(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        mtpModel: Gemma4AssistantDraftModel?,
        prefillHidden: MLXArray?,
        sharedKVStates: [String: Gemma4SharedKVState],
        prefillTokenCount: Int,
        mtpStatsTemplate: Gemma4MTPStats,
        jsonConstrained: Bool = false,
        noRepeatNgramSize: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        guard tokenBudget > 0 else {
            return Gemma4BatchedDecodeResult(
                generatedTokens: [],
                decodeSeconds: 0,
                firstTokenSeconds: nil,
                mtpStats: mtpStatsTemplate
            )
        }
        // JSON-constrained requests always decode serially: the batched rows share
        // one sampling path and cannot carry per-request scanner state.
        guard continuousBatchingEnabled, !jsonConstrained, noRepeatNgramSize == nil else {
            return try await decodeTokensSerially(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                promptTokens: promptTokens,
                mtpModel: mtpModel,
                prefillHidden: prefillHidden,
                sharedKVStates: sharedKVStates,
                prefillTokenCount: prefillTokenCount,
                mtpStatsTemplate: mtpStatsTemplate,
                jsonConstrained: jsonConstrained,
                noRepeatNgramSize: noRepeatNgramSize,
                progressHandler: progressHandler
            )
        }

        let rowID = UUID()
        let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
        let layerCachesBox = RuntimeUncheckedSendable(layerCaches)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let row = Gemma4BatchedDecodeRow(
                    id: rowID,
                    logits: initialLogitsBox.value,
                    layerCaches: layerCachesBox.value,
                    eosSet: eosSet,
                    generationConfig: generationConfig,
                    tokenBudget: tokenBudget,
                    repetitionHistory: promptTokens,
                    progressHandler: progressHandler,
                    continuation: continuation
                )
                enqueueDecodeRow(row, model: model, tokenizerAndTemplate: tokenizerAndTemplate)
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelDecodeRow(id: rowID)
            }
        }
    }

    func canUsePipelinedDecode(
        _ config: GenerationConfig,
        mtpModel: Gemma4AssistantDraftModel?,
        jsonConstrained: Bool
    ) -> Bool {
        // MTP verification and JSON-constrained decoding both need each token on
        // the CPU before the next forward; everything else can pipeline, sampling
        // included — the token stays on the GPU and only the previous step's
        // readback blocks. Prompt-lookup speculation runs as bursts inside the
        // pipelined loop itself, so it never forces the serial path.
        mtpModel == nil && !jsonConstrained
    }

    func decodeTokensPipelined(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        mtpStatsTemplate: Gemma4MTPStats,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        var layerCaches = layerCaches
        var mtpStats = mtpStatsTemplate
        var promptLookupMisses = 0
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = greedyRepetitionHistoryArray(
            promptTokens: promptTokens,
            config: generationConfig
        )
        let banMask = tokenBanMask(
            vocabularySize: initialLogits.dim(-1),
            dtype: initialLogits.dtype,
            tokens: generationConfig.bannedTokens
        )
        var firstTokenSeconds: Double?
        var pendingToken = sampledTokenArray(
            logits: initialLogits[0, -1, 0...],
            config: generationConfig,
            previousTokenIndices: repetitionHistory,
            banMask: banMask
        )
        MLX.asyncEval(pendingToken)
        let decodeStart = Date()
        let traceEnabled = Gemma4DecodeTrace.enabled
        var traceBuildSeconds = 0.0
        var traceWaitSeconds = 0.0
        var traceForwardSeconds = 0.0
        var traceSampleSeconds = 0.0
        var traceScheduleSeconds = 0.0

        decode: while generated.count < tokenBudget {
            try Task.checkCancellation()

            let buildStart = CFAbsoluteTimeGetCurrent()
            var forwardEnd = buildStart
            var sampleEnd = buildStart
            let scheduled: (token: MLXArray, history: MLXArray?)?
            if generated.count + 1 < tokenBudget {
                let nextInput = pendingToken.asType(.int32).reshaped(1, 1)
                let nextLogits = model.forward(inputIds: nextInput, cache: layerCaches)
                forwardEnd = CFAbsoluteTimeGetCurrent()
                let nextHistory = appendingGreedyRepetitionHistory(
                    repetitionHistory,
                    token: pendingToken,
                    config: generationConfig
                )
                let nextToken = sampledTokenArray(
                    logits: nextLogits[0, -1, 0...],
                    config: generationConfig,
                    previousTokenIndices: nextHistory,
                    banMask: banMask
                )
                sampleEnd = CFAbsoluteTimeGetCurrent()
                MLX.asyncEval(nextToken, nextLogits)
                scheduled = (nextToken, nextHistory)
            } else {
                scheduled = nil
            }
            let buildEnd = CFAbsoluteTimeGetCurrent()

            let next = pendingToken.item(Int.self)
            if traceEnabled {
                traceForwardSeconds += forwardEnd - buildStart
                traceSampleSeconds += sampleEnd - forwardEnd
                traceScheduleSeconds += buildEnd - sampleEnd
                traceBuildSeconds += buildEnd - buildStart
                traceWaitSeconds += CFAbsoluteTimeGetCurrent() - buildEnd
            }
            if eosSet.contains(next) {
                break
            }

            generated.append(next)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(decodeStart)
            }
            if let progressHandler {
                let piece = tokenizerAndTemplate.decode(token: next)
                if !piece.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: piece))
                }
            }

            if Self.promptLookupSpeculationEnabled,
               generationConfig.temperature <= 0,
               promptLookupMisses < 3,
               let scheduled,
               generated.count < tokenBudget {
                switch promptLookupBurst(
                    model: model,
                    tokenizerAndTemplate: tokenizerAndTemplate,
                    scheduledToken: scheduled.token,
                    promptTokens: promptTokens,
                    eosSet: eosSet,
                    generationConfig: generationConfig,
                    tokenBudget: tokenBudget,
                    generated: &generated,
                    layerCaches: &layerCaches,
                    pendingToken: &pendingToken,
                    repetitionHistory: &repetitionHistory,
                    mtpStats: &mtpStats,
                    promptLookupMisses: &promptLookupMisses,
                    progressHandler: progressHandler
                ) {
                case .notAttempted:
                    break
                case .resumed:
                    continue decode
                case .finished:
                    break decode
                }
            }

            guard let scheduled, generated.count < tokenBudget else {
                break
            }
            pendingToken = scheduled.token
            repetitionHistory = scheduled.history
        }

        if traceEnabled, !generated.isEmpty {
            let count = Double(generated.count)
            Gemma4DecodeTrace.emit(String(
                format: "[gemma4-decode-trace] mode=pipelined temp=\(generationConfig.temperature) tokens=%d build=%.2fms/tok (forward=%.2f sample=%.2f schedule=%.2f) wait=%.2fms/tok wall=%.2fms/tok",
                generated.count,
                traceBuildSeconds / count * 1000,
                traceForwardSeconds / count * 1000,
                traceSampleSeconds / count * 1000,
                traceScheduleSeconds / count * 1000,
                traceWaitSeconds / count * 1000,
                Date().timeIntervalSince(decodeStart) / count * 1000
            ))
        }

        return Gemma4BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds,
            mtpStats: mtpStats
        )
    }

    func greedyRepetitionHistoryArray(
        promptTokens: [Int],
        config: GenerationConfig
    ) -> MLXArray? {
        repetitionHistoryArray(promptTokens: promptTokens, config: config)
    }

    func appendingGreedyRepetitionHistory(
        _ history: MLXArray?,
        token: MLXArray,
        config: GenerationConfig
    ) -> MLXArray? {
        appendingRepetitionHistory(history, token: token, config: config)
    }

    func forkLayerCaches(_ caches: [Gemma4AttentionCache]) -> [Gemma4AttentionCache] {
        caches.map { $0.fork() }
    }

    func lastTokenLogits(_ logits: MLXArray) -> MLXArray {
        logits[0..., (logits.dim(1) - 1)..., 0...]
    }

    func lastTokenHidden(_ hidden: MLXArray) -> MLXArray {
        hidden[0..., (hidden.dim(1) - 1)..., 0...]
    }
}
