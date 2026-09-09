import Foundation
import MLX

extension LagunaGenerator {
    func generate(
        _ request: ChatRequest,
        dflashRouting: LagunaDFlashRoutingMode,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let config else {
            throw LagunaError.modelNotLoaded
        }

        let requestedContext = request.maxContextTokens ?? LagunaResources.defaultContextLength
        guard requestedContext > 0 else {
            throw LagunaError.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(requestedContext, config.maxPositionEmbeddings)

        progressHandler?(ChatProgress(stage: .encoding, message: "Encoding Laguna prompt"))
        let prefillStart = Date()
        let promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        guard !promptTokens.isEmpty else {
            throw LagunaError.generationFailed("The rendered prompt contained no tokens.")
        }

        let tokenBudget = max(0, min(
            request.maxTokens,
            effectiveContext - promptTokens.count
        ))
        let activeDFlash = request.logprobCapture.isEnabled ? nil : dflashModel.flatMap { dflash in
            let enabled = switch dflashRouting {
            case .automatic:
                LagunaDFlashRouting.shouldUseDFlash(
                    tokenBudget: tokenBudget,
                    minimumOutputTokens: dflashMinimumOutputTokens
                )
            case .targetOnly:
                false
            case .dflash:
                true
            }
            return enabled ? dflash : nil
        }
        if dflashModel != nil {
            if activeDFlash == nil {
                dflashBypassedRequests += 1
            } else {
                dflashRoutedRequests += 1
            }
        }

        let cache = model.makeCache()
        let prefill = try prefill(
            model: model,
            promptTokens: promptTokens,
            cache: cache,
            dflash: activeDFlash,
            progressHandler: progressHandler
        )
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let generationConfig = GenerationConfig(
            maxTokens: tokenBudget,
            temperature: Float(request.temperature),
            topK: request.topK ?? 0,
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: nil,
            repetitionContextSize: 64
        )
        let eosTokens = Self.resolvedEOSTokens(
            modelTokenIDs: config.eosTokenIDs,
            templateTokenIDs: tokenizerAndTemplate.stopTokenIDs,
            stopOnEOS: request.stopOnEOS
        )

        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decode = try await decodeTokens(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: prefill.logits,
            caches: cache,
            dflash: activeDFlash,
            dflashCache: prefill.dflashCache,
            eosTokens: eosTokens,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            promptTokens: promptTokens,
            dflashRouting: dflashRouting,
            logprobCapture: request.logprobCapture,
            logprobRegion: request.logprobRegionHint ?? .visible,
            progressHandler: progressHandler
        )

        let decoded = tokenizerAndTemplate.decode(tokens: decode.generatedTokens)
        let trimmed = TextGenerationStopSequences.trimming(
            decoded,
            sequences: request.stopSequences
        )
        let toolCalls: [ToolCall]? = request.tools?.isEmpty == false ? {
            let parsed = LagunaToolParser.parseToolCalls(trimmed.text)
            return parsed.isEmpty ? nil : parsed
        }() : nil
        let finishReason: ChatFinishReason
        if trimmed.matchedSequence != nil {
            finishReason = .stopSequence
        } else if decode.generatedTokens.count >= tokenBudget, tokenBudget > 0 {
            finishReason = .length
        } else {
            finishReason = .stop
        }

        return ChatResponse(
            generatedText: trimmed.text,
            tokensGenerated: decode.generatedTokens.count,
            showThinking: request.showThinking,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decode.decodeSeconds,
                firstTokenSeconds: decode.firstTokenSeconds,
                kvCacheMode: .default,
                prefillKVCache: "bf16",
                decodeKVCache: "bf16"
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            finishReason: finishReason,
            logprobs: decode.logprobs,
            acceleration: decode.acceleration
        )
    }

    static func resolvedEOSTokens(
        modelTokenIDs: [Int],
        templateTokenIDs: [Int],
        stopOnEOS: Bool
    ) -> Set<Int> {
        guard stopOnEOS else { return [] }
        return Set(modelTokenIDs + templateTokenIDs)
    }

    func decodeTokens(
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate,
        initialLogits: MLXArray,
        caches: [Gemma4AttentionCache],
        dflash: LagunaDFlashModel?,
        dflashCache: [Gemma4AttentionCache]?,
        eosTokens: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        dflashRouting: LagunaDFlashRoutingMode,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> LagunaDecodeResult {
        if let dflash, let dflashCache {
            if continuousBatchingEnabled, dflashRouting == .dflash {
                guard tokenBudget > 0 else {
                    return LagunaDecodeResult(
                        generatedTokens: [],
                        decodeSeconds: 0,
                        firstTokenSeconds: nil
                    )
                }
                let rowID = UUID()
                // The row and model stay confined to this generator actor. Swift
                // 6.0's targeted-concurrency checker still treats the continuation
                // closure as a send boundary, so make that ownership transfer
                // explicit just as the other continuous-batching runtimes do.
                let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
                let targetCachesBox = RuntimeUncheckedSendable(caches)
                let draftCachesBox = RuntimeUncheckedSendable(dflashCache)
                let modelBox = RuntimeUncheckedSendable(model)
                let dflashBox = RuntimeUncheckedSendable(dflash)
                return try await withTaskCancellationHandler {
                    try await withCheckedThrowingContinuation { continuation in
                        enqueueDFlashDecodeRow(
                            LagunaDFlashBatchedDecodeRow(
                                id: rowID,
                                logits: initialLogitsBox.value,
                                targetCaches: targetCachesBox.value,
                                draftCaches: draftCachesBox.value,
                                eosTokens: eosTokens,
                                generationConfig: generationConfig,
                                tokenBudget: tokenBudget,
                                repetitionHistory: promptTokens,
                                progressHandler: progressHandler,
                                continuation: continuation
                            ),
                            model: modelBox.value,
                            dflash: dflashBox.value,
                            tokenizerAndTemplate: tokenizerAndTemplate
                        )
                    }
                } onCancel: { [weak self] in
                    guard let self else { return }
                    Task {
                        await self.cancelDFlashDecodeRow(id: rowID)
                    }
                }
            }

            let result = try LagunaDFlashDecoder.decode(
                initialLogits: initialLogits,
                target: model,
                targetCache: caches,
                dflash: dflash,
                draftCache: dflashCache,
                generationConfig: generationConfig,
                eosTokens: eosTokens,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                speculativeTokens: dflashSpeculativeTokens,
                adaptiveMinimumAcceptanceRate: dflashRouting == .automatic
                    ? LagunaDFlashRouting.defaultMinimumAcceptanceRate
                    : nil,
                decodeToken: { tokenizerAndTemplate.decode(token: $0) },
                decodeTokens: { tokenizerAndTemplate.decode(tokens: $0) },
                emitPiece: { _, piece in
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            accumulateDFlashStats(result.stats)
            return LagunaDecodeResult(
                generatedTokens: result.generatedTokens,
                decodeSeconds: result.decodeSeconds,
                firstTokenSeconds: result.firstTokenSeconds,
                acceleration: ChatAccelerationDiagnostics(
                    route: "dflash-speculative",
                    draftModel: LagunaResources.dflashModelID,
                    rounds: result.stats.rounds,
                    draftedTokens: result.stats.draftedTokens,
                    acceptedDraftTokens: result.stats.acceptedDraftTokens
                )
            )
        }

        guard continuousBatchingEnabled && !logprobCapture.isEnabled else {
            let result = try AutoregressiveDecodeEngine.decode(
                AutoregressiveDecodeRequest(
                    initialLogits: initialLogits,
                    generationConfig: generationConfig,
                    eosTokens: eosTokens,
                    tokenBudget: tokenBudget,
                    historySeedTokens: promptTokens,
                    logprobCapture: logprobCapture,
                    logprobRegion: logprobRegion
                ),
                stepForward: { token in
                    model.lastPositionLogits(token, cache: caches)
                },
                decodeToken: { tokenizerAndTemplate.decode(token: $0) },
                decodeTokens: { tokenizerAndTemplate.decode(tokens: $0) },
                emitPiece: { _, piece in
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            return LagunaDecodeResult(
                generatedTokens: result.generatedTokens,
                decodeSeconds: result.decodeSeconds,
                firstTokenSeconds: result.firstTokenSeconds,
                logprobs: result.logprobs,
                acceleration: ChatAccelerationDiagnostics(route: "final-target-pipelined")
            )
        }

        guard tokenBudget > 0 else {
            return LagunaDecodeResult(
                generatedTokens: [],
                decodeSeconds: 0,
                firstTokenSeconds: nil
            )
        }

        let rowID = UUID()
        let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
        let cachesBox = RuntimeUncheckedSendable(caches)
        let modelBox = RuntimeUncheckedSendable(model)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                enqueueDecodeRow(
                    LagunaBatchedDecodeRow(
                        id: rowID,
                        logits: initialLogitsBox.value,
                        caches: cachesBox.value,
                        eosTokens: eosTokens,
                        generationConfig: generationConfig,
                        tokenBudget: tokenBudget,
                        repetitionHistory: promptTokens,
                        progressHandler: progressHandler,
                        continuation: continuation
                    ),
                    model: modelBox.value,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelDecodeRow(id: rowID)
            }
        }
    }

}
