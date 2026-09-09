import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    func chunkedPrefill(
        model: Q35Model,
        promptTokens: [Int],
        cache: [Q35LayerCache?],
        startIndex: Int = 0,
        existingLogits: MLXArray? = nil,
        existingHidden: MLXArray? = nil,
        modelPath: String? = nil,
        checkpointTokenCounts: Set<Int> = [],
        retainHidden: Bool = true,
        retainMTPHistory: Bool = false,
        mtpSession: Q35MTPDraftSession? = nil,
        prefillMTPModel: (any Q35MTPDraftModel)? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35PrefillOutput {
        guard !promptTokens.isEmpty else {
            throw Q35Error.generationFailed("Prompt is empty after tokenization.")
        }

        var processed = startIndex
        var logits = existingLogits
        var hidden = existingHidden
        var mtpHistoryChunks: [MLXArray] = []
        if processed > 0, processed < promptTokens.count {
            progressHandler?(ChatProgress(stage: .encoding, message: "Reusing \(processed) prompt KV tokens"))
        }
        while processed < promptTokens.count {
            try Task.checkCancellation()
            let chunkSize = Self.prefillChunkSize(
                modelId: modelId,
                activeRequestCount: activeChatRequestCount
            )
            let end = RuntimePrefillCheckpointPlanner.nextEnd(
                processed: processed,
                total: promptTokens.count,
                chunkSize: chunkSize,
                checkpoints: checkpointTokenCounts
            )
            if promptTokens.count > chunkSize {
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(promptTokens.count) tokens"))
            }
            let chunk = MLXArray(promptTokens[processed..<end].map(Int32.init))
                .reshaped(1, end - processed)
            if retainHidden {
                let output = model.forwardPrefill(
                    chunk,
                    cache: cache,
                    retainAllHidden: retainMTPHistory
                )
                let outputMTPHidden = output.mtpHidden ?? output.hidden
                MLX.eval(output.logits)
                MLX.eval(outputMTPHidden)
                logits = output.logits
                if let mtpSession, let prefillMTPModel {
                    if let hidden, processed > 0 {
                        mtpSession.recordCommittedTransitions(
                            hiddenStates: hidden, nextTokens: [promptTokens[processed]]
                        )
                    }
                    mtpSession.recordCommittedTransitions(
                        hiddenStates: outputMTPHidden,
                        nextTokens: Array(promptTokens[(processed + 1)..<end])
                    )
                    mtpSession.primeCommittedHistory(mtpModel: prefillMTPModel, baseModel: model)
                } else if retainMTPHistory {
                    mtpHistoryChunks.append(outputMTPHidden)
                }
                hidden = lastTokenHidden(outputMTPHidden)
                let priority: RuntimePrefixCacheEntryPriority? = mtpSession != nil || model.config.textConfig.isQwen4Exp
                    ? RuntimePrefillCheckpointPlanner.storagePriority(
                        tokenCount: end, total: promptTokens.count,
                        semanticCheckpoints: checkpointTokenCounts
                    )
                    : (checkpointTokenCounts.contains(end) ? .semantic : .chunk)
                if let modelPath, let priority {
                    storePrefixKVCache(
                        modelPath: modelPath,
                        promptTokens: promptTokens,
                        tokenCount: end,
                        cache: cache,
                        logits: output.logits,
                        hidden: lastTokenHidden(outputMTPHidden),
                        mtpSession: mtpSession,
                        priority: priority
                    )
                }
            } else {
                let output = model.forwardPrefill(chunk, cache: cache)
                MLX.eval(output.logits)
                logits = output.logits
                hidden = nil
            }
            clearMLXCacheUnderPressureIfNeeded()
            processed = end
            await Task.yield()
        }

        guard let logits else {
            throw Q35Error.generationFailed("Prefill did not produce logits.")
        }
        let mtpHistoryHidden = mtpHistoryChunks.isEmpty
            ? nil
            : MLX.concatenated(mtpHistoryChunks, axis: 1)
        if let mtpHistoryHidden {
            MLX.eval(mtpHistoryHidden)
        }
        return Q35PrefillOutput(
            logits: logits,
            hidden: hidden,
            mtpHistoryHidden: mtpHistoryHidden,
            mtpSession: mtpSession
        )
    }

    func chunkedPrefillEmbeddings(
        model: Q35Model,
        inputIds: MLXArray,
        inputEmbeddings: MLXArray,
        cache: [Q35LayerCache?],
        positionIds: MLXArray? = nil,
        retainHidden: Bool = true,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35PrefillOutput {
        let tokenCount = inputEmbeddings.dim(1)
        guard tokenCount > 0 else {
            throw Q35Error.generationFailed("Prompt embeddings are empty after tokenization.")
        }

        var processed = 0
        var logits: MLXArray?
        var hidden: MLXArray?
        while processed < tokenCount {
            try Task.checkCancellation()
            let chunkSize = Self.prefillChunkSize(
                modelId: modelId,
                activeRequestCount: activeChatRequestCount
            )
            let end = min(processed + chunkSize, tokenCount)
            if tokenCount > chunkSize {
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(tokenCount) tokens"))
            }
            let chunkEmbeddings = inputEmbeddings[0..., processed..<end, 0...]
            // Flash-Next PLE still consumes the original token IDs after
            // image embeddings replace the multimodal placeholder vectors.
            let chunkInput = inputIds[0..., processed..<end]
            let chunkPositionIds = positionIds?[0..., 0..., processed..<end]
            if retainHidden {
                let output = model.forwardPrefill(
                    chunkInput,
                    cache: cache,
                    inputEmbeddings: chunkEmbeddings,
                    positionIds: chunkPositionIds
                )
                let outputMTPHidden = output.mtpHidden ?? output.hidden
                MLX.eval(output.logits)
                MLX.eval(outputMTPHidden)
                logits = output.logits
                hidden = outputMTPHidden
            } else {
                let output = model.forwardPrefill(
                    chunkInput,
                    cache: cache,
                    inputEmbeddings: chunkEmbeddings,
                    positionIds: chunkPositionIds
                )
                MLX.eval(output.logits)
                logits = output.logits
                hidden = nil
            }
            clearMLXCacheUnderPressureIfNeeded()
            processed = end
            await Task.yield()
        }

        guard let logits else {
            throw Q35Error.generationFailed("Prefill did not produce logits.")
        }
        return Q35PrefillOutput(
            logits: logits,
            hidden: hidden,
            mtpHistoryHidden: nil,
            mtpSession: nil
        )
    }

    func semanticPrefixCheckpoints(
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        messages: [ChatMessage],
        tools: [ToolDefinition]?,
        includeThinking: Bool,
        reasoningEffort: String?,
        promptTokens: [Int],
        maxContextLength: Int
    ) -> Set<Int> {
        guard prefixKVCacheEnabled, messages.count > 1 else {
            return []
        }
        let prefixMessages = Array(messages.dropLast())
        guard !prefixMessages.isEmpty else {
            return []
        }
        guard let prefixTokens = try? tokenizerAndTemplate.encodeForGeneration(
            messages: prefixMessages,
            tools: tools,
            addGenerationPrompt: false,
            includeThinking: includeThinking,
            reasoningEffort: reasoningEffort,
            maxLength: maxContextLength
        ) else {
            return []
        }
        guard promptTokens.starts(with: prefixTokens) else {
            return []
        }
        return RuntimePrefillCheckpointPlanner.normalizedCheckpoints(
            [prefixTokens.count],
            total: promptTokens.count
        )
    }

    func prefixKVCacheSeed(
        modelPath: String,
        promptTokens: [Int],
        cacheMode: RuntimeKVCacheMode,
        requiresMTPSession: Bool = false
    ) -> (tokenCount: Int, caches: [Q35LayerCache?], logits: MLXArray,
          hidden: MLXArray, mtpSession: Q35MTPDraftSession?)? {
        guard prefixKVCacheEnabled else { return nil }
        let matchingKey = prefixKVCache.keys
            .filter { key in
                key.modelPath == modelPath
                    && key.cacheMode == cacheMode
                    && key.tokens.count <= promptTokens.count
                    && promptTokens.starts(with: key.tokens)
                    && (!requiresMTPSession || prefixKVCache[key]?.mtpSession != nil)
            }
            .max { $0.tokens.count < $1.tokens.count }

        guard let matchingKey,
              var entry = prefixKVCache[matchingKey] else {
            prefixKVCacheMisses += 1
            return nil
        }

        entry.lastAccess = Date()
        prefixKVCache[matchingKey] = entry
        prefixKVCacheHits += 1
        prefixKVCacheReusedTokens += matchingKey.tokens.count
        return (
            matchingKey.tokens.count,
            forkLayerCaches(entry.caches),
            entry.logits,
            entry.hidden,
            entry.mtpSession?.fork()
        )
    }

    private func storePrefixKVCache(
        modelPath: String,
        promptTokens: [Int],
        tokenCount: Int,
        cache: [Q35LayerCache?],
        logits: MLXArray,
        hidden: MLXArray,
        mtpSession: Q35MTPDraftSession? = nil,
        priority: RuntimePrefixCacheEntryPriority
    ) {
        guard prefixKVCacheEnabled, tokenCount > 0 else { return }
        let key = Q35PrefixKVCacheKey(
            modelPath: modelPath,
            cacheMode: cacheMode(for: cache),
            tokens: Array(promptTokens.prefix(tokenCount))
        )
        prefixKVCache[key] = Q35PrefixKVCacheEntry(
            caches: forkLayerCaches(cache),
            logits: logits,
            hidden: hidden,
            mtpSession: mtpSession?.fork(),
            priority: priority,
            lastAccess: Date()
        )
        prefixKVCacheStores += 1
        prunePrefixKVCache()
    }

    private func prunePrefixKVCache() {
        while prefixKVCache.count > Self.prefixKVCacheMaxEntries {
            let metadata = prefixKVCache.mapValues {
                RuntimePrefixCacheRetentionMetadata(
                    priority: $0.priority,
                    lastAccess: $0.lastAccess
                )
            }
            guard let oldest = RuntimePrefixCacheRetentionPlanner.keyToPrune(entries: metadata) else {
                return
            }
            prefixKVCache.removeValue(forKey: oldest)
        }
    }

    func resetPrefixKVCache() {
        prefixKVCache.removeAll()
    }

    func forkLayerCaches(_ caches: [Q35LayerCache?]) -> [Q35LayerCache?] {
        caches.map { $0?.fork() }
    }

    func restoreVerificationCaches(
        _ caches: [Q35LayerCache?],
        totalTokens: Int,
        tokenCount: Int
    ) -> Bool {
        let presentCaches = caches.compactMap { $0 }
        guard presentCaches.allSatisfy({ cache in
            cache.canRestoreVerificationPrefix(
                totalTokens: totalTokens,
                tokenCount: tokenCount
            )
        }) else {
            return false
        }
        for cache in presentCaches {
            guard cache.restoreVerificationPrefix(
                totalTokens: totalTokens,
                tokenCount: tokenCount
            ) else {
                return false
            }
        }
        return true
    }

    func commitVerificationCaches(_ caches: [Q35LayerCache?]) {
        for cache in caches.compactMap({ $0 }) {
            cache.commitVerification()
        }
    }
}
