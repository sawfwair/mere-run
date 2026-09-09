import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func semanticPrefixCheckpoints(
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        messages: [ChatMessage],
        tools: [ToolDefinition]?,
        includeThinking: Bool,
        promptTokens: [Int],
        maxContextLength: Int
    ) -> Set<Int> {
        guard prefixKVCacheEnabled, messages.count > 1 else {
            return []
        }
        let prefixMessages = Array(messages.dropLast())
        guard !prefixMessages.isEmpty,
              let prefixTokens = try? tokenizerAndTemplate.encodeForGeneration(
                  messages: prefixMessages,
                  tools: tools,
                  addGenerationPrompt: false,
                  includeThinking: includeThinking,
                  maxLength: maxContextLength
              ),
              promptTokens.starts(with: prefixTokens) else {
            return []
        }
        return RuntimePrefillCheckpointPlanner.normalizedCheckpoints(
            [prefixTokens.count],
            total: promptTokens.count
        )
    }

    func prefixKVCacheSeed(
        modelPath: String,
        quantization: Gemma4KVCacheQuantization,
        promptTokens: [Int]
    ) -> (tokenCount: Int, caches: [Gemma4AttentionCache], logits: MLXArray)? {
        guard prefixKVCacheEnabled else { return nil }
        let matchingKey = prefixKVCache.keys
            .filter { key in
                key.modelPath == modelPath
                    && key.quantization == quantization
                    && key.tokens.count <= promptTokens.count
                    && promptTokens.starts(with: key.tokens)
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
            entry.caches.map { $0.fork() },
            entry.logits
        )
    }

    func storePrefixKVCache(
        modelPath: String,
        quantization: Gemma4KVCacheQuantization,
        promptTokens: [Int],
        tokenCount: Int,
        cache: [Gemma4AttentionCache],
        logits: MLXArray,
        priority: RuntimePrefixCacheEntryPriority
    ) {
        guard prefixKVCacheEnabled, tokenCount > 0 else { return }
        let key = Gemma4PrefixKVCacheKey(
            modelPath: modelPath,
            quantization: quantization,
            tokens: Array(promptTokens.prefix(tokenCount))
        )
        prefixKVCache[key] = Gemma4PrefixKVCacheEntry(
            caches: cache.map { $0.fork() },
            logits: logits,
            priority: priority,
            lastAccess: Date()
        )
        prefixKVCacheStores += 1
        prunePrefixKVCache()
    }

    func prunePrefixKVCache() {
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
}
