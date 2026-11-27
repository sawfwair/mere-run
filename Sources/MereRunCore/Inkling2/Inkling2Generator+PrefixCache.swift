import Foundation
import MediaIO
import MLX
import MereRunInkling2Model

struct Inkling2PrefixKVCacheKey: Hashable {
    let modelPath: String
    /// Entries prefilled through DSpark also hold the draft's state.
    let speculative: Bool
    let tokens: [Int]
}

struct Inkling2PrefixKVCacheEntry {
    let cache: Inkling2Cache
    let draft: Inkling2DSparkPrefix?
    let priority: RuntimePrefixCacheEntryPriority
    var lastAccess: Date
}

struct Inkling2PrefixKVCacheSeed {
    let tokenCount: Int
    let cache: Inkling2Cache
    let draft: Inkling2DSparkPrefix?
}

extension Inkling2Generator {
    static let prefixKVCacheMaxEntries = 4

    static let imageFeatureCacheMaxEntries = 16

    /// The end of every message but the last, when it is an exact prefix of the prompt's key tokens.
    /// Earlier messages hold the prompt's first images, in order.
    func semanticPrefixCheckpoints(_ request: ChatRequest, template: Inkling2ChatTemplate, promptTokens: [Int],
                                   imageMarkers: [Int], imageCounts: [Int]) -> Set<Int> {
        guard prefixKVCacheEnabled, request.messages.count > 1, let tokenizer else { return [] }
        let effort = request.reasoningEffort ?? 0.9
        guard let rendered = try? template.render(messages: Array(request.messages.dropLast()), tools: request.tools, effort: effort) else {
            return []
        }
        let raw = tokenizer.encode(text: rendered, addSpecialTokens: false)
        guard let earlier = Inkling2Prompt.keyTokens(raw: raw, images: zip(imageMarkers, imageCounts).map { ($0, $1) }),
              promptTokens.starts(with: earlier) else { return [] }
        return RuntimePrefillCheckpointPlanner.normalizedCheckpoints([earlier.count], total: promptTokens.count)
    }

    /// The vision features of an image, encoded once while it stays among the recent images.
    func cachedImageFeatures(_ image: MediaImage, marker: Int, model: Inkling2Model) throws -> MLXArray {
        if let cached = imageFeatureCache[marker] {
            imageFeatureOrder.removeAll { $0 == marker }
            imageFeatureOrder.append(marker)
            return cached
        }
        let encoded = try Inkling2Prompt.encode(image, model: model)
        imageFeatureCache[marker] = encoded
        imageFeatureOrder.append(marker)
        while imageFeatureOrder.count > Self.imageFeatureCacheMaxEntries {
            imageFeatureCache.removeValue(forKey: imageFeatureOrder.removeFirst())
        }
        return encoded
    }

    /// The longest stored prefix of this prompt, leaving at least one token to prefill.
    func prefixKVCacheSeed(modelPath: String, speculative: Bool, promptTokens: [Int],
                           reusableTokens: Int) -> Inkling2PrefixKVCacheSeed? {
        guard prefixKVCacheEnabled else { return nil }
        let limit = min(reusableTokens, promptTokens.count - 1)
        let match = prefixKVCache.keys
            .filter { $0.modelPath == modelPath && $0.speculative == speculative && $0.tokens.count <= limit
                && promptTokens.starts(with: $0.tokens) }
            .max { $0.tokens.count < $1.tokens.count }
        guard let match, var entry = prefixKVCache[match] else {
            prefixKVCacheMisses += 1
            return nil
        }
        entry.lastAccess = Date()
        prefixKVCache[match] = entry
        prefixKVCacheHits += 1
        prefixKVCacheReusedTokens += match.tokens.count
        return Inkling2PrefixKVCacheSeed(tokenCount: match.tokens.count, cache: entry.cache.copy(), draft: entry.draft)
    }

    func storePrefixKVCache(modelPath: String, speculative: Bool, promptTokens: [Int], tokenCount: Int,
                            cache: Inkling2Cache, draft: Inkling2DSparkPrefix?, priority: RuntimePrefixCacheEntryPriority) {
        guard prefixKVCacheEnabled, tokenCount > 0, !speculative || draft != nil else { return }
        let key = Inkling2PrefixKVCacheKey(modelPath: modelPath, speculative: speculative, tokens: Array(promptTokens.prefix(tokenCount)))
        prefixKVCache[key] = Inkling2PrefixKVCacheEntry(cache: cache.copy(), draft: draft, priority: priority, lastAccess: Date())
        prefixKVCacheStores += 1
        while prefixKVCache.count > Self.prefixKVCacheMaxEntries {
            let metadata = prefixKVCache.mapValues { RuntimePrefixCacheRetentionMetadata(priority: $0.priority, lastAccess: $0.lastAccess) }
            guard let oldest = RuntimePrefixCacheRetentionPlanner.keyToPrune(entries: metadata) else { return }
            prefixKVCache.removeValue(forKey: oldest)
        }
    }

    func resetPrefixKVCache() { prefixKVCache.removeAll() }

    public func prefixKVCacheStats() -> PrefixKVCacheStats {
        PrefixKVCacheStats(
            enabled: prefixKVCacheEnabled, entries: prefixKVCache.count, maxEntries: Self.prefixKVCacheMaxEntries,
            hits: prefixKVCacheHits, misses: prefixKVCacheMisses, storedPrefixes: prefixKVCacheStores,
            reusedTokens: prefixKVCacheReusedTokens, storedTokens: prefixKVCache.keys.reduce(0) { $0 + $1.tokens.count }
        )
    }

    /// The end of the next prefill chunk. Chunks stay on the grid that a cold prefill uses, so a
    /// reused prefix and a cold prompt run the same shapes, and break early only at checkpoints.
    static func nextPrefillEnd(processed: Int, total: Int, chunkSize: Int, checkpoints: Set<Int>) -> Int {
        let gridEnd = min((processed / chunkSize + 1) * chunkSize, total)
        return checkpoints.filter { $0 > processed && $0 < gridEnd }.min() ?? gridEnd
    }
}
