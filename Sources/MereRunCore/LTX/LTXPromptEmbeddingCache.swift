import Foundation
import MLX

struct LTXPromptEmbeddingCacheKey: Hashable, Sendable {
    let prompt: String
    let maxLength: Int

    init(prompt: String, maxLength: Int) {
        self.prompt = prompt
        self.maxLength = maxLength
    }
}

struct LTXCachedPromptEmbeddings: @unchecked Sendable {
    let video: MLXArray
    let audio: MLXArray?
}

public struct LTXPromptCacheStatistics: Codable, Hashable, Sendable {
    public let capacity: Int
    public let entries: Int
    public let hits: Int
    public let misses: Int

    public init(capacity: Int, entries: Int, hits: Int, misses: Int) {
        self.capacity = capacity
        self.entries = entries
        self.hits = hits
        self.misses = misses
    }
}

struct LTXPromptEmbeddingCache {
    private(set) var capacity: Int
    private var values: [LTXPromptEmbeddingCacheKey: LTXCachedPromptEmbeddings] = [:]
    private var leastToMostRecentlyUsed: [LTXPromptEmbeddingCacheKey] = []
    private var hitCount = 0
    private var missCount = 0

    init(capacity: Int = 0) {
        self.capacity = max(0, capacity)
    }

    var isEnabled: Bool { capacity > 0 }

    func contains(_ key: LTXPromptEmbeddingCacheKey) -> Bool {
        values[key] != nil
    }

    mutating func value(for key: LTXPromptEmbeddingCacheKey) -> LTXCachedPromptEmbeddings? {
        guard let value = values[key] else {
            if isEnabled {
                missCount += 1
            }
            return nil
        }
        hitCount += 1
        touch(key)
        return value
    }

    mutating func insert(
        _ value: LTXCachedPromptEmbeddings,
        for key: LTXPromptEmbeddingCacheKey
    ) {
        guard isEnabled else { return }
        values[key] = value
        touch(key)
        while values.count > capacity, let evicted = leastToMostRecentlyUsed.first {
            leastToMostRecentlyUsed.removeFirst()
            values[evicted] = nil
        }
    }

    mutating func removeAll(keepingCapacity: Bool = true) {
        values.removeAll(keepingCapacity: false)
        leastToMostRecentlyUsed.removeAll(keepingCapacity: false)
        hitCount = 0
        missCount = 0
        if !keepingCapacity {
            capacity = 0
        }
    }

    func statistics() -> LTXPromptCacheStatistics {
        LTXPromptCacheStatistics(
            capacity: capacity,
            entries: values.count,
            hits: hitCount,
            misses: missCount
        )
    }

    private mutating func touch(_ key: LTXPromptEmbeddingCacheKey) {
        leastToMostRecentlyUsed.removeAll { $0 == key }
        leastToMostRecentlyUsed.append(key)
    }
}
