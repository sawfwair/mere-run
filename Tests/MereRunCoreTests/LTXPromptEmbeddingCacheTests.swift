import MLX
import XCTest
@testable import MereRunCore

final class LTXPromptEmbeddingCacheTests: MereRunCoreTestCase {
    func testDisabledCacheDoesNotRetainValuesOrCountMisses() {
        var cache = LTXPromptEmbeddingCache()
        let key = LTXPromptEmbeddingCacheKey(prompt: " fox ", maxLength: 1_024)

        cache.insert(
            LTXCachedPromptEmbeddings(video: MLXArray([Float(1)]), audio: nil),
            for: key
        )

        XCTAssertNil(cache.value(for: key))
        XCTAssertEqual(cache.statistics(), LTXPromptCacheStatistics(capacity: 0, entries: 0, hits: 0, misses: 0))
    }

    func testCacheRecordsExactPromptHits() throws {
        var cache = LTXPromptEmbeddingCache(capacity: 2)
        let inserted = LTXPromptEmbeddingCacheKey(prompt: "a fox in snow", maxLength: 1_024)
        let lookedUp = LTXPromptEmbeddingCacheKey(prompt: "a fox in snow", maxLength: 1_024)
        cache.insert(
            LTXCachedPromptEmbeddings(
                video: MLXArray([Float(3)]),
                audio: MLXArray([Float(4)])
            ),
            for: inserted
        )

        let value = try XCTUnwrap(cache.value(for: lookedUp))

        XCTAssertEqual(value.video.item(Float.self), 3)
        XCTAssertEqual(value.audio?.item(Float.self), 4)
        XCTAssertEqual(cache.statistics(), LTXPromptCacheStatistics(capacity: 2, entries: 1, hits: 1, misses: 0))
    }

    func testCacheKeepsWhitespaceSensitivePromptsDistinct() {
        var cache = LTXPromptEmbeddingCache(capacity: 2)
        let value = LTXCachedPromptEmbeddings(video: MLXArray([Float(1)]), audio: nil)
        cache.insert(value, for: LTXPromptEmbeddingCacheKey(prompt: "fox", maxLength: 128))

        XCTAssertNil(cache.value(for: LTXPromptEmbeddingCacheKey(prompt: " fox", maxLength: 128)))
    }

    func testCacheEvictsLeastRecentlyUsedEntry() {
        var cache = LTXPromptEmbeddingCache(capacity: 2)
        let first = LTXPromptEmbeddingCacheKey(prompt: "first", maxLength: 1)
        let second = LTXPromptEmbeddingCacheKey(prompt: "second", maxLength: 1)
        let third = LTXPromptEmbeddingCacheKey(prompt: "third", maxLength: 1)
        let value = LTXCachedPromptEmbeddings(video: MLXArray([Float(1)]), audio: nil)
        cache.insert(value, for: first)
        cache.insert(value, for: second)
        XCTAssertNotNil(cache.value(for: first))

        cache.insert(value, for: third)

        XCTAssertFalse(cache.contains(second))
        XCTAssertTrue(cache.contains(first))
        XCTAssertTrue(cache.contains(third))
    }

    func testCacheKeyIncludesMaximumLength() {
        var cache = LTXPromptEmbeddingCache(capacity: 2)
        let value = LTXCachedPromptEmbeddings(video: MLXArray([Float(1)]), audio: nil)
        cache.insert(value, for: LTXPromptEmbeddingCacheKey(prompt: "fox", maxLength: 128))

        XCTAssertNil(cache.value(for: LTXPromptEmbeddingCacheKey(prompt: "fox", maxLength: 1_024)))
        XCTAssertEqual(cache.statistics().misses, 1)
    }
}
