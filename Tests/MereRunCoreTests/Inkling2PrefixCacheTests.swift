import XCTest
import MereRunInkling2Model
@testable import MereRunCore

/// Checks that run inside the generator's isolation, so caches never leave the actor.
extension Inkling2Generator {
    func store(_ tokens: [Int], count: Int, modelPath: String = "/m", speculative: Bool = false,
               priority: RuntimePrefixCacheEntryPriority = .semantic) {
        storePrefixKVCache(modelPath: modelPath, speculative: speculative, promptTokens: tokens, tokenCount: count,
                           cache: Inkling2Cache(layerCount: 2), draft: nil, priority: priority)
    }

    func seedLength(_ tokens: [Int], reusable: Int? = nil, modelPath: String = "/m", speculative: Bool = false) -> Int? {
        prefixKVCacheSeed(modelPath: modelPath, speculative: speculative, promptTokens: tokens,
                          reusableTokens: reusable ?? tokens.count)?.tokenCount
    }

    func seedsAreIndependentCopies(_ tokens: [Int]) -> Bool {
        guard let first = prefixKVCacheSeed(modelPath: "/m", speculative: false, promptTokens: tokens, reusableTokens: tokens.count),
              let second = prefixKVCacheSeed(modelPath: "/m", speculative: false, promptTokens: tokens, reusableTokens: tokens.count),
              let stored = prefixKVCache.values.first?.cache else { return false }
        return first.cache !== second.cache && first.cache !== stored && second.cache !== stored
    }
}

final class Inkling2PrefixCacheTests: XCTestCase {
    func testPrefillChunksStayOnTheColdGridAndStopAtCheckpoints() {
        func ends(from start: Int, total: Int, checkpoints: Set<Int> = []) -> [Int] {
            var result: [Int] = [], processed = start
            while processed < total {
                processed = Inkling2Generator.nextPrefillEnd(processed: processed, total: total, chunkSize: 32, checkpoints: checkpoints)
                result.append(processed)
            }
            return result
        }
        XCTAssertEqual(ends(from: 0, total: 70), [32, 64, 70])
        // A reused prefix of 40 tokens rejoins the grid at 64, as a cold prefill would.
        XCTAssertEqual(ends(from: 40, total: 70), [64, 70])
        XCTAssertEqual(ends(from: 0, total: 70, checkpoints: [10, 64, 70]), [10, 32, 64, 70])
    }

    func testSeedIsTheLongestStoredPrefixAndLeavesATokenToPrefill() async {
        let generator = Inkling2Generator(prefixKVCacheEnabled: true)
        let prompt = Array(0..<100)
        for count in [20, 60, 100] { await generator.store(prompt, count: count) }
        let longer = prompt + [7, 8]
        let extended = await generator.seedLength(longer)
        XCTAssertEqual(extended, 100)
        // The whole prompt is stored, but a request must still prefill its last token for logits.
        let same = await generator.seedLength(prompt)
        XCTAssertEqual(same, 60)
        // A prefix may not reach an image.
        let image = await generator.seedLength(longer, reusable: 50)
        XCTAssertEqual(image, 20)
        let diverged = await generator.seedLength([1] + prompt)
        XCTAssertNil(diverged)
        let otherModel = await generator.seedLength(longer, modelPath: "/n")
        XCTAssertNil(otherModel)
        // DSpark needs its draft state, which a plain prefill does not keep.
        let speculative = await generator.seedLength(longer, speculative: true)
        XCTAssertNil(speculative)

        let stats = await generator.prefixKVCacheStats()
        XCTAssertTrue(stats.enabled)
        XCTAssertEqual(stats.entries, 3)
        XCTAssertEqual(stats.storedPrefixes, 3)
        XCTAssertEqual(stats.hits, 3)
        XCTAssertEqual(stats.misses, 3)
        XCTAssertEqual(stats.reusedTokens, 180)
        XCTAssertEqual(stats.storedTokens, 180)
    }

    func testSeedIsACopyThatPrefillCannotChange() async {
        let generator = Inkling2Generator(prefixKVCacheEnabled: true)
        await generator.store(Array(0..<10), count: 5)
        let independent = await generator.seedsAreIndependentCopies(Array(0..<10))
        XCTAssertTrue(independent)
    }

    func testPruningDropsTheLeastRecentlyUsedAndStorageIsOffWhenDisabled() async {
        let generator = Inkling2Generator(prefixKVCacheEnabled: true)
        let prompt = Array(0..<200)
        for count in stride(from: 20, through: 200, by: 20) { await generator.store(prompt, count: count) }
        let kept = await generator.prefixKVCacheStats()
        XCTAssertEqual(kept.entries, Inkling2Generator.prefixKVCacheMaxEntries)
        XCTAssertEqual(kept.storedTokens, 140 + 160 + 180 + 200)
        // A speculative entry without draft state is not stored.
        await generator.store(prompt, count: 30, speculative: true)
        let after = await generator.prefixKVCacheStats()
        XCTAssertEqual(after.storedPrefixes, 10)

        let disabled = Inkling2Generator(prefixKVCacheEnabled: false)
        await disabled.store(prompt, count: 10)
        let seed = await disabled.seedLength(prompt)
        XCTAssertNil(seed)
        let stats = await disabled.prefixKVCacheStats()
        XCTAssertFalse(stats.enabled)
        XCTAssertEqual(stats.entries, 0)
    }

    func testServerRuntimeReportsInklingPrefixReuse() async {
        let runtime = NativeChatRuntime.make(engine: .textChatInkling2, modelID: "vision-chat-inkling-2-mini", modelPath: "/m",
                                             prefixKVCacheEnabled: true)
        let stats = await runtime.prefixKVCacheStats()
        XCTAssertEqual(stats?.enabled, true)
    }
}
