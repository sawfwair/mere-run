import MLX
import XCTest
@testable import MereRunGemmaModel
import MereRunMLXTestSupport

final class Gemma4KVCacheForkTests: MLXTestCase {
    func testFullCacheForkKeepsSavedPromptWhenAnotherBranchAppends() throws {
        try checkBranches(Gemma4FullKVCache(), window: nil)
    }

    func testSlidingCacheForkKeepsSavedPromptBeforeWindowWrap() throws {
        try checkBranches(Gemma4SlidingKVCache(maxSize: 8), window: 8)
    }

    func testSlidingCacheForkKeepsSavedPromptAfterWindowWrap() throws {
        try checkBranches(Gemma4SlidingKVCache(maxSize: 2), window: 2)
    }

    private func append(_ value: Float, to cache: Gemma4AttentionCache) {
        cache.append(keys: MLXArray([value]).reshaped(1, 1, 1, 1),
                     values: MLXArray([value * 10]).reshaped(1, 1, 1, 1))
        cache.evaluateStorage()
    }

    private func checkBranches(_ cache: Gemma4AttentionCache, window: Int?) throws {
        append(1, to: cache)
        append(2, to: cache)
        let sharedPrompt = cache.fork()
        append(3, to: cache)
        let firstPrompt = cache.fork()
        let secondPrompt = sharedPrompt.fork()
        append(9, to: secondPrompt)

        let expected = Array([Float(1), 2, 3].suffix(window ?? 3))
        for preserved in [cache, firstPrompt] {
            let state = try XCTUnwrap(preserved.currentState())
            XCTAssertEqual(state.0.asArray(Float.self), expected)
            XCTAssertEqual(state.1.asArray(Float.self), expected.map { $0 * 10 })
            XCTAssertEqual(preserved.offset, 3)
        }
        let shared = try XCTUnwrap(sharedPrompt.currentState())
        XCTAssertEqual(shared.0.asArray(Float.self), [1, 2])
        XCTAssertEqual(shared.1.asArray(Float.self), [10, 20])
        XCTAssertEqual(sharedPrompt.offset, 2)
        let changed = try XCTUnwrap(secondPrompt.currentState())
        let changedExpected = Array([Float(1), 2, 9].suffix(window ?? 3))
        XCTAssertEqual(changed.0.asArray(Float.self), changedExpected)
        XCTAssertEqual(changed.1.asArray(Float.self), changedExpected.map { $0 * 10 })
    }
}
