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

    func testQuantizedCacheForkKeepsSavedPromptWhenAnotherBranchAppends() throws {
        try checkQuantizedBranches { try Self.uniformQuantizedCache(maxSize: nil) }
    }

    func testSlidingQuantizedCacheForkKeepsSavedPromptBeforeWindowWrap() throws {
        try checkQuantizedBranches { try Self.uniformQuantizedCache(maxSize: 8) }
    }

    func testSlidingQuantizedCacheForkKeepsSavedPromptAfterWindowWrap() throws {
        try checkQuantizedBranches { try Self.uniformQuantizedCache(maxSize: 2) }
    }

    func testPolarCacheForkKeepsSavedPromptWhenAnotherBranchAppends() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("PolarKV uses MLXFast Metal pack/unpack kernels; set MERERUN_TEST_MLX_DEVICE=gpu to run it.")
        }
        try checkQuantizedBranches {
            let config = try Gemma4KVCacheQuantization(
                bits: 2, scheme: .polar, groupSize: 32, quantizedStart: 0
            ).validated()
            return Gemma4PolarKVCache(configuration: config, maxSize: nil)
        }
    }

    private static func uniformQuantizedCache(maxSize: Int?) throws -> Gemma4QuantizedKVCache {
        let config = try Gemma4KVCacheQuantization(
            bits: 8, scheme: .uniform, groupSize: 32, quantizedStart: 0
        ).validated()
        return Gemma4QuantizedKVCache(configuration: config, maxSize: maxSize)
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

    // Quantized rows are lossy, so each branch is compared with a cache of the same
    // configuration that received the expected tokens without forking. A constant row
    // of 32 identical values also quantizes exactly, which keeps the per-row readout
    // in failure messages meaningful.
    private func appendRow(_ value: Float, to cache: Gemma4AttentionCache) {
        cache.append(keys: MLXArray.full([1, 1, 1, 32], values: MLXArray(value)),
                     values: MLXArray.full([1, 1, 1, 32], values: MLXArray(value * 10)))
        cache.evaluateStorage()
    }

    private func rows(_ array: MLXArray) -> [Float] {
        array[0, 0, 0..., 0].asArray(Float.self)
    }

    private func checkQuantizedBranches(makeCache: () throws -> Gemma4AttentionCache) throws {
        let cache = try makeCache()
        appendRow(1, to: cache)
        appendRow(2, to: cache)
        let sharedPrompt = cache.fork()
        appendRow(3, to: cache)
        let firstPrompt = cache.fork()
        let secondPrompt = sharedPrompt.fork()
        appendRow(9, to: secondPrompt)

        for (name, preserved) in [("cache", cache), ("firstPrompt", firstPrompt)] {
            try assertBranch(name, preserved, holds: [1, 2, 3], makeCache: makeCache)
        }
        try assertBranch("sharedPrompt", sharedPrompt, holds: [1, 2], makeCache: makeCache)
        try assertBranch("secondPrompt", secondPrompt, holds: [1, 2, 9], makeCache: makeCache)
    }

    private func assertBranch(
        _ name: String,
        _ branch: Gemma4AttentionCache,
        holds tokens: [Float],
        makeCache: () throws -> Gemma4AttentionCache
    ) throws {
        let reference = try makeCache()
        for token in tokens {
            appendRow(token, to: reference)
        }
        let expected = try XCTUnwrap(reference.currentState())
        let state = try XCTUnwrap(branch.currentState())
        XCTAssertEqual(rows(state.0), rows(expected.0), "\(name) keys")
        XCTAssertEqual(rows(state.1), rows(expected.1), "\(name) values")
        XCTAssertTrue(MLX.arrayEqual(state.0, expected.0).item(Bool.self), "\(name) keys")
        XCTAssertTrue(MLX.arrayEqual(state.1, expected.1).item(Bool.self), "\(name) values")
        XCTAssertEqual(branch.offset, tokens.count, "\(name) offset")
    }
}
