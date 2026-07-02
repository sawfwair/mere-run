import Foundation
import XCTest
import MLX
import MLXFast
import MLXRandom
@testable import MereRunCore

final class Gemma4SlidingKVCacheDecodeStateTests: MereRunCoreTestCase {
    private func appendTokens(_ cache: Gemma4SlidingKVCache, count: Int, heads: Int, dim: Int, seed: UInt64) {
        MLXRandom.seed(seed)
        for _ in 0..<count {
            let keys = MLXRandom.normal([1, heads, 1, dim]).asType(.float16)
            let values = MLXRandom.normal([1, heads, 1, dim]).asType(.float16)
            cache.append(keys: keys, values: values)
        }
    }

    func testDecodeStateMatchesTemporalOrderBeforeWrap() {
        let cache = Gemma4SlidingKVCache(maxSize: 64)
        appendTokens(cache, count: 20, heads: 2, dim: 8, seed: 3)

        guard let ordered = cache.currentState(), let decode = cache.decodeState() else {
            XCTFail("expected cache state after appends")
            return
        }
        XCTAssertEqual(ordered.0.shape, decode.0.shape)
        XCTAssertEqual(
            MLX.abs(ordered.0 - decode.0).max().item(Float.self), 0,
            "before the ring wraps, decodeState must equal temporal order"
        )
        XCTAssertEqual(MLX.abs(ordered.1 - decode.1).max().item(Float.self), 0)
    }

    func testDecodeStateAttentionMatchesTemporalOrderAfterWrap() {
        let heads = 2
        let dim = 8
        let window = 32
        let cache = Gemma4SlidingKVCache(maxSize: window)
        appendTokens(cache, count: 75, heads: heads, dim: dim, seed: 5)

        guard let ordered = cache.currentState(), let decode = cache.decodeState() else {
            XCTFail("expected cache state after appends")
            return
        }
        XCTAssertEqual(ordered.0.dim(2), window)
        XCTAssertEqual(decode.0.dim(2), window)

        // Same token set, different order — single-query unmasked attention
        // must produce the same result.
        MLXRandom.seed(9)
        let queries = MLXRandom.normal([1, heads, 1, dim]).asType(.float16)
        let attendedOrdered = MLXFast.scaledDotProductAttention(
            queries: queries, keys: ordered.0, values: ordered.1, scale: 1, mask: .none
        )
        let attendedDecode = MLXFast.scaledDotProductAttention(
            queries: queries, keys: decode.0, values: decode.1, scale: 1, mask: .none
        )
        let difference = MLX.abs(
            attendedOrdered.asType(.float32) - attendedDecode.asType(.float32)
        ).max().item(Float.self)
        XCTAssertLessThan(difference, 1e-2)
    }
}
