import XCTest
import MLX
import MLXRandom
@testable import MereRunCore

final class Gemma4KVQuantizationTests: MereRunCoreTestCase {
    func testUniformRejectsFractionalBits() {
        let config = Gemma4KVCacheQuantization(bits: 3.5, scheme: .uniform, groupSize: 64, quantizedStart: 0)
        XCTAssertThrowsError(try config.validated())
    }

    func testTurboquantUsesHalfBitSplit() throws {
        let config = try Gemma4KVCacheQuantization(bits: 3.5, scheme: .turboquant, groupSize: 64, quantizedStart: 0).validated()
        XCTAssertEqual(config.keyBits, 3)
        XCTAssertEqual(config.valueBits, 4)
    }

    func testQuantizedCacheRoundTripsShapeAndOffset() throws {
        MLXRandom.seed(7)
        let config = try Gemma4KVCacheQuantization(bits: 4, scheme: .uniform, groupSize: 64, quantizedStart: 0).validated()
        let cache = Gemma4QuantizedKVCache(configuration: config, maxSize: nil)

        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 3, 64]).asType(.bfloat16)
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 3, 64]).asType(.bfloat16)
        cache.append(keys: keys, values: values)
        let returned = try XCTUnwrap(cache.currentState())

        XCTAssertEqual(cache.offset, 3)
        XCTAssertEqual(returned.0.shape, [1, 2, 3, 64])
        XCTAssertEqual(returned.1.shape, [1, 2, 3, 64])

        let keyDelta = returned.0.asType(DType.float32) - keys.asType(DType.float32)
        let valueDelta = returned.1.asType(DType.float32) - values.asType(DType.float32)
        let keyMSE = MLX.mean(keyDelta * keyDelta)
        let valueMSE = MLX.mean(valueDelta * valueDelta)
        XCTAssertLessThan(keyMSE.item(Float.self), 0.02)
        XCTAssertLessThan(valueMSE.item(Float.self), 0.02)
    }

    func testFullCacheForkCanMutateIndependently() throws {
        MLXRandom.seed(8)
        let cache = Gemma4FullKVCache()

        let firstKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        let firstValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        cache.append(keys: firstKeys, values: firstValues)

        let forked = cache.fork()
        let nextKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 1, 8]).asType(.bfloat16)
        let nextValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 1, 8]).asType(.bfloat16)
        forked.append(keys: nextKeys, values: nextValues)

        let original = try XCTUnwrap(cache.currentState())
        let copy = try XCTUnwrap(forked.currentState())
        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(forked.offset, 3)
        XCTAssertEqual(original.0.shape, [1, 1, 2, 8])
        XCTAssertEqual(copy.0.shape, [1, 1, 3, 8])
    }

    func testQuantizedCacheForkCanMutateIndependently() throws {
        MLXRandom.seed(10)
        let config = try Gemma4KVCacheQuantization(bits: 4, scheme: .uniform, groupSize: 32, quantizedStart: 0).validated()
        let cache = Gemma4QuantizedKVCache(configuration: config, maxSize: nil)

        let firstKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 32]).asType(.bfloat16)
        let firstValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 32]).asType(.bfloat16)
        cache.append(keys: firstKeys, values: firstValues)

        let forked = cache.fork()
        let nextKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 1, 32]).asType(.bfloat16)
        let nextValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 1, 32]).asType(.bfloat16)
        forked.append(keys: nextKeys, values: nextValues)

        let original = try XCTUnwrap(cache.currentState())
        let copy = try XCTUnwrap(forked.currentState())
        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(forked.offset, 3)
        XCTAssertEqual(original.0.shape, [1, 1, 2, 32])
        XCTAssertEqual(copy.0.shape, [1, 1, 3, 32])
    }

    func testQuantizedCachesCanMergeAndSplitBatchRows() throws {
        MLXRandom.seed(12)
        let config = try Gemma4KVCacheQuantization(bits: 4, scheme: .uniform, groupSize: 32, quantizedStart: 0).validated()
        let caches = (0..<2).map { _ in Gemma4QuantizedKVCache(configuration: config, maxSize: nil) }

        for cache in caches {
            let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 3, 32]).asType(.bfloat16)
            let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 3, 32]).asType(.bfloat16)
            cache.append(keys: keys, values: values)
        }

        let batched = try XCTUnwrap(caches[0].batched(with: caches))
        let state = try XCTUnwrap(batched.currentState())
        XCTAssertEqual(batched.offset, 3)
        XCTAssertEqual(state.0.shape, [2, 1, 3, 32])
        XCTAssertEqual(state.1.shape, [2, 1, 3, 32])

        let rows = try XCTUnwrap(batched.unbatchedRows(count: 2))
        XCTAssertEqual(rows.count, 2)
        for row in rows {
            let rowState = try XCTUnwrap(row.currentState())
            XCTAssertEqual(row.offset, 3)
            XCTAssertEqual(rowState.0.shape, [1, 1, 3, 32])
            XCTAssertEqual(rowState.1.shape, [1, 1, 3, 32])
        }
    }

    func testSlidingQuantizedCacheRespectsMaxSize() throws {
        MLXRandom.seed(9)
        let config = try Gemma4KVCacheQuantization(bits: 4, scheme: .turboquant, groupSize: 64, quantizedStart: 0).validated()
        let cache = Gemma4QuantizedKVCache(configuration: config, maxSize: 4)

        let firstKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 3, 64]).asType(.bfloat16)
        let firstValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 3, 64]).asType(.bfloat16)
        cache.append(keys: firstKeys, values: firstValues)

        let secondKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 3, 64]).asType(.bfloat16)
        let secondValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 3, 64]).asType(.bfloat16)
        cache.append(keys: secondKeys, values: secondValues)
        let returned = try XCTUnwrap(cache.currentState())

        XCTAssertEqual(cache.offset, 6)
        XCTAssertEqual(returned.0.shape, [1, 1, 4, 64])
        XCTAssertEqual(returned.1.shape, [1, 1, 4, 64])
    }

    func testSpecializedAttentionMatchesDenseDecodeShapeAndValue() throws {
        MLXRandom.seed(13)
        let config = try Gemma4KVCacheQuantization(bits: 4, scheme: .turboquant, groupSize: 64, quantizedStart: 0).validated()
        let cache = Gemma4QuantizedKVCache(configuration: config, maxSize: nil)

        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 32, 64]).asType(.bfloat16)
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 32, 64]).asType(.bfloat16)
        cache.append(keys: keys, values: values)

        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 4, 1, 64]).asType(.bfloat16)
        let scale: Float = 1.0 / sqrt(64.0)

        let specialized = try XCTUnwrap(cache.specializedAttention(queries: queries, repeats: 2, scale: scale))
        let denseKeys = MLX.repeated(keys.asType(.float32), count: 2, axis: 1)
        let denseValues = MLX.repeated(values.asType(.float32), count: 2, axis: 1)
        var denseScores = MLX.matmul(queries.asType(.float32), denseKeys.transposed(0, 1, 3, 2)) * MLXArray(scale)
        denseScores = softmax(denseScores, axis: -1)
        let dense = MLX.matmul(denseScores, denseValues).asType(specialized.dtype)

        XCTAssertEqual(specialized.shape, [1, 4, 1, 64])
        let delta = specialized.asType(.float32) - dense.asType(.float32)
        let mse = MLX.mean(delta * delta)
        XCTAssertLessThan(mse.item(Float.self), 0.02)
    }

    func testFusedSpecializedAttentionMatchesDenseDecodeShapeAndValue() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("MLX fast Metal kernels require a GPU device.")
        }

        MLXRandom.seed(21)
        let config = try Gemma4KVCacheQuantization(bits: 4, scheme: .turboquant, groupSize: 64, quantizedStart: 0).validated()
        let cache = Gemma4QuantizedKVCache(configuration: config, maxSize: nil)

        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 48, 64]).asType(.bfloat16)
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 48, 64]).asType(.bfloat16)
        cache.append(keys: keys, values: values)

        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 4, 1, 64]).asType(.bfloat16)
        let scale: Float = 1.0 / sqrt(64.0)

        let fused = try XCTUnwrap(cache.fusedSpecializedAttention(queries: queries, repeats: 2, scale: scale))
        let denseKeys = MLX.repeated(keys.asType(.float32), count: 2, axis: 1)
        let denseValues = MLX.repeated(values.asType(.float32), count: 2, axis: 1)
        var denseScores = MLX.matmul(queries.asType(.float32), denseKeys.transposed(0, 1, 3, 2)) * MLXArray(scale)
        denseScores = softmax(denseScores, axis: -1)
        let dense = MLX.matmul(denseScores, denseValues).asType(fused.dtype)

        XCTAssertEqual(fused.shape, [1, 4, 1, 64])
        let delta = fused.asType(.float32) - dense.asType(.float32)
        let mse = MLX.mean(delta * delta)
        XCTAssertLessThan(mse.item(Float.self), 0.02)
    }
}
