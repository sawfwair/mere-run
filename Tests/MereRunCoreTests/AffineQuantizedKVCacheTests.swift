import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class AffineQuantizedKVCacheTests: MereRunCoreTestCase {
    func testFourBitCacheShrinksResidentStateAndReportsItsWidth() {
        MLXRandom.seed(7000)
        let cache = AffineQuantizedKVCache(groupSize: 64, bits: 4, step: 4)
        let keys = MLXRandom.uniform(low: -1, high: 1, [1, 2, 4, 64]).asType(.bfloat16)
        let values = MLXRandom.uniform(low: -1, high: 1, [1, 2, 4, 64]).asType(.bfloat16)
        let returned = cache.update(keys: keys, values: values)
        MLX.eval(returned.0, returned.1)

        let denseBytes = (keys.size * keys.itemSize) + (values.size * values.itemSize)
        XCTAssertEqual(cache.bitWidth, 4)
        XCTAssertEqual(Double(cache.storageBytes) / Double(denseBytes), 0.281_25, accuracy: 0.001)
        XCTAssertLessThan(meanSquaredError(returned.0, keys), 0.01)
        XCTAssertLessThan(meanSquaredError(returned.1, values), 0.01)
    }

    func testEightBitCacheRoundTripsIncrementalStateWithinTolerance() throws {
        MLXRandom.seed(7001)
        let cache = AffineQuantizedKVCache(groupSize: 64, bits: 8, step: 4)
        let firstKeys = MLXRandom.uniform(low: -1, high: 1, [1, 2, 3, 64]).asType(.bfloat16)
        let firstValues = MLXRandom.uniform(low: -1, high: 1, [1, 2, 3, 64]).asType(.bfloat16)
        let nextKeys = MLXRandom.uniform(low: -1, high: 1, [1, 2, 1, 64]).asType(.bfloat16)
        let nextValues = MLXRandom.uniform(low: -1, high: 1, [1, 2, 1, 64]).asType(.bfloat16)

        _ = cache.update(keys: firstKeys, values: firstValues)
        let returned = cache.update(keys: nextKeys, values: nextValues)
        let expectedKeys = concatenated([firstKeys, nextKeys], axis: 2)
        let expectedValues = concatenated([firstValues, nextValues], axis: 2)
        MLX.eval(returned.0, returned.1)

        XCTAssertEqual(cache.offset, 4)
        XCTAssertEqual(returned.0.shape, [1, 2, 4, 64])
        XCTAssertEqual(returned.1.shape, [1, 2, 4, 64])
        XCTAssertLessThan(meanSquaredError(returned.0, expectedKeys), 0.0001)
        XCTAssertLessThan(meanSquaredError(returned.1, expectedValues), 0.0001)
    }

    func testPackedResidentBuffersAreSmallerThanBF16() {
        MLXRandom.seed(7002)
        let cache = AffineQuantizedKVCache(groupSize: 64, bits: 8, step: 4)
        let keys = MLXRandom.uniform(low: -1, high: 1, [1, 2, 4, 64]).asType(.bfloat16)
        let values = MLXRandom.uniform(low: -1, high: 1, [1, 2, 4, 64]).asType(.bfloat16)
        _ = cache.update(keys: keys, values: values)

        let denseBytes = (keys.size * keys.itemSize) + (values.size * values.itemSize)
        let packedRatio = Double(cache.storageBytes) / Double(denseBytes)
        XCTAssertEqual(packedRatio, 0.531_25, accuracy: 0.001)
    }

    func testForkMutationDoesNotChangeParent() throws {
        MLXRandom.seed(7003)
        let cache = AffineQuantizedKVCache(groupSize: 64, bits: 8, step: 4)
        let first = MLXRandom.uniform(low: -1, high: 1, [1, 1, 2, 64]).asType(.bfloat16)
        _ = cache.update(keys: first, values: first)

        let fork = try XCTUnwrap(cache.fork() as? AffineQuantizedKVCache)
        let next = MLXRandom.uniform(low: -1, high: 1, [1, 1, 1, 64]).asType(.bfloat16)
        _ = fork.update(keys: next, values: next)

        let parent = try XCTUnwrap(cache.currentState())
        let child = try XCTUnwrap(fork.currentState())
        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(fork.offset, 3)
        XCTAssertEqual(parent.0.shape, [1, 1, 2, 64])
        XCTAssertEqual(child.0.shape, [1, 1, 3, 64])
    }

    func testSameOffsetCachesBatchAndSplit() throws {
        MLXRandom.seed(7004)
        let caches = (0..<2).map { _ in
            AffineQuantizedKVCache(groupSize: 64, bits: 8, step: 4)
        }
        for cache in caches {
            let keys = MLXRandom.uniform(low: -1, high: 1, [1, 1, 3, 64]).asType(.bfloat16)
            let values = MLXRandom.uniform(low: -1, high: 1, [1, 1, 3, 64]).asType(.bfloat16)
            _ = cache.update(keys: keys, values: values)
        }

        let batched = try XCTUnwrap(caches[0].batched(with: caches) as? AffineQuantizedKVCache)
        let state = try XCTUnwrap(batched.currentState())
        XCTAssertEqual(state.0.shape, [2, 1, 3, 64])
        XCTAssertEqual(state.1.shape, [2, 1, 3, 64])

        let rows = try XCTUnwrap(batched.unbatchedRows(count: 2))
        XCTAssertEqual(rows.count, 2)
        for row in rows {
            let typed = try XCTUnwrap(row as? AffineQuantizedKVCache)
            XCTAssertEqual(typed.offset, 3)
            XCTAssertEqual(try XCTUnwrap(typed.currentState()).0.shape, [1, 1, 3, 64])
        }
    }

    func testAffineHeadWidthsAndPaddedTailLayouts() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("BF16 padded KV-cache layouts require MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(7006)
        for width in [64, 128, 256] {
            for count in [4, 5, 256, 257] {
                let strided = MLXRandom.uniform(low: -1, high: 1, [1, count, 2, width])
                    .asType(.bfloat16).transposed(0, 2, 1, 3)
                for bits in [4, 8] {
                    for keys in [strided, MLX.contiguous(strided)] {
                        let cache = AffineQuantizedKVCache(groupSize: 64, bits: bits, step: 256)
                        let values = -keys
                        let returned = cache.update(keys: keys, values: values)
                        MLX.eval(returned.0, returned.1)
                        let tolerance: Float = bits == 8 ? 0.0001 : 0.01
                        XCTAssertLessThan(
                            meanSquaredError(returned.0, keys), tolerance,
                            "Keys changed at bits=\(bits), head width=\(width), tokens=\(count)"
                        )
                        XCTAssertLessThan(
                            meanSquaredError(returned.1, values), tolerance,
                            "Values changed at bits=\(bits), head width=\(width), tokens=\(count)"
                        )
                    }
                }
            }
        }
    }

    func testLongStridedEightBitCacheRoundTripAndForkIsolation() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Long BF16 KV-cache fixtures require MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(7005)
        for count in [64_740, 129_700] {
            let cache = AffineQuantizedKVCache(groupSize: 64, bits: 8, step: 256)
            let keys = MLXRandom.uniform(low: -1, high: 1, [1, count, 2, 256])
                .asType(.bfloat16).transposed(0, 2, 1, 3)
            let values = -keys
            let original = cache.update(keys: keys, values: values)
            MLX.eval(original.0, original.1)
            XCTAssertLessThan(meanSquaredError(original.0, keys), 0.0001)
            XCTAssertLessThan(meanSquaredError(original.1, values), 0.0001)
            XCTAssertLessThan((original.0.asType(.float32) - keys.asType(.float32)).abs().max().item(Float.self), 0.02)

            let child = try XCTUnwrap(cache.fork() as? AffineQuantizedKVCache)
            let next = MLXRandom.uniform(low: 0.25, high: 0.75, [1, 2, 4, 256]).asType(.bfloat16)
            let childState = child.update(keys: next, values: -next)
            MLX.eval(childState.0, childState.1)
            let parentState = cache.update(keys: -next, values: next)
            MLX.eval(parentState.0, parentState.1)
            let childAfterParentWrite = try XCTUnwrap(child.currentState())
            XCTAssertEqual(cache.offset, count + 4)
            XCTAssertEqual(child.offset, count + 4)
            for (old, parent, forked) in [
                (original.0, parentState.0, childAfterParentWrite.0),
                (original.1, parentState.1, childAfterParentWrite.1),
            ] {
                XCTAssertEqual((parent[0..., 0..., 0..<count, 0...] - old).abs().max().item(Float.self), 0)
                XCTAssertEqual((forked[0..., 0..., 0..<count, 0...] - old).abs().max().item(Float.self), 0)
            }
            XCTAssertLessThan(meanSquaredError(childAfterParentWrite.0[0..., 0..., count..., 0...], next), 0.0001)
            XCTAssertLessThan(meanSquaredError(parentState.0[0..., 0..., count..., 0...], -next), 0.0001)
        }
    }

    private func meanSquaredError(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let delta = lhs.asType(.float32) - rhs.asType(.float32)
        return MLX.mean(delta * delta).item(Float.self)
    }
}
