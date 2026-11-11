import MLX
import MLXRandom
import XCTest
import MereRunMLXTestSupport

final class M5AffineQ8AccelerationTests: MLXTestCase {
    private func requireMetal() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Run with MERERUN_TEST_MLX_DEVICE=gpu and MLX_ENABLE_TF32=0.")
        }
    }

    private func assertReference(_ actual: MLXArray, _ reference: MLXArray,
                                 label: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, reference.shape, label, file: file, line: line)
        let value = actual.asType(.float32)
        XCTAssertTrue(MLX.all(MLX.isFinite(value)).item(Bool.self), label, file: file, line: line)
        let difference = value - reference
        let relative = MLX.sqrt(MLX.sum(difference * difference) / MLX.sum(reference * reference)).item(Float.self)
        XCTAssertLessThan(relative, 0.01, "\(label): relative error \(relative)", file: file, line: line)
        let peak = MLX.max(MLX.abs(difference)) / MLX.sqrt(MLX.mean(reference * reference))
        XCTAssertLessThan(peak.item(Float.self), 0.08, "\(label): localized error", file: file, line: line)
    }

    private func check(rows: Int, input: Int, output: Int, group: Int, dtype: DType) {
        let x = MLXRandom.normal([rows, input], key: MLXRandom.key(1729)).asType(dtype)
        let w = MLXRandom.normal([output, input], key: MLXRandom.key(1730)).asType(dtype)
        let q = MLX.quantized(w, groupSize: group, bits: 8)
        let restored = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                                       groupSize: group, bits: 8, dtype: .float32)
        let actual = MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, groupSize: group, bits: 8)
        assertReference(actual, MLX.matmul(x.asType(.float32), restored.T),
                        label: "\(dtype) M=\(rows) K=\(input) N=\(output) group=\(group)")
    }

    func testHistoricalWideOutputThresholdAndTileEdges() throws {
        try requireMetal()
        for dtype: DType in [.float16, .bfloat16] {
            for rows in [33, 64, 65] {
                for group in [64, 128] {
                    for output in [9_216, 16_385] {
                        check(rows: rows, input: 512, output: output, group: group, dtype: dtype)
                    }
                }
            }
        }
    }

    func testReleasedH3ProjectionWidths() throws {
        try requireMetal()
        for (input, output) in [(5_376, 21_504), (5_376, 28_672), (7_168, 5_376), (14_336, 5_376)] {
            check(rows: 64, input: input, output: output, group: 64, dtype: .bfloat16)
            Memory.clearCache()
        }
    }

    func testLargeDimensionsAndRaggedEdges() throws {
        try requireMetal()
        for dtype: DType in [.float16, .bfloat16] {
            for (rows, input, output, group) in [
                (32_769, 64, 65, 64), (65, 64, 32_769, 64),
                (65, 512, 129, 32), (33, 512, 127, 128), (16, 96, 95, 32)
            ] {
                check(rows: rows, input: input, output: output, group: group, dtype: dtype)
            }
        }
    }

    func testNonTransposedGroupSizes() throws {
        try requireMetal()
        for dtype: DType in [.float16, .bfloat16] {
            for group in [32, 64, 128] {
                let x = MLXRandom.normal([65, 128], key: MLXRandom.key(13)).asType(dtype)
                let w = MLXRandom.normal([128, 256], key: MLXRandom.key(14)).asType(dtype)
                let q = MLX.quantized(w, groupSize: group, bits: 8)
                let restored = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                                               groupSize: group, bits: 8, dtype: .float32)
                let actual = MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases,
                                             transpose: false, groupSize: group, bits: 8)
                assertReference(actual, MLX.matmul(x.asType(.float32), restored), label: "NN \(dtype) group=\(group)")
            }
        }
    }

    func testGatheredAndSortedExperts() throws {
        try requireMetal()
        for dtype: DType in [.float16, .bfloat16] {
            for sorted in [false, true] {
                let x = MLXRandom.normal([4, 64, 512], key: MLXRandom.key(15)).asType(dtype)
                let w = MLXRandom.normal([3, 9_216, 512], key: MLXRandom.key(16)).asType(dtype)
                let q = MLX.quantized(w, groupSize: 128, bits: 8)
                let ids = MLXArray([UInt32(0), 0, 1, 2])
                let restored = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                                               groupSize: 128, bits: 8, dtype: .float32)
                let actual = MLX.gatherQuantizedMM(x, q.wq, scales: q.scales, biases: q.biases,
                    lhsIndices: sorted ? nil : MLXArray([UInt32(0), 1, 2, 3]), rhsIndices: ids,
                    groupSize: 128, bits: 8, sortedIndices: sorted)
                let reference = MLX.matmul(x.asType(.float32), MLX.take(restored, ids, axis: 0).swappedAxes(-1, -2))
                assertReference(actual, reference, label: "gather \(dtype) sorted=\(sorted)")
            }
        }
    }

    func testSortedSingleRowExpertsWithRaggedEdges() throws {
        try requireMetal()
        let ids = MLXArray((0..<33).map { UInt32($0 / 11) })
        for dtype: DType in [.float16, .bfloat16] {
            for (input, output, group) in [(96, 95, 32), (64, 129, 64), (512, 9_216, 128)] {
                let x = MLXRandom.normal([33, 1, input], key: MLXRandom.key(21)).asType(dtype)
                let w = MLXRandom.normal([3, output, input], key: MLXRandom.key(22)).asType(dtype)
                let q = MLX.quantized(w, groupSize: group, bits: 8)
                let restored = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                                               groupSize: group, bits: 8, dtype: .float32)
                for sorted in [false, true] {
                    let actual = MLX.gatherQuantizedMM(x, q.wq, scales: q.scales, biases: q.biases,
                        rhsIndices: ids, groupSize: group, bits: 8, sortedIndices: sorted)
                    let reference = MLX.matmul(x.asType(.float32),
                                               MLX.take(restored, ids, axis: 0).swappedAxes(-1, -2))
                    assertReference(actual, reference,
                                    label: "single row gather \(dtype) K=\(input) N=\(output) sorted=\(sorted)")
                }
            }
        }
    }

    func testSortedSingleRowNonTransposedExperts() throws {
        try requireMetal()
        let ids = MLXArray((0..<33).map { UInt32($0 / 11) })
        for dtype: DType in [.float16, .bfloat16] {
            for (input, output, group) in [(96, 96, 32), (128, 256, 128)] {
                let x = MLXRandom.normal([33, 1, input], key: MLXRandom.key(23)).asType(dtype)
                let w = MLXRandom.normal([3, input, output], key: MLXRandom.key(24)).asType(dtype)
                let q = MLX.quantized(w, groupSize: group, bits: 8)
                let restored = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                                               groupSize: group, bits: 8, dtype: .float32)
                for sorted in [false, true] {
                    let actual = MLX.gatherQuantizedMM(x, q.wq, scales: q.scales, biases: q.biases,
                        rhsIndices: ids, transpose: false, groupSize: group, bits: 8, sortedIndices: sorted)
                    let reference = MLX.matmul(x.asType(.float32), MLX.take(restored, ids, axis: 0))
                    assertReference(actual, reference,
                                    label: "NN single row gather \(dtype) K=\(input) N=\(output) sorted=\(sorted)")
                }
            }
        }
    }

    func testBroadcastedBatches() throws {
        try requireMetal()
        let x = MLXRandom.normal([2, 1, 65, 512], key: MLXRandom.key(17)).asType(.bfloat16)
        let w = MLXRandom.normal([1, 3, 129, 512], key: MLXRandom.key(18)).asType(.bfloat16)
        let q = MLX.quantized(w, groupSize: 64, bits: 8)
        let restored = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                                       groupSize: 64, bits: 8, dtype: .float32)
        let actual = MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: 8)
        assertReference(actual, MLX.matmul(x.asType(.float32), restored.swappedAxes(-1, -2)), label: "broadcast batch")
    }

    func testFloat32InputsMatchCPUReference() throws {
        try requireMetal()
        for weightType: DType in [.float32, .bfloat16] {
            let x = MLXRandom.normal([64, 512], key: MLXRandom.key(19))
            let w = MLXRandom.normal([9_216, 512], key: MLXRandom.key(20)).asType(weightType)
            let q = MLX.quantized(w, groupSize: 64, bits: 8)
            let restored = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases,
                                           groupSize: 64, bits: 8, dtype: .float32, stream: .cpu)
            let reference = MLX.matmul(x, restored.T, stream: .cpu)
            let actual = MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: 8)
            assertReference(actual, reference, label: "FP32 input, \(weightType) scales")
        }
    }
}
