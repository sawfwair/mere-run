#if os(macOS)
import MLX
import MLXRandom
import MereRunAdmission
import MereRunTensor
import XCTest
@testable import MereRunCore

final class MuseGlimmerNAXHeadTests: MereRunCoreTestCase {
    func testMatrixHeadMatchesBF16ReferenceAndSerialRows() throws {
        try XCTSkipUnless(M5AffineQ8Head.isSupported, "Requires the M5 GPU test device")
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true)
        )
        let lease = try coordinator.acquireBlocking(.init(label: "Glimmer Q8 NAX arithmetic", resourceClass: .large))
        defer { lease.release() }
        MLXRandom.seed(8_064)
        for (input, output) in [(512, 64), (6656, 128)] {
            let weight = MLXRandom.randInt(0..<UInt32.max, [output, input / 4])
            let scales = MLXRandom.uniform(0.001..<0.01, [output, input / 64]).asType(.bfloat16)
            let biases = (-128 * scales).asType(.bfloat16)
            let dense = dequantized(weight, scales: scales, biases: biases, groupSize: 64, bits: 8, dtype: .bfloat16)
            for magnitude in [Float(1), Float(200_000)] {
                let x = (MLXRandom.normal([1, 8, input]) * magnitude).asType(.bfloat16)
                let reference = matmul(x.asType(.float32), dense.asType(.float32).T).asType(.bfloat16)
                for splits in [1, 2, 4, 8] {
                    let serial = try (0..<8).map { index in
                        try XCTUnwrap(M5AffineQ8Head.matmul(x[0..., index..<(index + 1), 0...],
                            weight: weight, scales: scales, biases: biases, splits: splits))
                    }
                    let oracle = MLX.concatenated(serial, axis: 1)
                    for rows in [1, 2, 3, 4, 7, 8] {
                        let actual = try XCTUnwrap(M5AffineQ8Head.matmul(x[0..., ..<rows, 0...],
                            weight: weight, scales: scales, biases: biases, splits: splits))
                        XCTAssertEqual(actual.dtype, .bfloat16)
                        XCTAssertTrue(MLX.all(MLX.isFinite(actual)).item(Bool.self))
                        XCTAssertEqual((actual.asType(.float32) - oracle[0..., ..<rows, 0...].asType(.float32))
                            .abs().max().item(Float.self), 0)
                        let expected = reference[0..., ..<rows, 0...].asType(.float32)
                        let delta = actual.asType(.float32) - expected
                        let relativeRMS = MLX.sqrt((delta * delta).mean() / (expected * expected).mean()).item(Float.self)
                        XCTAssertLessThan(relativeRMS, 0.005, "K=\(input) M=\(rows) splits=\(splits)")
                    }
                }
            }
        }
    }
}
#endif
