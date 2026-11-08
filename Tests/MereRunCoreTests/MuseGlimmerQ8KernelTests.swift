#if os(macOS)
import MLX
import MLXRandom
import MereRunAdmission
import MereRunTensor
import XCTest
@testable import MereRunCore

final class MuseGlimmerQ8KernelTests: MereRunCoreTestCase {
    func testSharedQ8BlocksMatchNativeSerialWithWideActivationRange() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Q8 Metal parity requires the GPU test device")
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true)
        )
        let lease = try coordinator.acquireBlocking(.init(label: "Glimmer Q8 kernel parity", resourceClass: .large))
        defer { lease.release() }
        MLXRandom.seed(6_656)
        for (input, output) in [(256, 8), (512, 16), (6656, 256), (19968, 64)] {
            let weight = MLXRandom.randInt(0..<UInt32.max, [output, input / 4])
            let scales = MLXRandom.uniform(0.001..<0.01, [output, input / 64]).asType(.bfloat16)
            let biases = MLXRandom.uniform(-0.1..<0.1, [output, input / 64]).asType(.bfloat16)
            for rows in 2...8 {
                let x = (MLXRandom.normal([1, rows, input]) * 200_000).asType(.bfloat16)
                let reference = MLX.concatenated((0..<rows).map { row in
                    MLX.quantizedMM(x[0..., row..<(row + 1), 0...], weight,
                                    scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
                }, axis: 1)
                for grouping in [1, 2, 4] {
                    let actual = try XCTUnwrap(SmallBatchAffineQ8.matmul(
                        x, weight: weight, scales: scales, biases: biases, groupSize: 64, bits: 8,
                        mode: .affine, rowsPerSIMD: grouping
                    ))
                    XCTAssertEqual(actual.dtype, .bfloat16)
                    XCTAssertEqual((actual.asType(.float32) - reference.asType(.float32)).abs().max().item(Float.self),
                                   0, "K=\(input), N=\(output), M=\(rows), R=\(grouping)")
                }
            }
        }
        let x = MLXArray.zeros([1, 4, 256], dtype: .bfloat16)
        let weight = MLXArray.zeros([8, 64], dtype: .uint32)
        let metadata = MLXArray.ones([8, 4], dtype: .bfloat16)
        XCTAssertNil(SmallBatchAffineQ8.matmul(x, weight: weight, scales: MLXArray.ones([1], dtype: .bfloat16),
            biases: metadata, groupSize: 64, bits: 8, mode: .affine, rowsPerSIMD: 4))
        XCTAssertNil(SmallBatchAffineQ8.matmul(MLXArray.zeros([1, 4, 0], dtype: .bfloat16),
            weight: MLXArray.zeros([8, 0], dtype: .uint32), scales: MLXArray.zeros([8, 0], dtype: .bfloat16),
            biases: MLXArray.zeros([8, 0], dtype: .bfloat16), groupSize: 64, bits: 8, mode: .affine, rowsPerSIMD: 4))
        // Broadcast views exercise the index boundary without allocating a
        // multi-gigabyte checkpoint or dispatching its unsupported kernel.
        let largeMetadata = MLX.broadcast(MLXArray(Float(1)).asType(.bfloat16), to: [202_048, 256])
        XCTAssertNil(SmallBatchAffineQ8.matmul(MLXArray.zeros([1, 4, 16_384], dtype: .bfloat16),
            weight: MLX.broadcast(MLXArray(UInt32(0)), to: [202_048, 4096]),
            scales: largeMetadata, biases: largeMetadata, groupSize: 64, bits: 8, mode: .affine, rowsPerSIMD: 4))
    }
}
#endif
