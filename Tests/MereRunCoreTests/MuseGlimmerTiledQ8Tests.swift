#if os(macOS)
import MLX
import MLXNN
import MLXRandom
import MereRunAdmission
import MereRunTensor
import XCTest
@testable import MereRunCore

final class MuseGlimmerTiledQ8Tests: MereRunCoreTestCase {
    func testPackingProjectionAndFusedActivation() throws {
        try XCTSkipUnless(M5AffineQ8Head.isSupported, "Requires an M5 GPU")
        let lease = try MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true)
        ).acquireBlocking(.init(label: "Tiled Q8 qualification", resourceClass: .large))
        defer { lease.release() }
        MLXRandom.seed(32_864)
        for (input, output) in [(512, 64), (6656, 128), (19968, 64)] {
            let packed = MLXRandom.randInt(0..<UInt32.max, [output, input / 4])
            let scales = MLXRandom.uniform(0.001..<0.01, [output, input / 64]).asType(.bfloat16)
            let biases = MLXRandom.uniform(-0.1..<0.1, scales.shape).asType(.bfloat16)
            let weights = try XCTUnwrap(M5TiledAffineQ8.Weights(weight: packed, scales: scales, biases: biases))
            let up = try XCTUnwrap(M5TiledAffineQ8.Weights(
                weight: MLXRandom.randInt(0..<UInt32.max, packed.shape), scales: scales, biases: -biases))
            let restored = weights.codes.transposed(0, 2, 1, 3).reshaped(packed.shape)
            XCTAssertTrue(MLX.all(restored .== packed).item(Bool.self))
            XCTAssertTrue(MLX.all(weights.scales.transposed(0, 2, 1).reshaped(scales.shape) .== scales).item(Bool.self))
            XCTAssertTrue(MLX.all(weights.biases.transposed(0, 2, 1).reshaped(biases.shape) .== biases).item(Bool.self))
            let dense = dequantized(packed, scales: scales, biases: biases, groupSize: 64, bits: 8, dtype: .bfloat16)
            for magnitude in [Float(1), 200_000] {
                let x = (MLXRandom.normal([1, 8, input]) * magnitude).asType(.bfloat16)
                let expected = matmul(x.asType(.float32), dense.asType(.float32).T).asType(.bfloat16).asType(.float32)
                for splits in [1, 2, 4, 8] {
                    let serial = try concatenated((0..<8).map {
                        try XCTUnwrap(M5TiledAffineQ8.matmul(x[0..., $0..<($0 + 1), 0...], weights: weights, splits: splits))
                    }, axis: 1)
                    for width in [1, 2, 3, 4, 7, 8] {
                        let actual = try XCTUnwrap(M5TiledAffineQ8.matmul(x[0..., ..<width, 0...], weights: weights, splits: splits))
                        XCTAssertEqual(actual.dtype, .bfloat16)
                        XCTAssertTrue(MLX.all(MLX.isFinite(actual)).item(Bool.self))
                        XCTAssertEqual((actual.asType(.float32) - serial[0..., ..<width, 0...].asType(.float32)).abs().max().item(Float.self), 0)
                        let reference = expected[0..., ..<width, 0...]
                        let delta = actual.asType(.float32) - reference
                        XCTAssertLessThan(MLX.sqrt((delta * delta).mean() / (reference * reference).mean()).item(Float.self), 0.005)
                    }
                }
                let projected = try XCTUnwrap(M5TiledAffineQ8.matmul(x, weights: weights, splits: 1))
                let upProjected = try XCTUnwrap(M5TiledAffineQ8.matmul(x, weights: up, splits: 1))
                let fused = try XCTUnwrap(M5TiledAffineQ8.swiglu(x, gate: weights, up: up))
                let separate = MLXNN.silu(projected) * upProjected
                XCTAssertEqual((fused.asType(.float32) - separate.asType(.float32)).abs().max().item(Float.self), 0)
            }
            let source = QuantizedLinear(weight: packed, bias: nil, scales: scales, biases: biases, groupSize: 64, bits: 8)
            let wrapped = MuseGlimmerTiledQ8Linear(source, tiled: weights, splits: 1)
            XCTAssertEqual(Set(source.parameters().flattened().map(\.0)), Set(wrapped.parameters().flattened().map(\.0)))
            XCTAssertNil(M5TiledAffineQ8.matmul(MLXArray.zeros([1, 9, input], dtype: .bfloat16), weights: weights))
            XCTAssertNil(M5TiledAffineQ8.matmul(MLXArray.zeros([1, 3, input]), weights: weights))
            let replacement = MLXArray.zeros(packed.shape, dtype: .uint32)
            try wrapped.update(parameters: ModuleParameters.unflattened([("weight", replacement)]), verify: .none)
            XCTAssertFalse(wrapped.hasCurrentTiles)
            let probe = MLXArray.ones([1, 3, input], dtype: .bfloat16)
            let fallback = MLX.quantizedMM(probe, replacement, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
            XCTAssertEqual((wrapped(probe).asType(.float32) - fallback.asType(.float32)).abs().max().item(Float.self), 0)
        }
    }

    func testHeadSplitControlRejectsInvalidValues() throws {
        for value in ["0", "3", "16", "oops"] {
            XCTAssertThrowsError(try MuseGlimmerModel.tiledQ8HeadSplits(environment: ["MERERUN_MUSE_GLIMMER_Q8_TILED_HEAD_SPLITS": value]))
        }
        XCTAssertEqual(try MuseGlimmerModel.tiledQ8HeadSplits(environment: [:]), 1)
    }
}
#endif
