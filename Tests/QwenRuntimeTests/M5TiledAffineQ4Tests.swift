import Foundation
import MLX
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunTensor
@testable import MereRunQwenModel

#if os(macOS)
final class M5TiledAffineQ4Tests: MLXTestCase {
    func testTiledWeightsPreserveProjectionAndSerialArithmetic() throws {
        try requireM5()
        MLXRandom.seed(760)
        for inputSize in [512, 2_048, 5_120, 17_408] {
            let f = try fixtures(inputSize: inputSize, outputSize: 128)
            let rounded = MLX.dequantized(
                f.weights.originalWeight, scales: f.weights.originalScales, biases: f.weights.originalBiases,
                groupSize: 64, bits: 4, dtype: .float32
            ).asType(.float16).asType(.float32)
            for count in [1, 4, 8] {
                let x = f.input[0..., 0..<count, 0...]
                let expected = MLX.matmul(x.asType(.float32), rounded.T)
                for splits in (inputSize == 512 ? [1] : [1, 2, 4]) {
                    let actual = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: f.weights, splits: splits))
                    let error = actual.asType(.float32) - expected
                    XCTAssertLessThan((error.square().mean() / expected.square().mean()).sqrt().item(Float.self), 0.003)
                    let serial = try MLX.concatenated((0..<count).map {
                        try XCTUnwrap(M5TiledAffineQ4.matmul(x[0..., $0..<($0 + 1), 0...], weights: f.weights, splits: splits))
                    }, axis: 1)
                    XCTAssertEqual((actual.asType(.float32) - serial.asType(.float32)).abs().max().item(Float.self), 0)
                }
            }
        }
    }

    func testInterleavedGateUpMatchesSeparateTiledProjections() throws {
        try requireM5()
        MLXRandom.seed(761)
        for (inputSize, outputSize) in [(512, 128), (5_120, 17_408)] {
            let gate = try fixtures(inputSize: inputSize, outputSize: outputSize)
            let up = try fixtures(inputSize: inputSize, outputSize: outputSize)
            for count in [1, 4, 8] {
                let x = gate.input[0..., 0..<count, 0...]
                let expected = Q35CompiledOperations.current.swiglu(
                    try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: gate.weights)),
                    try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: up.weights))
                )
                let actual = try XCTUnwrap(M5TiledAffineQ4.swiglu(x, gate: gate.weights, up: up.weights))
                XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0)
                let serial = try MLX.concatenated((0..<count).map {
                    try XCTUnwrap(M5TiledAffineQ4.swiglu(
                        x[0..., $0..<($0 + 1), 0...], gate: gate.weights, up: up.weights
                    ))
                }, axis: 1)
                XCTAssertEqual((actual.asType(.float32) - serial.asType(.float32)).abs().max().item(Float.self), 0)
            }
        }
    }

    private struct Fixtures {
        let input: MLXArray
        let weights: M5TiledAffineQ4.Weights
    }

    private func fixtures(inputSize: Int, outputSize: Int) throws -> Fixtures {
        let input = MLXRandom.uniform(-1..<1, [1, 8, inputSize]).asType(.bfloat16)
        let weight = MLXRandom.randInt(0..<UInt32.max, [outputSize, inputSize / 8])
        let scales = MLXRandom.uniform(0.001..<0.01, [outputSize, inputSize / 64]).asType(.bfloat16)
        let biases = MLXRandom.uniform(-0.1..<0.1, [outputSize, inputSize / 64]).asType(.bfloat16)
        return Fixtures(input: input, weights: try XCTUnwrap(M5TiledAffineQ4.Weights(
            weight: weight, scales: scales, biases: biases
        )))
    }

    private func requireM5() throws {
        try XCTSkipUnless(M5AffineQ4Verification.isSupported,
                          "Tiled matrix experiments require an Apple10 GPU and MERERUN_TEST_MLX_DEVICE=gpu.")
    }
}
#endif
