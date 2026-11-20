import Foundation
import MLX
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunTensor
@testable import MereRunQwenModel

#if os(macOS)
final class M5TiledAffineQ4Tests: MLXTestCase {
    func testRegisterOperandsPreserveStagedProjectionAndSwiGLUExactly() throws {
        try requireM5()
        MLXRandom.seed(766)
        for width in [512, 5_120, 17_408] {
            let gate = try fixtures(inputSize: width, outputSize: 128, rows: 16)
            let up = try fixtures(inputSize: width, outputSize: 128, rows: 16)
            for count in 1...16 {
                let x = gate.input[0..., 0..<count, 0...]
                for splits in width == 512 ? [1] : [1, 2, 4] {
                    let staged = try XCTUnwrap(M5TiledAffineQ4.$registerOperandsEnabled.withValue(false) {
                        M5TiledAffineQ4.matmul(x, weights: gate.weights, splits: splits)
                    })
                    let registers = try XCTUnwrap(M5TiledAffineQ4.$registerOperandsEnabled.withValue(true) {
                        M5TiledAffineQ4.matmul(x, weights: gate.weights, splits: splits)
                    })
                    XCTAssertEqual((registers.asType(.float32) - staged.asType(.float32)).abs().max().item(Float.self), 0,
                                   "K=\(width), rows=\(count), splits=\(splits)")
                }
                let staged = try XCTUnwrap(M5TiledAffineQ4.$registerOperandsEnabled.withValue(false) {
                    M5TiledAffineQ4.swiglu(x, gate: gate.weights, up: up.weights)
                })
                let registers = try XCTUnwrap(M5TiledAffineQ4.$registerOperandsEnabled.withValue(true) {
                    M5TiledAffineQ4.swiglu(x, gate: gate.weights, up: up.weights)
                })
                XCTAssertEqual((registers.asType(.float32) - staged.asType(.float32)).abs().max().item(Float.self), 0)
            }
        }
    }

    func testTileGroupingPreservesExistingProjectionAndSwiGLUArithmetic() throws {
        try requireM5()
        MLXRandom.seed(762)
        let gate = try fixtures(inputSize: 5_120, outputSize: 192, rows: 16)
        let up = try fixtures(inputSize: 5_120, outputSize: 192, rows: 16)
        for count in 1...16 {
            let x = gate.input[0..., 0..<count, 0...]
            let expectedGLU = try XCTUnwrap(M5TiledAffineQ4.swiglu(x, gate: gate.weights, up: up.weights, tileGroups: 1))
            for splits in [1, 2, 4] {
                let expected = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: gate.weights, splits: splits, tileGroups: 2))
                for groups in [1, 2, 4, 8] {
                    let actual = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: gate.weights, splits: splits, tileGroups: groups))
                    XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0)
                    let glu = try XCTUnwrap(M5TiledAffineQ4.swiglu(x, gate: gate.weights, up: up.weights, tileGroups: groups))
                    XCTAssertEqual((glu.asType(.float32) - expectedGLU.asType(.float32)).abs().max().item(Float.self), 0)
                }
            }
        }
    }

    func testTileGroupingComponentBenchmark() throws {
        try requireM5()
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MERERUN_TEST_M5_TILE_BENCHMARK"] == "1",
                          "Component timings are diagnostic and opt-in.")
        MLXRandom.seed(763)
        for (inputSize, outputSize) in [(5_120, 17_408), (17_408, 5_120), (6_144, 5_120), (5_120, 248_320)] {
            let gate = try fixtures(inputSize: inputSize, outputSize: outputSize)
            let up = inputSize == 5_120 && outputSize == 17_408
                ? try fixtures(inputSize: inputSize, outputSize: outputSize) : nil
            for fused in up == nil ? [false] : [false, true] {
                func apply(_ groups: Int) throws -> MLXArray {
                    if fused, let up {
                        return try XCTUnwrap(M5TiledAffineQ4.swiglu(
                            gate.input, gate: gate.weights, up: up.weights, tileGroups: groups
                        ))
                    }
                    return try XCTUnwrap(M5TiledAffineQ4.matmul(gate.input, weights: gate.weights, tileGroups: groups))
                }
                let groups = [1, 2, 4, 8]
                for group in groups { MLX.eval(try apply(group)) }
                var samples: [String: [Double]] = [:]
                for iteration in 0..<11 {
                    for offset in 0..<groups.count {
                        let group = groups[(iteration + offset) % groups.count]
                        let start = DispatchTime.now().uptimeNanoseconds
                        MLX.eval(try (0..<8).map { _ in try apply(group) })
                        if iteration >= 2 {
                            samples[String(group), default: []].append(Double(DispatchTime.now().uptimeNanoseconds - start) / 8e6)
                        }
                    }
                }
                let row = TileTiming(inputSize: inputSize, outputSize: outputSize, fused: fused, milliseconds: samples)
                print("[m5-tile-groups] \(String(decoding: try JSONEncoder().encode(row), as: UTF8.self))")
            }
        }
    }

    private struct TileTiming: Encodable {
        let inputSize: Int
        let outputSize: Int
        let fused: Bool
        let milliseconds: [String: [Double]]
    }

    func testTiledWeightsPreserveProjectionAndSerialArithmetic() throws {
        try requireM5()
        MLXRandom.seed(760)
        for inputSize in [512, 2_048, 5_120, 17_408] {
            let f = try fixtures(inputSize: inputSize, outputSize: 128, rows: 16)
            let rounded = MLX.dequantized(
                f.weights.originalWeight, scales: f.weights.originalScales, biases: f.weights.originalBiases,
                groupSize: 64, bits: 4, dtype: .float32
            ).asType(.float16).asType(.float32)
            for count in 1...16 {
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
            let gate = try fixtures(inputSize: inputSize, outputSize: outputSize, rows: 16)
            let up = try fixtures(inputSize: inputSize, outputSize: outputSize, rows: 16)
            for count in 1...16 {
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

    private func fixtures(inputSize: Int, outputSize: Int, rows: Int = 8) throws -> Fixtures {
        let input = MLXRandom.uniform(-1..<1, [1, rows, inputSize]).asType(.bfloat16)
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
