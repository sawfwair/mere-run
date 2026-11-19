import Foundation
import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor

#if os(macOS)
final class M5GroupAffineQ4Tests: MLXTestCase {
    func testLongFFNRowReuseMatchesBothReferenceProjections() throws {
        try requireM5()
        MLXRandom.seed(905)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            try M5TiledAffineQ4.$groupAffinePrefillEnabledOverride.withValue(true) {
                func weights(_ k: Int, _ n: Int) throws -> M5TiledAffineQ4.Weights {
                    try XCTUnwrap(M5TiledAffineQ4.Weights(
                        weight: MLXRandom.randInt(0..<UInt32.max, [n, k / 8]),
                        scales: MLXRandom.uniform(0.001..<0.01, [n, k / 64]).asType(.bfloat16),
                        biases: MLXRandom.uniform(-0.1..<0.1, [n, k / 64]).asType(.bfloat16)))
                }
                let gate = try weights(5_120, 17_408), up = try weights(5_120, 17_408)
                let down = try weights(17_408, 5_120)
                let input = MLXRandom.uniform(-1..<1, [1, 1024, 5_120]).asType(.bfloat16)
                func run(_ reuse: Bool) throws -> (MLXArray, MLXArray) {
                    try M5TiledAffineQ4.$prefillFFNRowReuse.withValue(reuse) {
                        let activated = try XCTUnwrap(M5TiledAffineQ4.swiglu(input, gate: gate, up: up))
                        return (activated, try XCTUnwrap(M5TiledAffineQ4.matmul(activated, weights: down)))
                    }
                }
                let control = try run(false), trial = try run(true)
                XCTAssertEqual(delta(trial.0, control.0), 0)
                XCTAssertEqual(delta(trial.1, control.1), 0)
                M5TiledAffineQ4.$prefillFFNRowReuse.withValue(true) {
                    XCTAssertEqual(M5TiledAffineQ4.prefillFFNRowTiles(width: 1024, input: 5_120, output: 17_408), 2)
                    XCTAssertEqual(M5TiledAffineQ4.prefillFFNRowTiles(width: 1024, input: 17_408, output: 5_120), 4)
                    XCTAssertEqual(M5TiledAffineQ4.prefillFFNRowTiles(width: 526, input: 5_120, output: 17_408), 1)
                    XCTAssertEqual(M5TiledAffineQ4.prefillFFNRowTiles(width: 16, input: 17_408, output: 5_120), 1)
                }
            }
        }
    }

    func testPrefillBanksAndLayoutCacheInvalidation() throws {
        try requireM5()
        MLXRandom.seed(903)
        let raw = MLXRandom.randInt(0..<UInt32.max, [128, 64])
        let scales = MLXArray.ones([128, 8], dtype: .bfloat16) / 256
        let biases = MLXArray.ones([128, 8], dtype: .bfloat16) / -128
        let classic = try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(false) {
            try XCTUnwrap(M5TiledAffineQ4.Weights(weight: raw, scales: scales, biases: biases))
        }
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            let grouped = try XCTUnwrap(M5TiledAffineQ4.Weights(weight: raw, scales: scales, biases: biases))
            XCTAssertFalse(classic.matches(weight: raw, scales: scales, biases: biases))
            XCTAssertTrue(grouped.matches(weight: raw, scales: scales, biases: biases))
            try M5TiledAffineQ4.$groupAffinePrefillEnabledOverride.withValue(true) {
                let dense = MLX.dequantized(raw, scales: scales, biases: biases, groupSize: 64, bits: 4, dtype: .float32)
                for count in [17, 31, 64, 128, 129, 512] {
                    let x = MLXRandom.randInt(-3..<4, [1, count, 512]).asType(.bfloat16)
                    let expected = MLX.matmul(x.asType(.float32), dense.T).asType(.bfloat16)
                    let actual = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: grouped))
                    XCTAssertEqual(delta(actual, expected), 0, "bank rows=\(count)")
                    let activation = try XCTUnwrap(M5TiledAffineQ4.swiglu(x, gate: grouped, up: grouped))
                    XCTAssertEqual(delta(activation, Q35CompiledOperations.current.swiglu(expected, expected)), 0)
                }
            }
        }
    }

    func testRealFFNStackPreservesSeparateProjectionBoundaries() throws {
        try requireM5()
        MLXRandom.seed(904)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            try M5TiledAffineQ4.$groupAffinePrefillEnabledOverride.withValue(true) {
                func weights() throws -> M5TiledAffineQ4.Weights {
                    try XCTUnwrap(M5TiledAffineQ4.Weights(
                        weight: MLXRandom.randInt(0..<UInt32.max, [17_408, 640]),
                        scales: MLXRandom.uniform(0.001..<0.01, [17_408, 80]).asType(.bfloat16),
                        biases: MLXRandom.uniform(-0.1..<0.1, [17_408, 80]).asType(.bfloat16)))
                }
                let gate = try weights(), up = try weights()
                for count in [1, 8, 16, 17, 128] {
                    let x = MLXRandom.uniform(-1..<1, [1, count, 5_120]).asType(.bfloat16)
                    let g = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: gate))
                    let u = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: up))
                    let actual = try XCTUnwrap(M5TiledAffineQ4.swiglu(x, gate: gate, up: up))
                    XCTAssertEqual(delta(actual, Q35CompiledOperations.current.swiglu(g, u)), 0)
                }
            }
        }
    }

    func testExactAffineFixturesAndEveryWidthUseTheSameReference() throws {
        try requireM5()
        MLXRandom.seed(901)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            for k in [512, 5_120, 6_144, 17_408] {
                let packed = MLXRandom.randInt(0..<UInt32.max, [128, k / 8])
                let scales = (MLXRandom.randInt(-3..<4, [128, k / 64]).asType(.float32) / 256).asType(.bfloat16)
                let biases = (MLXRandom.randInt(-3..<4, [128, k / 64]).asType(.float32) / 256).asType(.bfloat16)
                let weights = try XCTUnwrap(M5TiledAffineQ4.Weights(weight: packed, scales: scales, biases: biases))
                let x = MLXRandom.randInt(-3..<4, [1, 16, k]).asType(.bfloat16)
                let dense = MLX.dequantized(packed, scales: scales, biases: biases, groupSize: 64, bits: 4, dtype: .float32)
                let expected = MLX.matmul(x.asType(.float32), dense.T).asType(.bfloat16)
                for splits in k == 512 ? [1] : [1, 2, 4] {
                    let full = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: weights, splits: splits))
                    XCTAssertEqual(delta(full, expected), 0, "K=\(k), split=\(splits)")
                    for count in 1...16 {
                        let batch = try XCTUnwrap(M5TiledAffineQ4.matmul(
                            x[0..., 0..<count, 0...], weights: weights, splits: splits))
                        XCTAssertEqual(delta(batch, full[0..., 0..<count, 0...]), 0, "rows=\(count)")
                    }
                }
            }
        }
    }

    func testRandomAffineAccuracyAndStackedSwiGLU() throws {
        try requireM5()
        MLXRandom.seed(902)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            for k in [512, 5_120, 17_408] {
                func weights() throws -> M5TiledAffineQ4.Weights {
                    try XCTUnwrap(M5TiledAffineQ4.Weights(
                        weight: MLXRandom.randInt(0..<UInt32.max, [128, k / 8]),
                        scales: MLXRandom.uniform(0.001..<0.01, [128, k / 64]).asType(.bfloat16),
                        biases: MLXRandom.uniform(-0.1..<0.1, [128, k / 64]).asType(.bfloat16)))
                }
                let gate = try weights(), up = try weights()
                let input = MLXRandom.uniform(-1..<1, [1, 16, k]).asType(.bfloat16)
                let dense = MLX.dequantized(gate.originalWeight, scales: gate.originalScales,
                    biases: gate.originalBiases, groupSize: 64, bits: 4, dtype: .float32)
                let reference = MLX.matmul(input.asType(.float32), dense.T)
                let actual = try XCTUnwrap(M5TiledAffineQ4.matmul(input, weights: gate, splits: 1))
                let error = actual.asType(.float32) - reference
                XCTAssertLessThan((error.square().mean() / reference.square().mean()).sqrt().item(Float.self), 0.003)
                let full = try XCTUnwrap(M5TiledAffineQ4.swiglu(input, gate: gate, up: up))
                for count in 1...16 {
                    let x = input[0..., 0..<count, 0...]
                    let g = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: gate, splits: 1))
                    let u = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: up, splits: 1))
                    let expected = Q35CompiledOperations.current.swiglu(g, u)
                    let fused = try XCTUnwrap(M5TiledAffineQ4.swiglu(x, gate: gate, up: up))
                    XCTAssertEqual(delta(fused, expected), 0)
                    XCTAssertEqual(delta(fused, full[0..., 0..<count, 0...]), 0)
                }
            }
        }
    }

    func testBF16RangeAndTinyAffineWeightsSurviveWithoutFP16Dequantization() throws {
        try requireM5()
        let codes = MLXArray(Array(repeating: UInt32(0x8888_8888), count: 64 * 64), [64, 64])
        func weights(scale: Float, bias: Float, group: Bool = true) throws -> M5TiledAffineQ4.Weights {
            try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(group) {
                try XCTUnwrap(M5TiledAffineQ4.Weights(weight: codes,
                    scales: MLXArray.full([64, 8], values: MLXArray(scale), dtype: .bfloat16),
                    biases: MLXArray.full([64, 8], values: MLXArray(bias), dtype: .bfloat16)))
            }
        }
        let tiny = try weights(scale: pow(2, -30), bias: pow(2, -31))
        let x = MLXArray.full([1, 1, 512], values: MLXArray(Float(256)), dtype: .bfloat16)
        let candidate = try XCTUnwrap(M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            M5TiledAffineQ4.matmul(x, weights: tiny)
        })
        let legacy = try weights(scale: pow(2, -30), bias: pow(2, -31), group: false)
        let staged = try XCTUnwrap(M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(false) {
            M5TiledAffineQ4.matmul(x, weights: legacy)
        })
        XCTAssertEqual(candidate.asType(.float32).asArray(Float.self), Array(repeating: Float(17) / 16_384, count: 64))
        XCTAssertEqual(staged.asType(.float32).asArray(Float.self), Array(repeating: 0, count: 64))
        let wide = try weights(scale: 1 / 65_536, bias: -7 / 65_536)
        let large = MLXArray.full([1, 1, 512], values: MLXArray(Float(65_536)), dtype: .bfloat16)
        let result = try XCTUnwrap(M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            M5TiledAffineQ4.matmul(large, weights: wide)
        })
        XCTAssertEqual(result.asType(.float32).asArray(Float.self), Array(repeating: 512, count: 64))
    }

    private func delta(_ a: MLXArray, _ b: MLXArray) -> Float {
        (a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self)
    }

    private func requireM5() throws {
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Group-affine projections require an M5 GPU.")
    }
}
#endif
