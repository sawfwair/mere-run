import Dispatch
import Foundation
import MLX
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunTensor
@testable import MereRunQwenModel

#if os(macOS)
final class M5AffineQ4VerificationTests: MLXTestCase {
    func testFusedGateUpMatchesSeparateCompiledSwiGLU() throws {
        try requireM5()
        MLXRandom.seed(754)
        for (inputSize, outputSize) in [(512, 128), (5_120, 17_408)] {
            let gate = fixtures(inputSize: inputSize, outputSize: outputSize)
            let up = fixtures(inputSize: inputSize, outputSize: outputSize)
            for count in [1, 4, 8] {
                let input = gate.input[0..., 0..<count, 0...]
                let expected = Q35CompiledOperations.current.swiglu(
                    try projection(input, values: gate, splits: 1),
                    try projection(input, values: up, splits: 1)
                )
                let actual = try fused(input, gate: gate, up: up)
                XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self),
                               0, "Fused FFN changed compiled BF16 activation boundaries")
                let serial = try MLX.concatenated((0..<count).map { row in
                    try fused(input[0..., row..<(row + 1), 0...], gate: gate, up: up)
                }, axis: 1)
                XCTAssertEqual((actual.asType(.float32) - serial.asType(.float32)).abs().max().item(Float.self), 0)
            }
        }
    }

    func testMatrixArithmeticAndSerialVerificationAgreement() throws {
        try requireM5()
        MLXRandom.seed(752)
        for inputSize in [512, 2_048] {
            let values = fixtures(inputSize: inputSize, outputSize: 128)
            let rounded = MLX.dequantized(
                values.weight, scales: values.scales, biases: values.biases,
                groupSize: 64, bits: 4, dtype: .float32
            ).asType(.float16).asType(.float32)
            for count in [1, 4, 8] {
                let input = values.input[0..., 0..<count, 0...]
                let reference = MLX.matmul(input.asType(.float32), rounded.T)
                for splits in (inputSize == 512 ? [1] : [1, 2, 4]) {
                    let actual = try projection(input, values: values, splits: splits)
                    let error = actual.asType(.float32) - reference
                    let relativeRMS = (error.square().mean() / reference.square().mean()).sqrt().item(Float.self)
                    XCTAssertLessThan(relativeRMS, 0.003, "BF16 output error: K=\(inputSize), M=\(count), splits=\(splits)")
                    let serial = try MLX.concatenated((0..<count).map { row in
                        try projection(input[0..., row..<(row + 1), 0...], values: values, splits: splits)
                    }, axis: 1)
                    XCTAssertEqual((actual.asType(.float32) - serial.asType(.float32)).abs().max().item(Float.self),
                                   0, "Matrix backend changed across serial/verification rows")
                }
            }
        }
    }

    /// Timed only after the external collector admits an idle compute window.
    func testProjectionComponentBenchmark() throws {
        try requireM5()
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MERERUN_TEST_M5_Q4_COMPONENT_BENCHMARK"] == "1",
                          "Matrix component timing is opt-in; not full-model throughput.")
        MLXRandom.seed(753)
        print("[m5-q4-device] \(GPU.deviceInfo().architecture)")
        for (inputSize, outputSize) in [
            (5_120, 17_408), (17_408, 5_120), (5_120, 16_384),
            (5_120, 14_336), (6_144, 5_120), (5_120, 248_320),
        ] {
            let values = fixtures(inputSize: inputSize, outputSize: outputSize)
            let modes = [0, 1, 2, 4, 8].filter { $0 == 0 || inputSize.isMultiple(of: 64 * $0) }
            for mode in modes { MLX.eval(try benchmarkProjection(values, mode: mode)) }
            let warmUntil = DispatchTime.now().uptimeNanoseconds + 1_000_000_000
            while DispatchTime.now().uptimeNanoseconds < warmUntil {
                MLX.eval(try benchmarkProjection(values, mode: 0))
            }
            var samples = Dictionary(uniqueKeysWithValues: modes.map { (String($0), [Double]()) })
            for iteration in 0..<11 {
                for offset in 0..<modes.count {
                    let mode = modes[(iteration + offset) % modes.count]
                    let start = DispatchTime.now().uptimeNanoseconds
                    let outputs = try (0..<8).map { _ in try benchmarkProjection(values, mode: mode) }
                    MLX.eval(outputs)
                    let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 8_000_000
                    if iteration >= 2 { samples[String(mode), default: []].append(milliseconds) }
                }
            }
            let report = ComponentResult(inputSize: inputSize, outputSize: outputSize, milliseconds: samples)
            print("[m5-q4-component] \(String(decoding: try JSONEncoder().encode(report), as: UTF8.self))")
        }
    }

    private struct ComponentResult: Codable {
        let inputSize: Int
        let outputSize: Int
        let milliseconds: [String: [Double]]
    }

    func testSwiGLUComponentBenchmark() throws {
        try requireM5()
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MERERUN_TEST_M5_Q4_COMPONENT_BENCHMARK"] == "1",
                          "Matrix component timing is opt-in; not full-model throughput.")
        MLXRandom.seed(755)
        let gate = fixtures(inputSize: 5_120, outputSize: 17_408)
        let up = fixtures(inputSize: 5_120, outputSize: 17_408)
        let modes = ["reference", "matrix", "fused"]
        for mode in modes { MLX.eval(try swigluProjection(gate: gate, up: up, mode: mode)) }
        let warmUntil = DispatchTime.now().uptimeNanoseconds + 1_000_000_000
        while DispatchTime.now().uptimeNanoseconds < warmUntil {
            MLX.eval(try swigluProjection(gate: gate, up: up, mode: "reference"))
        }
        var samples = Dictionary(uniqueKeysWithValues: modes.map { ($0, [Double]()) })
        for iteration in 0..<11 {
            for offset in 0..<modes.count {
                let mode = modes[(iteration + offset) % modes.count]
                let start = DispatchTime.now().uptimeNanoseconds
                let outputs = try (0..<8).map { _ in try swigluProjection(gate: gate, up: up, mode: mode) }
                MLX.eval(outputs)
                let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 8_000_000
                if iteration >= 2 { samples[mode, default: []].append(milliseconds) }
            }
        }
        let report = ComponentResult(inputSize: 5_120, outputSize: 17_408, milliseconds: samples)
        print("[m5-swiglu-component] \(String(decoding: try JSONEncoder().encode(report), as: UTF8.self))")
    }

    private struct Fixtures {
        let input: MLXArray
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private func fixtures(inputSize: Int, outputSize: Int) -> Fixtures {
        let values = Fixtures(
            input: MLXRandom.normal([1, 8, inputSize]).asType(.bfloat16),
            weight: MLXRandom.randInt(0..<UInt32.max, [outputSize, inputSize / 8]),
            scales: MLXRandom.uniform(0.001..<0.01, [outputSize, inputSize / 64]).asType(.bfloat16),
            biases: MLXRandom.uniform(-0.1..<0.1, [outputSize, inputSize / 64]).asType(.bfloat16)
        )
        MLX.eval(values.input, values.weight, values.scales, values.biases)
        return values
    }

    private func projection(_ input: MLXArray, values: Fixtures, splits: Int) throws -> MLXArray {
        try XCTUnwrap(M5AffineQ4Verification.matmul(
            input, weight: values.weight, scales: values.scales, biases: values.biases, splits: splits
        ))
    }

    private func fused(_ input: MLXArray, gate: Fixtures, up: Fixtures) throws -> MLXArray {
        try XCTUnwrap(M5AffineQ4Verification.swiglu(
            input, gateWeight: gate.weight, gateScales: gate.scales, gateBiases: gate.biases,
            upWeight: up.weight, upScales: up.scales, upBiases: up.biases
        ))
    }

    private func swigluProjection(gate: Fixtures, up: Fixtures, mode: String) throws -> MLXArray {
        if mode == "fused" { return try fused(gate.input, gate: gate, up: up) }
        let g = try mode == "matrix" ? projection(gate.input, values: gate, splits: 1)
            : benchmarkProjection(gate, mode: 0)
        let u: MLXArray
        if mode == "matrix" { u = try projection(gate.input, values: up, splits: 1) }
        else {
            u = try XCTUnwrap(SmallBatchAffineQMV.matmul(
                gate.input, weight: up.weight, scales: up.scales, biases: up.biases,
                groupSize: 64, bits: 4, mode: .affine, eightRowGrouping: .split
            ))
        }
        return Q35CompiledOperations.current.swiglu(g, u)
    }

    private func benchmarkProjection(_ values: Fixtures, mode: Int) throws -> MLXArray {
        if mode != 0 { return try projection(values.input, values: values, splits: mode) }
        return try XCTUnwrap(SmallBatchAffineQMV.matmul(
            values.input, weight: values.weight, scales: values.scales, biases: values.biases,
            groupSize: 64, bits: 4, mode: .affine, eightRowGrouping: .split
        ))
    }

    private func requireM5() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu && M5AffineQ4Verification.isSupported,
                          "Matrix-accelerator experiments require an Apple10 GPU and MERERUN_TEST_MLX_DEVICE=gpu.")
    }
}
#endif
