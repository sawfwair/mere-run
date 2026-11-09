import Dispatch
import Foundation
import MLX
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunTensor

#if os(macOS)
final class SmallBatchAffineQMVTests: MLXTestCase {
    func testSharedEightRowsPreserveSerialArithmetic() throws {
        try requireGPU()
        MLXRandom.seed(748)
        for (inputSize, outputSize) in [(512, 8), (5_120, 17_408), (17_408, 5_120), (6_144, 5_120)] {
            let values = fixtures(inputSize: inputSize, outputSize: outputSize)
            let reference = MLX.concatenated((0..<8).map { row in
                MLX.quantizedMM(
                    values.input[0..., row..<(row + 1), 0...], values.weight,
                    scales: values.scales, biases: values.biases, transpose: true,
                    groupSize: 64, bits: 4
                )
            }, axis: 1)
            for grouping in [SmallBatchAffineQMV.EightRowGrouping.split, .shared, .sharedTwo, .sharedOne] {
                let result = try XCTUnwrap(SmallBatchAffineQMV.matmul(
                    values.input, weight: values.weight, scales: values.scales,
                    biases: values.biases, groupSize: 64, bits: 4, mode: .affine,
                    eightRowGrouping: grouping
                ))
                XCTAssertEqual((result.asType(.float32) - reference.asType(.float32)).abs().max().item(Float.self),
                               0, "K=\(inputSize), N=\(outputSize), \(grouping)")
            }
        }
    }

    /// Synchronized component timings, including graph submission and launch overhead.
    /// Run separately from full-model throughput and only on an admitted idle GPU.
    func testEightRowComponentBenchmark() throws {
        try requireGPU()
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MERERUN_TEST_Q4_COMPONENT_BENCHMARK"] == "1",
                          "Component timing is opt-in; not full-model throughput.")
        MLXRandom.seed(749)
        for (inputSize, outputSize) in [
            (5_120, 17_408), (17_408, 5_120), (5_120, 16_384),
            (5_120, 14_336), (6_144, 5_120), (5_120, 248_320),
        ] {
            let values = fixtures(inputSize: inputSize, outputSize: outputSize)
            let modes = ["split", "shared", "sharedTwo", "sharedOne", "native"]
            var timings = Dictionary(uniqueKeysWithValues: modes.map { ($0, [Double]()) })
            for iteration in 0..<11 {
                // Rotate the order to reduce frequency and cache order effects.
                for offset in 0..<modes.count {
                    let mode = modes[(iteration + offset) % modes.count]
                    let start = DispatchTime.now().uptimeNanoseconds
                    let outputs = try (0..<8).map { _ -> MLXArray in
                        if mode == "native" {
                            return MLX.quantizedMM(
                                values.input, values.weight, scales: values.scales,
                                biases: values.biases, transpose: true, groupSize: 64, bits: 4
                            )
                        }
                        return try XCTUnwrap(SmallBatchAffineQMV.matmul(
                            values.input, weight: values.weight, scales: values.scales,
                            biases: values.biases, groupSize: 64, bits: 4, mode: .affine,
                            eightRowGrouping: grouping(mode)
                        ))
                    }
                    MLX.eval(outputs)
                    let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 8_000_000
                    if iteration >= 2 { timings[mode, default: []].append(milliseconds) }
                }
            }
            for mode in modes {
                let times = try XCTUnwrap(timings[mode]).sorted()
                print("[q4-eight-row] K=\(inputSize) N=\(outputSize) mode=\(mode) median_ms=\(times[4]) min_ms=\(times[0]) max_ms=\(times[8])")
            }
        }
    }

    private struct Fixtures {
        let input: MLXArray
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private func grouping(_ mode: String) -> SmallBatchAffineQMV.EightRowGrouping {
        switch mode {
        case "shared": .shared
        case "sharedTwo": .sharedTwo
        case "sharedOne": .sharedOne
        default: .split
        }
    }

    private func fixtures(inputSize: Int, outputSize: Int) -> Fixtures {
        // Generate packed arrays directly to avoid a full-precision vocabulary-head allocation.
        let values = Fixtures(
            input: MLXRandom.normal([1, 8, inputSize]).asType(.bfloat16),
            weight: MLXRandom.randInt(0..<UInt32.max, [outputSize, inputSize / 8]),
            scales: MLXRandom.uniform(0.001..<0.01, [outputSize, inputSize / 64]).asType(.bfloat16),
            biases: MLXRandom.uniform(-0.1..<0.1, [outputSize, inputSize / 64]).asType(.bfloat16)
        )
        MLX.eval(values.input, values.weight, values.scales, values.biases)
        return values
    }

    private func requireGPU() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu,
                          "Exact Q4 Metal tests require MERERUN_TEST_MLX_DEVICE=gpu.")
    }
}
#endif
