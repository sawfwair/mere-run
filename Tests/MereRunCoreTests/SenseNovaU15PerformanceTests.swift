import Foundation
import MLX
import MLXRandom
@testable import MereRunCore
import XCTest

/// Env-gated, in-process comparisons for exact SenseNova U1.5 optimization candidates.
/// Run with `MERERUN_SENSENOVA_BENCH=1 swift test --filter SenseNovaU15PerformanceTests`.
final class SenseNovaU15PerformanceTests: XCTestCase {
    private let hiddenSize = 4_096
    private let intermediateSize = 12_288
    private let keyValueSize = 1_024

    private func requireBenchmark() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_SENSENOVA_BENCH"] == "1" else {
            throw XCTSkip("Set MERERUN_SENSENOVA_BENCH=1 to run SenseNova performance tests")
        }
    }

    private var tokenCounts: [Int] {
        let configured = ProcessInfo.processInfo.environment["MERERUN_SENSENOVA_BENCH_TOKENS"]?
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            .filter { $0 > 0 } ?? []
        return configured.isEmpty ? [1_024, 4_070] : configured
    }

    private func measure(_ body: () -> MLXArray) -> Double {
        let start = CFAbsoluteTimeGetCurrent()
        MLX.eval(body())
        return CFAbsoluteTimeGetCurrent() - start
    }

    private func pairedTimes(
        rounds: Int = 4,
        first: @escaping () -> MLXArray,
        second: @escaping () -> MLXArray
    ) -> (first: Double, second: Double) {
        MLX.eval(first(), second())
        var firstTimes: [Double] = []
        var secondTimes: [Double] = []
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                firstTimes.append(measure(first))
                secondTimes.append(measure(second))
            } else {
                secondTimes.append(measure(second))
                firstTimes.append(measure(first))
            }
        }
        return (firstTimes.sorted()[rounds / 2], secondTimes.sorted()[rounds / 2])
    }

    func testFusedGenerationProjections() throws {
        try requireBenchmark()
        let queryWeight = MLXRandom.uniform(-0.02 ..< 0.02, [hiddenSize, hiddenSize]).asType(.bfloat16)
        let keyWeight = MLXRandom.uniform(-0.02 ..< 0.02, [keyValueSize, hiddenSize]).asType(.bfloat16)
        let valueWeight = MLXRandom.uniform(-0.02 ..< 0.02, [keyValueSize, hiddenSize]).asType(.bfloat16)
        let gateWeight = MLXRandom.uniform(-0.02 ..< 0.02, [intermediateSize, hiddenSize]).asType(.bfloat16)
        let upWeight = MLXRandom.uniform(-0.02 ..< 0.02, [intermediateSize, hiddenSize]).asType(.bfloat16)
        MLX.eval(queryWeight, keyWeight, valueWeight, gateWeight, upWeight)

        let fusedQKVWeight = MLX.concatenated([queryWeight, keyWeight, valueWeight], axis: 0)
        let fusedGateUpWeight = MLX.concatenated([gateWeight, upWeight], axis: 0)
        MLX.eval(fusedQKVWeight, fusedGateUpWeight)

        for tokenCount in tokenCounts {
            let input = MLXRandom.uniform(-0.5 ..< 0.5, [tokenCount, hiddenSize]).asType(.bfloat16)
            MLX.eval(input)

            let separateQKV = {
                MLX.concatenated([
                    MLX.matmul(input, queryWeight.T),
                    MLX.matmul(input, keyWeight.T),
                    MLX.matmul(input, valueWeight.T),
                ], axis: -1)
            }
            let fusedQKV = { MLX.matmul(input, fusedQKVWeight.T) }
            let qkvReference = separateQKV()
            let qkvCandidate = fusedQKV()
            MLX.eval(qkvReference, qkvCandidate)
            XCTAssertTrue(MLX.arrayEqual(qkvReference, qkvCandidate).item(Bool.self))
            let qkvTimes = pairedTimes(first: separateQKV, second: fusedQKV)

            let separateGateUp = {
                MLX.concatenated([
                    MLX.matmul(input, gateWeight.T),
                    MLX.matmul(input, upWeight.T),
                ], axis: -1)
            }
            let fusedGateUp = { MLX.matmul(input, fusedGateUpWeight.T) }
            let gateUpReference = separateGateUp()
            let gateUpCandidate = fusedGateUp()
            MLX.eval(gateUpReference, gateUpCandidate)
            XCTAssertTrue(MLX.arrayEqual(gateUpReference, gateUpCandidate).item(Bool.self))
            let gateUpTimes = pairedTimes(first: separateGateUp, second: fusedGateUp)

            print(
                String(
                    format: "[sensenova-bench] tokens=%d qkv %.3f -> %.3f ms (%.3fx)",
                    tokenCount,
                    qkvTimes.first * 1_000,
                    qkvTimes.second * 1_000,
                    qkvTimes.first / qkvTimes.second
                )
            )
            print(
                String(
                    format: "[sensenova-bench] tokens=%d gate-up %.3f -> %.3f ms (%.3fx)",
                    tokenCount,
                    gateUpTimes.first * 1_000,
                    gateUpTimes.second * 1_000,
                    gateUpTimes.first / gateUpTimes.second
                )
            )
        }
    }

    func testStaticKVUpdateAgainstConcatenation() throws {
        try requireBenchmark()
        let prefixCount = 320
        for tokenCount in tokenCounts {
            let prefix = MLXRandom.uniform(
                -0.5 ..< 0.5,
                [1, 8, prefixCount, 128]
            ).asType(.bfloat16)
            let current = MLXRandom.uniform(
                -0.5 ..< 0.5,
                [1, 8, tokenCount, 128]
            ).asType(.bfloat16)
            let template = MLX.concatenated([
                prefix,
                MLXArray.zeros([1, 8, tokenCount, 128], dtype: .bfloat16),
            ], axis: 2)
            MLX.eval(prefix, current, template)

            let concatenate = { MLX.concatenated([prefix, current], axis: 2) }
            let staticUpdate = {
                let output = template
                output[0..., 0..., prefixCount..., 0...] = current
                return output
            }
            let reference = concatenate()
            let candidate = staticUpdate()
            MLX.eval(reference, candidate)
            XCTAssertTrue(MLX.arrayEqual(reference, candidate).item(Bool.self))
            let times = pairedTimes(first: concatenate, second: staticUpdate)
            print(
                String(
                    format: "[sensenova-bench] tokens=%d kv-concat %.3f vs update %.3f ms (%.3fx)",
                    tokenCount,
                    times.first * 1_000,
                    times.second * 1_000,
                    times.first / times.second
                )
            )
        }
    }

    func testAllocatorCacheLimits() throws {
        try requireBenchmark()
        let input = MLXRandom.uniform(-0.5 ..< 0.5, [1_024, hiddenSize]).asType(.bfloat16)
        let weight = MLXRandom.uniform(-0.02 ..< 0.02, [hiddenSize, hiddenSize]).asType(.bfloat16)
        MLX.eval(input, weight)
        let originalLimit = Memory.cacheLimit
        defer {
            Memory.cacheLimit = originalLimit
            Memory.clearCache()
        }

        for limitMiB in [0, 256, 1_024, 2_048] {
            Memory.clearCache()
            Memory.cacheLimit = limitMiB * 1_048_576
            var times: [Double] = []
            for _ in 0..<6 {
                times.append(measure { MLX.matmul(input, weight.T) })
            }
            let median = times.sorted()[times.count / 2]
            let snapshot = Memory.snapshot()
            print(
                String(
                    format: "[sensenova-bench] allocator=%d MiB %.3f ms active=%.2f GiB cache=%.2f GiB",
                    limitMiB,
                    median * 1_000,
                    Double(snapshot.activeMemory) / 1_073_741_824,
                    Double(snapshot.cacheMemory) / 1_073_741_824
                )
            )
        }
    }
}
