import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class MiniMaxH3MPPProjectionTests: MereRunCoreTestCase {
    func testSelectsMeasuredFeedForwardOutputTile() {
        XCTAssertEqual(
            MiniMaxH3MPPProjection.tile(
                inputDimension: 14_336,
                outputDimension: 5_376
            ),
            .init(rows: 64, columns: 128, simdgroups: 8)
        )
        XCTAssertEqual(
            MiniMaxH3MPPProjection.tile(
                inputDimension: 5_376,
                outputDimension: 21_504
            ),
            .init(rows: 32, columns: 64, simdgroups: 2)
        )
    }

    #if os(macOS)
    func testSmallProjectionMatchesMLXBitExactly() throws {
        guard MiniMaxH3MPPProjection.isAvailable else {
            throw XCTSkip(
                "MPP projection parity requires macOS 26 and a Metal GPU."
            )
        }

        MLXRandom.seed(2_026_081_015)
        let source = MLXRandom.uniform(-0.5 ..< 0.5, [2, 37, 128])
            .asType(.bfloat16)
        let weight = MLXRandom.uniform(-0.5 ..< 0.5, [192, 128])
            .asType(.bfloat16)
        let reference = MLX.matmul(source, weight.T)
        for tile in [
            MiniMaxH3MPPProjection.standardTile,
            MiniMaxH3MPPProjection.feedForwardOutputTile,
        ] {
            let candidate = try XCTUnwrap(
                MiniMaxH3MPPProjection.project(
                    source: source,
                    weight: weight,
                    tile: tile
                )
            )
            MLX.eval(reference, candidate)

            XCTAssertEqual(candidate.shape, [2, 37, 192])
            XCTAssertEqual(candidate.dtype, .bfloat16)
            XCTAssertTrue(MLX.arrayEqual(reference, candidate).item(Bool.self))
        }
    }

    func testProductionShapeReleaseBenchmark() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_H3_MPP_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_MPP_BENCH=1 to run the H3 MPP projection benchmark."
            )
        }
        guard MiniMaxH3MPPProjection.isAvailable else {
            throw XCTSkip("The H3 MPP projection benchmark requires macOS 26 and a Metal GPU.")
        }

        let rows = max(
            1,
            Int(ProcessInfo.processInfo.environment["MERERUN_H3_BENCH_ROWS"] ?? "")
                ?? 14_958
        )
        let rounds = max(
            2,
            Int(ProcessInfo.processInfo.environment["MERERUN_H3_BENCH_ROUNDS"] ?? "")
                ?? 4
        )
        for (name, inputDimension, outputDimension) in [
            ("qkv", 5_376, 21_504),
            ("attention-output", 7_168, 5_376),
            ("feed-forward-input", 5_376, 28_672),
            ("feed-forward-output", 14_336, 5_376),
        ] {
            try compareProductionShape(
                name: name,
                rows: rows,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                rounds: rounds
            )
            MLX.Memory.clearCache()
        }
    }

    private func compareProductionShape(
        name: String,
        rows: Int,
        inputDimension: Int,
        outputDimension: Int,
        rounds: Int
    ) throws {
        let source = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, rows, inputDimension]
        ).asType(.bfloat16)
        let weight = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [outputDimension, inputDimension]
        ).asType(.bfloat16)
        let reference = MLX.matmul(source, weight.T)
        let candidate = try XCTUnwrap(
            MiniMaxH3MPPProjection.project(source: source, weight: weight)
        )
        MLX.eval(source, weight, reference, candidate)
        XCTAssertTrue(MLX.arrayEqual(reference, candidate).item(Bool.self))

        var bestMLX = Double.greatestFiniteMagnitude
        var bestMPP = Double.greatestFiniteMagnitude
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                bestMLX = min(bestMLX, measure { MLX.matmul(source, weight.T) })
                bestMPP = min(bestMPP, measureMPP(source: source, weight: weight))
            } else {
                bestMPP = min(bestMPP, measureMPP(source: source, weight: weight))
                bestMLX = min(bestMLX, measure { MLX.matmul(source, weight.T) })
            }
        }

        print(String(
            format: "[h3-lab] mpp rows=%d projection=%@ %d->%d mlx_ms=%.3f "
                + "mpp_ms=%.3f speedup=%.3fx exact=true",
            rows,
            name,
            inputDimension,
            outputDimension,
            bestMLX * 1_000,
            bestMPP * 1_000,
            bestMLX / bestMPP
        ))
    }

    private func measure(_ body: () -> MLXArray) -> Double {
        let started = CFAbsoluteTimeGetCurrent()
        MLX.eval(body())
        return CFAbsoluteTimeGetCurrent() - started
    }

    private func measureMPP(source: MLXArray, weight: MLXArray) -> Double {
        measure {
            MiniMaxH3MPPProjection.project(source: source, weight: weight)
                ?? MLX.matmul(source, weight.T)
        }
    }
    #endif
}
