import Foundation
import XCTest
import MLX
@testable import MereRunCore

/// Micro-benchmark for the per-call cost of mlx-swift compiled functions,
/// used to evaluate CompiledFunction.call patches. Skipped unless explicitly
/// requested — it prints timing, it does not assert performance.
final class MLXCompiledFunctionOverheadTests: MereRunCoreTestCase {
    func testCompiledCallOverheadMicrobench() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_BENCHMARK_COMPILE_OVERHEAD"] == "1" else {
            throw XCTSkip("Set MERERUN_BENCHMARK_COMPILE_OVERHEAD=1 to run the compiled-call micro-benchmark.")
        }

        let compiled = MLX.compile { (x: MLXArray) -> MLXArray in
            x + 1
        }
        let input = MLXArray([Float](repeating: 1, count: 64))
        MLX.eval(compiled(input))

        let iterations = 2_000

        var chained = input
        var start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            chained = compiled(chained)
        }
        let compiledBuildSeconds = CFAbsoluteTimeGetCurrent() - start
        MLX.eval(chained)
        let compiledTotalSeconds = CFAbsoluteTimeGetCurrent() - start

        var raw = input
        start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            raw = raw + 1
        }
        let rawBuildSeconds = CFAbsoluteTimeGetCurrent() - start
        MLX.eval(raw)
        let rawTotalSeconds = CFAbsoluteTimeGetCurrent() - start

        let scale = 1_000_000.0 / Double(iterations)
        print(String(
            format: "[compile-overhead] compiled build=%.1fus/call total=%.1fus/call | raw build=%.1fus/call total=%.1fus/call",
            compiledBuildSeconds * scale,
            compiledTotalSeconds * scale,
            rawBuildSeconds * scale,
            rawTotalSeconds * scale
        ))

        XCTAssertEqual(chained.shape, raw.shape)
        XCTAssertLessThan(
            MLX.abs(chained - raw).max().item(Float.self), 1e-3,
            "compiled and raw chains must agree"
        )
    }
}
