import Foundation
import MLX
import MLXFast
import XCTest
@testable import MereRunGemmaModel

final class SOVHeadDispatchProfileTests: MereRunCoreTestCase {
    func testProfileHeadGraphConstruction() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let output = env["MERERUN_PROFILE_SOV_HEAD_DISPATCH"] else {
            throw XCTSkip("Set GPU device and head-dispatch profile output.")
        }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("Sources/MereRunGemmaModel/Kernels/Gemma4HeadNormRoPE.metal"), encoding: .utf8)
        let generic = MLXFast.metalKernel(name: "gemma4_head_norm_rope",
            inputNames: ["q_raw", "k_raw", "v_raw", "q_weight", "k_weight", "eps", "offset", "freqs", "log_base"],
            outputNames: ["q_out", "k_out", "v_out"], source: source)
        let prepared = try Gemma4HeadNormRoPE(heads: 16, kvHeads: 2, dim: 512, rotatedDims: 128,
                                            proportional: true, base: 1_000_000, eps: 1e-6)
        let q = MLXArray.ones([1, 16, 8192], dtype: .bfloat16), k = MLXArray.ones([1, 16, 1024], dtype: .bfloat16)
        let weight = MLXArray.ones([512], dtype: .bfloat16), position = MLXArray([Int32(779)])
        let eps = MLXArray([Float(1e-6)]), logBase = MLXArray([log2(Float(1_000_000))])
        let frequencies = pow(MLXArray(Float(1_000_000)), MLXArray(stride(from: Float(0), to: 128, by: 2)) / Float(512))
        eval(q, k, weight, position, eps, logBase, frequencies)
        func reference() -> [MLXArray] {
            guard Gemma4MetalExperts.isSupported, [256, 512].contains(512),
                  q.ndim == 3, q.dim(1) <= 16, q.dtype == .bfloat16, k.dtype == .bfloat16, k.dtype == .bfloat16,
                  weight.dtype == .bfloat16, weight.dtype == .bfloat16 else { return [] }
            let batch = q.dim(0), count = q.dim(1)
            return generic([q, k, k, weight, weight, eps, position, frequencies, logBase],
                template: [("QH", 16), ("KH", 2), ("D", 512), ("T", count), ("RotHalf", 64), ("Proportional", true)],
                grid: (128, batch * count * 20, 1), threadGroup: (128, 1, 1),
                outputShapes: [[batch, 16, count, 512], [batch, 2, count, 512], [batch, 2, count, 512]],
                outputDTypes: [.bfloat16, .bfloat16, .bfloat16])
        }
        let actual = try XCTUnwrap(prepared(queries: q, keys: k, values: k, queryWeight: weight, keyWeight: weight, offset: position))
        for (a, b) in zip([actual.0, actual.1, actual.2], reference()) {
            XCTAssertEqual(abs(a - b).max().item(Float.self), 0)
        }
        let operations: [(String, () -> Void)] = [
            ("generic", { _ = reference() }),
            ("prepared", { _ = prepared(queries: q, keys: k, values: k, queryWeight: weight, keyWeight: weight, offset: position) })
        ]
        for (_, operation) in operations { for _ in 0..<100 { operation() } }
        var samples: [String: [Double]] = [:]
        for iteration in 0..<30 {
            for (name, operation) in iteration.isMultiple(of: 2) ? operations : Array(operations.reversed()) {
                let start = ProcessInfo.processInfo.systemUptime
                for _ in 0..<100 { operation() }
                samples[name, default: []].append((ProcessInfo.processInfo.systemUptime - start) / 100)
            }
        }
        // Deliberately no eval in the timed loop: this isolates host graph construction.
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(samples).write(to: URL(fileURLWithPath: output))
    }
}
