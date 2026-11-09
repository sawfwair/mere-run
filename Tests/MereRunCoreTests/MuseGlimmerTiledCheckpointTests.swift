#if os(macOS)
import Foundation
import MLX
import MLXNN
import MereRunTensor
import XCTest
@testable import MereRunCore

extension MuseGlimmerCheckpointQualificationTests {
    func testTiledQ8Components() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"] else { throw XCTSkip("Set Glimmer checkpoint root") }
        try XCTSkipUnless(M5AffineQ8Head.isSupported, "Requires M5")
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let root = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let target = try loadTarget(root, config: config)
        defer { Memory.clearCache() }
        let projections = target.namedModules().compactMap { $0.1 as? QuantizedLinear }.filter { $0.bits == 8 }
        for (input, output) in [(6656, 19968), (19968, 6656), (6656, 4096), (6656, 256), (6656, 202048)] {
            let projection = try XCTUnwrap(projections.first { $0.weight.shape == [output, input / 4] })
            let weights = try XCTUnwrap(M5TiledAffineQ8.Weights(
                weight: projection.weight, scales: projection.scales, biases: XCTUnwrap(projection.biases)))
            for rows in [1, 3, 4, 8] {
                let x = MLX.sin(MLXArray(0..<(rows * input)).asType(.float32) * 0.017)
                    .asType(.bfloat16).reshaped(1, rows, input)
                func operation(_ splits: Int) throws -> MLXArray {
                    if splits == 0 {
                        if output >= 65_536, let shared = SmallBatchAffineQ8.matmul(x, weight: projection.weight,
                            scales: projection.scales, biases: projection.biases!, groupSize: 64, bits: 8, mode: .affine, rowsPerSIMD: 4) {
                            return shared
                        }
                        return MLX.quantizedMM(x, projection.weight, scales: projection.scales, biases: projection.biases,
                            transpose: true, groupSize: 64, bits: 8)
                    }
                    return try XCTUnwrap(M5TiledAffineQ8.matmul(x, weights: weights, splits: splits))
                }
                let reference = try operation(0).asType(.float32)
                MLX.eval(reference)
                var errors: [Int: (Float, Float)] = [:]
                for mode in [0, 1, 2, 4, 8] {
                    let actual = try operation(mode).asType(.float32)
                    let delta = actual - reference
                    errors[mode] = (delta.abs().max().item(Float.self), sqrt((delta * delta).mean() / (reference * reference).mean()).item(Float.self))
                    if mode > 0 {
                        let serial = try concatenated((0..<rows).map {
                            try XCTUnwrap(M5TiledAffineQ8.matmul(x[0..., $0..<($0 + 1), 0...], weights: weights, splits: mode))
                        }, axis: 1).asType(.float32)
                        XCTAssertEqual((actual - serial).abs().max().item(Float.self), 0)
                    }
                }
                let modes = [0, 1, 2, 4, 8]
                for repetition in 0..<9 {
                    for offset in modes.indices {
                        let mode = modes[(repetition + offset) % modes.count]
                        let start = Date()
                        let values = try (0..<4).map { _ in try operation(mode) }
                        MLX.eval(values)
                        let seconds = Date().timeIntervalSince(start) / 4
                        if repetition == 0 { continue }
                        var row = Record(kind: "tiled-q8-operator", modelRoot: path, bits: 8,
                            id: "\(output)x\(input)", width: rows, repetition: repetition)
                        row.verificationKernel = mode == 0 ? "head4" : "tiled\(mode)"
                        row.operatorSeconds = seconds; row.maximumDifference = errors[mode]!.0
                        row.relativeRMSError = errors[mode]!.1
                        try record(row, env: env)
                    }
                }
            }
        }
    }

    func testTiledQ8FFNComponents() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"] else { throw XCTSkip("Set Glimmer checkpoint root") }
        try XCTSkipUnless(M5AffineQ8Head.isSupported, "Requires M5")
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let root = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let target = try loadTarget(root, config: config)
        defer { Memory.clearCache() }
        let mlp = try XCTUnwrap(target.namedModules().compactMap { $0.1 as? MuseGlimmerTextMLP }.first)
        let gate = try XCTUnwrap(mlp.gateProj as? QuantizedLinear)
        let up = try XCTUnwrap(mlp.upProj as? QuantizedLinear)
        let g = try XCTUnwrap(M5TiledAffineQ8.Weights(weight: gate.weight, scales: gate.scales, biases: XCTUnwrap(gate.biases)))
        let u = try XCTUnwrap(M5TiledAffineQ8.Weights(weight: up.weight, scales: up.scales, biases: XCTUnwrap(up.biases)))
        for rows in [1, 3, 4, 8] {
            let x = MLX.sin(MLXArray(0..<(rows * 6656)).asType(.float32) * 0.017).asType(.bfloat16).reshaped(1, rows, 6656)
            func operation(_ mode: Int) throws -> MLXArray {
                if mode == 0 { return MLXNN.silu(gate(x)) * up(x) }
                if mode == 1 {
                    return try MLXNN.silu(XCTUnwrap(M5TiledAffineQ8.matmul(x, weights: g, splits: 1)))
                        * XCTUnwrap(M5TiledAffineQ8.matmul(x, weights: u, splits: 1))
                }
                return try XCTUnwrap(M5TiledAffineQ8.swiglu(x, gate: g, up: u))
            }
            let separate = try operation(1)
            let fused = try operation(2)
            XCTAssertEqual((separate.asType(.float32) - fused.asType(.float32)).abs().max().item(Float.self), 0)
            let modes = [0, 1, 2]
            for repetition in 0..<9 {
                for offset in modes.indices {
                    let mode = modes[(repetition + offset) % modes.count]
                    let start = Date()
                    let values = try (0..<4).map { _ in try operation(mode) }
                    MLX.eval(values)
                    let seconds = Date().timeIntervalSince(start) / 4
                    if repetition == 0 { continue }
                    var row = Record(kind: "tiled-q8-ffn", modelRoot: path, bits: 8, id: "gate-up-19968x6656", width: rows, repetition: repetition)
                    row.verificationKernel = ["native", "tiled-separate", "tiled-fused"][mode]
                    row.operatorSeconds = seconds
                    try record(row, env: env)
                }
            }
        }
    }

    func testSmallBatchTeacherForcedLikelihood() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"],
              let text = env["MERERUN_GLIMMER_QUALIFICATION_TEXT"] else { throw XCTSkip("Set Glimmer root and held-out text") }
        let requestedWidth = Int(env["MERERUN_GLIMMER_QUALIFICATION_ROWS"] ?? "4")
        let width = try XCTUnwrap(requestedWidth.flatMap { [1, 2, 4, 8].contains($0) ? $0 : nil },
            "Qualification rows must be 1, 2, 4, or 8.")
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let root = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let tokenizer = try await loadTokenizer(root)
        let tokens = tokenizer.tokenizer.encode(text: try String(contentsOfFile: text, encoding: .utf8))
        let target = try loadTarget(root, config: config)
        defer { Memory.clearCache() }
        try target.materializeTiledWarmup(checkCancellation: { try Task.checkCancellation() })
        let length = 2049
        XCTAssertGreaterThan(tokens.count, 16 * length)
        for window in 0..<16 {
            let start = window * (tokens.count - length) / 15
            let sequence = Array(tokens[start..<(start + length)])
            let cache = target.makeCache()
            var losses: [Float] = [], predictions: [Int] = []
            for offset in stride(from: 0, to: length - 1, by: width) {
                try Task.checkCancellation()
                let input = MLXArray(sequence[offset..<(offset + width)].map(Int32.init)).reshaped(1, width)
                let logits = target.forward(input, cache: cache).logits[0].asType(.float32)
                let labels = MLXArray(sequence[(offset + 1)...(offset + width)].map(Int32.init)).expandedDimensions(axis: -1)
                let nll = (logSumExp(logits, axis: -1, keepDims: true) - takeAlong(logits, labels, axis: -1)).reshaped(width)
                let best = argMax(logits, axis: -1).asType(.int32)
                MLX.eval(nll, best); evaluateGemma4CacheStorage(cache)
                losses += nll.asArray(Float.self); predictions += best.asArray(Int32.self).map(Int.init)
            }
            var row = Record(kind: "likelihood", modelRoot: path, bits: 8, id: "wikitext2-test-\(window)",
                tokens: sequence, nll: losses, top1: predictions)
            row.verificationKernel = env["MERERUN_MUSE_GLIMMER_Q8_VERIFY"] ?? "native"
            row.width = width
            XCTAssertEqual(losses.count, 2048); XCTAssertTrue(losses.allSatisfy { $0.isFinite })
            try record(row, env: env)
            Memory.clearCache()
        }
    }
}
#endif
