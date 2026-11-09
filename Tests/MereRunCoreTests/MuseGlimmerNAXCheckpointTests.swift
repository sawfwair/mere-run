#if os(macOS)
import Foundation
import MLX
import MLXNN
import MereRunTensor
import XCTest
@testable import MereRunCore

extension MuseGlimmerCheckpointQualificationTests {
    func testNAXHeadComponents() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"] else { throw XCTSkip("Set Glimmer checkpoint root") }
        try XCTSkipUnless(M5AffineQ8Head.isSupported, "Requires M5")
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let root = URL(fileURLWithPath: rootPath)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let target = try loadTarget(root, config: config)
        let head = try XCTUnwrap(target.lmHead as? QuantizedLinear)
        let biases = try XCTUnwrap(head.biases)
        let modes = [0, 1, 2, 4, 8]
        for rows in [1, 3, 4, 7, 8] {
            let x = MLX.sin(MLXArray(0..<(rows * config.textConfig.hiddenSize)).asType(.float32) * 0.017)
                .asType(.bfloat16).reshaped(1, rows, config.textConfig.hiddenSize)
            func operation(_ mode: Int) throws -> MLXArray {
                if mode == 0 {
                    return SmallBatchAffineQ8.matmul(x, weight: head.weight, scales: head.scales, biases: biases,
                        groupSize: 64, bits: 8, mode: .affine, rowsPerSIMD: 4)
                        ?? MLX.quantizedMM(x, head.weight, scales: head.scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
                }
                return try XCTUnwrap(M5AffineQ8Head.matmul(x, weight: head.weight, scales: head.scales, biases: biases, splits: mode))
            }
            let reference = try operation(0).asType(.float32)
            MLX.eval(reference)
            var errors: [Int: (Float, Float)] = [:]
            for mode in modes {
                let actual = try operation(mode).asType(.float32)
                let delta = actual - reference
                errors[mode] = (delta.abs().max().item(Float.self), MLX.sqrt((delta * delta).mean() / (reference * reference).mean()).item(Float.self))
                XCTAssertTrue(MLX.all(MLX.isFinite(actual)).item(Bool.self))
                if mode > 0 {
                    let serial = try (0..<rows).map { index in
                        try XCTUnwrap(M5AffineQ8Head.matmul(x[0..., index..<(index + 1), 0...],
                            weight: head.weight, scales: head.scales, biases: biases, splits: mode))
                    }
                    XCTAssertEqual((actual - MLX.concatenated(serial, axis: 1).asType(.float32)).abs().max().item(Float.self), 0)
                }
            }
            for repetition in 0..<9 {
                for offset in modes.indices {
                    let mode = modes[(repetition + offset) % modes.count]
                    let start = Date()
                    let values = try (0..<4).map { _ in try operation(mode) }
                    MLX.eval(values)
                    let seconds = Date().timeIntervalSince(start) / 4
                    if repetition > 0 {
                        var row = Record(kind: "nax-head-operator", modelRoot: rootPath, bits: 8,
                            id: "\(config.textConfig.vocabSize)x\(config.textConfig.hiddenSize)", width: rows, repetition: repetition)
                        row.verificationKernel = mode == 0 ? "head4" : "nax-head\(mode)"
                        row.operatorSeconds = seconds; row.maximumDifference = errors[mode]!.0
                        row.relativeRMSError = errors[mode]!.1; row.thermalState = ProcessInfo.processInfo.thermalState.rawValue
                        try record(row, env: env)
                    }
                }
            }
        }
    }

    func testNAXHeadLikelihoodPairs() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"],
              let textPath = env["MERERUN_GLIMMER_QUALIFICATION_TEXT"] else { throw XCTSkip("Set Glimmer root and held-out text") }
        let splits = try XCTUnwrap(Int(env["MERERUN_GLIMMER_NAX_SPLITS"] ?? "1"))
        XCTAssertTrue([1, 2, 4, 8].contains(splits))
        try XCTSkipUnless(M5AffineQ8Head.isSupported, "Requires M5")
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let root = URL(fileURLWithPath: rootPath)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let target = try loadTarget(root, config: config)
        let head = try XCTUnwrap(target.lmHead as? QuantizedLinear)
        let biases = try XCTUnwrap(head.biases)
        let tokenizer = try await loadTokenizer(root)
        let tokens = tokenizer.tokenizer.encode(text: try String(contentsOfFile: textPath, encoding: .utf8))
        let length = 2049
        XCTAssertGreaterThan(tokens.count, 16 * length)
        func logits(_ x: MLXArray, mode: Int) throws -> MLXArray {
            var result: MLXArray
            if mode == 0 {
                result = try XCTUnwrap(SmallBatchAffineQ8.matmul(x, weight: head.weight, scales: head.scales,
                    biases: biases, groupSize: 64, bits: 8, mode: .affine, rowsPerSIMD: 4))
            } else {
                result = try XCTUnwrap(M5AffineQ8Head.matmul(x, weight: head.weight, scales: head.scales, biases: biases, splits: mode))
            }
            result = result * config.textConfig.outputMultiplier
            let cap = config.textConfig.finalLogitSoftcapping
            if cap > 0 { result = MLX.tanh(result / cap) * cap }
            return result[0].asType(.float32)
        }
        for window in 0..<16 {
            let start = window * (tokens.count - length) / 15
            let sequence = Array(tokens[start..<(start + length)])
            let cache = target.makeCache()
            var losses: [Int: [Float]] = [0: [], splits: []]
            var top1: [Int: [Int]] = [0: [], splits: []]
            for offset in stride(from: 0, to: length - 1, by: 128) {
                try Task.checkCancellation()
                let input = MLXArray(sequence[offset..<(offset + 128)].map(Int32.init)).reshaped(1, 128)
                let hidden = target.model.languageModel.forward(target.inputEmbeddings(for: input), cache: cache).hidden
                MLX.eval(hidden); evaluateGemma4CacheStorage(cache)
                for block in stride(from: 0, to: 128, by: 8) {
                    let x = hidden[0..., block..<(block + 8), 0...]
                    let labels = MLXArray(sequence[(offset + block + 1)...(offset + block + 8)].map(Int32.init)).expandedDimensions(axis: -1)
                    for mode in [0, splits] {
                        let projected = try logits(x, mode: mode)
                        let nll = (logSumExp(projected, axis: -1, keepDims: true) - takeAlong(projected, labels, axis: -1)).reshaped(8)
                        let best = argMax(projected, axis: -1).asType(.int32)
                        MLX.eval(nll, best)
                        losses[mode]! += nll.asArray(Float.self)
                        top1[mode]! += best.asArray(Int32.self).map(Int.init)
                    }
                }
            }
            for mode in [0, splits] {
                var row = Record(kind: "nax-head-likelihood", modelRoot: rootPath, bits: 8,
                    id: "wikitext2-test-\(window)", tokens: sequence, nll: losses[mode], top1: top1[mode])
                row.verificationKernel = mode == 0 ? "head4" : "nax-head\(mode)"
                XCTAssertEqual(row.nll?.count, length - 1)
                XCTAssertTrue(row.nll!.allSatisfy { $0.isFinite })
                try record(row, env: env)
            }
            Memory.clearCache()
        }
    }

}
#endif
