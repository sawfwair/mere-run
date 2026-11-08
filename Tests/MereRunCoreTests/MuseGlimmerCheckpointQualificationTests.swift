#if os(macOS)
import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

/// Opt-in measurements against the exact runtime, weights, and tokenizer used in serving.
/// Finite outputs qualify execution; task/quality gates are evaluated separately.
final class MuseGlimmerCheckpointQualificationTests: MereRunCoreTestCase {
    private struct Plan: Decodable {
        struct Prompt: Decodable { let id: String; let text: String }
        let prompts: [Prompt]
        let widths: [Int]
        let chunkSizes: [Int]
        let repetitions: Int
        let outputTokens: Int
    }

    private struct Record: Encodable {
        let kind: String
        let modelRoot: String
        let bits: Int?
        let id: String
        var width: Int?
        var chunk: Int?
        var repetition: Int?
        var promptTokens: Int?
        var outputTokens: Int?
        var prefillSeconds: Double?
        var decodeSeconds: Double?
        var firstTokenSeconds: Double?
        var stats: MuseGlimmerDFlashStats?
        var text: String?
        var tokens: [Int]?
        var nll: [Float]?
        var top1: [Int]?
        var activeBytes: Int?
        var thermalState: Int?
        var peakBytes: Int?
        var prefillRoute: String?
        var activationDType: String?
    }

    func testNativeDecodeSweep() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"],
              let planPath = env["MERERUN_GLIMMER_QUALIFICATION_PLAN"] else {
            throw XCTSkip("Set Glimmer qualification root, plan, and output for checkpoint measurements")
        }
        let root = URL(fileURLWithPath: rootPath)
        let plan = try JSONDecoder().decode(Plan.self, from: Data(contentsOf: URL(fileURLWithPath: planPath)))
        XCTAssertTrue(plan.widths.allSatisfy { (0...15).contains($0) })
        XCTAssertTrue(plan.chunkSizes.allSatisfy { (128...2048).contains($0) })
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let tokenizer = try await loadTokenizer(root)
        let target = try loadTarget(root, config: config)
        let densePrefill = target.namedModules().contains { $0.1 is MuseGlimmerPrefillLinear }
        let assistant = try plan.widths.contains(where: { $0 > 0 }) ? loadAssistant(env) : nil
        defer { Memory.clearCache() }
        let sampling = GenerationConfig(maxTokens: plan.outputTokens, temperature: 0, topK: 0, topP: 1, minP: 0, repetitionPenalty: 1)
        // Materialize production serial/speculative kernels outside measured trials.
        _ = try MuseGlimmerGenerator.warmUp(model: target, assistant: assistant, speculativeTokens: 3)
        for prompt in plan.prompts {
            let tokens = try tokenizer.encodeForGeneration(messages: [.init(role: .user, content: prompt.text)],
                reasoningStrength: "low", currentDate: "2026-11-08", maxLength: 65536, truncate: false)
            for chunk in plan.chunkSizes {
                for repetition in 0..<plan.repetitions {
                    for width in plan.widths {
                        let cache = target.makeCache()
                        let draftCache = width > 0 ? assistant?.makeCache() : nil
                        let start = Date()
                        var logits: MLXArray?
                        for offset in stride(from: 0, to: tokens.count, by: chunk) {
                            try Task.checkCancellation()
                            let end = min(tokens.count, offset + chunk)
                            let input = MLXArray(tokens[offset..<end].map(Int32.init)).reshaped(1, end - offset)
                            let result = try target.forwardPrefillDetailed(inputIds: input, imageBatch: nil, cache: cache,
                                captureLayerIndices: width > 0 ? Set(assistant!.config.targetLayerIds) : [])
                            if let assistant, let draftCache {
                                assistant.appendTargetContext(result.capturedHiddenStates, cache: draftCache)
                                evaluateGemma4CacheStorage(draftCache)
                            }
                            MLX.eval(result.logits)
                            evaluateGemma4CacheStorage(cache)
                            logits = result.logits
                        }
                        let prefill = Date().timeIntervalSince(start)
                        var row = Record(kind: "decode", modelRoot: rootPath, bits: config.quantization?.bits, id: prompt.id,
                            width: width, chunk: chunk, repetition: repetition, promptTokens: tokens.count, prefillSeconds: prefill)
                        if width > 0 {
                            let result = try MuseGlimmerDFlashDecoder.decode(initialLogits: XCTUnwrap(logits), target: target,
                                targetCache: cache, assistant: XCTUnwrap(assistant), assistantCache: XCTUnwrap(draftCache),
                                generationConfig: sampling, eosTokens: [], tokenBudget: plan.outputTokens, historySeedTokens: tokens,
                                speculativeTokens: width, assistantModelPath: env["MERERUN_GLIMMER_QUALIFICATION_ASSISTANT"],
                                checkCancellation: { try Task.checkCancellation() })
                            row.tokens = result.generatedTokens; row.decodeSeconds = result.decodeSeconds
                            row.firstTokenSeconds = result.firstTokenSeconds; row.stats = result.stats
                        } else {
                            let result = try AutoregressiveDecodeEngine.decode(AutoregressiveDecodeRequest(initialLogits: XCTUnwrap(logits),
                                generationConfig: sampling, eosTokens: [], tokenBudget: plan.outputTokens, historySeedTokens: tokens),
                                stepForward: { target($0, cache: cache) }, checkCancellation: { try Task.checkCancellation() })
                            row.tokens = result.generatedTokens; row.decodeSeconds = result.decodeSeconds
                            row.firstTokenSeconds = result.firstTokenSeconds
                        }
                        row.outputTokens = row.tokens?.count
                        row.text = tokenizer.decode(tokens: row.tokens ?? [])
                        row.activeBytes = Memory.activeMemory
                        row.thermalState = ProcessInfo.processInfo.thermalState.rawValue
                        row.activationDType = String(describing: target.model.languageModel.embedTokens.weight.dtype)
                        row.peakBytes = Memory.peakMemory
                        row.prefillRoute = densePrefill ? "dense-activation-reconstruction" : "native"
                        XCTAssertEqual(row.outputTokens, plan.outputTokens)
                        try record(row, env: env)
                        Memory.clearCache()
                    }
                }
            }
        }
    }

    func testTeacherForcedLikelihood() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"],
              let textPath = env["MERERUN_GLIMMER_QUALIFICATION_TEXT"] else {
            throw XCTSkip("Set Glimmer qualification root, held-out text, and output for likelihood measurements")
        }
        let root = URL(fileURLWithPath: rootPath)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let tokenizer = try await loadTokenizer(root)
        let text = try String(contentsOfFile: textPath, encoding: .utf8)
        let tokens = tokenizer.tokenizer.encode(text: text)
        let length = 2049
        XCTAssertGreaterThan(tokens.count, 16 * length)
        let target = try loadTarget(root, config: config)
        defer { Memory.clearCache() }
        for window in 0..<16 {
            let start = window * (tokens.count - length) / 15
            let sequence = Array(tokens[start..<(start + length)])
            let cache = target.makeCache()
            var nll: [Float] = []
            var top1: [Int] = []
            for offset in stride(from: 0, to: length - 1, by: 128) {
                try Task.checkCancellation()
                let end = min(length - 1, offset + 128)
                let input = MLXArray(sequence[offset..<end].map(Int32.init)).reshaped(1, end - offset)
                let logits = target.forward(input, cache: cache).logits[0].asType(.float32)
                let labels = MLXArray(sequence[(offset + 1)...end].map(Int32.init)).expandedDimensions(axis: -1)
                let losses = (logSumExp(logits, axis: -1, keepDims: true) - takeAlong(logits, labels, axis: -1)).reshaped(end - offset)
                let best = argMax(logits, axis: -1).asType(.int32)
                MLX.eval(losses, best)
                evaluateGemma4CacheStorage(cache)
                nll += losses.asArray(Float.self)
                top1 += best.asArray(Int32.self).map(Int.init)
            }
            XCTAssertTrue(nll.allSatisfy { $0.isFinite })
            XCTAssertEqual(nll.count, length - 1)
            try record(Record(kind: "likelihood", modelRoot: rootPath, bits: config.quantization?.bits,
                id: "wikitext2-test-\(window)", tokens: sequence, nll: nll, top1: top1), env: env)
            Memory.clearCache()
        }
    }

    private func loadTokenizer(_ root: URL) async throws -> MuseGlimmerTokenizerAndTemplate {
        let generation = try JSONDecoder().decode(MuseGlimmerGenerationConfig.self,
            from: Data(contentsOf: root.appendingPathComponent("generation_config.json")))
        return try await MuseGlimmerTokenizerAndTemplate.load(from: root, generationConfig: generation)
    }

    private func loadTarget(_ root: URL, config: MuseGlimmerConfig) throws -> MuseGlimmerModel {
        let target = MuseGlimmerModel(config: config)
        let index = root.appendingPathComponent("model.safetensors.index.json")
        if let quantization = config.quantization {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(indexURL: index, to: target,
                groupSize: quantization.groupSize, bits: quantization.bits,
                keyMapper: MuseGlimmerWeightKeys.normalized, mapper: MuseGlimmerWeightKeys.mapped)
        } else {
            try HFSafetensorsWeightsLoader.applyShardedWeights(indexURL: index, to: target, mapper: MuseGlimmerWeightKeys.mapped)
        }
        FileHandle.standardError.write(Data("[glimmer-qualification-loaded] \(root.lastPathComponent)\n".utf8))
        let formats = Dictionary(grouping: target.namedModules().compactMap { _, module -> String? in
            guard let q = module as? QuantizedLinear else { return nil }
            return "bits=\(q.bits),group=\(q.groupSize),scales=\(q.scales.dtype)"
        }, by: { $0 }).mapValues(\.count)
        FileHandle.standardError.write(Data("[glimmer-qualification-formats] \(formats)\n".utf8))
        let count = try target.installQuantizedPrefill()
        FileHandle.standardError.write(Data("[glimmer-qualification-prefill] dense_modules=\(count)\n".utf8))
        return target
    }

    private func loadAssistant(_ env: [String: String]) throws -> MuseGlimmerAssistantModel {
        let root = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_GLIMMER_QUALIFICATION_ASSISTANT"]))
        let config = try JSONDecoder().decode(MuseGlimmerAssistantConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let assistant = MuseGlimmerAssistantModel(config: config)
        try HFSafetensorsWeightsLoader.applyWeights(url: root.appendingPathComponent("model.safetensors"), to: assistant,
            mapper: MuseGlimmerDFlash2WeightKeys.mapped)
        return assistant
    }

    private func record(_ row: Record, env: [String: String]) throws {
        let output = try XCTUnwrap(env["MERERUN_GLIMMER_QUALIFICATION_OUTPUT"])
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(row) + Data("\n".utf8)
        if !FileManager.default.fileExists(atPath: output) { FileManager.default.createFile(atPath: output, contents: nil) }
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
        defer { try? handle.close() }
        try handle.seekToEnd(); try handle.write(contentsOf: data)
        FileHandle.standardError.write(Data("[glimmer-qualification-row] \(row.kind) \(row.id) width=\(row.width ?? 0)\n".utf8))
    }
}
#endif
