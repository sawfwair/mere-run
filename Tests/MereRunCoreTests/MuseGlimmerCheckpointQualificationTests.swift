#if os(macOS)
import Foundation
import MLX
import MLXNN
import MLXRandom
import MereRunAdmission
import MereRunTensor
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
        var schedulings: [MuseGlimmerDFlashScheduling]?
        var profile: Bool?
        var temperature: Float?
        var topK: Int?
        var topP: Float?
        var reasoningStrength: String?
        var seed: UInt64?
        var acceptanceRounds: Int?
        var minimumAcceptance: Double?
        var fallback: MuseGlimmerDFlashFallback?
    }

    struct Record: Encodable {
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
        var scheduling: String?
        var temperature: Float?
        var topK: Int?
        var topP: Float?
        var reasoningStrength: String?
        var seed: UInt64?
        var phaseSeconds: [String: Double]?
        var admissionWaitSeconds: Double?
        var checkpointPreparationSeconds: Double?
        var fusedNormalization: Bool?
        var fallback: String?
        var verificationKernel: String?
        var operatorSeconds: Double?
        var maximumDifference: Float?
        var relativeRMSError: Float?
    }

    private final class PhaseProfile {
        var seconds: [String: Double] = [:]
        private var mark = Date()

        private func finish(_ phase: String) {
            seconds[phase, default: 0] += Date().timeIntervalSince(mark)
            mark = Date()
        }

        func observe(_ event: MuseGlimmerDFlashEvent) {
            switch event {
            case .beginRound: mark = Date()
            case .drafted(let arrays):
                MLX.eval(arrays)
                finish("draft")
            case .verificationSubmitted: finish("verificationSubmission")
            case .verified: finish("verificationWait")
            case .accepted: finish("acceptance")
            case .committed(let arrays):
                MLX.eval(arrays)
                finish("cacheUpdate")
            case .fallback(let arrays):
                MLX.eval(arrays)
                finish("serialFallback")
            }
        }
    }

    func testNativeDecodeSweep() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"],
              let planPath = env["MERERUN_GLIMMER_QUALIFICATION_PLAN"] else {
            throw XCTSkip("Set Glimmer qualification root, plan, and output for checkpoint measurements")
        }
        let admissionStart = Date()
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let admissionWait = Date().timeIntervalSince(admissionStart)
        let preparationStart = Date()
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
        let sampling = GenerationConfig(maxTokens: plan.outputTokens, temperature: plan.temperature ?? 0,
            topK: plan.topK ?? 0, topP: plan.topP ?? 1, minP: 0, repetitionPenalty: 1)
        // Materialize production serial/speculative kernels outside measured trials.
        let schedulings = plan.schedulings ?? [.eager]
        for scheduling in schedulings {
            _ = try MuseGlimmerGenerator.warmUp(model: target, assistant: assistant, speculativeTokens: 3, scheduling: scheduling)
        }
        let preparationSeconds = Date().timeIntervalSince(preparationStart)
        for prompt in plan.prompts {
            let tokens = try tokenizer.encodeForGeneration(messages: [.init(role: .user, content: prompt.text)],
                reasoningStrength: plan.reasoningStrength ?? "low", currentDate: "2026-11-08", maxLength: 65536, truncate: false)
            for chunk in plan.chunkSizes {
                for repetition in 0..<plan.repetitions {
                    for width in plan.widths {
                        let order = repetition.isMultiple(of: 2) ? schedulings : Array(schedulings.reversed())
                        for scheduling in width == 0 ? Array(order.prefix(1)) : order {
                            if let seed = plan.seed { MLXRandom.seed(seed + UInt64(repetition)) }
                            var acceptanceSeed = (plan.seed ?? 0) + UInt64(repetition)
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
                            row.verificationKernel = env["MERERUN_MUSE_GLIMMER_Q8_VERIFY"] ?? "native"
                            row.fusedNormalization = env["MERERUN_MUSE_GLIMMER_FUSED_NORM"] == "1"
                            row.fallback = (plan.fallback ?? .selected).rawValue
                            row.scheduling = scheduling.rawValue
                            row.admissionWaitSeconds = admissionWait
                            row.checkpointPreparationSeconds = preparationSeconds
                            row.temperature = sampling.temperature
                            row.topK = sampling.topK; row.topP = sampling.topP
                            row.reasoningStrength = plan.reasoningStrength ?? "low"
                            row.seed = plan.seed.map { $0 + UInt64(repetition) }
                            let profile = plan.profile == true ? PhaseProfile() : nil
                            let observer: ((MuseGlimmerDFlashEvent) -> Void)? = profile.map { profile in { profile.observe($0) } }
                            if width > 0 {
                                let result = try MuseGlimmerDFlashDecoder.decode(initialLogits: XCTUnwrap(logits), target: target,
                                    targetCache: cache, assistant: XCTUnwrap(assistant), assistantCache: XCTUnwrap(draftCache),
                                    generationConfig: sampling, eosTokens: [], tokenBudget: plan.outputTokens, historySeedTokens: tokens,
                                    speculativeTokens: width, assistantModelPath: env["MERERUN_GLIMMER_QUALIFICATION_ASSISTANT"],
                                    checkCancellation: { try Task.checkCancellation() }, scheduling: scheduling,
                                    minimumAcceptanceRate: plan.minimumAcceptance ?? MuseGlimmerDFlashPolicy.minimumAcceptanceRate(),
                                    acceptanceEvaluationRounds: plan.acceptanceRounds ?? MuseGlimmerDFlashPolicy.defaultAcceptanceEvaluationRounds,
                                    fallback: plan.fallback ?? .selected, observer: observer,
                                    acceptanceDraw: {
                                        guard plan.seed != nil else { return Float.random(in: 0..<1) }
                                        acceptanceSeed = acceptanceSeed &* 6_364_136_223_846_793_005 &+ 1
                                        return Float(acceptanceSeed >> 40) / Float(1 << 24)
                                    })
                                row.tokens = result.generatedTokens; row.decodeSeconds = result.decodeSeconds
                                row.firstTokenSeconds = result.firstTokenSeconds; row.stats = result.stats
                            } else {
                                let result = try AutoregressiveDecodeEngine.decode(AutoregressiveDecodeRequest(initialLogits: XCTUnwrap(logits),
                                    generationConfig: sampling, eosTokens: [], tokenBudget: plan.outputTokens, historySeedTokens: tokens),
                                    stepForward: { target($0, cache: cache) }, checkCancellation: { try Task.checkCancellation() })
                                row.tokens = result.generatedTokens; row.decodeSeconds = result.decodeSeconds
                                row.firstTokenSeconds = result.firstTokenSeconds
                            }
                            row.phaseSeconds = profile?.seconds
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
    }

    func testQ8VerificationComponents() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"] else {
            throw XCTSkip("Set the Glimmer checkpoint root and output for component measurements")
        }
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let root = URL(fileURLWithPath: rootPath)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let target = try loadTarget(root, config: config)
        defer { Memory.clearCache() }
        let projections = target.namedModules().compactMap { $0.1 as? QuantizedLinear }.filter { $0.bits == 8 }
        let hidden = config.textConfig.hiddenSize
        let intermediate = config.textConfig.intermediateSize
        let shapes = [(hidden, intermediate), (intermediate, hidden), (hidden, 4096), (4096, hidden),
                      (hidden, 256), (hidden, config.textConfig.vocabSize)]
        for (input, output) in shapes {
            let projection = try XCTUnwrap(projections.first { $0.weight.dim(0) == output && $0.weight.dim(1) == input / 4 })
            let biases = try XCTUnwrap(projection.biases)
            for rows in [3, 4, 7, 8] {
                let x = MLX.sin(MLXArray(0..<(rows * input)).asType(.float32) * 0.017).asType(.bfloat16).reshaped(1, rows, input)
                let reference = concatenated((0..<rows).map { row in
                    MLX.quantizedMM(x[0..., row..<(row + 1), 0...], projection.weight,
                                    scales: projection.scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
                }, axis: 1)
                MLX.eval(reference)
                func operation(_ mode: Int) throws -> MLXArray {
                    if mode == 0 {
                        return MLX.quantizedMM(x, projection.weight, scales: projection.scales, biases: biases,
                                               transpose: true, groupSize: 64, bits: 8)
                    }
                    return try XCTUnwrap(SmallBatchAffineQ8.matmul(x, weight: projection.weight,
                        scales: projection.scales, biases: biases, groupSize: 64, bits: 8, mode: .affine, rowsPerSIMD: mode))
                }
                let modes = [0, 1, 2, 4]
                var differences: [Int: Float] = [:]
                for mode in modes {
                    let actual = try operation(mode)
                    differences[mode] = (actual.asType(.float32) - reference.asType(.float32)).abs().max().item(Float.self)
                    if mode != 0 { XCTAssertEqual(differences[mode], 0, "K=\(input), N=\(output), M=\(rows), R=\(mode)") }
                }
                for repetition in 0..<9 {
                    for offset in modes.indices {
                        let mode = modes[(repetition + offset) % modes.count]
                        let start = Date()
                        let values = try (0..<4).map { _ in try operation(mode) }
                        MLX.eval(values)
                        let seconds = Date().timeIntervalSince(start) / 4
                        if repetition > 0 {
                            var row = Record(kind: "q8-operator", modelRoot: rootPath, bits: 8, id: "\(output)x\(input)",
                                             width: rows, repetition: repetition)
                            row.verificationKernel = mode == 0 ? "native" : "shared\(mode)"
                            row.operatorSeconds = seconds
                            row.maximumDifference = differences[mode]
                            row.thermalState = ProcessInfo.processInfo.thermalState.rawValue
                            try record(row, env: env)
                        }
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
        let lease = try await checkpointAdmission()
        defer { lease.release() }
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

    func checkpointAdmission() async throws -> MachineInferenceLease {
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true)
        )
        return try await coordinator.acquire(.init(label: "Glimmer checkpoint qualification", resourceClass: .large))
    }

    func loadTokenizer(_ root: URL) async throws -> MuseGlimmerTokenizerAndTemplate {
        let generation = try JSONDecoder().decode(MuseGlimmerGenerationConfig.self,
            from: Data(contentsOf: root.appendingPathComponent("generation_config.json")))
        return try await MuseGlimmerTokenizerAndTemplate.load(from: root, generationConfig: generation)
    }

    func loadTarget(_ root: URL, config: MuseGlimmerConfig) throws -> MuseGlimmerModel {
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
        target.configureFusedNormalization(ProcessInfo.processInfo.environment["MERERUN_MUSE_GLIMMER_FUSED_NORM"] == "1")
        let parameterKeys = Set(target.parameters().flattened().map(\.0))
        let count = try target.installQuantizedPrefill()
        #if os(macOS)
        if let mode = ProcessInfo.processInfo.environment["MERERUN_MUSE_GLIMMER_Q8_VERIFY"], mode.hasPrefix("tiled-") {
            let packed = target.namedModules().compactMap { path, module -> (String, MuseGlimmerTiledQ8Linear)? in
                (module as? MuseGlimmerTiledQ8Linear).map { (path, $0) }
            }
            XCTAssertEqual(Set(target.parameters().flattened().map(\.0)), parameterKeys)
            XCTAssertEqual(packed.contains { $0.0 == "lm_head" }, mode != "tiled-ffn")
            XCTAssertTrue(packed.allSatisfy { $0.0 == "lm_head" || $0.0.hasPrefix("model.language_model.layers.") })
            if mode == "tiled-head" { XCTAssertEqual(packed.count, 1) } else { XCTAssertGreaterThan(packed.count, 100) }
            let fused = target.namedModules().compactMap { $0.1 as? MuseGlimmerTextMLP }.filter(\.useTiledQ8).count
            XCTAssertEqual(fused, ["tiled-all", "tiled-ffn"].contains(mode) ? config.textConfig.numHiddenLayers : 0)
            if mode == "tiled-ffn" {
                XCTAssertEqual(packed.count, config.textConfig.numHiddenLayers * 3)
                XCTAssertTrue(packed.allSatisfy { $0.0.contains(".mlp.") })
            }
            let bytes = packed.reduce(0) { $0 + $1.1.tiled!.codes.size * 4 + $1.1.tiled!.scales.size * 4 }
            FileHandle.standardError.write(Data("[glimmer-tiled-q8] projections=\(packed.count) fused=\(fused) packed_bytes=\(bytes)\n".utf8))
        }
        #endif
        if let mode = ProcessInfo.processInfo.environment["MERERUN_MUSE_GLIMMER_Q8_VERIFY"], mode.hasPrefix("nax-head") {
            let selected = target.namedModules().compactMap { path, module -> String? in
                (module as? MuseGlimmerPrefillLinear)?.naxHeadSplits == nil ? nil : path
            }
            XCTAssertEqual(selected, ["lm_head"])
        }
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

    func record(_ row: Record, env: [String: String]) throws {
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
