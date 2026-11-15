import AudioCodecs
import Foundation
import MLX
import MLXNN
import MereRunTensor
import Tokenizers
import XCTest
@testable import AudioSTT
@testable import AudioSOVTranscribeModel
@testable import MereRunGemmaModel

/// Checkpoint-only diagnostic: synchronized stages are not end-to-end release timings.
final class SOVDecoderKernelProfileTests: MereRunCoreTestCase {
    private struct Stage: Encodable {
        let name: String
        let seconds: Double
    }
    private struct Row: Encodable {
        let kind: String
        let tokens: Int
        let uniqueExperts: Int
        let expertCounts: [Int: Int]
        let stages: [Stage]
    }

    func testProfileRealPrefillAndFirstVerificationRoutes() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let targetPath = env["MERERUN_TEST_SOV_PATH"],
              let draftPath = env["MERERUN_TEST_SOV_DFLASH_PATH"], let prefix = env["MERERUN_PROFILE_SOV_DECODER_OUTPUT"],
              let projected = env["MERERUN_PROFILE_SOV_EMBEDDINGS"] else {
            throw XCTSkip("Set SOV checkpoints, embeddings, GPU device, and decoder profile output for diagnostic profiling.")
        }
        try await MLXRequestStreams.withStream {
            await Task.yield()
            let root = URL(fileURLWithPath: targetPath)
            let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
            let target = Gemma4TextCausalLM(config: config.textConfig)
            target.prepareForFullPrecisionExpertWeights()
            try SOVTranscribeWeights.load(root: root, into: target, target: true)
            let policy = env["MERERUN_PROFILE_SOV_EXPERTS"] ?? "bf16"
            if policy != "bf16" {
                try target.prepareExperimentalExperts(groupSize: 64, metalVerification: policy == "q8-metal")
            }
            target.setExperimentalAttention(env["MERERUN_PROFILE_SOV_OPTIMIZED_ATTENTION"] == "1")
            try target.setExperimentalHeadNorm(env["MERERUN_PROFILE_SOV_HEAD_NORM"] == "1")
            let draftRoot = URL(fileURLWithPath: draftPath)
            let draftConfig = try JSONDecoder().decode(SOVDFlashConfig.self, from: Data(contentsOf: draftRoot.appendingPathComponent("config.json")))
            let draft = SOVDFlashModel(config: draftConfig)
            try SOVTranscribeWeights.load(root: draftRoot, into: draft)
            let audio = try MLX.loadArrays(url: URL(fileURLWithPath: projected))["audio_embeddings"]!.asType(.bfloat16)
            let tokenizer = try await AutoTokenizer.from(modelFolder: root)
            let prompt = SOVTranscribeGenerator.prompt(instruction: nil, language: nil, audioTokens: audio.dim(1))
            let ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
            let first = try XCTUnwrap(ids.firstIndex(of: config.audioTokenID))
            let inputIDs = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            let text = target.inputEmbeddings(for: inputIDs)
            let embeddings = concatenated([text[0..., ..<first, 0...], audio,
                text[0..., (first + audio.dim(1))..., 0...]], axis: 1)
            let cache = target.makeAttentionCache()
            let captured = Set(draftConfig.dflash.targetLayerIDs + [0])
            let prefill = target.forwardEmbeddings(embeddings, inputIDs: inputIDs, cache: cache,
                captureLayerIndices: captured, lastLogitsOnly: true)
            eval([prefill.logits] + Array(prefill.states.values))
            let dc = draft.makeCache()
            draft.appendContext(prefill.states, cache: dc)
            let anchor = argMax(prefill.logits[0, -1, 0...]).item(Int.self)
            let candidates = [anchor] + draft.propose(anchor: anchor, count: 15, target: target, cache: dc)
            let candidateIDs = MLXArray(candidates.map(Int32.init)).reshaped(1, candidates.count)
            let verificationInput = target.inputEmbeddings(for: candidateIDs)
            if env["MERERUN_PROFILE_SOV_COMPONENTS"] == "1" {
                try self.profileComponents(target: target, draft: draft, draftCache: dc, cache: cache,
                    input: verificationInput, ids: candidateIDs, captured: captured, anchor: anchor, prefix: prefix)
            }
            let layer = target.languageModel.layers[0]
            var arrays: [String: MLXArray] = [:]
            var rows: [Row] = []
            for (kind, input, layerCache) in [
                ("prefill", embeddings, target.makeAttentionCache()[0]),
                ("verification", verificationInput, cache[0].fork())
            ] {
                var stages: [Stage] = []
                func timed(_ name: String, _ operation: () -> MLXArray) -> MLXArray {
                    let start = ProcessInfo.processInfo.systemUptime
                    let result = operation(); eval(result)
                    stages.append(Stage(name: name, seconds: ProcessInfo.processInfo.systemUptime - start))
                    return result
                }
                eval(input)
                let normalized = timed("inputNorm") { layer.inputLayerNorm(input) }
                let attention = timed("attention") { layer.selfAttention(normalized, cache: Gemma4ForwardAttentionCache(layerCache)) }
                let hidden = timed("attentionResidual") { input + layer.postAttentionLayerNorm(attention) }
                _ = timed("denseMLP") { layer.postFeedforwardLayerNorm1!(layer.mlp(layer.preFeedforwardLayerNorm(hidden))) }
                let started = ProcessInfo.processInfo.systemUptime
                let route = layer.router!(hidden); eval(route.indices, route.weights)
                stages.append(Stage(name: "router", seconds: ProcessInfo.processInfo.systemUptime - started))
                let sparseInput = timed("expertInputNorm") { layer.preFeedforwardLayerNorm2!(hidden) }
                arrays[kind + "_input"] = sparseInput
                arrays[kind + "_indices"] = route.indices
                arrays[kind + "_weights"] = route.weights
                let count = route.indices.size, topK = route.indices.dim(-1)
                let order = argSort(route.indices.reshaped(count))
                let sorted = take(route.indices.reshaped(count), order)
                let flat = timed("routeGather") {
                    sparseInput.reshaped(-1, sparseInput.dim(-1)).take(order.floorDivide(topK), axis: 0).reshaped(count, 1, sparseInput.dim(-1))
                }
                let glu = layer.experts!.switchGLU
                let up: MLXArray, gate: MLXArray
                if let pair = glu.gateUpProj {
                    let output = timed("expertGateUp") { pair.applyFlat(flat, indices: sorted, sortedIndices: true) }
                    let half = output.dim(-1) / 2
                    gate = output[.ellipsis, ..<half]; up = output[.ellipsis, half...]
                } else {
                    up = timed("expertUp") { glu.upProj!.applyFlat(flat, indices: sorted, sortedIndices: true) }
                    gate = timed("expertGate") { glu.gateProj!.applyFlat(flat, indices: sorted, sortedIndices: true) }
                }
                let activation = timed("expertActivation") { geluApproximate(gate) * up }
                _ = timed("expertDown") { glu.downProj.applyFlat(activation, indices: sorted, sortedIndices: true) }
                let routed = route.indices.asArray(Int.self)
                let histogram = Dictionary(grouping: routed, by: { $0 }).mapValues(\.count)
                rows.append(Row(kind: kind, tokens: input.dim(1), uniqueExperts: histogram.count, expertCounts: histogram, stages: stages))
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(rows).write(to: URL(fileURLWithPath: prefix + ".json"))
            try MLX.save(arrays: arrays, url: URL(fileURLWithPath: prefix + ".safetensors"))
        }
    }

    private func profileComponents(
        target: Gemma4TextCausalLM, draft: SOVDFlashModel, draftCache: [Gemma4AttentionCache],
        cache: [Gemma4AttentionCache], input: MLXArray, ids: MLXArray, captured: Set<Int>, anchor: Int, prefix: String
    ) throws {
        var runs: [[String: Double]] = []
        let draftIDs = MLXArray(([anchor] + Array(repeating: draft.config.dflash.maskTokenID, count: 15)).map(Int32.init)).reshaped(1, 16)
        for iteration in 0..<7 {
            var times: [String: Double] = [:]
            var start = ProcessInfo.processInfo.systemUptime
            let denoised = draft.denoise(target.rawInputEmbeddings(draftIDs), cache: draftCache)
            eval(denoised)
            times["draftBody"] = ProcessInfo.processInfo.systemUptime - start
            start = ProcessInfo.processInfo.systemUptime
            _ = argMax(target.outputLogits(denoised[0..., 1..., 0...]), axis: -1).asArray(Int.self)
            times["draftHeadAndArgmax"] = ProcessInfo.processInfo.systemUptime - start
            start = ProcessInfo.processInfo.systemUptime
            let body = target.languageModel.forwardDetailed(embeddings: input, inputIds: ids,
                cache: cache.map { $0.fork() }, captureSharedKV: false, captureLayerIndices: captured)
            eval([body.hidden] + Array(body.capturedLayerStates.values))
            times["targetBody"] = ProcessInfo.processInfo.systemUptime - start
            start = ProcessInfo.processInfo.systemUptime
            _ = argMax(target.outputLogits(body.hidden), axis: -1).asArray(Int.self)
            times["targetHeadAndArgmax"] = ProcessInfo.processInfo.systemUptime - start
            start = ProcessInfo.processInfo.systemUptime
            draft.appendContext(body.capturedLayerStates, cache: draftCache.map { $0.fork() })
            times["draftContextAppendFullBlock"] = ProcessInfo.processInfo.systemUptime - start
            if iteration >= 2 { runs.append(times) }
        }
        // Stage barriers intentionally disable overlap. These diagnostics are
        // separate from the release end-to-end benchmark.
        var stages: [[String: Double]] = []
        let trial = cache.map { $0.fork() }
        let forward = trial.map(Gemma4ForwardAttentionCache.init)
        var hidden = input
        XCTAssertEqual(target.config.hiddenSizePerLayerInput, 0)
        for (index, layer) in target.languageModel.layers.enumerated() {
            if ProcessInfo.processInfo.environment["MERERUN_PROFILE_SOV_ATTENTION"] == "1", index == 0 || index == 5 {
                try profileAttention(layer: layer, input: hidden, cache: trial[target.languageModel.layerIndexToCacheIndex[index]].fork(),
                    config: target.config, layerIndex: index, prefix: prefix)
            }
            var times: [String: Double] = [:]
            func timed(_ name: String, _ operation: () -> MLXArray) -> MLXArray {
                let start = ProcessInfo.processInfo.systemUptime
                let value = operation(); eval(value)
                times[name] = ProcessInfo.processInfo.systemUptime - start
                return value
            }
            let context = forward[target.languageModel.layerIndexToCacheIndex[index]]
            let attention = timed("attentionAndInputNorm") { layer.selfAttention(layer.inputLayerNorm(hidden), cache: context) }
            let residual = timed("attentionResidual") { hidden + layer.postAttentionLayerNorm(attention) }
            let dense = timed("denseMLPAndNorms") { layer.postFeedforwardLayerNorm1!(layer.mlp(layer.preFeedforwardLayerNorm(residual))) }
            let start = ProcessInfo.processInfo.systemUptime
            let route = layer.router!(residual); eval(route.indices, route.weights)
            times["router"] = ProcessInfo.processInfo.systemUptime - start
            let sparse = timed("expertsAndNorms") {
                layer.postFeedforwardLayerNorm2!(layer.experts!(layer.preFeedforwardLayerNorm2!(residual), indices: route.indices, weights: route.weights))
            }
            hidden = timed("finalNormAndResidual") { (residual + layer.postFeedforwardLayerNorm(dense + sparse)) * layer.layerScalar }
            times["layer"] = Double(index)
            stages.append(times)
        }
        let expected = target.languageModel.forwardDetailed(embeddings: input, inputIds: ids,
            cache: cache.map { $0.fork() }, captureSharedKV: false).hidden
        let error = abs(target.languageModel.norm(hidden) - expected).max().item(Float.self)
        XCTAssertEqual(error, 0, "Manual diagnostic stages must preserve the target graph's arithmetic.")
        try MLX.save(arrays: ["verified_hidden": expected, "candidate_ids": ids],
            url: URL(fileURLWithPath: prefix + "-hidden.safetensors"))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(runs).write(to: URL(fileURLWithPath: prefix + "-components.json"))
        try encoder.encode(stages).write(to: URL(fileURLWithPath: prefix + "-layers.json"))
    }
}
