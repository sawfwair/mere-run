@testable import AudioSTT
import Foundation
import MLX
import MereRunTensor
import Tokenizers
import XCTest
@testable import AudioSOVTranscribeModel
@testable import MereRunGemmaModel

/// Optional checkpoint diagnostic. Stage timing includes synchronization and is separate from CLI release samples.
final class SOVGenerationProfileTests: MereRunCoreTestCase {
    private struct Sample: Encodable {
        let pipelined: Bool
        let tokens: [Int]
        let acceptedLengths: [Int]
        let emittedDraftTokens: Int
        let rounds: [SOVDecodeRoundTiming]
    }

    func testCheckpointPipelineParityAndMaterializationProfile() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let modelPath = env["MERERUN_TEST_SOV_PATH"],
              let draftPath = env["MERERUN_TEST_SOV_DFLASH_PATH"], let projected = env["MERERUN_PROFILE_SOV_EMBEDDINGS"],
              let output = env["MERERUN_PROFILE_SOV_GENERATION"] else {
            throw XCTSkip("Set GPU device, SOV checkpoints, projected embeddings, and generation profile output.")
        }
        try await MLXRequestStreams.withStream {
            let root = URL(fileURLWithPath: modelPath)
            let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
            let target = Gemma4TextCausalLM(config: config.textConfig)
            target.prepareForFullPrecisionExpertWeights()
            try SOVTranscribeWeights.load(root: root, into: target, target: true)
            try target.prepareExperimentalExperts(groupSize: 64, metalVerification: true)
            target.setExperimentalAttention(true)
            let draftRoot = URL(fileURLWithPath: draftPath)
            let dc = try JSONDecoder().decode(SOVDFlashConfig.self, from: Data(contentsOf: draftRoot.appendingPathComponent("config.json")))
            let draft = SOVDFlashModel(config: dc)
            try SOVTranscribeWeights.load(root: draftRoot, into: draft)
            let tokenizer = try await AutoTokenizer.from(modelFolder: root)
            let audio = try XCTUnwrap(MLX.loadArrays(url: URL(fileURLWithPath: projected))["audio_embeddings"]).asType(.bfloat16)
            let prompt = SOVTranscribeGenerator.prompt(instruction: nil, language: nil, audioTokens: audio.dim(1))
            let ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
            let first = try XCTUnwrap(ids.firstIndex(of: config.audioTokenID))
            let inputIDs = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            let text = target.inputEmbeddings(for: inputIDs)
            let embeddings = concatenated([text[0..., ..<first, 0...], audio, text[0..., (first + audio.dim(1))..., 0...]], axis: 1)
            eval(embeddings)

            // Compare the complete first verification body with identical GPU-resident candidate IDs.
            let cache = target.makeAttentionCache()
            let captured = Set(dc.dflash.targetLayerIDs)
            let prefill = target.forwardEmbeddings(embeddings, inputIDs: inputIDs, cache: cache,
                captureLayerIndices: captured, lastLogitsOnly: true)
            eval([prefill.logits] + Array(prefill.states.values))
            let draftCache = draft.makeCache()
            draft.appendContext(prefill.states, cache: draftCache)
            let anchor = argMax(prefill.logits[0, -1, 0...]).item(Int.self)
            let deviceIDs = draft.proposalIDs(anchor: anchor, count: 15, target: target, cache: draftCache)
            asyncEval(deviceIDs)
            let candidates = concatenated([MLXArray([Int32(anchor)]), deviceIDs]).reshaped(1, 16)
            let actual = target.languageModel.forwardDetailed(embeddings: target.inputEmbeddings(for: candidates), inputIds: candidates,
                cache: cache.map { $0.fork() }, captureSharedKV: false).hidden
            let hostIDs = MLXArray(([anchor] + deviceIDs.asArray(Int.self)).map(Int32.init)).reshaped(1, 16)
            if let research = env["MERERUN_PROFILE_SOV_REMAINING"] {
                try profileRemaining(target: target, cache: cache, candidates: hostIDs, embeddings: embeddings, output: research)
            }
            let reference = target.languageModel.forwardDetailed(embeddings: target.inputEmbeddings(for: hostIDs), inputIds: hostIDs,
                cache: cache.map { $0.fork() }, captureSharedKV: false).hidden
            XCTAssertEqual(abs(actual - reference).max().item(Float.self), 0)
            let sharedCache = cache.map { $0.fork() }
            for case let entry as Gemma4SlidingKVCache in sharedCache { entry.reuseAttentionStorage = true }
            let sharedHidden = target.languageModel.forwardDetailed(embeddings: target.inputEmbeddings(for: hostIDs), inputIds: hostIDs,
                cache: sharedCache, captureSharedKV: false).hidden
            XCTAssertEqual(abs(sharedHidden - reference).max().item(Float.self), 0)
            try MLX.save(arrays: ["candidate_ids": candidates, "verified_hidden": actual],
                url: URL(fileURLWithPath: output + ".safetensors"))

            let draftIDs = MLXArray(([anchor] + Array(repeating: dc.dflash.maskTokenID, count: 15)).map(Int32.init)).reshaped(1, 16)
            let denoised = draft.denoise(target.rawInputEmbeddings(draftIDs), cache: draftCache)[0..., 1..., 0...]
            eval(denoised)
            let head = target.makeQuantizedDraftHead()
            eval(head.parameters())
            var headTimes: [String: [Double]] = [:]
            for iteration in 0..<14 {
                for quantized in iteration.isMultiple(of: 2) ? [false, true] : [true, false] {
                    let started = ProcessInfo.processInfo.systemUptime
                    let logits = quantized ? target.softcapOutputLogits(head(denoised)) : target.outputLogits(denoised)
                    _ = argMax(logits, axis: -1).asArray(Int.self)
                    if iteration >= 4 { headTimes[quantized ? "q8" : "bf16", default: []].append(ProcessInfo.processInfo.systemUptime - started) }
                }
            }
            let headEncoder = JSONEncoder(); headEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try headEncoder.encode(headTimes).write(to: URL(fileURLWithPath: output + "-heads.json"))

            var samples: [Sample] = []
            for pipelined in [false, true, true, false] {
                var timings: [SOVDecodeRoundTiming] = []
                let result = try SOVTranscribeDecoder.decode(target: target, embeddings: embeddings, inputIDs: inputIDs, draft: draft,
                    maxTokens: 448, eosTokens: [1, 106, 50], pipelined: pipelined, roundTiming: { timings.append($0) })
                if let reference = samples.first {
                    XCTAssertEqual(result.tokens, reference.tokens)
                    XCTAssertEqual(result.acceptedLengths, reference.acceptedLengths)
                }
                samples.append(Sample(pipelined: pipelined, tokens: result.tokens, acceptedLengths: result.acceptedLengths,
                    emittedDraftTokens: result.emittedDraftTokens, rounds: timings))
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(samples).write(to: URL(fileURLWithPath: output + ".json"))
        }
    }
}
