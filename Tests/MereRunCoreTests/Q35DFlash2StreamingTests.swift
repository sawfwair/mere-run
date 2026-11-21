import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Q35DFlash2StreamingTests: MereRunCoreTestCase {
    func testDenseSubmissionBoundariesPreservePrefillVerificationAndRewind() throws {
        MLXRandom.seed(766)
        let json = Self.target.replacingOccurrences(of: "\"num_hidden_layers\":2", with: "\"num_hidden_layers\":8")
            .replacingOccurrences(of: "[\"linear_attention\",\"full_attention\"]",
                with: "[\"linear_attention\",\"full_attention\",\"linear_attention\",\"full_attention\",\"linear_attention\",\"full_attention\",\"linear_attention\",\"full_attention\"]")
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(json.utf8))
        let model = Q35Model(config: config, asynchronousDecodeBlocks: true)
        model.update(parameters: model.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
        var reference: [MLXArray]?
        for stride in [0, 1, 4, 8] {
            let values = Q35DenseSubmissionPolicy.$decodeStride.withValue(stride) {
                Q35DenseSubmissionPolicy.$prefillStride.withValue(stride) {
                    let caches: [Q35LayerCache?] = (0..<8).map {
                        $0.isMultiple(of: 2) ? .linear(Q35LinearCache()) : .full(KVCacheSimple())
                    }
                    let prompt = model.forwardPrefill(MLXArray(Array(1...21).map(Int32.init)).reshaped(1, 21), cache: caches)
                    let verified = model.forward(MLXArray(Array(22...29).map(Int32.init)).reshaped(1, 8), cache: caches, targetVerify: true)
                    MLX.eval(prompt.logits, verified.logits)
                    for cache in caches.compactMap({ $0 }) {
                        XCTAssertTrue(cache.restoreVerificationPrefix(totalTokens: 8, tokenCount: 3))
                        cache.commitVerification()
                    }
                    let resumed = model.forward(MLXArray([Int32(30)]).reshaped(1, 1), cache: caches)
                    MLX.eval(resumed.logits)
                    return [prompt.logits, verified.logits, resumed.logits]
                }
            }
            if let reference {
                for (actual, expected) in zip(values, reference) {
                    XCTAssertEqual((actual - expected).abs().max().item(Float.self), 0)
                }
            } else { reference = values }
        }
        XCTAssertEqual(Q35DenseSubmissionPolicy.stride(nil), 0)
        XCTAssertEqual(Q35DenseSubmissionPolicy.stride("invalid"), 0)
        XCTAssertEqual(Q35DenseSubmissionPolicy.stride("4"), 4)
    }

    func testUncachedPrefillShrinksWhenAnotherRequestIsAdmitted() async throws {
        let generator = Q35Generator(modelId: Q35Resources.q36NanoModelId, prefixKVCacheEnabled: true)
        try await Stream.withNewDefaultStream {
            try await Self.qualifyUncachedChunkSizing(generator)
        }
    }

    private static func qualifyUncachedChunkSizing(_ generator: isolated Q35Generator) async throws {
        let base = Q35Generator.prefillChunkSize(modelId: Q35Resources.q36NanoModelId, activeRequestCount: 1)
        try XCTSkipUnless(base > Q35Generator.contendedPrefillChunkSize, "Configured chunks already fit contention.")
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(target.utf8))
        let model = Q35Model(config: config)
        let draftModel = Q35DFlash2Model(config: try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(draft.utf8)))
        let total = base + Q35Generator.contendedPrefillChunkSize + 7
        let trace = PrefillTrace()
        _ = try await generator.prefillDFlash2(
            model: model, draft: draftModel,
            draftCache: draftModel.makeCache(offset: total - draftModel.config.slidingWindow),
            promptTokens: Array(repeating: 3, count: total), cache: generator.makeLayerCaches(config: config),
            progressHandler: { progress in
                if let message = progress.message { trace.append(message) }
                generator.assumeIsolated { $0.activeChatRequestCount = 2 }
            }
        )
        XCTAssertEqual(trace.messages, [
            "Prefilling \(base)/\(total) tokens",
            "Prefilling \(base + Q35Generator.contendedPrefillChunkSize)/\(total) tokens",
            "Prefilling \(total)/\(total) tokens"
        ])
        XCTAssertEqual(generator.prefixKVCacheStats().entries, 0)
    }

    private final class PrefillTrace: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String] = []
        var messages: [String] { lock.withLock { values } }
        func append(_ message: String) { lock.withLock { values.append(message) } }
    }

    func testPrefixSnapshotsResumeTheSameGridAndForkBothHistories() async throws {
        let generator = Q35Generator(modelId: Q35Resources.q38TwentySevenB4BitModelId, prefixKVCacheEnabled: true)
        try await Stream.withNewDefaultStream {
            try await Self.qualifyPrefixSnapshots(generator)
        }
    }

    private static func qualifyPrefixSnapshots(_ generator: isolated Q35Generator) async throws {
        MLXRandom.seed(615)
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(target.utf8))
        let model = Q35Model(config: config)
        model.update(parameters: model.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
        let draftModel = Q35DFlash2Model(config: try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(draft.utf8)))
        let tokens = (0..<173).map { $0 % 31 + 1 }
        let context = draftModel.makeCache()
        let original = try await generator.prefillDFlash2(
            model: model, draft: draftModel, draftCache: context, promptTokens: tokens,
            cache: generator.makeLayerCaches(config: config), chunkSize: 64,
            modelPath: "fixture", progressHandler: nil
        )
        let exactValue = generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: tokens, cacheMode: .default, dflash2ChunkSize: 64
        )
        let exact = try XCTUnwrap(exactValue)
        XCTAssertEqual(exact.tokenCount, 173)
        let exactHistory = try XCTUnwrap(exact.dflash2Cache)
        XCTAssertEqual(exactHistory.map(\.offset), [173])
        XCTAssertEqual(exactHistory[0].keys?.dim(2), 4)
        let followup = tokens + Array(2...11)
        let seedValue = generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: followup, cacheMode: .default, dflash2ChunkSize: 64
        )
        let seed = try XCTUnwrap(seedValue)
        XCTAssertEqual(seed.tokenCount, 128, "An off-grid final prompt cannot seed a longer cold prompt")
        let resumedHistory = try XCTUnwrap(seed.dflash2Cache)
        let resumed = try await generator.prefillDFlash2(
            model: model, draft: draftModel, draftCache: resumedHistory, promptTokens: followup,
            cache: seed.caches, chunkSize: 64, startIndex: seed.tokenCount,
            existingLogits: seed.logits, existingHidden: seed.hidden, modelPath: "fixture", progressHandler: nil
        )
        let savedValue = generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: tokens, cacheMode: .default, dflash2ChunkSize: 64
        )
        let saved = try XCTUnwrap(savedValue)
        XCTAssertEqual((saved.logits - original.logits).abs().max().item(Float.self), 0)
        let savedHistory = try XCTUnwrap(saved.dflash2Cache)
        XCTAssertEqual(savedHistory.map(\.offset), [173])
        for (actual, expected) in zip(savedHistory, exactHistory) {
            XCTAssertEqual((try XCTUnwrap(actual.keys) - XCTUnwrap(expected.keys)).abs().max().item(Float.self), 0)
            XCTAssertEqual((try XCTUnwrap(actual.values) - XCTUnwrap(expected.values)).abs().max().item(Float.self), 0)
        }
        XCTAssertNil(generator.prefixKVCacheSeed(modelPath: "fixture", promptTokens: tokens, cacheMode: .default))
        for mode in [Q35PrefillProjectionPolicy.Mode.reference, .all, .gateUp] where mode != Q35PrefillProjectionPolicy.mode {
            let incompatible = Q35PrefillProjectionPolicy.$modeOverride.withValue(mode) {
                generator.prefixKVCacheSeed(modelPath: "fixture", promptTokens: tokens, cacheMode: .default, dflash2ChunkSize: 64)
            }
            XCTAssertNil(incompatible, "Different prefill arithmetic cannot share a snapshot")
        }
        XCTAssertNil(generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: tokens, cacheMode: .default, dflash2ChunkSize: 128
        ))
        XCTAssertNil(generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: tokens, cacheMode: .affine4, dflash2ChunkSize: 64
        ))
        var changed = tokens
        changed[0] = 63
        XCTAssertNil(generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: changed, cacheMode: .default, dflash2ChunkSize: 64
        ))
        generator.resetPrefixKVCache()
        let freshHistory = draftModel.makeCache()
        let fresh = try await generator.prefillDFlash2(
            model: model, draft: draftModel, draftCache: freshHistory, promptTokens: followup,
            cache: generator.makeLayerCaches(config: config), chunkSize: 64,
            modelPath: "fresh", progressHandler: nil
        )
        XCTAssertEqual((resumed.logits - fresh.logits).abs().max().item(Float.self), 0)
        for (actual, expected) in zip(resumedHistory, freshHistory) {
            XCTAssertEqual((try XCTUnwrap(actual.keys) - XCTUnwrap(expected.keys)).abs().max().item(Float.self), 0)
            XCTAssertEqual((try XCTUnwrap(actual.values) - XCTUnwrap(expected.values)).abs().max().item(Float.self), 0)
        }
        XCTAssertLessThanOrEqual(generator.prefixKVCacheStats().entries, Q35Generator.prefixKVCacheMaxEntries)
    }

    func testAdaptiveDiagnosticsRetainCostsAndDecodeOlderPayloads() throws {
        let old = Data(#"{"route":"dflash2","rounds":4}"#.utf8)
        let decoded = try JSONDecoder().decode(ChatAccelerationDiagnostics.self, from: old)
        XCTAssertNil(decoded.targetOnlyRounds)
        XCTAssertNil(decoded.roundSecondsByDraftDepth)
        XCTAssertNil(decoded.rejectedRoundSecondsByDraftDepth)
        XCTAssertNil(decoded.draftRoundsByDepth)
        XCTAssertNil(decoded.treeRoundSecondsByNodeCount)
        let diagnostics = ChatAccelerationDiagnostics(
            route: "dflash2", targetOnlyRounds: 8, roundSecondsByDraftDepth: [0.01, 0.02],
            rejectedRoundSecondsByDraftDepth: [0.01, 0.03], draftRoundsByDepth: [8, 4],
            treeRoundSecondsByNodeCount: [nil, nil, 0.04]
        )
        XCTAssertEqual(try JSONDecoder().decode(
            ChatAccelerationDiagnostics.self, from: JSONEncoder().encode(diagnostics)
        ), diagnostics)
    }

    func testConfirmedDFlashTokensStreamReasoningAndSplitUnicodeBeforeCompletion() throws {
        let target = Q35Model(config: try JSONDecoder().decode(Q35Config.self, from: Data(Self.target.utf8)))
        target.update(parameters: target.parameters().mapValues { MLXArray.zeros(like: $0) })
        let draft = Q35DFlash2Model(config: try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(Self.draft.utf8)))
        draft.update(parameters: draft.parameters().mapValues { MLXArray.zeros(like: $0) })
        let caches: [Q35LayerCache?] = [.linear(Q35LinearCache()), .full(KVCacheSimple())]
        let prompt = [1, 2, 3]
        let output = target.forwardPrefill(
            MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: caches, captureLayerIndices: [0, 1]
        )
        let history = draft.makeCache()
        draft.appendTargetContext(output.capturedHiddenStates, cache: history)
        let observer = NativeChatOutputObserver(initialReasoning: true, parseTools: false)
        let recorded = RecordedEvents()
        let bytes = Array("work 🧮</think>42".utf8)
        let result = try Q35Generator.decodeDFlash2(
            model: target, draft: draft, draftCache: history, initialLogits: output.logits,
            layerCaches: caches, eosSet: [], tokenBudget: bytes.count,
            generationConfig: GenerationConfig(temperature: 0, repetitionPenalty: 1), promptTokens: prompt,
            decodeToken: { _ in "\u{FFFD}" },
            decodeTokens: { String(decoding: bytes.prefix($0.count), as: UTF8.self) },
            progressHandler: { observer.observe($0, handler: { recorded.append($0) }) }
        )
        XCTAssertEqual(result.generatedTokens.count, bytes.count)
        XCTAssertEqual(result.acceleration?.route, "dflash2")
        // Check callbacks before finish: final reconciliation must not hide a
        // decoder that buffers everything until the response is assembled.
        XCTAssertEqual(recorded.events.compactMap { if case .reasoning(let value) = $0 { return value }; return nil }.joined(), "work 🧮")
        XCTAssertEqual(recorded.events.compactMap { if case .text(let value) = $0 { return value }; return nil }.joined(), "42")
        XCTAssertEqual(observer.finish(String(decoding: bytes, as: UTF8.self), handler: nil), [.reasoning("work 🧮"), .text("42")])
    }

    func testDFlashStopsInsideAVerifiedBlockAtTheCompletedToolCall() throws {
        let target = Q35Model(config: try JSONDecoder().decode(Q35Config.self, from: Data(Self.target.utf8)))
        target.update(parameters: target.parameters().mapValues { MLXArray.zeros(like: $0) })
        let draft = Q35DFlash2Model(config: try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(Self.draft.utf8)))
        draft.update(parameters: draft.parameters().mapValues { MLXArray.zeros(like: $0) })
        let caches: [Q35LayerCache?] = [.linear(Q35LinearCache()), .full(KVCacheSimple())]
        let prompt = [1, 2, 3]
        let output = target.forwardPrefill(
            MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: caches, captureLayerIndices: [0, 1]
        )
        let history = draft.makeCache()
        draft.appendTargetContext(output.capturedHiddenStates, cache: history)
        let pieces = ["<tool_call>\n<function=read>\n</function>\n", "</tool_", "call>"]
        var emitted = 0
        let result = try Q35Generator.decodeDFlash2(
            model: target, draft: draft, draftCache: history, initialLogits: output.logits,
            layerCaches: caches, eosSet: [], tokenBudget: 8,
            generationConfig: GenerationConfig(temperature: 0, repetitionPenalty: 1), promptTokens: prompt,
            stopAtCompletedToolCall: true,
            decodeToken: { _ in
                defer { emitted += 1 }
                return emitted < pieces.count ? pieces[emitted] : "unexpected suffix"
            },
            decodeTokens: { pieces.prefix($0.count).joined() },
            progressHandler: nil
        )
        XCTAssertEqual(result.generatedTokens.count, pieces.count)
        XCTAssertEqual(result.acceleration?.acceptedDraftTokens, 2)
    }

    private final class RecordedEvents: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [ChatGenerationEvent] = []
        var events: [ChatGenerationEvent] { lock.withLock { values } }
        func append(_ progress: ChatProgress) {
            if let event = progress.event { lock.withLock { values.append(event) } }
        }
    }

    private static let target = #"""
    {"model_type":"qwen3_5","text_config":{
      "model_type":"qwen3_5_text","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
      "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
      "layer_types":["linear_attention","full_attention"],
      "linear_num_key_heads":1,"linear_num_value_heads":2,
      "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
      "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,"eos_token_id":63,"vocab_size":64,
      "max_position_embeddings":4096,"rms_norm_eps":0.000001,
      "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1}}}
    """#

    private static let draft = #"""
    {"architectures":["DFlash2DraftModel"],"model_type":"qwen3","is_causal":false,
     "attention_bias":false,"hidden_act":"silu","tie_word_embeddings":false,
     "hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,"num_target_layers":2,
     "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":64,
     "rms_norm_eps":0.000001,"sliding_window":4,"layer_types":["sliding_attention"],
     "rope_parameters":{"rope_theta":10000,"rope_type":"default"},
     "dflash_config":{"block_size":4,"conv_group_size":16,"conv_kernel_size":2,
       "mask_token_id":63,"selector_rank":1,"selector_top_k":2,"target_layer_ids":[0,1]}}
    """#
}
