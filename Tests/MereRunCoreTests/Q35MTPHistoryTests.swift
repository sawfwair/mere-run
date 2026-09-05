import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q35MTPHistoryTests: MereRunCoreTestCase {
    func testModelAdmissionPreservesPrecisionAndExplicitOverrides() {
        for (modelID, moe, expected) in [
            (Q35Resources.q38TwentySevenB4BitModelId, false, true),
            (Q35Resources.q38TwentySevenBModelId, false, false),
            (Q35Resources.ornith35BMLX4BitModelId, true, true),
            (Q35Resources.q38FlashNext3BitNativePLEModelId, true, true),
            (Q35Resources.q36NanoModelId, true, false),
        ] {
            for (environment, enabled) in [
                ([:], expected), (["MERERUN_Q35_MTP_SPECULATION": "0"], false),
                (["MERERUN_Q35_MTP_SPECULATION": "1"], true),
            ] {
                XCTAssertEqual(Q35Generator.shouldSpeculate(
                    modelId: modelID, usesMoE: moe, promptTokenCount: 32,
                    maxContextTokens: 8_192, environment: environment
                ), enabled, modelID)
            }
        }
    }

    func testLongHistoryIsStreamedOnlyForEligibleGreedyRequests() {
        for (modelID, moe) in [(Q35Resources.q38TwentySevenB4BitModelId, false),
                               (Q35Resources.ornith35BMLX4BitModelId, true)] {
            XCTAssertEqual(Q35Generator.mtpHistoryMode(
                modelId: modelID, isQwen4Exp: false, usesMoE: moe,
                speculationEligible: true, greedy: true, promptTokenCount: 128,
                environment: ["MERERUN_Q35_MTP_STREAM_HISTORY": "none"]
            ), .none, "History ablation preserves MTP admission without priming the drafter")
            for (eligible, greedy, expected) in [(true, true, Q35MTPHistoryMode.streaming),
                                               (false, true, .none), (true, false, .none)] {
                XCTAssertEqual(Q35Generator.mtpHistoryMode(
                    modelId: modelID, isQwen4Exp: false, usesMoE: moe,
                    speculationEligible: eligible, greedy: greedy, promptTokenCount: 32_768,
                    environment: [:]
                ), expected)
            }
            XCTAssertEqual(Q35Generator.mtpHistoryMode(
                modelId: modelID, isQwen4Exp: false, usesMoE: moe,
                speculationEligible: true, greedy: true, promptTokenCount: 32_768,
                environment: ["MERERUN_Q35_MTP_STREAM_HISTORY": "0"]
            ), .none)
        }
    }

    func testStreamedHistoryMatchesRetainedHistoryAndForks() async throws {
        for moe in [false, true] {
            let modelID = moe ? Q35Resources.ornith35BMLX4BitModelId : Q35Resources.q38TwentySevenB4BitModelId
            let generator = Q35Generator(modelId: modelID, prefixKVCacheEnabled: true)
            try await Stream.withNewDefaultStream {
                try await Self.qualifyHistory(generator, moe: moe)
            }
        }
    }

    private static func qualifyHistory(_ generator: isolated Q35Generator, moe: Bool) async throws {
        MLXRandom.seed(511)
        let config = try configuration(moe: moe)
        let model = Q35Model(config: config)
        let mtp = Q35MTPModel(config: config)
        let tokens = (0..<601).map { $0 % 30 + 1 }
        let checkpoints: Set<Int> = [257, 512]
        let retained = try await generator.chunkedPrefill(
            model: model, promptTokens: tokens, cache: makeLayerCaches(config: config),
            checkpointTokenCounts: checkpoints, retainMTPHistory: true, progressHandler: nil
        )
        let session = Q35MTPDraftSession()
        let streamed = try await generator.chunkedPrefill(
            model: model, promptTokens: tokens, cache: makeLayerCaches(config: config),
            modelPath: "fixture", checkpointTokenCounts: checkpoints, retainMTPHistory: true,
            mtpSession: session, prefillMTPModel: mtp, progressHandler: nil
        )
        XCTAssertNil(streamed.mtpHistoryHidden)
        XCTAssertEqual(session.committedHistoryCount, tokens.count - 1)
        XCTAssertEqual(session.pendingHistoryCount, (tokens.count - 1) % 256)
        assertExact(streamed.logits, retained.logits)
        let seedValue = generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: tokens, cacheMode: .default, requiresMTPSession: true
        )
        let seed = try XCTUnwrap(seedValue)
        XCTAssertEqual(seed.tokenCount, tokens.count)
        let seededSession = try XCTUnwrap(seed.mtpSession)
        let retainedSession = Q35MTPDraftSession(promptTokens: tokens, promptHidden: retained.mtpHistoryHidden)
        let hidden = try XCTUnwrap(streamed.hidden)
        let expected = mtp.draftBlock(lastToken: 7, hidden: hidden, blockSize: 8,
                                     session: retainedSession, baseModel: model)
        let actual = mtp.draftBlock(lastToken: 7, hidden: hidden, blockSize: 8,
                                   session: seededSession, baseModel: model)
        XCTAssertEqual(actual.tokens, expected.tokens)
        XCTAssertEqual(session.committedHistoryCount, tokens.count - 1, "Snapshot must fork mutable history")

        let followup = tokens + [3, 4, 5]
        let resumed = try await generator.chunkedPrefill(
            model: model, promptTokens: followup, cache: seed.caches, startIndex: tokens.count,
            existingLogits: seed.logits, existingHidden: seed.hidden, retainMTPHistory: true,
            mtpSession: session.fork(), prefillMTPModel: mtp, progressHandler: nil
        )
        XCTAssertEqual(resumed.mtpSession?.committedHistoryCount, followup.count - 1)
        XCTAssertEqual(session.committedHistoryCount, tokens.count - 1)
        let freshSeedValue = generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: tokens, cacheMode: .default, requiresMTPSession: true
        )
        let freshSeed = try XCTUnwrap(freshSeedValue)
        XCTAssertEqual(freshSeed.mtpSession?.committedHistoryCount, tokens.count - 1)
        assertExact(freshSeed.logits, streamed.logits)
        let stats = generator.prefixKVCacheStats()
        XCTAssertEqual(stats.entries, 3, "Only semantic boundaries and the final prompt are retained")
    }

    func testTargetOnlyPrefixCannotSeedMTP() async throws {
        try await Stream.withNewDefaultStream {
            try await Self.qualifyTargetOnlyPrefix(Q35Generator(prefixKVCacheEnabled: true))
        }
    }

    private static func qualifyTargetOnlyPrefix(_ generator: isolated Q35Generator) async throws {
        let config = try configuration(moe: false)
        let model = Q35Model(config: config)
        let tokens = [1, 2, 3]
        _ = try await generator.chunkedPrefill(
            model: model, promptTokens: tokens, cache: makeLayerCaches(config: config),
            modelPath: "fixture", progressHandler: nil
        )
        let seed = generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: tokens, cacheMode: .default, requiresMTPSession: true
        )
        XCTAssertNil(seed)
    }

    private static func assertExact(_ actual: MLXArray, _ expected: MLXArray) {
        XCTAssertEqual((actual - expected).abs().max().item(Float.self), 0)
    }

    private static func makeLayerCaches(config: Q35Config) -> [Q35LayerCache?] {
        config.textConfig.layerTypes.map { _ in .full(KVCacheSimple()) }
    }

    private static func configuration(moe: Bool) throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data("""
        {"model_type":"qwen3_5","text_config":{
          "model_type":"qwen3_5_text","hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
          "num_experts":\(moe ? 8 : 0),"num_experts_per_tok":\(moe ? 2 : 0),
          "moe_intermediate_size":16,"shared_expert_intermediate_size":16,
          "layer_types":["full_attention"],"mtp_num_hidden_layers":1,
          "linear_num_key_heads":1,"linear_num_value_heads":2,
          "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
          "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,"eos_token_id":63,"vocab_size":64,"max_position_embeddings":4096,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1}}}
        """.utf8))
    }
}
