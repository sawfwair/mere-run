import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Q35GreedyVerificationTests: MereRunCoreTestCase {
    func testPackedMTPReadbackReusesBonusAndReplacementTokensWithoutChangingOutput() async throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/PPLXEmbedV2/tokenizer.json")
        try FileManager.default.copyItem(at: fixture, to: root.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"[PAD]","model_max_length":4096}"#.utf8)
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        let generator = Q35Generator(
            modelId: Q35Resources.q38TwentySevenB4BitModelId, prefixKVCacheEnabled: false,
            continuousBatchingEnabled: false, dflash2ModelPath: nil
        )
        try await generator.withRequestStream { @Sendable in
            try await Self.qualify(generator, root: root)
        }
    }

    private static func qualify(_ generator: isolated Q35Generator, root: URL) async throws {
        let template = try Q35TokenizerAndTemplate.load(from: root)
        MLXRandom.seed(918)
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(configuration.utf8))
        let model = Q35Model(config: config)
        model.update(parameters: model.parameters().mapValues { MLXRandom.normal($0.shape) * 0.01 })
        var embedding = [Float](repeating: 0, count: 64 * 32)
        var head = embedding
        for token in 0..<64 { embedding[token * 32 + token % 32] = 1 }
        for token in 0..<32 { head[token * 32 + (token + 31) % 32] = 10 }
        try model.update(parameters: ModuleParameters.unflattened([
            ("model.embed_tokens.weight", MLXArray(embedding, [64, 32])),
            ("lm_head.weight", MLXArray(head, [64, 32])),
        ]), verify: .noUnusedKeys)
        for (packed, reject) in [(false, false), (false, true), (true, false), (true, true)] {
            for eosSet in [Set<Int>(), [9]] {
                let caches = generator.makeLayerCaches(config: config)
                let prompt = [1, 2, 3]
                let output = model.forwardPrefill(
                    MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: caches
                )
                let result = try await generator.decodeTokensSerially(
                    model: model, tokenizerAndTemplate: template, initialLogits: output.logits,
                    initialHidden: output.hidden, prefillMTPHidden: nil, prefillMTPSession: nil,
                    mtpModel: CyclicDraft(reject: reject), layerCaches: caches, eosSet: eosSet,
                    generationConfig: GenerationConfig(temperature: 0, repetitionPenalty: 1),
                    packedGreedyVerification: packed,
                    tokenBudget: 17, prefillTokenCount: prompt.count, mropeRopeDelta: nil,
                    promptTokens: prompt, jsonConstrained: false, stopAtCompletedToolCall: false,
                    logprobCapture: .none, logprobRegion: .visible, progressHandler: nil
                )
                XCTAssertEqual(result.generatedTokens, eosSet.isEmpty ? Array(4...20) : Array(4..<9))
                XCTAssertEqual(result.acceleration?.route, "mtp-speculative")
                XCTAssertGreaterThan(result.acceleration?.acceptedDraftTokens ?? 0, 0)
                if reject {
                    XCTAssertLessThan(result.acceleration?.acceptedDraftTokens ?? 0,
                                      result.acceleration?.draftedTokens ?? 0)
                }
            }
        }
    }

    /// A deterministic proposal source that forces an accepted prefix followed
    /// by rejection at 7, exercising the real hybrid-cache restoration path.
    private final class CyclicDraft: Q35MTPDraftModel {
        let diagnosticsID = "fixture"
        let reject: Bool
        init(reject: Bool) { self.reject = reject }

        func forwardDraft(inputEmbeddings: MLXArray, hiddenStates: MLXArray, cache: KVCache) -> Q35MTPDraftOutput {
            Q35MTPDraftOutput(logitsHidden: hiddenStates, recurrentHidden: hiddenStates)
        }

        func draftLogits(token: Int, previousHidden: MLXArray, positionOffset: Int, baseModel: Q35Model) -> MLXArray {
            baseModel.logits(from: previousHidden)
        }

        func draftBlock(
            lastToken: Int, hidden: MLXArray, blockSize: Int,
            session: Q35MTPDraftSession, baseModel: Q35Model
        ) -> Q35MTPDraftBlock {
            let tokens = (1..<blockSize).map { index -> Int32 in
                let next = (lastToken + index) % 32
                return Int32(reject && next == 7 ? 0 : next)
            }
            return Q35MTPDraftBlock(tokenIDs: MLXArray(tokens).reshaped(1, tokens.count))
        }
    }

    private static let configuration = #"""
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
}
