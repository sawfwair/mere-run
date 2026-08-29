import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q38PLEVerificationTests: MereRunCoreTestCase {
    func testEmbeddingPrefillPreservesOriginalPLETokenIDs() async throws {
        let config = try configuration()
        try await Stream.withNewDefaultStream {
            try await Self.qualifyEmbeddingPrefill(
                generator: Q35Generator(modelId: Q35Resources.q38FlashNextMixedModelId), config: config
            )
        }
    }

    private static func qualifyEmbeddingPrefill(generator: isolated Q35Generator, config: Q35Config) async throws {
        MLXRandom.seed(99)
        let model = Q35Model(config: config)
        let ple = try XCTUnwrap(model.model.layers[0].ple)
        let (weight, scales, biases) = MLX.quantized(
            MLXRandom.normal([36, 32]) * 0.1, groupSize: 32, bits: 4
        )
        ple.pleEmbedding.installShards([
            PreQuantizedEmbedding(weight: weight, scales: scales, biases: biases, groupSize: 32, bits: 4),
        ])
        let input = MLXArray([Int32(4), 5, 6, 7]).reshaped(1, 4)
        // Stand in for image-replaced embeddings: PLE must use the original
        // token sequence, not infer IDs from these vectors or receive zeros.
        let embeddings = MLXRandom.normal([1, 4, 8]) * 0.1
        let expectedCache = Q35LinearCache()
        let expected = model.forwardPrefill(input, cache: [.linear(expectedCache)], inputEmbeddings: embeddings)
        MLX.eval(expected.logits)
        let actualCache = Q35LinearCache()
        let actual = try await generator.chunkedPrefillEmbeddings(
            model: model, inputIds: input, inputEmbeddings: embeddings,
            cache: [.linear(actualCache)], progressHandler: nil
        )
        let actualIDs = try XCTUnwrap(actualCache.pleTokenContext)
        let expectedIDs = try XCTUnwrap(expectedCache.pleTokenContext)
        XCTAssertEqual(actualIDs.asArray(Int32.self), expectedIDs.asArray(Int32.self))
        XCTAssertFalse(actualIDs.asArray(Int32.self).allSatisfy { $0 == 0 })
        XCTAssertLessThan((actual.logits - expected.logits).abs().max().item(Float.self), 0.0001)
    }

    func testRejectedDraftsRestorePLEHistoryAndContinuation() throws {
        MLXRandom.seed(98)
        let model = Q35Model(config: try configuration())
        let ple = try XCTUnwrap(model.model.layers[0].ple)
        let (weight, scales, biases) = MLX.quantized(
            MLXRandom.normal([36, 32]) * 0.1, groupSize: 32, bits: 4
        )
        ple.pleEmbedding.installShards([
            PreQuantizedEmbedding(weight: weight, scales: scales, biases: biases, groupSize: 32, bits: 4),
        ])
        // Include EOS in the candidate block: n-gram boundaries must also
        // rewind when a speculative token is rejected.
        let candidate = MLXArray([Int32(4), 5, 31, 7, 8, 9, 10, 11, 12]).reshaped(1, 9)
        for prefix in [[], [Int32(1), 2, 3]] {
            let base = Q35LinearCache()
            if !prefix.isEmpty {
                let output = model.forward(MLXArray(prefix).reshaped(1, prefix.count), cache: [.linear(base)])
                MLX.eval(output.logits)
            }
            for committed in 1...9 {
                let restored = base.fork()
                let verified = model.forward(candidate, cache: [.linear(restored)], targetVerify: true)
                MLX.eval(verified.logits)
                XCTAssertTrue(restored.restoreVerificationPrefix(tokenCount: committed))
                let serial = base.fork()
                for row in 0..<committed {
                    let output = model.forward(candidate[0..., row..<(row + 1)], cache: [.linear(serial)])
                    MLX.eval(output.logits)
                }
                let actualTokens = try XCTUnwrap(restored.pleTokenContext)
                let expectedTokens = try XCTUnwrap(serial.pleTokenContext)
                XCTAssertEqual(actualTokens.asArray(Int32.self), expectedTokens.asArray(Int32.self))
                let actualConv = try XCTUnwrap(restored.pleConvState)
                let expectedConv = try XCTUnwrap(serial.pleConvState)
                XCTAssertLessThan((actualConv - expectedConv).abs().max().item(Float.self), 0.0001)
                let next = MLXArray([Int32(13)]).reshaped(1, 1)
                let actual = model.forward(next, cache: [.linear(restored)])
                let expected = model.forward(next, cache: [.linear(serial)])
                XCTAssertLessThan(
                    (actual.logits - expected.logits).abs().max().item(Float.self), 0.0001,
                    "Rejected PLE history changed continuation after accepting \(committed) tokens"
                )
            }
        }
    }

    private func configuration() throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","tie_word_embeddings":false,"text_config":{
          "model_type":"qwen4_exp_text","hidden_size":8,"num_hidden_layers":1,
          "intermediate_size":8,"num_experts":2,"num_experts_per_tok":1,
          "shared_expert_intermediate_size":8,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
          "layer_types":["linear_attention"],"linear_num_value_heads":1,
          "linear_num_key_heads":1,"linear_key_head_dim":4,"linear_value_head_dim":4,
          "linear_conv_kernel_dim":2,"max_position_embeddings":32768,
          "rms_norm_eps":0.000001,"attention_bias":false,"attention_dropout":0,
          "attn_output_gate":true,"output_gate_type":"sigmoid","hc_count":4,"hc_lowrank":4,
          "indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":4,
          "indexer_budget":2048,"indexer_compress_ratio":4,"ple_layer_ids":[1],
          "ple_embed_dim":128,"ple_conv_kernel_size":4,"ngram_size":3,
          "heads_per_ngram":2,"ngram_vocab_size_base":5,"seed":1234,
          "vocab_size":32,"eos_token_id":31,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.5}
        }}
        """#.utf8))
    }
}
