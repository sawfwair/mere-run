import Foundation
import XCTest
import MLX
@testable import MereRunCore

final class Gemma4ModelTests: MereRunCoreTestCase {
    private func decodeTextConfig(_ object: [String: Any]) throws -> Gemma4TextConfig {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(Gemma4TextConfig.self, from: data)
    }

    private func makeBaseConfig() -> [String: Any] {
        [
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 16,
            "hidden_size_per_layer_input": 4,
            "rope_parameters": [
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                ],
                "full_attention": [
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
            "sliding_window": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "num_kv_shared_layers": 0,
            "tie_word_embeddings": true,
        ]
    }

    func testPerLayerModelProjectionIsUnscaledLinearWeightPath() throws {
        let config = try decodeTextConfig(makeBaseConfig())
        let model = Gemma4LanguageModel(config: config)

        let hidden = MLXArray.ones([1, 1, config.hiddenSize], dtype: .float32)
        let projected = model.perLayerModelProjection(hidden)

        XCTAssertEqual(String(describing: type(of: model.perLayerModelProjection)), "Linear")
        XCTAssertEqual(projected.shape, [1, 1, config.numHiddenLayers * config.hiddenSizePerLayerInput])
    }

    func testAttentionKEqVDisablesValueProjectionForFullAttentionLayers() throws {
        var configObject = makeBaseConfig()
        configObject["attention_k_eq_v"] = true
        configObject["num_global_key_value_heads"] = 1
        let config = try decodeTextConfig(configObject)

        let slidingAttention = Gemma4Attention(config: config, layerIndex: 0)
        let fullAttention = Gemma4Attention(config: config, layerIndex: 1)

        XCTAssertNotNil(slidingAttention.vProj)
        XCTAssertNil(fullAttention.vProj)
    }

    func testAttentionKEqVFullAttentionStillProducesExpectedShape() throws {
        let config = try decodeTextConfig([
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 16,
            "hidden_size_per_layer_input": 4,
            "rope_parameters": [
                "full_attention": [
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
            "sliding_window": 32,
            "layer_types": ["full_attention"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": true,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "num_kv_shared_layers": 0,
            "tie_word_embeddings": true,
        ])
        let attention = Gemma4Attention(config: config, layerIndex: 0)
        let hidden = MLXArray.ones([1, 2, config.hiddenSize], dtype: .float32)

        let output = attention(hidden, cache: nil)

        XCTAssertNil(attention.vProj)
        XCTAssertEqual(output.shape, [1, 2, config.hiddenSize])
    }
}
