import Foundation
import XCTest
@testable import MereRunCore

final class Q35ConfigDecodingTests: MereRunCoreTestCase {

    private func decodeConfig(_ object: [String: Any]) throws -> Q35Config {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(Q35Config.self, from: data)
    }

    private func makeBaseConfig() -> [String: Any] {
        let textConfig: [String: Any] = [
            "model_type": "qwen3_moe",
            "hidden_size": 3072,
            "num_hidden_layers": 48,
            "intermediate_size": 4096,
            "shared_expert_intermediate_size": 1024,
            "moe_intermediate_size": 1024,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "layer_types": Array(repeating: "linear_attention", count: 48),
            "mlp_only_layers": [],
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 262_144,
            "rms_norm_eps": 0.000001,
            "attn_output_gate": true,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "full_attention_interval": 4,
            "vocab_size": 151_936,
            "eos_token_id": 151_645,
            "rope_parameters": [
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 1.0,
            ],
        ]

        let visionConfig: [String: Any] = [
            "model_type": "qwen3_vit",
            "depth": 24,
            "hidden_act": "gelu",
            "hidden_size": 1280,
            "intermediate_size": 3420,
            "num_heads": 16,
            "out_hidden_size": 3584,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "in_channels": 3,
        ]

        return [
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3ForConditionalGeneration"],
            "tie_word_embeddings": false,
            "eos_token_id": [151_645],
            "text_config": textConfig,
            "vision_config": visionConfig,
        ]
    }

    func testIntermediateSizeFallsBackToSharedWhenNull() throws {
        var config = makeBaseConfig()
        var textConfig = config["text_config"] as? [String: Any] ?? [:]
        textConfig["intermediate_size"] = NSNull()
        textConfig["shared_expert_intermediate_size"] = 2048
        textConfig["moe_intermediate_size"] = 1024
        config["text_config"] = textConfig

        let decoded = try decodeConfig(config)
        XCTAssertEqual(decoded.textConfig.intermediateSize, 2048)
    }

    func testIntermediateSizeFallsBackWhenMissing() throws {
        var config = makeBaseConfig()
        var textConfig = config["text_config"] as? [String: Any] ?? [:]
        textConfig["intermediate_size"] = nil
        textConfig["shared_expert_intermediate_size"] = 1536
        textConfig["moe_intermediate_size"] = 1024
        config["text_config"] = textConfig

        let decoded = try decodeConfig(config)
        XCTAssertEqual(decoded.textConfig.intermediateSize, 1536)
    }

    func testIntermediateSizeFailsWithExplicitMessageWhenAllCandidatesMissing() throws {
        var config = makeBaseConfig()
        var textConfig = config["text_config"] as? [String: Any] ?? [:]
        textConfig["intermediate_size"] = nil
        textConfig["shared_expert_intermediate_size"] = nil
        textConfig["moe_intermediate_size"] = nil
        config["text_config"] = textConfig

        XCTAssertThrowsError(try decodeConfig(config)) { error in
            guard case let DecodingError.dataCorrupted(context) = error else {
                XCTFail("Expected DecodingError.dataCorrupted, got: \(error)")
                return
            }
            XCTAssertTrue(context.debugDescription.contains("Missing intermediate size"))
        }
    }

    func testExplicitIntermediateSizeIsPreserved() throws {
        var config = makeBaseConfig()
        var textConfig = config["text_config"] as? [String: Any] ?? [:]
        textConfig["intermediate_size"] = 4096
        textConfig["shared_expert_intermediate_size"] = 1024
        textConfig["moe_intermediate_size"] = 1024
        config["text_config"] = textConfig

        let decoded = try decodeConfig(config)
        XCTAssertEqual(decoded.textConfig.intermediateSize, 4096)
    }
}
