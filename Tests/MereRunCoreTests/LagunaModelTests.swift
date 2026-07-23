import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class LagunaModelTests: MereRunCoreTestCase {
    private func makeConfig() throws -> LagunaConfig {
        let object: [String: Any] = [
            "model_type": "laguna",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_attention_heads_per_layer": [2, 2],
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "attention_bias": false,
            "gating": "per-head",
            "layer_types": ["full_attention", "sliding_attention"],
            "sliding_window": 8,
            "mlp_layer_types": ["dense", "sparse"],
            "mlp_only_layers": [0],
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 6,
            "shared_expert_intermediate_size": 5,
            "moe_routed_scaling_factor": 2.5,
            "moe_router_logit_softcapping": 0.0,
            "norm_topk_prob": true,
            "decoder_sparse_step": 1,
            "moe_apply_router_weight_on_input": false,
            "tie_word_embeddings": false,
            "eos_token_id": [2, 24],
            "rope_parameters": [
                "full_attention": [
                    "rope_type": "yarn",
                    "rope_theta": 500000.0,
                    "factor": 32.0,
                    "original_max_position_embeddings": 8,
                    "beta_slow": 1.0,
                    "beta_fast": 32.0,
                    "attention_factor": 1.3465735902799727,
                    "partial_rotary_factor": 0.5,
                ],
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: object)
        return try JSONDecoder().decode(LagunaConfig.self, from: data)
    }

    func testOfficialConfigurationFieldsDecode() throws {
        let config = try makeConfig()

        XCTAssertEqual(config.modelType, "laguna")
        XCTAssertEqual(config.attentionHeads(layerIndex: 1), 2)
        XCTAssertEqual(config.ropeParameters(layerIndex: 0).ropeType, "yarn")
        XCTAssertEqual(config.ropeParameters(layerIndex: 0).partialRotaryFactor, 0.5)
        XCTAssertEqual(config.moeRoutedScalingFactor, 2.5)
        XCTAssertEqual(config.eosTokenIDs, [2, 24])
        XCTAssertFalse(config.isSparse(layerIndex: 0))
        XCTAssertTrue(config.isSparse(layerIndex: 1))
    }

    func testModelParameterPathsMatchOfficialMLXCheckpoint() throws {
        let model = LagunaCausalLM(config: try makeConfig())
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.e_score_correction_bias"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.shared_expert.down_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.g_proj.weight"))
        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testNVFP4ExpertLayoutMatchesOfficialCheckpointHeader() throws {
        let quantization = try JSONDecoder().decode(
            LagunaQuantizationConfig.self,
            from: Data(#"{"group_size":16,"bits":4,"mode":"nvfp4"}"#.utf8)
        )
        let projection = LagunaSwitchLinear(
            inputDimensions: 3_072,
            outputDimensions: 1_024,
            expertCount: 256,
            quantization: quantization
        )

        XCTAssertEqual(projection.weight.shape, [256, 1_024, 384])
        XCTAssertEqual(projection.weight.dtype, .uint32)
        XCTAssertEqual(projection.scales?.shape, [256, 1_024, 192])
    }

    func testTinyMixedAttentionMoEModelProducesFiniteLogits() throws {
        MLXRandom.seed(42)
        let config = try makeConfig()
        let model = LagunaCausalLM(config: config)
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3))

        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, config.vocabSize])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testToolParserConvertsLagunaMarkup() {
        let calls = LagunaToolParser.parseToolCalls(
            """
            <tool_call>mere_email_search<arg_key>workspace</arg_key><arg_value>sawfwair</arg_value><arg_key>limit</arg_key><arg_value>5</arg_value></tool_call>
            """
        )

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "mere_email_search")
        XCTAssertEqual(calls.first?.arguments, ["workspace": "sawfwair", "limit": "5"])
    }

    func testOfficialTokenizerAndTemplateWhenMetadataIsAvailable() async throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_LAGUNA_TOKENIZER_PATH"] else {
            throw XCTSkip("Set MERERUN_LAGUNA_TOKENIZER_PATH to run the official tokenizer contract test.")
        }
        let tokenizer = try await LagunaTokenizerAndTemplate.load(
            from: URL(fileURLWithPath: path),
            maxLength: 4_096
        )
        let tokens = try tokenizer.encodeForGeneration(
            messages: [ChatMessage(role: .user, content: "Return exactly: ready")],
            includeThinking: false,
            maxLength: 4_096
        )
        let rendered = tokenizer.decode(tokens: tokens)

        XCTAssertFalse(tokens.isEmpty)
        XCTAssertTrue(rendered.contains("<system>"))
        XCTAssertTrue(rendered.contains("<user>Return exactly: ready</user>"))
        XCTAssertTrue(rendered.hasSuffix("<assistant></think>"))
        XCTAssertEqual(tokenizer.eosTokenID, 2)
        XCTAssertEqual(tokenizer.assistantEndTokenID, 24)
    }

    func testOfficialCheckpointParameterInventoryWhenMetadataIsAvailable() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_LAGUNA_CHECKPOINT_PATH"] else {
            throw XCTSkip("Set MERERUN_LAGUNA_CHECKPOINT_PATH to run the official parameter inventory contract test.")
        }
        let rootURL = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(
            LagunaConfig.self,
            from: Data(contentsOf: rootURL.appending(path: "config.json"))
        )
        let index = try JSONDecoder().decode(
            LagunaSafetensorIndex.self,
            from: Data(contentsOf: rootURL.appending(path: "model.safetensors.index.json"))
        )
        let modelKeys = Set(LagunaCausalLM(config: config).parameters().flattened().map(\.0))
        let checkpointKeys = Set(index.weightMap.keys)
        let missingModelKeys = checkpointKeys.subtracting(modelKeys)
        let derivedRuntimeKeys = modelKeys.subtracting(checkpointKeys)

        XCTAssertEqual(checkpointKeys.count, 955)
        XCTAssertTrue(missingModelKeys.isEmpty, "Missing checkpoint parameters: \(missingModelKeys.sorted())")
        XCTAssertEqual(derivedRuntimeKeys.count, 12)
        XCTAssertTrue(
            derivedRuntimeKeys.allSatisfy { $0.hasSuffix(".self_attn.rope.frequencies") },
            "Unexpected derived runtime parameters: \(derivedRuntimeKeys.sorted())"
        )
    }

    func testAvailableOfficialCheckpointShardsApplyWithShapeVerification() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_LAGUNA_CHECKPOINT_PATH"] else {
            throw XCTSkip("Set MERERUN_LAGUNA_CHECKPOINT_PATH to run the official shard loading contract test.")
        }
        let rootURL = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(
            LagunaConfig.self,
            from: Data(contentsOf: rootURL.appending(path: "config.json"))
        )
        let shardURLs = try FileManager.default.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: nil
        )
            .filter { $0.lastPathComponent.hasSuffix(".safetensors") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shardURLs.isEmpty else {
            throw XCTSkip("No finalized Laguna checkpoint shards are available yet.")
        }

        let model = LagunaCausalLM(config: config)
        for shardURL in shardURLs {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: shardURL,
                to: model,
                dtype: nil,
                verify: .shapeMismatch
            )
        }

        if shardURLs.contains(where: { $0.lastPathComponent == "model-00001-of-00014.safetensors" }) {
            let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            XCTAssertEqual(parameters["model.embed_tokens.weight"]?.dtype, .bfloat16)
            XCTAssertEqual(
                parameters["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.dtype,
                .uint32
            )
            XCTAssertEqual(
                parameters["model.layers.1.mlp.switch_mlp.gate_proj.scales"]?.dtype,
                .uint8
            )
        }
    }

    func testCachedDecodeMatchesFullForwardAtLastPosition() throws {
        MLXRandom.seed(7)
        let model = LagunaCausalLM(config: try makeConfig())
        let tokens = [1, 7, 4, 9]
        let full = model.lastPositionLogits(MLXArray(tokens).reshaped(1, tokens.count))
        MLX.eval(full)

        let cache = model.makeCache()
        var cached: MLXArray?
        for token in tokens {
            cached = model.lastPositionLogits(MLXArray([token]).reshaped(1, 1), cache: cache)
            MLX.eval(cached!)
        }

        let fullValues = full.asArray(Float.self)
        let cachedValues = try XCTUnwrap(cached).asArray(Float.self)
        let maximumDifference = zip(fullValues, cachedValues).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maximumDifference, 0.0001)
    }
}

private struct LagunaSafetensorIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
