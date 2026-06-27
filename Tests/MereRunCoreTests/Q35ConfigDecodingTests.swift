import Foundation
import XCTest
import MLX
import MLXRandom
@testable import MereRunCore

final class Q35ConfigDecodingTests: MereRunCoreTestCase {
    func testQ35TemplatePrefillsClosedThinkBlockWhenThinkingIsHidden() {
        let rendered = Q35TokenizerAndTemplate.renderPrompt(
            messages: [ChatMessage(role: .user, content: "Reply with READY only.")],
            addGenerationPrompt: true,
            includeThinking: false
        )

        XCTAssertTrue(rendered.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    func testQ35TemplateExpandsImagePadCount() {
        let rendered = Q35TokenizerAndTemplate.renderPrompt(
            messages: [ChatMessage(role: .user, content: "Read it.", imageUrl: "/tmp/page.png")],
            addGenerationPrompt: true,
            includeThinking: false,
            imageTokenCounts: [3]
        )

        XCTAssertTrue(rendered.contains("<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"))
    }

    func testQ35Qwen3VLTargetSizeUsesInfinityParserPixelBudget() {
        let target = Q35Generator.qwen3VLTargetSize(
            originalWidth: 2_108,
            originalHeight: 1_094,
            patchSize: 16,
            temporalPatchSize: 2,
            spatialMergeSize: 2
        )

        XCTAssertEqual(target.width, 2_112)
        XCTAssertEqual(target.height, 1_088)
        XCTAssertEqual((target.width / 16) * (target.height / 16) / 4, 2_244)
    }

    func testQ35LinearAttentionConvWeightsKeepMLXLayout() {
        let mlxLayout = MLXArray.zeros([8, 4, 1])
        let normalizedMLX = Q35Generator.normalizedLinearAttentionConv1DWeight(mlxLayout)
        XCTAssertEqual(normalizedMLX.shape, [8, 4, 1])

        let pytorchDepthwiseLayout = MLXArray.zeros([8, 1, 4])
        let normalizedPyTorch = Q35Generator.normalizedLinearAttentionConv1DWeight(pytorchDepthwiseLayout)
        XCTAssertEqual(normalizedPyTorch.shape, [8, 4, 1])
    }

    func testQ35RMSNormOffsetConversionSkipsLinearAttentionGateNorm() {
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.layers.0.input_layernorm.weight"))
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.layers.0.post_attention_layernorm.weight"))
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.layers.3.self_attn.q_norm.weight"))
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.norm.weight"))
        XCTAssertFalse(Q35Generator.isOffsetRMSNormWeight("model.layers.0.linear_attn.norm.weight"))
    }

    func testQ35TemplateLeavesThinkingOpenWhenRequested() {
        let rendered = Q35TokenizerAndTemplate.renderPrompt(
            messages: [ChatMessage(role: .user, content: "Explain.")],
            addGenerationPrompt: true,
            includeThinking: true
        )

        XCTAssertTrue(rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
        XCTAssertFalse(rendered.hasSuffix("</think>\n\n"))
    }

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

    private func makeTinyRuntimeConfig(layerTypes: [String]) -> [String: Any] {
        var config = makeBaseConfig()
        var textConfig = config["text_config"] as? [String: Any] ?? [:]
        textConfig["hidden_size"] = 8
        textConfig["num_hidden_layers"] = layerTypes.count
        textConfig["intermediate_size"] = 8
        textConfig["shared_expert_intermediate_size"] = 8
        textConfig["moe_intermediate_size"] = 8
        textConfig["num_attention_heads"] = 2
        textConfig["num_key_value_heads"] = 1
        textConfig["head_dim"] = 4
        textConfig["num_experts"] = 2
        textConfig["num_experts_per_tok"] = 1
        textConfig["layer_types"] = layerTypes
        textConfig["mlp_only_layers"] = []
        textConfig["linear_num_value_heads"] = 1
        textConfig["linear_num_key_heads"] = 1
        textConfig["linear_key_head_dim"] = 4
        textConfig["linear_value_head_dim"] = 4
        textConfig["linear_conv_kernel_dim"] = 2
        textConfig["max_position_embeddings"] = 128
        textConfig["attn_output_gate"] = false
        textConfig["vocab_size"] = 32
        textConfig["eos_token_id"] = 31
        config["text_config"] = textConfig
        config["eos_token_id"] = [31]
        return config
    }

    private func makeLayerCaches(config: Q35Config) -> [Q35LayerCache?] {
        let text = config.textConfig
        let mlpOnly = Set(text.mlpOnlyLayers)
        return (0..<text.numHiddenLayers).map { layerIndex in
            if mlpOnly.contains(layerIndex) {
                return nil
            }
            let layerType = layerIndex < text.layerTypes.count ? text.layerTypes[layerIndex] : "linear_attention"
            if layerType == "full_attention" {
                return .full(KVCacheSimple(step: 8))
            }
            return .linear(Q35LinearCache())
        }
    }

    func testSamplingProbabilitiesGreedyReturnsOneHotDistribution() {
        let logits = MLXArray([0.1, 2.0, -1.0, 0.7])
        let config = GenerationConfig(
            maxTokens: 1,
            temperature: 0,
            topK: 0,
            topP: 1,
            repetitionPenalty: nil
        )

        let probs = samplingProbabilities(logits: logits, config: config, previousTokens: [])
        MLX.eval(probs)

        XCTAssertEqual(probs[1].item(Float.self), 1, accuracy: 0.0001)
        XCTAssertEqual(probs.sum().item(Float.self), 1, accuracy: 0.0001)
    }

    func testQ35MTPDraftLogitsSupportsDenseExpertWeightLayout() throws {
        MLXRandom.seed(39)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["full_attention"]))
        let model = Q35Model(config: config)
        let mtp = Q35MTPModel(config: config)
        let tokens = [1, 2, 3]
        let cache = makeLayerCaches(config: config)
        let input = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        let output = model.forward(input, cache: cache)
        MLX.eval(output.hidden)

        let lastIndex = output.hidden.dim(1) - 1
        let previousHidden = output.hidden[0..., lastIndex..<(lastIndex + 1), 0...]
        let draftLogits = mtp.draftLogits(
            token: 4,
            previousHidden: previousHidden,
            positionOffset: tokens.count,
            baseModel: model
        )
        MLX.eval(draftLogits)

        XCTAssertEqual(draftLogits.shape, [1, 1, config.textConfig.vocabSize])
        XCTAssertTrue(MLX.max(MLX.abs(draftLogits.asType(.float32))).item(Float.self).isFinite)
    }

    func testQ35MTPDraftBlockReturnsRequestedGreedyTokens() throws {
        MLXRandom.seed(40)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["full_attention"]))
        let model = Q35Model(config: config)
        let mtp = Q35MTPModel(config: config)
        let tokens = [1, 2, 3]
        let cache = makeLayerCaches(config: config)
        let input = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        let output = model.forward(input, cache: cache)
        MLX.eval(output.hidden)

        let lastIndex = output.hidden.dim(1) - 1
        let previousHidden = output.hidden[0..., lastIndex..<(lastIndex + 1), 0...]
        let draftTokens = mtp.draftBlock(
            lastToken: 4,
            hidden: previousHidden,
            positionOffset: tokens.count,
            blockSize: 4,
            baseModel: model
        )

        XCTAssertEqual(draftTokens.count, 3)
        XCTAssertTrue(draftTokens.allSatisfy { $0 >= 0 && $0 < config.textConfig.vocabSize })
    }

    func testQ35MTPSpeculationRequiresAdaptiveContextWindow() {
        XCTAssertFalse(
            Q35Generator.shouldSpeculate(
                promptTokenCount: 8_192,
                maxContextTokens: 4_096,
                environment: ["MERERUN_Q35_MTP_SPECULATION": "1"]
            )
        )
        XCTAssertTrue(
            Q35Generator.shouldSpeculate(
                promptTokenCount: 8_192,
                maxContextTokens: 8_192,
                environment: ["MERERUN_Q35_MTP_SPECULATION": "1"]
            )
        )
        XCTAssertTrue(
            Q35Generator.shouldSpeculate(
                promptTokenCount: 2_048,
                maxContextTokens: 4_096,
                environment: [
                    "MERERUN_Q35_MTP_SPECULATION": "1",
                    "MERERUN_Q35_MTP_MIN_PROMPT_TOKENS": "2048",
                ]
            )
        )
    }

    func testQ35MTPBlockSizeUsesEnvironmentClamp() {
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: [:]), 4)
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: ["MERERUN_Q35_MTP_BLOCK_SIZE": "1"]), 4)
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: ["MERERUN_Q35_MTP_BLOCK_SIZE": "6"]), 6)
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: ["MERERUN_Q35_MTP_BLOCK_SIZE": "32"]), 16)
    }

    func testQ35ConfigAllowsTextOnlyQwen36Layout() throws {
        var configObject = makeBaseConfig()
        configObject["model_type"] = "qwen3_5_moe"
        configObject["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        if var textConfig = configObject["text_config"] as? [String: Any] {
            textConfig.removeValue(forKey: "mlp_only_layers")
            configObject["text_config"] = textConfig
        }
        configObject.removeValue(forKey: "vision_config")

        let config = try decodeConfig(configObject)

        XCTAssertEqual(config.modelType, "qwen3_5_moe")
        XCTAssertNil(config.visionConfig)
        XCTAssertEqual(config.textConfig.mlpOnlyLayers, [])
        XCTAssertEqual(config.textConfig.numExperts, 256)
    }

    func testQ35ConfigAllowsDenseInfinityParser2FlashLayout() throws {
        var configObject = makeBaseConfig()
        configObject["model_type"] = "qwen3_5"
        configObject["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        configObject["eos_token_id"] = 248_046
        configObject["image_token_id"] = 248_056
        configObject["vision_start_token_id"] = 248_053
        configObject["vision_end_token_id"] = 248_054
        if var textConfig = configObject["text_config"] as? [String: Any] {
            textConfig["model_type"] = "qwen3_5_text"
            textConfig["hidden_size"] = 2048
            textConfig["num_hidden_layers"] = 24
            textConfig["intermediate_size"] = 6144
            textConfig["num_attention_heads"] = 8
            textConfig["head_dim"] = 256
            textConfig.removeValue(forKey: "num_experts")
            textConfig.removeValue(forKey: "num_experts_per_tok")
            textConfig.removeValue(forKey: "moe_intermediate_size")
            textConfig.removeValue(forKey: "shared_expert_intermediate_size")
            configObject["text_config"] = textConfig
        }
        if var visionConfig = configObject["vision_config"] as? [String: Any] {
            visionConfig["model_type"] = "qwen3_5"
            visionConfig["hidden_size"] = 1024
            visionConfig["intermediate_size"] = 4096
            visionConfig["out_hidden_size"] = 2048
            visionConfig["patch_size"] = 16
            visionConfig["spatial_merge_size"] = 2
            visionConfig["num_position_embeddings"] = 2304
            configObject["vision_config"] = visionConfig
        }

        let config = try decodeConfig(configObject)

        XCTAssertEqual(config.modelType, "qwen3_5")
        XCTAssertEqual(config.eosTokenIds, [248_046])
        XCTAssertEqual(config.imageTokenId, 248_056)
        XCTAssertFalse(config.textConfig.usesMoE)
        XCTAssertEqual(config.textConfig.numExperts, 0)
        XCTAssertEqual(config.textConfig.numExpertsPerTok, 0)
        XCTAssertEqual(config.visionConfig?.patchSize, 16)
        XCTAssertEqual(config.visionConfig?.spatialMergeSize, 2)
    }

    func testQ35ConfigAllowsOrnithOptiQQuantizationMetadata() throws {
        var configObject = makeBaseConfig()
        configObject["model_type"] = "qwen3_5"
        configObject["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        configObject["quantization"] = [
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "language_model.model.layers.0.linear_attn.in_proj_qkv": [
                "bits": 8,
                "group_size": 64,
            ],
        ]
        if var textConfig = configObject["text_config"] as? [String: Any] {
            textConfig["model_type"] = "qwen3_5_text"
            textConfig["hidden_size"] = 4096
            textConfig["num_hidden_layers"] = 32
            textConfig["intermediate_size"] = 12288
            textConfig["num_attention_heads"] = 16
            textConfig["num_key_value_heads"] = 4
            textConfig["head_dim"] = 256
            textConfig.removeValue(forKey: "num_experts")
            textConfig.removeValue(forKey: "num_experts_per_tok")
            textConfig.removeValue(forKey: "moe_intermediate_size")
            textConfig.removeValue(forKey: "shared_expert_intermediate_size")
            configObject["text_config"] = textConfig
        }

        let config = try decodeConfig(configObject)

        XCTAssertEqual(config.modelType, "qwen3_5")
        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertEqual(config.quantization?.groupSize, 64)
        XCTAssertEqual(config.quantization?.mode, "affine")
        XCTAssertFalse(config.textConfig.usesMoE)
        XCTAssertEqual(config.textConfig.hiddenSize, 4096)
    }

    func testQ35DenseFeedForwardRuntimeProducesLogits() throws {
        var configObject = makeTinyRuntimeConfig(layerTypes: ["full_attention"])
        if var textConfig = configObject["text_config"] as? [String: Any] {
            textConfig.removeValue(forKey: "num_experts")
            textConfig.removeValue(forKey: "num_experts_per_tok")
            textConfig.removeValue(forKey: "moe_intermediate_size")
            textConfig.removeValue(forKey: "shared_expert_intermediate_size")
            textConfig["intermediate_size"] = 8
            configObject["text_config"] = textConfig
        }
        let config = try decodeConfig(configObject)
        let model = Q35Model(config: config)
        let input = MLXArray([Int32(1), Int32(2)]).reshaped(1, 2)
        let output = model.forward(input, cache: makeLayerCaches(config: config))
        MLX.eval(output.logits)

        XCTAssertFalse(config.textConfig.usesMoE)
        XCTAssertEqual(output.logits.shape, [1, 2, config.textConfig.vocabSize])
        XCTAssertTrue(MLX.max(MLX.abs(output.logits.asType(.float32))).item(Float.self).isFinite)
    }

    func testQ35TiedEmbeddingsRuntimeProducesLogits() throws {
        var configObject = makeTinyRuntimeConfig(layerTypes: ["full_attention"])
        configObject["tie_word_embeddings"] = true
        let config = try decodeConfig(configObject)
        let model = Q35Model(config: config)
        let input = MLXArray([Int32(1), Int32(2)]).reshaped(1, 2)
        let output = model.forward(input, cache: makeLayerCaches(config: config))
        MLX.eval(output.logits)

        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(output.logits.shape, [1, 2, config.textConfig.vocabSize])
        XCTAssertTrue(MLX.max(MLX.abs(output.logits.asType(.float32))).item(Float.self).isFinite)
    }

    private func mergeLayerCaches(_ rowCaches: [[Q35LayerCache?]]) -> [Q35LayerCache?]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }
        var mergedCaches: [Q35LayerCache?] = []
        mergedCaches.reserveCapacity(first.count)
        for index in first.indices {
            let layerCaches = rowCaches.map { $0[index] }
            if layerCaches.allSatisfy({ $0 == nil }) {
                mergedCaches.append(nil)
                continue
            }
            let nonNil = layerCaches.compactMap { $0 }
            guard nonNil.count == layerCaches.count,
                  let merged = nonNil[0].batched(with: nonNil) else {
                return nil
            }
            mergedCaches.append(merged)
        }
        return mergedCaches
    }

    private func splitLayerCaches(
        _ caches: [Q35LayerCache?],
        rowCount: Int
    ) -> [[Q35LayerCache?]]? {
        var rows = Array(repeating: [Q35LayerCache?](), count: rowCount)
        for cache in caches {
            guard let cache else {
                for index in 0..<rowCount {
                    rows[index].append(nil)
                }
                continue
            }
            guard let split = cache.unbatchedRows(count: rowCount), split.count == rowCount else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
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

    func testFullAttentionKVCacheForkCanMutateIndependently() {
        MLXRandom.seed(31)
        let cache = KVCacheSimple(step: 2)
        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        _ = cache.update(keys: keys, values: values)

        let forked = cache.fork()
        let nextKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 1, 8]).asType(.bfloat16)
        let nextValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 1, 8]).asType(.bfloat16)
        _ = forked.update(keys: nextKeys, values: nextValues)

        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(forked.offset, 3)
    }

    func testFullAttentionKVCacheBatchingRejectsDifferentGrowthSteps() {
        MLXRandom.seed(33)
        let first = KVCacheSimple(step: 2)
        let second = KVCacheSimple(step: 4)
        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        _ = first.update(keys: keys, values: values)
        _ = second.update(keys: keys, values: values)

        XCTAssertNil(first.batched(with: [first, second]))
    }

    func testFullAttentionKVCacheBatchesDifferentOffsetsAsRaggedRows() throws {
        MLXRandom.seed(35)
        let first = KVCacheSimple(step: 2)
        let second = KVCacheSimple(step: 2)
        let firstKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        let firstValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 2, 8]).asType(.bfloat16)
        let secondKeys = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 4, 8]).asType(.bfloat16)
        let secondValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 1, 4, 8]).asType(.bfloat16)
        _ = first.update(keys: firstKeys, values: firstValues)
        _ = second.update(keys: secondKeys, values: secondValues)

        let batched = try XCTUnwrap(first.batched(with: [first, second]))
        XCTAssertEqual(batched.rowOffsets, [2, 4])

        let nextKeys = MLXRandom.uniform(0.0 ..< 1.0, [2, 1, 1, 8]).asType(.bfloat16)
        let nextValues = MLXRandom.uniform(0.0 ..< 1.0, [2, 1, 1, 8]).asType(.bfloat16)
        let updated = batched.update(keys: nextKeys, values: nextValues)
        XCTAssertEqual(batched.rowOffsets, [3, 5])
        XCTAssertEqual(updated.0.shape, [2, 1, 5, 8])
        XCTAssertEqual(updated.1.shape, [2, 1, 5, 8])

        let split = try XCTUnwrap(batched.unbatchedRows(count: 2))
        XCTAssertEqual(split[0].offset, 3)
        XCTAssertEqual(split[1].offset, 5)
    }

    func testLinearCacheForkCanMutateIndependently() {
        MLXRandom.seed(37)
        let cache = Q35LinearCache()
        cache.convState = MLXRandom.uniform(0.0 ..< 1.0, [1, 3, 8]).asType(.bfloat16)
        cache.recurrentState = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 4, 8]).asType(.bfloat16)

        let forked = cache.fork()
        forked.convState = MLXArray.zeros([1, 3, 8], dtype: .bfloat16)
        forked.recurrentState = MLXArray.zeros([1, 2, 4, 8], dtype: .bfloat16)

        let convMax = MLX.max(MLX.abs(cache.convState!.asType(.float32))).item(Float.self)
        let recurrentMax = MLX.max(MLX.abs(cache.recurrentState!.asType(.float32))).item(Float.self)
        let forkConvMax = MLX.max(MLX.abs(forked.convState!.asType(.float32))).item(Float.self)
        let forkRecurrentMax = MLX.max(MLX.abs(forked.recurrentState!.asType(.float32))).item(Float.self)

        XCTAssertGreaterThan(convMax, 0)
        XCTAssertGreaterThan(recurrentMax, 0)
        XCTAssertEqual(forkConvMax, 0)
        XCTAssertEqual(forkRecurrentMax, 0)
    }

    func testQ35PrefixCacheForkMatchesFullForwardForSuffixLogits() throws {
        MLXRandom.seed(41)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["linear_attention", "full_attention"]))
        let model = Q35Model(config: config)
        let tokens = [1, 2, 3, 4, 5]
        let prefixCount = 2

        let fullCache = makeLayerCaches(config: config)
        let fullInput = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        let fullLogits = model(fullInput, cache: fullCache)
        MLX.eval(fullLogits)

        let prefixCache = makeLayerCaches(config: config)
        let prefixInput = MLXArray(tokens.prefix(prefixCount).map(Int32.init)).reshaped(1, prefixCount)
        let prefixLogits = model(prefixInput, cache: prefixCache)
        MLX.eval(prefixLogits)

        let forkedCache = prefixCache.map { $0?.fork() }
        let suffixTokens = Array(tokens.dropFirst(prefixCount))
        let suffixInput = MLXArray(suffixTokens.map(Int32.init)).reshaped(1, suffixTokens.count)
        let cachedSuffixLogits = model(suffixInput, cache: forkedCache)
        MLX.eval(cachedSuffixLogits)

        let expectedSuffixLogits = fullLogits[0..., prefixCount..., 0...]
        let maxDiff = MLX.max(MLX.abs(
            expectedSuffixLogits.asType(.float32) - cachedSuffixLogits.asType(.float32)
        )).item(Float.self)
        XCTAssertLessThan(maxDiff, 0.0001)
    }

    func testQ35MergedIndependentCachesBatchedDecodeMatchesIndependentRows() throws {
        MLXRandom.seed(43)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["linear_attention", "full_attention"]))
        let model = Q35Model(config: config)
        let prefixes = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        let nextTokens = [7, 8]
        let secondTokens = [9, 10]

        var rowCaches: [[Q35LayerCache?]] = []
        for prefix in prefixes {
            let cache = makeLayerCaches(config: config)
            let prefixInput = MLXArray(prefix.map(Int32.init)).reshaped(1, prefix.count)
            let prefixLogits = model(prefixInput, cache: cache)
            MLX.eval(prefixLogits)
            rowCaches.append(cache)
        }

        let batchedCaches = try XCTUnwrap(mergeLayerCaches(rowCaches))
        let nextBatch = MLXArray(nextTokens.map(Int32.init)).reshaped(nextTokens.count, 1)
        let batchedLogits = model(nextBatch, cache: batchedCaches)
        MLX.eval(batchedLogits)
        let splitCaches = try XCTUnwrap(splitLayerCaches(batchedCaches, rowCount: nextTokens.count))

        for rowIndex in prefixes.indices {
            let rowNext = MLXArray([Int32(nextTokens[rowIndex])]).reshaped(1, 1)
            let rowLogits = model(rowNext, cache: rowCaches[rowIndex])
            MLX.eval(rowLogits)
            var maxDiff = MLX.max(MLX.abs(
                batchedLogits[rowIndex, 0..., 0...].asType(.float32) - rowLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)

            let rowSecond = MLXArray([Int32(secondTokens[rowIndex])]).reshaped(1, 1)
            let splitLogits = model(rowSecond, cache: splitCaches[rowIndex])
            let independentLogits = model(rowSecond, cache: rowCaches[rowIndex])
            MLX.eval(splitLogits, independentLogits)
            maxDiff = MLX.max(MLX.abs(
                splitLogits[0, 0..., 0...].asType(.float32) - independentLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)
        }
    }

    func testQ35LinearOnlyDifferentLengthCachesBatchedDecodeMatchesIndependentRows() throws {
        MLXRandom.seed(47)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["linear_attention", "linear_attention"]))
        let model = Q35Model(config: config)
        let prefixes = [
            [1, 2],
            [3, 4, 5, 6],
        ]
        let nextTokens = [7, 8]
        let secondTokens = [9, 10]

        var rowCaches: [[Q35LayerCache?]] = []
        for prefix in prefixes {
            let cache = makeLayerCaches(config: config)
            let prefixInput = MLXArray(prefix.map(Int32.init)).reshaped(1, prefix.count)
            let prefixLogits = model(prefixInput, cache: cache)
            MLX.eval(prefixLogits)
            rowCaches.append(cache)
        }

        let batchedCaches = try XCTUnwrap(mergeLayerCaches(rowCaches))
        let nextBatch = MLXArray(nextTokens.map(Int32.init)).reshaped(nextTokens.count, 1)
        let batchedLogits = model(nextBatch, cache: batchedCaches)
        MLX.eval(batchedLogits)
        let splitCaches = try XCTUnwrap(splitLayerCaches(batchedCaches, rowCount: nextTokens.count))

        for rowIndex in prefixes.indices {
            let rowNext = MLXArray([Int32(nextTokens[rowIndex])]).reshaped(1, 1)
            let rowLogits = model(rowNext, cache: rowCaches[rowIndex])
            MLX.eval(rowLogits)
            var maxDiff = MLX.max(MLX.abs(
                batchedLogits[rowIndex, 0..., 0...].asType(.float32) - rowLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)

            let rowSecond = MLXArray([Int32(secondTokens[rowIndex])]).reshaped(1, 1)
            let splitLogits = model(rowSecond, cache: splitCaches[rowIndex])
            let independentLogits = model(rowSecond, cache: rowCaches[rowIndex])
            MLX.eval(splitLogits, independentLogits)
            maxDiff = MLX.max(MLX.abs(
                splitLogits[0, 0..., 0...].asType(.float32) - independentLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)
        }
    }

    func testQ35FullAttentionDifferentLengthCachesBatchedDecodeMatchesIndependentRows() throws {
        MLXRandom.seed(49)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["full_attention"]))
        let model = Q35Model(config: config)
        let prefixes = [
            [1, 2],
            [3, 4, 5, 6],
        ]
        let nextTokens = [7, 8]
        let secondTokens = [9, 10]

        var rowCaches: [[Q35LayerCache?]] = []
        for prefix in prefixes {
            let cache = makeLayerCaches(config: config)
            let prefixInput = MLXArray(prefix.map(Int32.init)).reshaped(1, prefix.count)
            let prefixLogits = model(prefixInput, cache: cache)
            MLX.eval(prefixLogits)
            rowCaches.append(cache)
        }

        let batchedCaches = try XCTUnwrap(mergeLayerCaches(rowCaches))
        let nextBatch = MLXArray(nextTokens.map(Int32.init)).reshaped(nextTokens.count, 1)
        let batchedLogits = model(nextBatch, cache: batchedCaches)
        MLX.eval(batchedLogits)
        let splitCaches = try XCTUnwrap(splitLayerCaches(batchedCaches, rowCount: nextTokens.count))

        for rowIndex in prefixes.indices {
            let rowNext = MLXArray([Int32(nextTokens[rowIndex])]).reshaped(1, 1)
            let rowLogits = model(rowNext, cache: rowCaches[rowIndex])
            MLX.eval(rowLogits)
            var maxDiff = MLX.max(MLX.abs(
                batchedLogits[rowIndex, 0..., 0...].asType(.float32) - rowLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)

            let rowSecond = MLXArray([Int32(secondTokens[rowIndex])]).reshaped(1, 1)
            let splitLogits = model(rowSecond, cache: splitCaches[rowIndex])
            let independentLogits = model(rowSecond, cache: rowCaches[rowIndex])
            MLX.eval(splitLogits, independentLogits)
            maxDiff = MLX.max(MLX.abs(
                splitLogits[0, 0..., 0...].asType(.float32) - independentLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)
        }
    }
}
