import Foundation
import XCTest
import MLX
import MLXRandom
@testable import MereRunCore

final class Gemma4ModelTests: MereRunCoreTestCase {
    private func decodeTextConfig(_ object: [String: Any]) throws -> Gemma4TextConfig {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(Gemma4TextConfig.self, from: data)
    }

    private func decodeConfig(_ object: [String: Any]) throws -> Gemma4Config {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(Gemma4Config.self, from: data)
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

    private func mergeLayerCaches(_ rowCaches: [[Gemma4AttentionCache]]) -> [Gemma4AttentionCache]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }
        var mergedCaches: [Gemma4AttentionCache] = []
        mergedCaches.reserveCapacity(first.count)
        for index in first.indices {
            let layerCaches = rowCaches.map { $0[index] }
            guard let merged = layerCaches[0].batched(with: layerCaches) else { return nil }
            mergedCaches.append(merged)
        }
        return mergedCaches
    }

    private func splitLayerCaches(
        _ caches: [Gemma4AttentionCache],
        rowCount: Int
    ) -> [[Gemma4AttentionCache]]? {
        var rows = Array(repeating: [Gemma4AttentionCache](), count: rowCount)
        for cache in caches {
            guard let split = cache.unbatchedRows(count: rowCount), split.count == rowCount else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    func testPerLayerModelProjectionIsUnscaledLinearWeightPath() throws {
        let config = try decodeTextConfig(makeBaseConfig())
        let model = Gemma4LanguageModel(config: config)

        let hidden = MLXArray.ones([1, 1, config.hiddenSize], dtype: .float32)
        let projected = model.perLayerModelProjection(hidden)

        XCTAssertEqual(String(describing: type(of: model.perLayerModelProjection)), "Linear")
        XCTAssertEqual(projected.shape, [1, 1, config.numHiddenLayers * config.hiddenSizePerLayerInput])
    }

    func testMakeCacheUsesPolarKVCacheWhenRequested() throws {
        let config = try decodeTextConfig(makeBaseConfig())
        let model = Gemma4LanguageModel(config: config)
        let quantization = try Gemma4KVCacheQuantization(bits: 2, scheme: .polar, groupSize: 64, quantizedStart: 0)
            .validated()

        let caches = model.makeCache(quantization: quantization)

        XCTAssertEqual(caches.count, 2)
        XCTAssertTrue(caches.allSatisfy { $0 is Gemma4PolarKVCache })
    }

    func testUnifiedModelReplacesImageTokenEmbeddings() throws {
        var textConfig = makeBaseConfig()
        textConfig["num_hidden_layers"] = 1
        textConfig["layer_types"] = ["full_attention"]
        textConfig["hidden_size_per_layer_input"] = 0
        textConfig["num_kv_shared_layers"] = 0
        textConfig["vocab_size"] = 64
        let config = try decodeConfig([
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "tie_word_embeddings": true,
            "eos_token_id": [1, 2],
            "image_token_id": 5,
            "boi_token_id": 6,
            "eoi_token_id": 7,
            "text_config": textConfig,
            "vision_config": [
                "model_type": "gemma4_unified_vision",
                "patch_size": 1,
                "pooling_kernel_size": 2,
                "model_patch_size": 2,
                "mm_embed_dim": 8,
                "mm_posemb_size": 4,
                "num_soft_tokens": 2,
                "rms_norm_eps": 0.000001,
                "output_proj_dims": 8,
            ],
        ])
        let model = try Gemma4UnifiedCausalLM(config: config)
        let inputIds = MLXArray([Int32(6), Int32(5), Int32(5), Int32(7)])
            .reshaped(1, 4)
        let pixelValues = MLXRandom.uniform(0.0 ..< 1.0, [1, 2, 12])
        let imagePositionIds = MLXArray([
            Int32(0), Int32(0),
            Int32(1), Int32(0),
        ], [1, 2, 2])
        let mmTokenTypeIds = MLXArray([Int32(0), Int32(1), Int32(1), Int32(0)])
            .reshaped(1, 4)

        let logits = try model.forward(
            inputIds: inputIds,
            pixelValues: pixelValues,
            imagePositionIds: imagePositionIds,
            mmTokenTypeIds: mmTokenTypeIds
        )

        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [1, 4, 64])
    }

    func testUnifiedConfigDecodesMultimodalTokenIds() throws {
        let config = try decodeConfig([
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "tie_word_embeddings": true,
            "eos_token_id": [1, 106, 50],
            "image_token_id": 258_880,
            "audio_token_id": 258_881,
            "video_token_id": 258_884,
            "boi_token_id": 255_999,
            "boa_token_id": 256_000,
            "eoa_token_index": 258_883,
            "eoi_token_id": 258_882,
            "text_config": makeBaseConfig(),
        ])

        XCTAssertEqual(config.imageTokenId, 258_880)
        XCTAssertEqual(config.audioTokenId, 258_881)
        XCTAssertEqual(config.videoTokenId, 258_884)
        XCTAssertEqual(config.boiTokenId, 255_999)
        XCTAssertEqual(config.boaTokenId, 256_000)
        XCTAssertEqual(config.eoaTokenId, 258_883)
        XCTAssertEqual(config.eoiTokenId, 258_882)
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

    func testDetailedForwardCapturesPreNormHiddenAndSharedKV() throws {
        var configObject = makeBaseConfig()
        configObject["num_hidden_layers"] = 2
        configObject["layer_types"] = ["full_attention", "full_attention"]
        configObject["num_kv_shared_layers"] = 1
        configObject["hidden_size_per_layer_input"] = 0
        let config = try decodeTextConfig(configObject)
        let model = Gemma4TextCausalLM(config: config)
        let caches = try XCTUnwrap(model.makeCache(quantization: nil) as? [Gemma4AttentionCache])
        let inputIds = MLXArray([Int32(3), Int32(4)]).reshaped(1, 2)

        let output = model.forwardForSpeculation(inputIds: inputIds, cache: caches as [AnyObject])

        MLX.eval(output.logits, output.hidden)
        XCTAssertEqual(output.logits.shape, [1, 2, config.vocabSize])
        XCTAssertEqual(output.hidden.shape, [1, 2, config.hiddenSize])
        XCTAssertNotNil(output.sharedKVStates["full_attention"])
        XCTAssertEqual(output.sharedKVStates["full_attention"]?.offset, 2)
    }

    func testAssistantConfigDecodesMTPDefaults() throws {
        var textConfig = makeBaseConfig()
        textConfig["num_hidden_layers"] = 1
        textConfig["layer_types"] = ["full_attention"]
        let data = try JSONSerialization.data(withJSONObject: [
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 8,
            "text_config": textConfig,
        ], options: [])

        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: data)

        XCTAssertEqual(config.modelType, "gemma4_assistant")
        XCTAssertEqual(config.backboneHiddenSize, 8)
        XCTAssertEqual(config.blockSize, Gemma4MTPResources.defaultBlockSize)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertFalse(config.useOrderedEmbeddings)
    }

    func testGemma4MTPPolicyGatesOnGreedyPromptAndEnvironment() {
        let greedy = GenerationConfig(maxTokens: 16, temperature: 0, topP: 1)
        let sampled = GenerationConfig(maxTokens: 16, temperature: 0.7, topP: 0.9)
        let env = [
            "MERERUN_GEMMA4_MTP_MIN_PROMPT_TOKENS": "8",
            "MERERUN_GEMMA4_MTP_BLOCK_SIZE": "6",
        ]

        XCTAssertNil(Gemma4MTPPolicy.activationReason(
            assistantAvailable: true,
            promptTokenCount: 8,
            generationConfig: greedy,
            prefixSeedWasUsed: false,
            environment: env
        ))
        XCTAssertEqual(Gemma4MTPPolicy.activationReason(
            assistantAvailable: true,
            promptTokenCount: 8,
            generationConfig: sampled,
            prefixSeedWasUsed: false,
            environment: env
        ), "non-greedy sampling")
        XCTAssertEqual(Gemma4MTPPolicy.activationReason(
            assistantAvailable: false,
            promptTokenCount: 8,
            generationConfig: greedy,
            prefixSeedWasUsed: false,
            environment: env
        ), "assistant not installed")
        XCTAssertEqual(Gemma4MTPPolicy.blockSize(configured: 4, environment: env), 6)
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

    func testMoEDecoderLayerInstallsRouterAndExperts() throws {
        var configObject = makeBaseConfig()
        configObject["enable_moe_block"] = true
        configObject["num_experts"] = 4
        configObject["top_k_experts"] = 2
        configObject["moe_intermediate_size"] = 4
        let config = try decodeTextConfig(configObject)
        let layer = Gemma4DecoderLayer(config: config, layerIndex: 0)

        XCTAssertNotNil(layer.router)
        XCTAssertNotNil(layer.experts)
    }

    func testPrefixCacheForkMatchesFullForwardForSuffixLogits() throws {
        MLXRandom.seed(19)
        let config = try decodeTextConfig(makeBaseConfig())
        let model = Gemma4TextCausalLM(config: config)
        let tokens = [1, 2, 3, 4, 5]
        let prefixCount = 2

        let fullCache = try XCTUnwrap(model.makeCache() as? [Gemma4AttentionCache])
        let fullInput = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        let fullLogits = model(fullInput, cache: fullCache as [AnyObject])
        MLX.eval(fullLogits)

        let prefixCache = try XCTUnwrap(model.makeCache() as? [Gemma4AttentionCache])
        let prefixInput = MLXArray(tokens.prefix(prefixCount).map(Int32.init)).reshaped(1, prefixCount)
        let prefixLogits = model(prefixInput, cache: prefixCache as [AnyObject])
        MLX.eval(prefixLogits)

        let forkedCache = prefixCache.map { $0.fork() }
        let suffixTokens = Array(tokens.dropFirst(prefixCount))
        let suffixInput = MLXArray(suffixTokens.map(Int32.init)).reshaped(1, suffixTokens.count)
        let cachedSuffixLogits = model(suffixInput, cache: forkedCache as [AnyObject])
        MLX.eval(cachedSuffixLogits)

        let expectedSuffixLogits = fullLogits[0..., prefixCount..., 0...]
        let maxDiff = MLX.max(MLX.abs(
            expectedSuffixLogits.asType(.float32) - cachedSuffixLogits.asType(.float32)
        )).item(Float.self)
        XCTAssertLessThan(maxDiff, 0.0001)
    }

    func testEqualLengthBatchedForwardMatchesIndependentRows() throws {
        MLXRandom.seed(23)
        let config = try decodeTextConfig(makeBaseConfig())
        let model = Gemma4TextCausalLM(config: config)
        let rows = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]

        let batchInput = MLXArray(rows.flatMap { $0 }.map(Int32.init)).reshaped(rows.count, rows[0].count)
        let batchLogits = model(batchInput)
        MLX.eval(batchLogits)

        for (rowIndex, tokens) in rows.enumerated() {
            let rowInput = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
            let rowLogits = model(rowInput)
            MLX.eval(rowLogits)
            let maxDiff = MLX.max(MLX.abs(
                batchLogits[rowIndex, 0..., 0...].asType(.float32) - rowLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)
        }
    }

    func testEqualLengthBatchedCachedDecodeMatchesIndependentRows() throws {
        MLXRandom.seed(29)
        let config = try decodeTextConfig(makeBaseConfig())
        let model = Gemma4TextCausalLM(config: config)
        let prefixes = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        let nextTokens = [7, 8]

        let batchCache = try XCTUnwrap(model.makeCache() as? [Gemma4AttentionCache])
        let prefixBatch = MLXArray(prefixes.flatMap { $0 }.map(Int32.init)).reshaped(prefixes.count, prefixes[0].count)
        let prefillLogits = model(prefixBatch, cache: batchCache as [AnyObject])
        MLX.eval(prefillLogits)
        let nextBatch = MLXArray(nextTokens.map(Int32.init)).reshaped(nextTokens.count, 1)
        let batchDecodeLogits = model(nextBatch, cache: batchCache as [AnyObject])
        MLX.eval(batchDecodeLogits)

        for (rowIndex, prefix) in prefixes.enumerated() {
            let rowCache = try XCTUnwrap(model.makeCache() as? [Gemma4AttentionCache])
            let rowPrefix = MLXArray(prefix.map(Int32.init)).reshaped(1, prefix.count)
            let rowPrefillLogits = model(rowPrefix, cache: rowCache as [AnyObject])
            MLX.eval(rowPrefillLogits)
            let rowNext = MLXArray([Int32(nextTokens[rowIndex])]).reshaped(1, 1)
            let rowDecodeLogits = model(rowNext, cache: rowCache as [AnyObject])
            MLX.eval(rowDecodeLogits)

            let maxDiff = MLX.max(MLX.abs(
                batchDecodeLogits[rowIndex, 0..., 0...].asType(.float32) - rowDecodeLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)
        }
    }

    func testMergedIndependentCachesBatchedDecodeMatchesIndependentRows() throws {
        MLXRandom.seed(31)
        let config = try decodeTextConfig(makeBaseConfig())
        let model = Gemma4TextCausalLM(config: config)
        let prefixes = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        let nextTokens = [7, 8]
        let secondTokens = [9, 10]

        var rowCaches: [[Gemma4AttentionCache]] = []
        for prefix in prefixes {
            let cache = try XCTUnwrap(model.makeCache() as? [Gemma4AttentionCache])
            let prefixInput = MLXArray(prefix.map(Int32.init)).reshaped(1, prefix.count)
            let prefixLogits = model(prefixInput, cache: cache as [AnyObject])
            MLX.eval(prefixLogits)
            rowCaches.append(cache)
        }

        let batchedCaches = try XCTUnwrap(mergeLayerCaches(rowCaches))
        let nextBatch = MLXArray(nextTokens.map(Int32.init)).reshaped(nextTokens.count, 1)
        let batchedLogits = model(nextBatch, cache: batchedCaches as [AnyObject])
        MLX.eval(batchedLogits)
        let splitCaches = try XCTUnwrap(splitLayerCaches(batchedCaches, rowCount: nextTokens.count))

        for rowIndex in prefixes.indices {
            let rowNext = MLXArray([Int32(nextTokens[rowIndex])]).reshaped(1, 1)
            let rowLogits = model(rowNext, cache: rowCaches[rowIndex] as [AnyObject])
            MLX.eval(rowLogits)
            var maxDiff = MLX.max(MLX.abs(
                batchedLogits[rowIndex, 0..., 0...].asType(.float32) - rowLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)

            let rowSecond = MLXArray([Int32(secondTokens[rowIndex])]).reshaped(1, 1)
            let splitLogits = model(rowSecond, cache: splitCaches[rowIndex] as [AnyObject])
            let independentLogits = model(rowSecond, cache: rowCaches[rowIndex] as [AnyObject])
            MLX.eval(splitLogits, independentLogits)
            maxDiff = MLX.max(MLX.abs(
                splitLogits[0, 0..., 0...].asType(.float32) - independentLogits[0, 0..., 0...].asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.0001)
        }
    }
}
