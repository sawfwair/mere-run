import Foundation
import XCTest
import MLX
import MLXFast
import MLXNN
import MLXRandom
@testable import MereRunCore

final class LFM2ConfigAndModelTests: MereRunCoreTestCase {
    private func decodeConfig(_ object: [String: Any]) throws -> LFM2Config {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(LFM2Config.self, from: data)
    }

    private func makeBaseConfig(includeQuantization: Bool = true) -> [String: Any] {
        var config: [String: Any] = [
            "architectures": ["Lfm2MoeForCausalLM"],
            "model_type": "lfm2_moe",
            "vocab_size": 124_928,
            "hidden_size": 2_048,
            "intermediate_size": 7_168,
            "moe_intermediate_size": 1_792,
            "num_hidden_layers": 24,
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "norm_topk_prob": true,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 128_000,
            "use_expert_bias": true,
            "num_dense_layers": 2,
            "norm_eps": 0.000001,
            "conv_bias": false,
            "conv_L_cache": 3,
            "rope_theta": 5_000_000.0,
            "rope_parameters": [
                "rope_theta": 5_000_000.0,
                "rope_type": "default",
            ],
            "layer_types": [
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
            ],
            "eos_token_id": 124_900,
            "tie_word_embeddings": true,
        ]
        if includeQuantization {
            config["quantization"] = [
                "group_size": 64,
                "bits": 8,
                "mode": "affine",
            ]
        }
        return config
    }

    private func makeTinyConfig() throws -> LFM2Config {
        var config = makeBaseConfig(includeQuantization: false)
        config["vocab_size"] = 32
        config["hidden_size"] = 16
        config["intermediate_size"] = 32
        config["moe_intermediate_size"] = 8
        config["num_hidden_layers"] = 4
        config["num_experts"] = 4
        config["num_experts_per_tok"] = 2
        config["num_attention_heads"] = 4
        config["num_key_value_heads"] = 2
        config["max_position_embeddings"] = 128
        config["num_dense_layers"] = 2
        config["conv_L_cache"] = 2
        config["layer_types"] = ["conv", "full_attention", "conv", "full_attention"]
        config["eos_token_id"] = [31]
        return try decodeConfig(config)
    }

    private func makeLayerCaches(config: LFM2Config) -> [LFM2LayerCache?] {
        let attentionLayers = config.fullAttentionLayerIndexes
        return (0..<config.numHiddenLayers).map { layerIndex in
            if attentionLayers.contains(layerIndex) {
                return .attention(KVCacheSimple(step: 8))
            }
            return .conv(LFM2ConvCache())
        }
    }

    func testDecodesLiquidAILFM25Config() throws {
        let config = try decodeConfig(makeBaseConfig())

        XCTAssertEqual(config.modelType, "lfm2_moe")
        XCTAssertEqual(config.architectures, ["Lfm2MoeForCausalLM"])
        XCTAssertEqual(config.hiddenSize, 2_048)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.numHiddenLayers, 24)
        XCTAssertEqual(config.fullAttentionLayerIndexes, Set([2, 6, 10, 14, 18, 21]))
        XCTAssertEqual(config.numDenseLayers, 2)
        XCTAssertEqual(config.numExperts, 32)
        XCTAssertEqual(config.numExpertsPerTok, 4)
        XCTAssertEqual(config.moeIntermediateSize, 1_792)
        XCTAssertEqual(config.convLCache, 3)
        XCTAssertEqual(config.eosTokenIds, [124_900])
        XCTAssertEqual(config.quantization?.bits, 8)
        XCTAssertEqual(config.quantization?.groupSize, 64)
        XCTAssertEqual(config.ropeParameters?.ropeTheta, 5_000_000)
    }

    func testDecodesQuantizationConfigAliasAndArrayEOS() throws {
        var object = makeBaseConfig(includeQuantization: false)
        object["eos_token_id"] = [1, 2]
        object["quantization_config"] = [
            "group_size": 32,
            "bits": 4,
            "mode": "affine",
        ]

        let config = try decodeConfig(object)

        XCTAssertEqual(config.eosTokenIds, [1, 2])
        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertEqual(config.quantization?.groupSize, 32)
    }

    func testRendersLiquidAIChatTemplateShape() throws {
        let prompt = try LFM2TokenizerAndTemplate.renderPrompt(
            messages: [
                ChatMessage(role: .system, content: "Be brief."),
                ChatMessage(role: .user, content: "What is 2+2?"),
            ],
            addGenerationPrompt: true,
            includeThinking: false
        )

        XCTAssertEqual(
            prompt,
            """
            <|startoftext|><|im_start|>system
            Be brief.<|im_end|>
            <|im_start|>user
            What is 2+2?<|im_end|>
            <|im_start|>assistant

            """
        )
    }

    func testTinyLFM2ForwardProducesLogits() throws {
        MLXRandom.seed(42)
        let config = try makeTinyConfig()
        let model = LFM2Model(config: config)
        let caches = makeLayerCaches(config: config)
        let input = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)

        let logits = model(input, cache: caches)
        MLX.eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, config.vocabSize])
        XCTAssertTrue(MLX.max(MLX.abs(logits.asType(.float32))).item(Float.self).isFinite)
    }

    func testRaggedBatchedDecodeMatchesIndependentRowsAndSplitsCaches() throws {
        MLXRandom.seed(421)
        let config = try makeTinyConfig()
        let model = LFM2Model(config: config)
        let firstCaches = makeLayerCaches(config: config)
        let secondCaches = makeLayerCaches(config: config)
        let firstPrompt = MLXArray([Int32(1), 2]).reshaped(1, 2)
        let secondPrompt = MLXArray([Int32(3), 4, 5, 6]).reshaped(1, 4)
        let firstPrefill = model(firstPrompt, cache: firstCaches)
        let secondPrefill = model(secondPrompt, cache: secondCaches)
        MLX.eval(firstPrefill, secondPrefill)

        let serialFirstCaches = firstCaches.map { $0?.fork() }
        let serialSecondCaches = secondCaches.map { $0?.fork() }
        let batchedCaches: [LFM2LayerCache?] = try firstCaches.indices.map { layerIndex in
            let rows = try XCTUnwrap([firstCaches[layerIndex], secondCaches[layerIndex]].compactMap { $0 })
            XCTAssertEqual(rows.count, 2)
            return try XCTUnwrap(rows[0].batched(with: rows))
        }

        let batched = model(
            MLXArray([Int32(7), 8]).reshaped(2, 1),
            cache: batchedCaches
        )
        let first = model(MLXArray([Int32(7)]).reshaped(1, 1), cache: serialFirstCaches)
        let second = model(MLXArray([Int32(8)]).reshaped(1, 1), cache: serialSecondCaches)
        MLX.eval(batched, first, second)

        XCTAssertLessThan(
            MLX.max(MLX.abs(batched[0] - first[0])).item(Float.self),
            2e-4
        )
        XCTAssertLessThan(
            MLX.max(MLX.abs(batched[1] - second[0])).item(Float.self),
            2e-4
        )

        for cache in batchedCaches.compactMap({ $0 }) {
            let rows = try XCTUnwrap(cache.unbatchedRows(count: 2))
            XCTAssertEqual(rows.count, 2)
            XCTAssertTrue(rows.allSatisfy { row in
                switch row {
                case .attention(let value):
                    return value.offset == 3 || value.offset == 5
                case .conv(let value):
                    return value.state?.dim(0) == 1
                }
            })
        }
    }

    func testLFM2ContinuousBatchingStatsExposeOptInState() async {
        let generator = LFM2Generator(continuousBatchingEnabled: true)

        let stats = await generator.continuousBatchingStats()

        XCTAssertTrue(stats.enabled)
        XCTAssertEqual(stats.batchedDecodeSteps, 0)
        XCTAssertEqual(stats.maxBatchSize, 0)
    }

    func testNativeGroupedQueryAttentionMatchesExpandedReference() throws {
        MLXRandom.seed(812)
        let config = try makeTinyConfig()
        let attention = LFM2Attention(config: config)
        let input = MLXRandom.normal([2, 7, config.hiddenSize]).asType(.float32)

        let actual = attention(input, mask: .causal, cache: nil)

        let batch = input.dim(0)
        let sequence = input.dim(1)
        var queries = attention.qLayerNorm(
            attention.qProj(input).reshaped(batch, sequence, config.numAttentionHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        var keys = attention.kLayerNorm(
            attention.kProj(input).reshaped(batch, sequence, config.numKeyValueHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        var values = attention.vProj(input)
            .reshaped(batch, sequence, config.numKeyValueHeads, config.headDim)
            .transposed(0, 2, 1, 3)
        let rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeParameters?.ropeTheta ?? config.ropeTheta
        )
        queries = rope(queries, offset: 0)
        keys = rope(keys, offset: 0)
        let repeats = config.numAttentionHeads / config.numKeyValueHeads
        keys = MLX.repeated(keys, count: repeats, axis: 1)
        values = MLX.repeated(values, count: repeats, axis: 1)
        let expanded = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / Float(config.headDim).squareRoot(),
            mask: .causal
        )
        let expected = attention.outProj(
            expanded.transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, config.numAttentionHeads * config.headDim)
        )

        MLX.eval(actual, expected)
        let maxDifference = MLX.max(MLX.abs(actual - expected)).item(Float.self)
        XCTAssertLessThan(maxDifference, 1e-5)
    }

    func testTransposesConvertedConvWeightsWhenNeeded() {
        let value = MLXArray(0..<12).reshaped(1, 3, 4)

        let mapped = LFM2Resources.mapWeight(key: "model.layers.0.conv.conv.weight", value: value)

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].0, "model.layers.0.conv.conv.weight")
        XCTAssertEqual(mapped[0].1.shape, [1, 4, 3])
    }
}
