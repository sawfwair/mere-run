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

    private func decodeVisionConfig(_ object: [String: Any]) throws -> LFM2VLConfig {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(LFM2VLConfig.self, from: data)
    }

    private func decodeDSparkConfig(_ object: [String: Any]) throws -> LFM2DSparkConfig {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(LFM2DSparkConfig.self, from: data)
    }

    private func makeDSparkConfig(
        vocabularySize: Int = 128_000,
        targetLayerCount: Int = 30,
        targetLayerIDs: [Int] = [2, 9, 17, 21, 27]
    ) -> [String: Any] {
        [
            "architectures": ["Lfm2DSparkDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 2_048,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "intermediate_size": 6_144,
            "rms_norm_eps": 0.00001,
            "vocab_size": vocabularySize,
            "rope_theta": 10_000_000,
            "max_position_embeddings": 32_768,
            "layer_types": Array(repeating: "full_attention", count: 5),
            "block_size": 9,
            "dflash_config": [
                "mask_token_id": 125_017,
                "target_layer_ids": targetLayerIDs,
                "num_target_layers": targetLayerCount,
            ],
            "markov_rank": 256,
            "rope_is_neox_style": false,
            "enable_confidence_head": true,
            "markov_head_type": "vanilla",
        ]
    }

    private func makeTinyVisionConfig() -> [String: Any] {
        [
            "architectures": ["Lfm2VlForConditionalGeneration"],
            "model_type": "lfm2_vl",
            "image_token_id": 63,
            "downsample_factor": 2,
            "encoder_patch_size": 2,
            "min_image_tokens": 1,
            "max_image_tokens": 1,
            "max_num_patches": 4,
            "projector_bias": true,
            "projector_hidden_size": 16,
            "projector_use_layernorm": false,
            "quantization": [
                "group_size": 64,
                "bits": 8,
                "mode": "affine",
            ],
            "text_config": [
                "architectures": ["Lfm2ForCausalLM"],
                "model_type": "lfm2",
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "norm_eps": 0.00001,
                "conv_bias": false,
                "conv_L_cache": 2,
                "layer_types": ["conv", "full_attention"],
                "eos_token_id": 62,
                "tie_word_embeddings": true,
            ],
            "vision_config": [
                "model_type": "siglip2_vision_model",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "num_patches": 4,
                "patch_size": 2,
                "layer_norm_eps": 0.000001,
            ],
        ]
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

    private func makeTinyDenseConfig() throws -> LFM2Config {
        try decodeConfig([
            "architectures": ["Lfm2ForCausalLM"],
            "model_type": "lfm2",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_bias": false,
            "conv_L_cache": 2,
            "rope_theta": 10_000_000,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"],
            "eos_token_id": [31],
            "tie_word_embeddings": true,
        ])
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

    func testDecodesLiquidAILFM25DenseConfigWithoutMoEFields() throws {
        let config = try makeTinyDenseConfig()

        XCTAssertEqual(config.modelType, "lfm2")
        XCTAssertEqual(config.architectures, ["Lfm2ForCausalLM"])
        XCTAssertEqual(config.moeIntermediateSize, config.intermediateSize)
        XCTAssertEqual(config.numExperts, 0)
        XCTAssertEqual(config.numExpertsPerTok, 1)
        XCTAssertEqual(config.numDenseLayers, config.numHiddenLayers)
        XCTAssertEqual(config.fullAttentionLayerIndexes, Set([1, 3]))
    }

    func testDecodesLFM25AutoAdjustedSwiGLUWidth() throws {
        var config = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data("""
            {
              "architectures": ["Lfm2ForCausalLM"],
              "model_type": "lfm2",
              "vocab_size": 65536,
              "hidden_size": 2048,
              "intermediate_size": 12288,
              "block_auto_adjust_ff_dim": true,
              "block_ffn_dim_multiplier": 1.0,
              "block_multiple_of": 256,
              "num_hidden_layers": 16,
              "num_attention_heads": 32,
              "num_key_value_heads": 8,
              "max_position_embeddings": 128000,
              "norm_eps": 0.00001,
              "conv_L_cache": 3,
              "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv"]
            }
            """.utf8)) as? [String: Any]
        )
        config["eos_token_id"] = 7

        XCTAssertEqual(try decodeConfig(config).intermediateSize, 8_192)
    }

    func testDecodesOfficialLiquidAILFM25DSparkContract() throws {
        let config = try decodeDSparkConfig(makeDSparkConfig())

        XCTAssertEqual(config.architectures, ["Lfm2DSparkDraftModel"])
        XCTAssertEqual(config.hiddenLayerCount, 5)
        XCTAssertEqual(config.blockSize, 9)
        XCTAssertEqual(config.markovRank, 256)
        XCTAssertEqual(config.features.targetLayerIDs, [2, 9, 17, 21, 27])
        XCTAssertEqual(config.features.targetLayerCount, 30)
        XCTAssertFalse(config.ropeUsesNeoXLayout)
        XCTAssertTrue(config.confidenceHeadEnabled)
    }

    func testRejectsAlteredLFM25DSparkArchitectureContract() throws {
        var object = makeDSparkConfig()
        object["block_size"] = 8

        XCTAssertThrowsError(try decodeDSparkConfig(object))
    }

    func testDecodesLFM25VisionConfigAndFallsBackToImageTokenID() throws {
        let config = try decodeVisionConfig(makeTinyVisionConfig())

        XCTAssertEqual(config.modelType, "lfm2_vl")
        XCTAssertEqual(config.imageTokenId, 63)
        XCTAssertEqual(config.imageTokenIndex, 63)
        XCTAssertEqual(config.textConfig.modelType, "lfm2")
        XCTAssertEqual(config.textConfig.intermediateSize, 32)
        XCTAssertEqual(config.visionConfig.modelType, "siglip2_vision_model")
        XCTAssertEqual(config.visionConfig.patchSize, 2)
        XCTAssertEqual(config.downsampleFactor, 2)
        XCTAssertEqual(config.quantization?.bits, 8)
    }

    func testLFM25VisionSmartResizeAndPromptExpansionMatchDownsampledGrid() throws {
        let size = LFM2VLImageProcessor.smartResize(
            width: 1_600,
            height: 900,
            encoderPatchSize: 16,
            downsampleFactor: 2,
            minImageTokens: 64,
            maxImageTokens: 256
        )
        XCTAssertEqual(size.width, 672)
        XCTAssertEqual(size.height, 384)

        let expanded = try LFM2VLImageProcessor.expandedPromptTokens(
            [10, 63, 11],
            grids: [LFM2VLImageGrid(rows: 24, columns: 42)],
            downsampleFactor: 2,
            imageTokenId: 63,
            imageStartTokenId: 60,
            imageEndTokenId: 61
        )
        XCTAssertEqual(expanded.count, 256)
        XCTAssertEqual(expanded.prefix(3), [10, 60, 63])
        XCTAssertEqual(expanded.suffix(2), [61, 11])
        XCTAssertEqual(expanded.filter { $0 == 63 }.count, 252)
    }

    func testLFM25VisionModelReplacesImageTokenWithProjectedFeature() throws {
        let config = try decodeVisionConfig(makeTinyVisionConfig())
        let model = LFM2VLModel(config: config)
        let embeddings = try model.inputEmbeddings(
            inputTokens: [1, config.imageTokenIndex, 2],
            pixelValues: MLXArray.zeros([1, 4, 12]),
            grids: [LFM2VLImageGrid(rows: 2, columns: 2)]
        )

        MLX.eval(embeddings)
        XCTAssertEqual(embeddings.shape, [1, 3, 16])
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

    func testRendersDenseLFM25ThinkingGenerationPrefix() throws {
        let prompt = try LFM2TokenizerAndTemplate.renderPrompt(
            messages: [ChatMessage(role: .user, content: "What is 2+2?")],
            addGenerationPrompt: true,
            includeThinking: false,
            generationPromptSuffix: "<think>"
        )

        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>"))
    }

    func testRenderedConversationHistoryIsAReusablePromptPrefix() throws {
        let history = [
            ChatMessage(role: .user, content: "Question one"),
            ChatMessage(role: .assistant, content: "Answer one"),
        ]
        let historyPrompt = try LFM2TokenizerAndTemplate.renderPrompt(
            messages: history,
            addGenerationPrompt: false,
            includeThinking: false
        )
        let nextPrompt = try LFM2TokenizerAndTemplate.renderPrompt(
            messages: history + [ChatMessage(role: .user, content: "Question two")],
            addGenerationPrompt: true,
            includeThinking: false
        )

        XCTAssertTrue(nextPrompt.hasPrefix(historyPrompt))
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

    func testTrainingLogitsMatchFullProjectionAtSelectedPositions() throws {
        MLXRandom.seed(73)
        let config = try makeTinyConfig()
        let model = LFM2Model(config: config)
        let input = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(2, 3)
        let positions = MLXArray([Int32(1), Int32(3), Int32(5)])
        let full = model.trainingForward(input).reshaped(-1, config.vocabSize)
        let expected = take(full, positions, axis: 0)
        let gathered = model.trainingLogits(
            inputIDs: input,
            flatTargetPositions: positions
        )
        MLX.eval(expected, gathered)

        XCTAssertEqual(gathered.shape, [3, config.vocabSize])
        XCTAssertEqual(
            MLX.max(MLX.abs(expected - gathered)).item(Float.self),
            0,
            accuracy: 1e-6
        )
    }

    func testNativeLFM2TrainerUpdatesAttentionLoRA() throws {
        MLXRandom.seed(74)
        let model = LFM2Model(config: try makeTinyConfig())
        let layers = try LFM2TextLoRAInjector.inject(into: model, rank: 2)
        let inputTokenIDs = (0..<40).map { ($0 % 8) + 1 }
        let labelTokenIDs = Array(inputTokenIDs.dropFirst()) + [1]

        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [
                TextSFTTokenizedExample(
                    inputTokenIds: inputTokenIDs,
                    labelTokenIds: labelTokenIDs,
                    lossMask: [0] + Array(repeating: 1, count: 39)
                ),
            ],
            config: TextLoRATrainingConfig(
                trainingSteps: 2,
                batchSize: 1,
                learningRate: 0.01
            ),
            gatheredForward: { model, inputIDs, positions in
                model.trainingLogits(
                    inputIDs: inputIDs,
                    flatTargetPositions: positions
                )
            }
        ) { model, inputIDs in
            model.trainingForward(inputIDs)
        }

        XCTAssertEqual(report.steps, 2)
        XCTAssertEqual(report.layerCount, 8)
        XCTAssertTrue(report.finalLoss?.isFinite == true)
        XCTAssertTrue(layers.values.contains {
            MLX.sum(MLX.abs($0.loraUp)).item(Float.self) > 0
        })
    }

    func testNativeLFM2AdapterReloadsSavedTrainingWeights() async throws {
        MLXRandom.seed(75)
        let config = try makeTinyConfig()
        let source = LFM2Model(config: config)
        let sourceLayers = try LFM2TextLoRAInjector.inject(
            into: source,
            rank: 2,
            alpha: 4
        )
        for layer in sourceLayers.values {
            layer.loraDown = MLXArray.ones(like: layer.loraDown) * 0.125
            layer.loraUp = MLXArray.ones(like: layer.loraUp) * 0.25
        }

        let directory = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: directory) }
        let adapterURL = directory.appendingPathComponent("lfm2-attention.safetensors")
        try LoRASafetensorsWriter.save(
            loraLayers: sourceLayers,
            to: adapterURL,
            metadata: ["format": TextLoRATrainingManifest.lfm2Format]
        )

        MLXRandom.seed(75)
        let target = LFM2Model(config: config)
        let input = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let baseline = target.trainingForward(input)
        MLX.eval(baseline)
        let report = try await LFM2TextLoRAAdapter.apply(
            .local(path: adapterURL.path, scale: 1),
            to: target
        )
        let adapted = target.trainingForward(input)
        MLX.eval(adapted)

        XCTAssertEqual(report.matchedLayerCount, sourceLayers.count)
        XCTAssertEqual(report.injectedLayerCount, sourceLayers.count)
        XCTAssertGreaterThan(
            MLX.max(MLX.abs(adapted - baseline)).item(Float.self),
            0
        )
    }

    func testTinyDenseLFM2ForwardUsesOfficialWeightNames() throws {
        MLXRandom.seed(43)
        let config = try makeTinyDenseConfig()
        let model = LFM2Model(config: config)
        let caches = makeLayerCaches(config: config)
        let input = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)

        let modulePaths = Set(model.leafModules().flattened().map(\.0))
        XCTAssertTrue(modulePaths.contains("model.layers.0.feed_forward.w1"))
        XCTAssertTrue(modulePaths.contains("model.layers.0.feed_forward.w2"))
        XCTAssertTrue(modulePaths.contains("model.layers.0.feed_forward.w3"))
        XCTAssertFalse(modulePaths.contains("model.layers.0.feed_forward.switch_mlp.gate_proj"))

        let logits = model(input, cache: caches)
        MLX.eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, config.vocabSize])
        XCTAssertTrue(MLX.max(MLX.abs(logits.asType(.float32))).item(Float.self).isFinite)
    }

    func testLFM2ForwardCapturesConfiguredTargetLayerOutputs() throws {
        MLXRandom.seed(44)
        let config = try makeTinyDenseConfig()
        let model = LFM2Model(config: config)
        let input = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let output = model.forward(
            input,
            cache: makeLayerCaches(config: config),
            captureLayerIndices: [0, 2]
        )
        let embeddings = model.embeddings(for: input)
        let expectedLayerZeroOutput = model.model.layers[0](
            embeddings,
            attentionMask: createAttentionMask(h: embeddings, cache: nil),
            cache: nil
        )

        MLX.eval(
            output.logits,
            output.capturedHiddenStates[0]!,
            output.capturedHiddenStates[2]!,
            expectedLayerZeroOutput
        )
        XCTAssertEqual(Set(output.capturedHiddenStates.keys), Set([0, 2]))
        XCTAssertEqual(output.capturedHiddenStates[0]?.shape, [1, 3, config.hiddenSize])
        XCTAssertEqual(output.capturedHiddenStates[2]?.shape, [1, 3, config.hiddenSize])
        XCTAssertLessThan(
            MLX.max(
                MLX.abs(output.capturedHiddenStates[0]! - expectedLayerZeroOutput)
            ).item(Float.self),
            1e-4
        )
    }

    func testLFM2BlockVerificationMatchesSerialDecodeAfterCachedPrefix() throws {
        MLXRandom.seed(45)
        let config = try makeTinyDenseConfig()
        let model = LFM2Model(config: config)
        let blockCache = makeLayerCaches(config: config)
        let serialCache = makeLayerCaches(config: config)
        let prefix = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        MLX.eval(model(prefix, cache: blockCache), model(prefix, cache: serialCache))

        let block = model.forward(
            MLXArray([Int32(4), 5, 6]).reshaped(1, 3),
            cache: blockCache
        ).logits
        let serialRows = [4, 5, 6].map { token in
            model.forward(
                MLXArray([Int32(token)]).reshaped(1, 1),
                cache: serialCache
            ).logits
        }
        let serial = MLX.concatenated(serialRows, axis: 1)
        MLX.eval(block, serial)

        XCTAssertLessThan(
            MLX.max(MLX.abs(block - serial)).item(Float.self),
            1e-4
        )
    }

    func testLFM2SpeculativeRollbackMatchesCommittedPrefixDecode() throws {
        MLXRandom.seed(46)
        let config = try makeTinyDenseConfig()
        let model = LFM2Model(config: config)
        let speculativeCache = makeLayerCaches(config: config)
        let referenceCache = makeLayerCaches(config: config)
        let prefix = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        MLX.eval(model(prefix, cache: speculativeCache), model(prefix, cache: referenceCache))

        let candidate = MLXArray([Int32(4), 5, 6, 7]).reshaped(1, 4)
        let speculative = model.forward(
            candidate,
            cache: speculativeCache,
            captureSpeculativeState: true
        ).logits
        MLX.eval(speculative, model.speculativeCacheStorageArrays(speculativeCache))

        model.rollbackSpeculativeCache(
            speculativeCache,
            candidateTokenCount: 4,
            committedTokenCount: 2
        )
        let committed = model.forward(
            MLXArray([Int32(4), 5]).reshaped(1, 2),
            cache: referenceCache
        ).logits
        MLX.eval(committed)

        let replacement = MLXArray([Int32(8)]).reshaped(1, 1)
        let rolledBack = model(replacement, cache: speculativeCache)
        let reference = model(replacement, cache: referenceCache)
        MLX.eval(rolledBack, reference)

        XCTAssertLessThan(
            MLX.max(MLX.abs(rolledBack - reference)).item(Float.self),
            1e-4
        )
    }

    func testLFM25DSparkResourceMappingAndPolicy() throws {
        XCTAssertEqual(
            LFM2Resources.dsparkModelID(for: LFM2Resources.a1bBF16ModelId),
            LFM2Resources.defaultDSparkModelId
        )
        XCTAssertEqual(
            LFM2Resources.dsparkModelID(for: LFM2Resources.smallModelId),
            LFM2Resources.smallDSparkModelId
        )
        XCTAssertEqual(
            LFM2Resources.dsparkModelID(for: LFM2Resources.denseBF16ModelId),
            LFM2Resources.denseDSparkModelId
        )
        XCTAssertNil(LFM2Resources.dsparkModelID(for: LFM2Resources.defaultModelId))
        XCTAssertNil(LFM2Resources.dsparkModelID(for: LFM2Resources.denseModelId))
        XCTAssertFalse(LFM2DSparkPolicy.enabled(environment: ["MERERUN_LFM25_DSPARK": "off"]))
        XCTAssertTrue(LFM2DSparkPolicy.enabled(environment: [:]))
    }

    func testFusedGatherAffine8SwiGLUMatchesNativeGathers() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The fused affine routed MoE kernel requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(45)
        let expertCount = 4
        let tokenCount = 2
        let topK = 3
        let routeCount = tokenCount * topK
        let inputDimensions = 512
        let outputDimensions = 64
        let groupSize = 64
        let bits = 8
        let input = MLXRandom.uniform(
            low: -0.5,
            high: 0.5,
            [1, tokenCount, inputDimensions]
        ).asType(.bfloat16)
        let gate = MLX.quantized(
            MLXRandom.uniform(
                low: -0.25,
                high: 0.25,
                [expertCount, outputDimensions, inputDimensions]
            ).asType(.bfloat16),
            groupSize: groupSize,
            bits: bits,
            mode: .affine
        )
        let up = MLX.quantized(
            MLXRandom.uniform(
                low: -0.25,
                high: 0.25,
                [expertCount, outputDimensions, inputDimensions]
            ).asType(.bfloat16),
            groupSize: groupSize,
            bits: bits,
            mode: .affine
        )
        let gateBiases = try XCTUnwrap(gate.biases)
        let upBiases = try XCTUnwrap(up.biases)
        let indices = MLXArray(
            [Int32(0), 2, 1, 3, 0, 2],
            [1, tokenCount, topK]
        )
        let routedInput = input
            .reshaped([tokenCount, inputDimensions])
            .take(MLXArray([Int32(0), 0, 0, 1, 1, 1]), axis: 0)
            .reshaped([routeCount, 1, inputDimensions])
        let flattenedIndices = indices.reshaped([routeCount])

        let gateOutput = portableGatherQuantizedMM(
            routedInput,
            gate.wq,
            scales: gate.scales,
            biases: gateBiases,
            rhsIndices: flattenedIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: false
        )
        let upOutput = portableGatherQuantizedMM(
            routedInput,
            up.wq,
            scales: up.scales,
            biases: upBiases,
            rhsIndices: flattenedIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: false
        )
        let reference = MLXNN.silu(gateOutput) * upOutput
        let actual = try XCTUnwrap(
            RoutedMoERouting.fusedGatherAffine8SwiGLU(
                input,
                gateWeight: gate.wq,
                gateScales: gate.scales,
                gateBiases: gateBiases,
                upWeight: up.wq,
                upScales: up.scales,
                upBiases: upBiases,
                expertIndices: indices,
                topK: topK,
                groupSize: groupSize,
                bits: bits
            )
        )
        MLX.eval(reference, actual)

        XCTAssertEqual(actual.shape, reference.shape)
        let maximumDifference = MLX.max(
            MLX.abs(reference.asType(.float32) - actual.asType(.float32))
        ).item(Float.self)
        XCTAssertEqual(maximumDifference, 0)
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

    func testLFM2PrefillThroughputUsesPromptTokensAndNativeTiming() throws {
        let throughput = try XCTUnwrap(LFM2Generator.prefillTokensPerSecond(
            promptTokenCount: 5_902,
            prefillSeconds: 2.5
        ))

        XCTAssertEqual(throughput, 2_360.8, accuracy: 0.0001)
        XCTAssertNil(LFM2Generator.prefillTokensPerSecond(
            promptTokenCount: 0,
            prefillSeconds: 2.5
        ))
        XCTAssertNil(LFM2Generator.prefillTokensPerSecond(
            promptTokenCount: 5_902,
            prefillSeconds: 0
        ))
    }

    func testLFM2DecodeLoopEpochCancelsStaleLoopAndAllowsImmediateReuse() {
        var state = LFM2DecodeLoopEpochState()
        let originalEpoch = state.residencyEpoch
        XCTAssertTrue(state.startLoopIfCurrent(epoch: originalEpoch))

        let replacementEpoch = state.beginResidencyTransition()

        XCTAssertFalse(state.isCurrent(originalEpoch))
        XCTAssertFalse(state.startLoopIfCurrent(epoch: originalEpoch))
        XCTAssertTrue(state.startLoopIfCurrent(epoch: replacementEpoch))
        XCTAssertEqual(state.runningEpoch, replacementEpoch)
    }

    func testLFM2StaleLoopCompletionCannotClearReplacementLoop() {
        var state = LFM2DecodeLoopEpochState()
        let originalEpoch = state.residencyEpoch
        XCTAssertTrue(state.startLoopIfCurrent(epoch: originalEpoch))
        let replacementEpoch = state.beginResidencyTransition()
        XCTAssertTrue(state.startLoopIfCurrent(epoch: replacementEpoch))

        state.finishLoop(epoch: originalEpoch)
        XCTAssertEqual(state.runningEpoch, replacementEpoch)

        state.finishLoop(epoch: replacementEpoch)
        XCTAssertNil(state.runningEpoch)
    }

    func testLFM2UnloadAdvancesResidencyEpoch() async {
        let generator = LFM2Generator(continuousBatchingEnabled: true)
        let before = await generator.decodeLoopEpochStateForTesting()

        await generator.unload()

        let after = await generator.decodeLoopEpochStateForTesting()
        XCTAssertEqual(after.residencyEpoch, before.residencyEpoch + 1)
        XCTAssertFalse(after.isCurrent(before.residencyEpoch))
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
