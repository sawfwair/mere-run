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

    private func makeDFlashConfig() throws -> LagunaDFlashConfig {
        let object: [String: Any] = [
            "model_type": "laguna",
            "vocab_size": 32,
            "draft_vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "attention_bias": false,
            "gating": "per-head",
            "layer_types": ["sliding_attention", "sliding_attention"],
            "sliding_window": 8,
            "rope_theta": 10_000.0,
            "eagle_aux_hidden_state_layer_ids": [1, 2],
            "dflash_config": [
                "block_size": 4,
                "mask_token_id": 12,
                "num_target_layers": 2,
                "target_layer_ids": [0, 1],
                "causal": true,
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: object)
        return try JSONDecoder().decode(LagunaDFlashConfig.self, from: data)
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

    func testDFlashConfigurationAndParameterPathsMatchOfficialContract() throws {
        let config = try makeDFlashConfig()
        let model = LagunaDFlashModel(config: config)
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertEqual(config.dflash.targetLayerIDs, [0, 1])
        XCTAssertEqual(config.eagleAuxHiddenStateLayerIDs, [1, 2])
        XCTAssertEqual(config.dflash.blockSize, 4)
        XCTAssertTrue(keys.contains("aux_hidden_norms.0.weight"))
        XCTAssertTrue(keys.contains("fc.weight"))
        XCTAssertTrue(keys.contains("hidden_norm.weight"))
        XCTAssertTrue(keys.contains("layers.0.self_attn.qkv_proj.weight"))
        XCTAssertTrue(keys.contains("layers.1.self_attn.g_proj.weight"))
        XCTAssertTrue(keys.contains("layers.1.mlp.down_proj.weight"))
        XCTAssertTrue(keys.contains("norm.weight"))
        XCTAssertEqual(keys.count, 25)
    }

    func testDFlashContextProjectionAndParallelDraftProduceFiniteLogits() throws {
        MLXRandom.seed(47)
        let target = LagunaCausalLM(config: try makeConfig())
        let dflash = LagunaDFlashModel(config: try makeDFlashConfig())
        let targetCache = target.makeCache()
        let targetOutput = target.forward(
            MLXArray([1, 2, 3]).reshaped(1, 3),
            cache: targetCache,
            captureLayerIndices: [0, 1]
        )
        let context = dflash.combineTargetHiddenStates(
            targetOutput.capturedHiddenStates
        )
        let draftCache = dflash.makeCache()
        dflash.appendTargetContext(context, cache: draftCache)
        let logits = dflash.draftLogits(
            anchorTokens: MLXArray([4]).reshaped(1, 1),
            speculativeTokenCount: 2,
            cache: draftCache,
            target: target
        )

        MLX.eval(logits)
        XCTAssertEqual(draftCache.map(\.offset), [6, 6])
        XCTAssertEqual(logits.shape, [1, 2, 32])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testDFlashRetainedWindowMatchesFullPromptContext() throws {
        MLXRandom.seed(49)
        let target = LagunaCausalLM(config: try makeConfig())
        let dflash = LagunaDFlashModel(config: try makeDFlashConfig())
        let prompt = Array(1...12)
        let output = target.forward(
            MLXArray(prompt).reshaped(1, prompt.count),
            captureLayerIndices: [0, 1]
        )

        let fullCache = dflash.makeCache()
        dflash.appendTargetContext(
            dflash.combineTargetHiddenStates(output.capturedHiddenStates),
            cache: fullCache
        )
        let retainedStart = prompt.count - dflash.config.slidingWindow
        let retainedStates = output.capturedHiddenStates.mapValues {
            $0[0..., retainedStart..., 0...]
        }
        let retainedCache = dflash.makeCache(initialOffset: retainedStart)
        dflash.appendTargetContext(
            dflash.combineTargetHiddenStates(retainedStates),
            cache: retainedCache
        )

        let fullLogits = dflash.draftLogits(
            anchorTokens: MLXArray([13]).reshaped(1, 1),
            speculativeTokenCount: 2,
            cache: fullCache,
            target: target
        )
        let retainedLogits = dflash.draftLogits(
            anchorTokens: MLXArray([13]).reshaped(1, 1),
            speculativeTokenCount: 2,
            cache: retainedCache,
            target: target
        )
        MLX.eval(fullLogits, retainedLogits)

        XCTAssertEqual(fullCache.map(\.offset), retainedCache.map(\.offset))
        let maximumDifference = MLX.max(
            MLX.abs(fullLogits.asType(.float32) - retainedLogits.asType(.float32))
        ).item(Float.self)
        XCTAssertLessThan(maximumDifference, 0.0001)
    }

    func testRaggedDFlashBatchMatchesIndependentRows() throws {
        MLXRandom.seed(53)
        let target = LagunaCausalLM(config: try makeConfig())
        let dflash = LagunaDFlashModel(config: try makeDFlashConfig())

        func contextCache(tokens: [Int]) -> [Gemma4AttentionCache] {
            let output = target.forward(
                MLXArray(tokens).reshaped(1, tokens.count),
                captureLayerIndices: [0, 1]
            )
            let caches = dflash.makeCache()
            dflash.appendTargetContext(
                dflash.combineTargetHiddenStates(output.capturedHiddenStates),
                cache: caches
            )
            return caches
        }

        let first = contextCache(tokens: [1, 2, 3])
        let second = contextCache(tokens: [7, 8, 9, 10, 11])
        let firstLogits = dflash.draftLogits(
            anchorTokens: MLXArray([4]).reshaped(1, 1),
            speculativeTokenCount: 2,
            cache: first.map { $0.fork() },
            target: target
        )
        let secondLogits = dflash.draftLogits(
            anchorTokens: MLXArray([12]).reshaped(1, 1),
            speculativeTokenCount: 2,
            cache: second.map { $0.fork() },
            target: target
        )
        let batchedCache = first.indices.map { layerIndex in
            LagunaRaggedKVCache(rows: [
                first[layerIndex].fork(),
                second[layerIndex].fork(),
            ])!
        }
        let batchedLogits = dflash.draftLogits(
            anchorTokens: MLXArray([4, 12]).reshaped(2, 1),
            speculativeTokenCount: 2,
            cache: batchedCache,
            target: target
        )
        MLX.eval(firstLogits, secondLogits, batchedLogits)

        let expected = concatenated([firstLogits, secondLogits], axis: 0)
        let maximumDifference = MLX.max(
            MLX.abs(expected.asType(.float32) - batchedLogits.asType(.float32))
        ).item(Float.self)
        XCTAssertLessThan(maximumDifference, 0.0001)
    }

    func testGreedyDFlashVerificationExactlyMatchesSerialTargetDecode() throws {
        MLXRandom.seed(59)
        let target = LagunaCausalLM(config: try makeConfig())
        let dflash = LagunaDFlashModel(config: try makeDFlashConfig())
        let prompt = [1, 2, 3, 4]
        let generationConfig = GenerationConfig(
            maxTokens: 12,
            temperature: 0,
            topK: 0,
            topP: 1,
            repetitionPenalty: nil,
            repetitionContextSize: 64
        )

        let serialCache = target.makeCache()
        let serialLogits = target.lastPositionLogits(
            MLXArray(prompt).reshaped(1, prompt.count),
            cache: serialCache
        )
        let serial = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: serialLogits,
                generationConfig: generationConfig,
                eosTokens: [],
                tokenBudget: 12,
                historySeedTokens: prompt
            ),
            stepForward: { token in
                target.lastPositionLogits(token, cache: serialCache)
            }
        )

        let speculativeTargetCache = target.makeCache()
        let speculativePrefill = target.forward(
            MLXArray(prompt).reshaped(1, prompt.count),
            cache: speculativeTargetCache,
            captureLayerIndices: [0, 1],
            lastPositionOnly: true
        )
        let draftCache = dflash.makeCache()
        dflash.appendTargetContext(
            dflash.combineTargetHiddenStates(
                speculativePrefill.capturedHiddenStates
            ),
            cache: draftCache
        )
        let speculative = try LagunaDFlashDecoder.decode(
            initialLogits: speculativePrefill.logits,
            target: target,
            targetCache: speculativeTargetCache,
            dflash: dflash,
            draftCache: draftCache,
            generationConfig: generationConfig,
            eosTokens: [],
            tokenBudget: 12,
            historySeedTokens: prompt,
            speculativeTokens: 3
        )

        XCTAssertEqual(speculative.generatedTokens, serial.generatedTokens)
        XCTAssertGreaterThan(speculative.stats.rounds, 0)
        XCTAssertGreaterThanOrEqual(
            speculative.stats.draftedTokens,
            speculative.stats.acceptedDraftTokens
                + speculative.stats.rejectedDraftTokens
        )

        let adaptiveTargetCache = target.makeCache()
        let adaptivePrefill = target.forward(
            MLXArray(prompt).reshaped(1, prompt.count),
            cache: adaptiveTargetCache,
            captureLayerIndices: [0, 1],
            lastPositionOnly: true
        )
        let adaptiveDraftCache = dflash.makeCache()
        dflash.appendTargetContext(
            dflash.combineTargetHiddenStates(
                adaptivePrefill.capturedHiddenStates
            ),
            cache: adaptiveDraftCache
        )
        let adaptive = try LagunaDFlashDecoder.decode(
            initialLogits: adaptivePrefill.logits,
            target: target,
            targetCache: adaptiveTargetCache,
            dflash: dflash,
            draftCache: adaptiveDraftCache,
            generationConfig: generationConfig,
            eosTokens: [],
            tokenBudget: 12,
            historySeedTokens: prompt,
            speculativeTokens: 3,
            adaptiveMinimumAcceptanceRate: 1.1,
            adaptiveAcceptanceEvaluationRounds: 1
        )

        XCTAssertEqual(adaptive.generatedTokens, serial.generatedTokens)
        XCTAssertEqual(adaptive.stats.adaptiveFallbacks, 1)
        XCTAssertGreaterThan(adaptive.stats.targetFallbackForwards, 0)
    }

    func testDFlashRejectionDistributionUsesPositiveTargetResidual() {
        let target = MLXArray([Float(0.1), 0.7, 0.2])
        let draft = MLXArray([Float(0.4), 0.2, 0.4])
        let corrected = LagunaDFlashDecoder.rejectionDistribution(
            target: target,
            draft: draft
        )
        MLX.eval(corrected)

        let probabilities = corrected.asArray(Float.self)
        XCTAssertEqual(probabilities[0], 0, accuracy: 0.0001)
        XCTAssertEqual(probabilities[1], 1, accuracy: 0.0001)
        XCTAssertEqual(probabilities[2], 0, accuracy: 0.0001)
    }

    func testDFlashRejectionDistributionFallsBackWhenResidualIsEmpty() {
        let target = MLXArray([Float(0.2), 0.3, 0.5])
        let corrected = LagunaDFlashDecoder.rejectionDistribution(
            target: target,
            draft: target
        )
        MLX.eval(corrected)

        XCTAssertEqual(corrected.asArray(Float.self), [0.2, 0.3, 0.5])
    }

    func testDFlashRoutingUsesEffectiveOutputBudgetBoundary() {
        let minimum = LagunaDFlashRouting.defaultMinimumOutputTokens

        XCTAssertEqual(minimum, 32)
        XCTAssertFalse(LagunaDFlashRouting.shouldUseDFlash(
            tokenBudget: minimum - 1,
            minimumOutputTokens: minimum
        ))
        XCTAssertTrue(LagunaDFlashRouting.shouldUseDFlash(
            tokenBudget: minimum,
            minimumOutputTokens: minimum
        ))
        XCTAssertTrue(LagunaDFlashRouting.shouldUseDFlash(
            tokenBudget: minimum + 1,
            minimumOutputTokens: minimum
        ))
        XCTAssertEqual(LagunaDFlashRouting.defaultSpeculativeTokens, 12)
        XCTAssertEqual(LagunaDFlashRouting.immediateFallbackAcceptanceRate, 0.25)
        XCTAssertEqual(LagunaDFlashRouting.defaultMinimumAcceptanceRate, 0.6)
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

    func testOfficialDFlashCheckpointLoadsWithExactParameterContractWhenAvailable() throws {
        guard let path = ProcessInfo.processInfo.environment[
            "MERERUN_LAGUNA_DFLASH_PATH"
        ] else {
            throw XCTSkip(
                "Set MERERUN_LAGUNA_DFLASH_PATH to run the official DFlash loading contract test."
            )
        }
        let rootURL = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(
            LagunaDFlashConfig.self,
            from: Data(contentsOf: rootURL.appending(path: "config.json"))
        )
        let model = LagunaDFlashModel(config: config)
        let parameterKeys = Set(model.parameters().flattened().map(\.0))

        XCTAssertEqual(config.numHiddenLayers, 6)
        XCTAssertEqual(config.dflash.blockSize, 16)
        XCTAssertEqual(config.dflash.maskTokenID, 12)
        XCTAssertEqual(config.dflash.targetLayerIDs, [1, 10, 19, 29, 38, 47])
        XCTAssertEqual(parameterKeys.count, 69)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: rootURL.appending(path: "model.safetensors"),
            to: model,
            dtype: nil,
            verify: .all
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

    func testChunkedPrefillMatchesFullForwardAcrossSlidingWindowBoundary() throws {
        MLXRandom.seed(11)
        let model = LagunaCausalLM(config: try makeConfig())
        let tokens = Array(1...20).map { $0 % 31 }
        let full = model.lastPositionLogits(MLXArray(tokens).reshaped(1, tokens.count))
        MLX.eval(full)

        let cache = model.makeCache()
        let first = model.lastPositionLogits(
            MLXArray(Array(tokens[..<11])).reshaped(1, 11),
            cache: cache
        )
        MLX.eval(first)
        let chunked = model.lastPositionLogits(
            MLXArray(Array(tokens[11...])).reshaped(1, tokens.count - 11),
            cache: cache
        )
        MLX.eval(chunked)

        let fullValues = full.asArray(Float.self)
        let chunkedValues = chunked.asArray(Float.self)
        let maximumDifference = zip(fullValues, chunkedValues).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maximumDifference, 0.0001)
    }

    func testForwardCapturesRequestedLayerOutputsWithoutChangingLogits() throws {
        MLXRandom.seed(13)
        let model = LagunaCausalLM(config: try makeConfig())
        let tokens = MLXArray([1, 4, 7]).reshaped(1, 3)
        let baseline = model(tokens)
        let captured = model.forward(tokens, captureLayerIndices: [0, 1])
        MLX.eval([baseline, captured.logits] + Array(captured.capturedHiddenStates.values))

        XCTAssertEqual(captured.capturedHiddenStates.keys.sorted(), [0, 1])
        XCTAssertEqual(captured.capturedHiddenStates[0]?.shape, [1, 3, 8])
        XCTAssertEqual(captured.capturedHiddenStates[1]?.shape, [1, 3, 8])
        let maximumDifference = zip(
            baseline.asArray(Float.self),
            captured.logits.asArray(Float.self)
        ).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maximumDifference, 0.0001)
    }

    func testRaggedDecodeBatchMatchesIndependentRows() throws {
        MLXRandom.seed(17)
        let model = LagunaCausalLM(config: try makeConfig())
        let firstCaches = model.makeCache()
        let secondCaches = model.makeCache()
        let firstPrompt = MLXArray([1, 2, 3]).reshaped(1, 3)
        let secondPrompt = MLXArray([4, 5, 6, 7, 8]).reshaped(1, 5)
        MLX.eval(
            model.lastPositionLogits(firstPrompt, cache: firstCaches),
            model.lastPositionLogits(secondPrompt, cache: secondCaches)
        )

        let serialFirstCaches = firstCaches.map { $0.fork() }
        let serialSecondCaches = secondCaches.map { $0.fork() }
        let serialFirst = model.lastPositionLogits(
            MLXArray([9]).reshaped(1, 1),
            cache: serialFirstCaches
        )
        let serialSecond = model.lastPositionLogits(
            MLXArray([10]).reshaped(1, 1),
            cache: serialSecondCaches
        )

        let raggedCaches = zip(firstCaches, secondCaches).map {
            LagunaRaggedKVCache(rows: [$0.0, $0.1])!
        }
        let batched = model.lastPositionLogits(
            MLXArray([9, 10]).reshaped(2, 1),
            cache: raggedCaches
        )
        MLX.eval(serialFirst, serialSecond, batched)

        let firstDifference = zip(
            serialFirst.asArray(Float.self),
            batched[0..<1, 0..., 0...].asArray(Float.self)
        ).map { abs($0 - $1) }.max() ?? 0
        let secondDifference = zip(
            serialSecond.asArray(Float.self),
            batched[1..<2, 0..., 0...].asArray(Float.self)
        ).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(firstDifference, 0.0001)
        XCTAssertLessThan(secondDifference, 0.0001)

        for cache in raggedCaches {
            let rows = try XCTUnwrap(cache.unbatchedRows(count: 2))
            XCTAssertEqual(rows.map(\.offset), [4, 6])
        }
    }
}

private struct LagunaSafetensorIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
