import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class LagunaModelTests: MereRunCoreTestCase {
    private func makeConfig(quantizedSharedExperts: Bool = false) throws -> LagunaConfig {
        let hiddenSize = quantizedSharedExperts ? 16 : 8
        let headDimension = quantizedSharedExperts ? 8 : 4
        let sharedExpertSize = quantizedSharedExperts ? 16 : 5
        var object: [String: Any] = [
            "model_type": "laguna",
            "vocab_size": 32,
            "hidden_size": hiddenSize,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_attention_heads_per_layer": [2, 2],
            "num_key_value_heads": 1,
            "head_dim": headDimension,
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
            "moe_intermediate_size": quantizedSharedExperts ? 16 : 6,
            "shared_expert_intermediate_size": sharedExpertSize,
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
                    "attention_factor": 1.0,
                    "partial_rotary_factor": 0.5,
                ],
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
        ]
        if quantizedSharedExperts {
            object["quantization"] = [
                "group_size": 16,
                "bits": 4,
                "mode": "nvfp4",
            ]
        }
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

    private func makeXSAttentionConfig() throws -> LagunaConfig {
        let object: [String: Any] = [
            "model_type": "laguna",
            "vocab_size": 32,
            "hidden_size": 2_048,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 64,
            "num_attention_heads_per_layer": [64],
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "attention_bias": false,
            "gating": "per-head",
            "layer_types": ["sliding_attention"],
            "sliding_window": 8,
            "mlp_layer_types": ["dense"],
            "mlp_only_layers": [0],
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "moe_routed_scaling_factor": 2.5,
            "moe_router_logit_softcapping": 0.0,
            "norm_topk_prob": true,
            "decoder_sparse_step": 1,
            "moe_apply_router_weight_on_input": false,
            "tie_word_embeddings": false,
            "eos_token_id": [2],
            "rope_parameters": [
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
        ]
        return try JSONDecoder().decode(
            LagunaConfig.self,
            from: JSONSerialization.data(withJSONObject: object)
        )
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

    func testYarnUsesRuntimeDerivedMscaleInsteadOfMetadataAttentionFactor() throws {
        let config = try makeConfig()
        let rope = LagunaRoPE(
            headDim: config.headDim,
            parameters: config.ropeParameters(layerIndex: 0)
        )
        let input = MLXArray.ones([1, 1, 1, config.headDim], dtype: .bfloat16)
        let output = rope(input, offset: 0)
        MLX.eval(output)

        let values = output.asArray(Float.self)
        let expectedMscale = 1 + 0.1 * log(Float(32))
        XCTAssertEqual(values[0], expectedMscale, accuracy: 0.01)
        XCTAssertEqual(values[1], expectedMscale, accuracy: 0.01)
        XCTAssertEqual(values[2], 1, accuracy: 0.001)
        XCTAssertEqual(values[3], 1, accuracy: 0.001)
    }

    func testXSLayoutQuantizesOnlySharedExpertLinears() throws {
        let model = LagunaCausalLM(
            config: try makeConfig(quantizedSharedExperts: true),
            quantizedSharedExperts: true
        )
        let dense = try XCTUnwrap(model.model.layers[0].mlp as? LagunaDenseMLP)
        let sparse = try XCTUnwrap(model.model.layers[1].mlp as? LagunaSparseMoE)

        XCTAssertFalse(dense.gateProj is QuantizedLinear)
        XCTAssertTrue(sparse.sharedExpert.gateProj is QuantizedLinear)
        XCTAssertTrue(sparse.sharedExpert.upProj is QuantizedLinear)
        XCTAssertTrue(sparse.sharedExpert.downProj is QuantizedLinear)
    }

    func testManagedResourceContractCoversOfficialTargetAndDFlashPayloads() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let dflashRoot = root.appendingPathComponent("dflash", isDirectory: true)
        try FileManager.default.createDirectory(at: dflashRoot, withIntermediateDirectories: true)

        let targetFiles = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ] + (1...14).map {
            String(format: "model-%05d-of-00014.safetensors", $0)
        }
        for file in targetFiles {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: Data()
            ))
        }
        let shardEntries = (1...14).map { index in
            let shard = String(format: "model-%05d-of-00014.safetensors", index)
            return "\"model.layers.\(index).weight\": \"\(shard)\""
        }.joined(separator: ",")
        let indexData = Data("{\"weight_map\":{\(shardEntries)}}".utf8)
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model.safetensors.index.json").path,
            contents: indexData
        ))
        for file in ["config.json", "model.safetensors"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: dflashRoot.appendingPathComponent(file).path,
                contents: Data()
            ))
        }

        XCTAssertTrue(LagunaResources.missingTargetFiles(rootURL: root).isEmpty)
        XCTAssertTrue(LagunaResources.missingDFlashFiles(rootURL: dflashRoot).isEmpty)
        XCTAssertTrue(LagunaResources.handles(modelSpec: LagunaResources.modelID))
        XCTAssertTrue(LagunaResources.handles(modelSpec: LagunaResources.upstreamModelID))
        XCTAssertTrue(LagunaResources.handles(modelSpec: "/tmp/Laguna-S-2.1-NVFP4-mlx"))
        XCTAssertTrue(LagunaResources.handles(modelSpec: LagunaResources.xsModelID))
        XCTAssertTrue(LagunaResources.handles(modelSpec: LagunaResources.xsUpstreamModelID))
        XCTAssertTrue(LagunaResources.handles(modelSpec: "/tmp/Laguna-XS-2.1-NVFP4-mlx"))
        XCTAssertEqual(
            LagunaResources.managedModelID(for: LagunaResources.xsUpstreamModelID),
            LagunaResources.xsModelID
        )
        XCTAssertNil(LagunaResources.installedDFlashPath(for: LagunaResources.xsModelID))

        try FileManager.default.removeItem(
            at: root.appendingPathComponent("model-00014-of-00014.safetensors")
        )
        XCTAssertEqual(
            LagunaResources.missingTargetFiles(rootURL: root).map(\.lastPathComponent),
            ["model-00014-of-00014.safetensors"]
        )
    }

    func testTargetResourceContractUsesShardNamesFromSafetensorsIndex() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
        ] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: Data()
            ))
        }
        let indexData = Data(
            """
            {"weight_map":{
              "model.embed_tokens.weight":"model-00001-of-00005.safetensors",
              "model.norm.weight":"model-00002-of-00005.safetensors"
            }}
            """.utf8
        )
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model.safetensors.index.json").path,
            contents: indexData
        ))

        XCTAssertTrue(LagunaResources.missingTargetFiles(rootURL: root).isEmpty)

        try FileManager.default.removeItem(
            at: root.appendingPathComponent("model-00002-of-00005.safetensors")
        )
        XCTAssertEqual(
            LagunaResources.missingTargetFiles(rootURL: root).map(\.lastPathComponent),
            ["model-00002-of-00005.safetensors"]
        )
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

    func testDFlashRejectionCorrectionConsumesMinPFilteredDistributions() {
        let config = GenerationConfig(
            temperature: 1,
            topK: 20,
            topP: 1,
            minP: 0.15,
            repetitionPenalty: nil
        )
        let target = samplingProbabilities(
            logits: MLXArray([Float(0.7), 0.2, 0.07, 0.03].map(log)),
            config: config,
            previousTokens: []
        )
        let draft = samplingProbabilities(
            logits: MLXArray([Float(0.5), 0.3, 0.15, 0.05].map(log)),
            config: config,
            previousTokens: []
        )
        let corrected = LagunaDFlashDecoder.rejectionDistribution(
            target: target,
            draft: draft
        )
        MLX.eval(target, draft, corrected)

        XCTAssertEqual(target.sum().item(Float.self), 1, accuracy: 0.0001)
        XCTAssertEqual(draft.sum().item(Float.self), 1, accuracy: 0.0001)
        XCTAssertEqual(target[2].item(Float.self), 0, accuracy: 0.0001)
        XCTAssertGreaterThan(draft[2].item(Float.self), 0)
        XCTAssertEqual(corrected.sum().item(Float.self), 1, accuracy: 0.0001)
        XCTAssertEqual(corrected[2].item(Float.self), 0, accuracy: 0.0001)
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

    func testLagunaEOSResolutionHonorsRequestStopPolicy() {
        XCTAssertEqual(
            LagunaGenerator.resolvedEOSTokens(
                modelTokenIDs: [1, 2],
                templateTokenIDs: [2, 3],
                stopOnEOS: true
            ),
            Set([1, 2, 3])
        )
        XCTAssertTrue(
            LagunaGenerator.resolvedEOSTokens(
                modelTokenIDs: [1, 2],
                templateTokenIDs: [2, 3],
                stopOnEOS: false
            ).isEmpty
        )
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

    func testMoEAccelerationBooleanParsing() {
        XCTAssertTrue(LagunaMoEAccelerationPolicy.parseBoolean("on", default: false))
        XCTAssertTrue(LagunaMoEAccelerationPolicy.parseBoolean(" YES ", default: false))
        XCTAssertFalse(LagunaMoEAccelerationPolicy.parseBoolean("off", default: true))
        XCTAssertFalse(LagunaMoEAccelerationPolicy.parseBoolean("0", default: true))
        XCTAssertTrue(LagunaMoEAccelerationPolicy.parseBoolean(nil, default: true))
        XCTAssertFalse(LagunaMoEAccelerationPolicy.parseBoolean("unexpected", default: false))
    }

    func testGraphAccelerationPolicyParsing() {
        XCTAssertTrue(LagunaGraphAccelerationPolicy.parseBoolean(nil, default: true))
        XCTAssertFalse(LagunaGraphAccelerationPolicy.parseBoolean("off", default: true))
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLadderStride(nil, default: 8),
            8
        )
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLadderStride("off", default: 8),
            0
        )
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLadderStride("4", default: 8),
            4
        )
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLadderStride("invalid", default: 8),
            8
        )
    }

    func testFusedPrefillResidualRMSNormMatchesStockOperations() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The fused Laguna prefill kernel requires a Metal GPU.")
        }
        MLXRandom.seed(61)
        let hiddenSize = 2_048
        let residual = MLXRandom.uniform(
            low: -0.5,
            high: 0.5,
            [1, 3, hiddenSize]
        ).asType(.bfloat16)
        let branch = MLXRandom.uniform(
            low: -0.5,
            high: 0.5,
            [1, 3, hiddenSize]
        ).asType(.bfloat16)
        let weight = MLXRandom.uniform(
            low: 0.75,
            high: 1.25,
            [hiddenSize]
        ).asType(.bfloat16)
        let stockSummed = residual + branch
        let stockNormalized = MLXFast.rmsNorm(
            stockSummed,
            weight: weight,
            eps: 0.000_001
        )
        let fused = try XCTUnwrap(LagunaFusedPrefill.residualRMSNorm(
            residual: residual,
            branch: branch,
            weight: weight
        ))

        MLX.eval(stockSummed, stockNormalized, fused.summed, fused.normalized)
        let summedDifference = MLX.max(
            MLX.abs(stockSummed.asType(.float32) - fused.summed.asType(.float32))
        ).item(Float.self)
        let normalizedDifference = MLX.max(
            MLX.abs(stockNormalized.asType(.float32) - fused.normalized.asType(.float32))
        ).item(Float.self)
        XCTAssertEqual(summedDifference, 0)
        XCTAssertEqual(normalizedDifference, 0)
    }

    func testFusedPrefillQKNormRoPEMatchesStockFamilies() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The fused Laguna prefill kernels require a Metal GPU.")
        }
        MLXRandom.seed(67)
        let headDimension = 128
        let keyValueHeads = 8
        let length = 3
        let offset = 7
        let queryWeight = MLXRandom.uniform(
            low: 0.75,
            high: 1.25,
            [headDimension]
        ).asType(.bfloat16)
        let keyWeight = MLXRandom.uniform(
            low: 0.75,
            high: 1.25,
            [headDimension]
        ).asType(.bfloat16)

        func verify(
            kind: LagunaFusedPrefill.QKNormRoPEKind,
            queryHeads: Int,
            parameters: LagunaRopeParameters
        ) throws {
            let rawQueries = MLXRandom.uniform(
                low: -0.5,
                high: 0.5,
                [1, length, queryHeads * headDimension]
            ).asType(.bfloat16)
            let rawKeys = MLXRandom.uniform(
                low: -0.5,
                high: 0.5,
                [1, length, keyValueHeads * headDimension]
            ).asType(.bfloat16)
            let rope = LagunaRoPE(headDim: headDimension, parameters: parameters)
            let atlas = try XCTUnwrap(rope.angleAtlas(
                length: LagunaFusedPrefill.ropeAngleAtlasLength
            ))
            let stockQueries = rope(
                MLXFast.rmsNorm(
                    rawQueries.reshaped(1, length, queryHeads, headDimension),
                    weight: queryWeight,
                    eps: 0.000_001
                ).transposed(0, 2, 1, 3),
                offset: offset
            )
            let stockKeys = rope(
                MLXFast.rmsNorm(
                    rawKeys.reshaped(1, length, keyValueHeads, headDimension),
                    weight: keyWeight,
                    eps: 0.000_001
                ).transposed(0, 2, 1, 3),
                offset: offset
            )
            let fused = try XCTUnwrap(LagunaFusedPrefill.qkNormRoPE(
                kind: kind,
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                queryWeight: queryWeight,
                keyWeight: keyWeight,
                angleAtlas: atlas,
                offset: offset,
                length: length
            ))

            MLX.eval(stockQueries, stockKeys, fused.queries, fused.keys)
            let queryDifference = MLX.max(
                MLX.abs(stockQueries.asType(.float32) - fused.queries.asType(.float32))
            ).item(Float.self)
            let keyDifference = MLX.max(
                MLX.abs(stockKeys.asType(.float32) - fused.keys.asType(.float32))
            ).item(Float.self)
            XCTAssertEqual(queryDifference, 0)
            XCTAssertEqual(keyDifference, 0)
        }

        try verify(
            kind: .sliding,
            queryHeads: 64,
            parameters: LagunaRopeParameters(
                ropeType: "default",
                ropeTheta: 10_000,
                partialRotaryFactor: 1
            )
        )
        try verify(
            kind: .fullYaRN,
            queryHeads: 48,
            parameters: LagunaRopeParameters(
                ropeType: "yarn",
                ropeTheta: 500_000,
                factor: 32,
                originalMaxPositionEmbeddings: 8_192,
                betaSlow: 1,
                betaFast: 64,
                attentionFactor: 1,
                partialRotaryFactor: 0.5
            )
        )
    }

    func testFusedGatherNVFP4SwiGLUMatchesNativeGathers() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The fused NVFP4 routed MoE kernel requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(55)
        let expertCount = 4
        let tokenCount = 2
        let topK = 3
        let routeCount = tokenCount * topK
        let inputDimensions = 512
        let outputDimensions = 64
        let groupSize = 16
        let bits = 4
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
            ),
            groupSize: groupSize,
            bits: bits,
            mode: .nvfp4
        )
        let up = MLX.quantized(
            MLXRandom.uniform(
                low: -0.25,
                high: 0.25,
                [expertCount, outputDimensions, inputDimensions]
            ),
            groupSize: groupSize,
            bits: bits,
            mode: .nvfp4
        )
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
            biases: gate.biases,
            rhsIndices: flattenedIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .nvfp4,
            sortedIndices: false
        )
        let upOutput = portableGatherQuantizedMM(
            routedInput,
            up.wq,
            scales: up.scales,
            biases: up.biases,
            rhsIndices: flattenedIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .nvfp4,
            sortedIndices: false
        )
        let reference = MLXNN.silu(gateOutput) * upOutput
        let variants = try [1, 2, 4].map { rowsPerSIMDGroup in
            (
                rowsPerSIMDGroup,
                try XCTUnwrap(
                    RoutedMoERouting.fusedGatherNVFP4SwiGLU(
                        input,
                        gateWeight: gate.wq,
                        gateScales: gate.scales,
                        upWeight: up.wq,
                        upScales: up.scales,
                        expertIndices: indices,
                        topK: topK,
                        groupSize: groupSize,
                        bits: bits,
                        rowsPerSIMDGroup: rowsPerSIMDGroup
                    )
                )
            )
        }
        MLX.eval(reference)

        for (rowsPerSIMDGroup, actual) in variants {
            MLX.eval(actual)
            XCTAssertEqual(actual.shape, reference.shape)
            let maximumDifference = MLX.max(
                MLX.abs(reference.asType(.float32) - actual.asType(.float32))
            ).item(Float.self)
            XCTAssertEqual(
                maximumDifference,
                0,
                "rowsPerSIMDGroup=\(rowsPerSIMDGroup)"
            )
        }
    }

    func testDecodeNVFP4RowsPerSIMDGroupParsing() {
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.defaultDecodeRowsPerSIMDGroup(
                architecture: "applegpu_g17s"
            ),
            2
        )
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.defaultDecodeRowsPerSIMDGroup(
                architecture: "applegpu_g16s"
            ),
            4
        )
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup(
                hiddenSize: 2_048,
                intermediateSize: 512,
                topK: 8,
                xsCandidate: 2
            ),
            2
        )
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup(
                hiddenSize: 3_072,
                intermediateSize: 1_024,
                topK: 8,
                xsCandidate: 2
            ),
            4
        )
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup("1", default: 4),
            1
        )
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup("2", default: 4),
            2
        )
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup("4", default: 2),
            4
        )
        XCTAssertEqual(
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup("3", default: 4),
            4
        )
    }

    func testNativeAffineQKVLayerCountParsing() {
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLayerCount(nil, default: 28),
            28
        )
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLayerCount("16", default: 28),
            16
        )
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLayerCount("-1", default: 28),
            0
        )
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLayerCount("41", default: 28),
            40
        )
        XCTAssertEqual(
            LagunaGraphAccelerationPolicy.parseLayerCount("bad", default: 28),
            28
        )
    }

    func testNativeAffineWeightBuildsGroup32SideLayout() throws {
        MLXRandom.seed(61)
        let weight = MLXRandom.uniform(
            low: -0.25,
            high: 0.25,
            [64, 32]
        ).asType(.bfloat16)
        let affine = try XCTUnwrap(lagunaNativeAffineWeight(weight))
        MLX.eval(affine.arrays)

        XCTAssertEqual(affine.originalShape, [64, 32])
        XCTAssertEqual(affine.packedCodes.shape, [64, 8])
        XCTAssertEqual(affine.scales.shape, [64, 1])
        XCTAssertEqual(affine.biases.shape, [64, 1])
        XCTAssertEqual(affine.packedCodes.dtype, .uint32)
        XCTAssertEqual(affine.scales.dtype, .bfloat16)
        XCTAssertEqual(affine.biases.dtype, .bfloat16)
    }

    func testNativeAffineQKVPreparesAndRunsXSDecodeShape() throws {
        guard LagunaGraphAccelerationPolicy.nativeAffineQKVEnabled else {
            throw XCTSkip("Set MERERUN_LAGUNA_NATIVE_AFFINE_QKV=1 to exercise the side layout.")
        }
        MLXRandom.seed(64)
        let attention = LagunaAttention(config: try makeXSAttentionConfig(), layerIndex: 0)
        attention.update(parameters: attention.parameters().mapValues {
            $0.asType(.bfloat16)
        })
        let prepared = attention.prepareNativeAffineQKV()
        guard prepared.count == 3 else {
            XCTFail("Expected one packed QKV side layout.")
            return
        }
        MLX.eval(prepared)

        XCTAssertEqual(prepared[0].shape, [10_240, 512])
        XCTAssertEqual(prepared[1].shape, [10_240, 64])
        XCTAssertEqual(prepared[2].shape, [10_240, 64])

        XCTAssertTrue(
            attention.prepareNativeAffineQKV().isEmpty,
            "Preparation must retain exactly one side layout per attention layer."
        )

        let input = MLXRandom.uniform(
            low: -0.5,
            high: 0.5,
            [1, 1, 2_048]
        ).asType(.bfloat16)
        let output = attention(input, cache: nil)
        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 1, 2_048])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))

        for _ in 0..<4 {
            autoreleasepool {
                MLX.eval(attention(input, cache: nil))
            }
        }
        Memory.clearCache()
        let activeBefore = Memory.activeMemory
        for _ in 0..<32 {
            autoreleasepool {
                MLX.eval(attention(input, cache: nil))
            }
        }
        Memory.clearCache()
        XCTAssertLessThanOrEqual(
            Memory.activeMemory,
            activeBefore + 1_048_576,
            "Repeated affine-QKV decode must not retain per-token MLX buffers."
        )
    }

    func testTerminalPrefillProjectionBanksPreserveLastAttentionRow() throws {
        MLXRandom.seed(65)
        let attention = LagunaAttention(config: try makeXSAttentionConfig(), layerIndex: 0)
        attention.update(parameters: attention.parameters().mapValues {
            $0.asType(.bfloat16)
        })
        let prepared = attention.prepareTerminalPrefillProjectionWeights(enabled: true)
        guard prepared.count == 2 else {
            XCTFail("Expected terminal [Q; gate] and [K; V] side banks.")
            return
        }
        MLX.eval(prepared)

        XCTAssertEqual(prepared[0].shape, [8_256, 2_048])
        XCTAssertEqual(prepared[1].shape, [2_048, 2_048])
        XCTAssertTrue(
            attention.prepareTerminalPrefillProjectionWeights(enabled: true).isEmpty,
            "Preparation must retain exactly one terminal projection-bank pair."
        )

        let input = MLXRandom.uniform(
            low: -0.25,
            high: 0.25,
            [1, 4, 2_048]
        ).asType(.bfloat16)
        let fullCache = Gemma4SlidingKVCache(maxSize: 8)
        let terminalReferenceCache = Gemma4SlidingKVCache(maxSize: 8)
        let terminalBankedCache = Gemma4SlidingKVCache(maxSize: 8)
        let full = attention(input, cache: fullCache)[0..., 3..., 0...]
        let terminalReference = attention.callLastPrefillRow(
            input,
            cache: terminalReferenceCache,
            useProjectionBanks: false
        )
        let terminalBanked = attention.callLastPrefillRow(
            input,
            cache: terminalBankedCache,
            useProjectionBanks: true
        )
        MLX.eval(full, terminalReference, terminalBanked)

        XCTAssertEqual(fullCache.offset, 4)
        XCTAssertEqual(terminalReferenceCache.offset, 4)
        XCTAssertEqual(terminalBankedCache.offset, 4)
        XCTAssertEqual(full.shape, terminalBanked.shape)
        let bankDifference = MLX.max(
            MLX.abs(
                terminalReference.asType(.float32)
                    - terminalBanked.asType(.float32)
            )
        ).item(Float.self)
        XCTAssertEqual(bankDifference, 0)
        let terminalDifference = MLX.max(
            MLX.abs(full.asType(.float32) - terminalBanked.asType(.float32))
        ).item(Float.self)
        XCTAssertLessThanOrEqual(terminalDifference, 0.0005)

        Memory.clearCache()
        let activeBefore = Memory.activeMemory
        for _ in 0..<8 {
            autoreleasepool {
                MLX.eval(attention.callLastPrefillRow(
                    input,
                    cache: nil,
                    useProjectionBanks: true
                ))
            }
        }
        Memory.clearCache()
        XCTAssertLessThanOrEqual(
            Memory.activeMemory,
            activeBefore + 1_048_576,
            "Terminal prefill must not retain per-forward MLX buffers."
        )
    }

    func testLagunaXSFusedDownResidualPreservesZeroBranch() throws {
        #if os(macOS)
        let architecture = GPU.deviceInfo().architecture
        guard Device.defaultDevice().deviceType == .gpu,
              ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26,
              architecture == "applegpu_g16s" || architecture == "applegpu_g17s" else {
            throw XCTSkip("The Laguna XS fused down kernel requires M4 Max or M5 Max on macOS 26.")
        }
        MLXRandom.seed(62)
        let residual = MLXRandom.uniform(
            low: -1,
            high: 1,
            [1, 1, 2_048]
        ).asType(.bfloat16)
        let actual = try XCTUnwrap(
            RoutedMoERouting.fusedLagunaXSRoutedSharedDownResidual(
                routedActivated: MLXArray.zeros([8, 1, 512], dtype: .bfloat16),
                routedDownWeight: MLXArray.zeros(
                    [256, 2_048, 64],
                    dtype: .uint32
                ),
                routedDownScales: MLXArray.zeros(
                    [256, 2_048, 32],
                    dtype: .uint8
                ),
                indices: MLXArray((0..<8).map(UInt32.init)).reshaped([1, 1, 8]),
                routerWeights: MLXArray.ones([1, 1, 8], dtype: .bfloat16),
                sharedActivated: MLXArray.zeros([1, 1, 512], dtype: .bfloat16),
                sharedDownWeight: MLXArray.zeros([2_048, 64], dtype: .uint32),
                sharedDownScales: MLXArray.zeros([2_048, 32], dtype: .uint8),
                residual: residual
            )
        )
        MLX.eval(residual, actual)

        XCTAssertEqual(actual.shape, residual.shape)
        XCTAssertEqual(
            MLX.max(
                MLX.abs(actual.asType(.float32) - residual.asType(.float32))
            ).item(Float.self),
            0
        )
        #else
        throw XCTSkip("The Laguna XS fused down kernel is Metal-only.")
        #endif
    }

    func testLagunaXSFusedDownResidualMatchesNativeOperations() throws {
        #if os(macOS)
        let architecture = GPU.deviceInfo().architecture
        guard Device.defaultDevice().deviceType == .gpu,
              ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26,
              architecture == "applegpu_g16s" || architecture == "applegpu_g17s" else {
            throw XCTSkip("The Laguna XS fused down kernel requires M4 Max or M5 Max on macOS 26.")
        }
        MLXRandom.seed(63)
        let routedBase = MLX.quantized(
            MLXRandom.uniform(
                low: -0.2,
                high: 0.2,
                [1, 2_048, 512]
            ),
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
        let routedWeight = MLX.repeated(routedBase.wq, count: 256, axis: 0)
        let routedScales = MLX.repeated(routedBase.scales, count: 256, axis: 0)
        let shared = MLX.quantized(
            MLXRandom.uniform(
                low: -0.2,
                high: 0.2,
                [2_048, 512]
            ),
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
        let routedActivated = MLXRandom.uniform(
            low: -0.5,
            high: 0.5,
            [8, 1, 512]
        ).asType(.bfloat16)
        let sharedActivated = MLXRandom.uniform(
            low: -0.5,
            high: 0.5,
            [1, 1, 512]
        ).asType(.bfloat16)
        let indices = MLXArray((0..<8).map(UInt32.init)).reshaped([1, 1, 8])
        let routerWeights = MLXRandom.uniform(
            low: 0.01,
            high: 0.3,
            [1, 1, 8]
        ).asType(.bfloat16)
        let residual = MLXRandom.uniform(
            low: -1,
            high: 1,
            [1, 1, 2_048]
        ).asType(.bfloat16)

        let routedRows = portableGatherQuantizedMM(
            routedActivated,
            routedWeight,
            scales: routedScales,
            biases: nil,
            rhsIndices: indices.reshaped([8]),
            transpose: true,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            sortedIndices: false
        ).reshaped([1, 1, 8, 2_048])
        let routed = (
            routedRows
                * MLX.expandedDimensions(routerWeights, axis: routerWeights.ndim)
        ).sum(axis: -2) * Float(2.5)
        let sharedOutput = MLX.quantizedMM(
            sharedActivated,
            shared.wq,
            scales: shared.scales,
            biases: nil,
            transpose: true,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
        let reference = residual + (routed + sharedOutput)
        let actual = try XCTUnwrap(
            RoutedMoERouting.fusedLagunaXSRoutedSharedDownResidual(
                routedActivated: routedActivated,
                routedDownWeight: routedWeight,
                routedDownScales: routedScales,
                indices: indices,
                routerWeights: routerWeights,
                sharedActivated: sharedActivated,
                sharedDownWeight: shared.wq,
                sharedDownScales: shared.scales,
                residual: residual
            )
        )
        MLX.eval(reference, actual)

        XCTAssertEqual(actual.shape, reference.shape)
        XCTAssertEqual(
            MLX.max(
                MLX.abs(actual.asType(.float32) - reference.asType(.float32))
            ).item(Float.self),
            0
        )
        #else
        throw XCTSkip("The Laguna XS fused down kernel is Metal-only.")
        #endif
    }

    func testFusedSortedNVFP4SwiGLUMatchesNativeGathers() throws {
        guard Device.defaultDevice().deviceType == .gpu,
              GPU.deviceInfo().architecture == "applegpu_g16s" else {
            throw XCTSkip("The fused sorted kernel requires an M4 Max GPU.")
        }
        MLXRandom.seed(56)
        let expertCount = 4
        let inputDimensions = 512
        let outputDimensions = 512
        let groupSize = 16
        let bits = 4
        let gate = MLX.quantized(
            MLXRandom.uniform(
                low: -0.25,
                high: 0.25,
                [expertCount, outputDimensions, inputDimensions]
            ),
            groupSize: groupSize,
            bits: bits,
            mode: .nvfp4
        )
        let up = MLX.quantized(
            MLXRandom.uniform(
                low: -0.25,
                high: 0.25,
                [expertCount, outputDimensions, inputDimensions]
            ),
            groupSize: groupSize,
            bits: bits,
            mode: .nvfp4
        )
        for routeCount in [64, 512, 1_024] {
            let input = MLXRandom.uniform(
                low: -0.5,
                high: 0.5,
                [routeCount, 1, inputDimensions]
            ).asType(.bfloat16)
            let rowsPerExpert = max(1, routeCount / expertCount)
            let indices = MLXArray(
                (0..<routeCount).map {
                    Int32(min(expertCount - 1, $0 / rowsPerExpert))
                }
            )
            let gateOutput = portableGatherQuantizedMM(
                input,
                gate.wq,
                scales: gate.scales,
                biases: gate.biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .nvfp4,
                sortedIndices: true
            )
            let upOutput = portableGatherQuantizedMM(
                input,
                up.wq,
                scales: up.scales,
                biases: up.biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .nvfp4,
                sortedIndices: true
            )
            let reference = MLXNN.silu(gateOutput) * upOutput
            let actual = try XCTUnwrap(
                RoutedMoERouting.fusedSortedNVFP4SwiGLU(
                    input,
                    gateWeight: gate.wq,
                    gateScales: gate.scales,
                    upWeight: up.wq,
                    upScales: up.scales,
                    sortedExpertIndices: indices,
                    groupSize: groupSize,
                    bits: bits
                )
            )
            MLX.eval(reference, actual)

            XCTAssertEqual(actual.shape, reference.shape)
            let maximumDifference = MLX.max(
                MLX.abs(reference.asType(.float32) - actual.asType(.float32))
            ).item(Float.self)
            XCTAssertEqual(maximumDifference, 0, "routeCount=\(routeCount)")
        }
    }

    func testSortedNVFP4DownProjectionMatchesNativeGather() throws {
        guard Device.defaultDevice().deviceType == .gpu,
              GPU.deviceInfo().architecture == "applegpu_g16s" else {
            throw XCTSkip("The expert-aligned sorted kernel requires an M4 Max GPU.")
        }
        MLXRandom.seed(58)
        let expertCount = 4
        let inputDimensions = 512
        let outputDimensions = 2_048
        let groupSize = 16
        let bits = 4
        let projection = MLX.quantized(
            MLXRandom.uniform(
                low: -0.25,
                high: 0.25,
                [expertCount, outputDimensions, inputDimensions]
            ),
            groupSize: groupSize,
            bits: bits,
            mode: .nvfp4
        )
        for routeCount in [64, 512, 1_024] {
            let input = MLXRandom.uniform(
                low: -0.5,
                high: 0.5,
                [routeCount, 1, inputDimensions]
            ).asType(.bfloat16)
            let rowsPerExpert = max(1, routeCount / expertCount)
            let indices = MLXArray(
                (0..<routeCount).map {
                    Int32(min(expertCount - 1, $0 / rowsPerExpert))
                }
            )
            let reference = portableGatherQuantizedMM(
                input,
                projection.wq,
                scales: projection.scales,
                biases: projection.biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .nvfp4,
                sortedIndices: true
            )
            let actual = try XCTUnwrap(RoutedMoERouting.sortedNVFP4Projection(
                input,
                weight: projection.wq,
                scales: projection.scales,
                sortedExpertIndices: indices,
                groupSize: groupSize,
                bits: bits
            ))
            MLX.eval(reference, actual)

            XCTAssertEqual(actual.shape, reference.shape)
            let maximumDifference = MLX.max(
                MLX.abs(reference.asType(.float32) - actual.asType(.float32))
            ).item(Float.self)
            XCTAssertEqual(maximumDifference, 0, "routeCount=\(routeCount)")
        }
    }

    func testPermutationInversionMatchesSecondSort() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The permutation inversion requires a GPU.")
        }
        let routeCount = 513
        let order = MLXArray(
            (0..<routeCount).map {
                Int32(($0 * 17) % routeCount)
            }
        )
        let reference = argSort(order, axis: 0).asType(.int32)
        let actual = try XCTUnwrap(
            RoutedMoERouting.invertPermutation(order)
        )
        MLX.eval(reference, actual)

        XCTAssertEqual(actual.shape, reference.shape)
        let maximumDifference = MLX.max(
            MLX.abs(reference - actual)
        ).item(Int32.self)
        XCTAssertEqual(maximumDifference, 0)
    }

    func testRankedLagunaPrefillRouteStagingMatchesNativeGathers() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Ranked Laguna route staging requires a GPU.")
        }
        MLXRandom.seed(43)
        let tokenCount = 512
        let hiddenSize = 2_048
        let topK = 8
        let routeCount = tokenCount * topK
        let input = MLXRandom.uniform(
            low: -1,
            high: 1,
            [tokenCount, hiddenSize]
        ).asType(.bfloat16)
        let flatIndices = MLXArray(
            (0..<routeCount).map {
                UInt32(($0 * 73 + $0 / topK * 11) % 256)
            }
        )
        let order = argSort(flatIndices, axis: 0)
        let referenceInput = input
            .take(order.floorDivide(topK), axis: 0)
            .reshaped([routeCount, 1, hiddenSize])
        let referenceIndices = flatIndices.take(order, axis: 0)
        let referenceInverse = argSort(order, axis: 0)
        let actual = try XCTUnwrap(
            RoutedMoERouting.stageRankedLagunaPrefillRoute(
                input,
                flatIndices: flatIndices,
                order: order,
                topK: topK
            )
        )
        MLX.eval(
            referenceInput,
            referenceIndices,
            referenceInverse,
            actual.sortedInput,
            actual.sortedIndices,
            actual.inverseOrder
        )

        XCTAssertEqual(actual.sortedInput.shape, referenceInput.shape)
        XCTAssertEqual(
            MLX.max(
                MLX.abs(
                    actual.sortedInput.asType(.float32)
                        - referenceInput.asType(.float32)
                )
            ).item(Float.self),
            0
        )
        XCTAssertEqual(
            MLX.max(
                MLX.abs(
                    actual.sortedIndices.asType(.int32)
                        - referenceIndices.asType(.int32)
                )
            ).item(Int32.self),
            0
        )
        XCTAssertEqual(
            MLX.max(
                MLX.abs(
                    actual.inverseOrder.asType(.int32)
                        - referenceInverse.asType(.int32)
                )
            ).item(Int32.self),
            0
        )
    }

    func testSortedMoERoutingMatchesUnsortedRouting() throws {
        MLXRandom.seed(41)
        let switchGLU = LagunaSwitchGLU(config: try makeConfig())
        let sequenceLength = 32
        let hiddenSize = 8
        let topK = 2
        let input = MLXArray(
            (0..<(sequenceLength * hiddenSize)).map {
                Float(($0 % 29) - 14) / 17
            },
            [1, sequenceLength, hiddenSize]
        )
        let indices = MLXArray(
            (0..<(sequenceLength * topK)).map {
                Int32(($0 * 2 + 1) % 3)
            },
            [1, sequenceLength, topK]
        )

        let unsorted = switchGLU.unsorted(input, indices: indices)
        let sorted = switchGLU.sorted(input, indices: indices)
        MLX.eval(unsorted, sorted)

        XCTAssertEqual(sorted.shape, unsorted.shape)
        let maximumDifference = MLX.max(
            MLX.abs(sorted.asType(.float32) - unsorted.asType(.float32))
        ).item(Float.self)
        XCTAssertLessThan(maximumDifference, 0.0001)
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

    func testTrainingLogitsMatchFullVocabularyProjectionAtSelectedPositions() throws {
        MLXRandom.seed(74)
        let model = LagunaCausalLM(config: try makeConfig())
        let input = MLXArray([1, 2, 3, 4, 5, 6]).reshaped(2, 3)
        let positions = MLXArray([Int32(1), Int32(3), Int32(5)])
        let full = model(input).reshaped(-1, model.config.vocabSize)
        let expected = take(full, positions, axis: 0)
        let gathered = model.trainingLogits(
            inputIDs: input,
            flatTargetPositions: positions
        )
        MLX.eval(expected, gathered)

        XCTAssertEqual(gathered.shape, [3, model.config.vocabSize])
        XCTAssertEqual(
            MLX.max(MLX.abs(expected - gathered)).item(Float.self),
            0,
            accuracy: 1e-6
        )
    }

    func testNativeLagunaAdapterWeightsRoundTripIntoRuntimePaths() async throws {
        MLXRandom.seed(75)
        let source = LagunaCausalLM(config: try makeConfig())
        let sourceLayers = try LagunaTextLoRAInjector.inject(
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
        let adapterURL = directory.appendingPathComponent("laguna.safetensors")
        try LoRASafetensorsWriter.save(
            loraLayers: sourceLayers,
            to: adapterURL,
            metadata: ["format": TextLoRATrainingManifest.lagunaFormat]
        )

        MLXRandom.seed(75)
        let target = LagunaCausalLM(config: try makeConfig())
        let baseline = target(MLXArray([1, 2, 3]).reshaped(1, 3))
        MLX.eval(baseline)
        let report = try await LagunaTextLoRAAdapter.apply(
            .local(path: adapterURL.path, scale: 1),
            to: target
        )
        let adapted = target(MLXArray([1, 2, 3]).reshaped(1, 3))
        MLX.eval(adapted)

        XCTAssertEqual(report.matchedLayerCount, sourceLayers.count)
        XCTAssertEqual(report.injectedLayerCount, sourceLayers.count)
        XCTAssertGreaterThan(
            MLX.max(MLX.abs(adapted - baseline)).item(Float.self),
            0
        )
    }

    func testNativeLagunaAdapterDerivesNonDefaultTargetPaths() async throws {
        MLXRandom.seed(751)
        let source = LagunaCausalLM(config: try makeConfig())
        let sourceLayers = try LagunaTextLoRAInjector.inject(
            into: source,
            rank: 2,
            alpha: 4,
            targetSuffixes: ["lm_head"]
        )
        for layer in sourceLayers.values {
            layer.loraDown = MLXArray.ones(like: layer.loraDown) * 0.125
            layer.loraUp = MLXArray.ones(like: layer.loraUp) * 0.25
        }

        let directory = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: directory) }
        let adapterURL = directory.appendingPathComponent("laguna-lm-head.safetensors")
        try LoRASafetensorsWriter.save(
            loraLayers: sourceLayers,
            to: adapterURL,
            metadata: ["format": TextLoRATrainingManifest.lagunaFormat]
        )

        MLXRandom.seed(751)
        let target = LagunaCausalLM(config: try makeConfig())
        let input = MLXArray([1, 2, 3]).reshaped(1, 3)
        let baseline = target(input)
        MLX.eval(baseline)
        let report = try await LagunaTextLoRAAdapter.apply(
            .local(path: adapterURL.path, scale: 1),
            to: target
        )
        let adapted = target(input)
        MLX.eval(adapted)

        XCTAssertEqual(report.matchedLayerCount, 1)
        XCTAssertEqual(report.injectedLayerCount, 1)
        XCTAssertGreaterThan(
            MLX.max(MLX.abs(adapted - baseline)).item(Float.self),
            0
        )
    }

    func testNativeLagunaTrainerUpdatesAttentionLoRA() throws {
        MLXRandom.seed(76)
        let model = LagunaCausalLM(config: try makeConfig())
        let layers = try LagunaTextLoRAInjector.inject(
            into: model,
            rank: 2
        )
        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [
                TextSFTTokenizedExample(
                    inputTokenIds: [1, 2, 3],
                    labelTokenIds: [2, 3, 4],
                    lossMask: [0, 1, 1]
                ),
            ],
            config: TextLoRATrainingConfig(
                trainingSteps: 1,
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
            model(inputIDs)
        }

        XCTAssertEqual(report.steps, 1)
        XCTAssertEqual(report.layerCount, 8)
        XCTAssertNotNil(report.finalLoss)
        let updatedLayers = layers.values.filter {
            MLX.sum(MLX.abs($0.loraUp)).item(Float.self) > 0
        }
        XCTAssertFalse(updatedLayers.isEmpty)
    }

    func testNativeLagunaTrainerCrossesQuantizedSharedExpertPath() throws {
        MLXRandom.seed(77)
        let model = LagunaCausalLM(
            config: try makeConfig(quantizedSharedExperts: true),
            quantizedSharedExperts: true
        )
        let sparse = try XCTUnwrap(model.model.layers[1].mlp as? LagunaSparseMoE)
        for projection in [
            sparse.switchMLP.gateProj,
            sparse.switchMLP.upProj,
            sparse.switchMLP.downProj,
        ] {
            try projection.update(
                parameters: ModuleParameters.unflattened([
                    (
                        "scales",
                        MLXArray.ones(
                            projection.scales?.shape ?? [],
                            dtype: .uint8
                        )
                    ),
                ]),
                verify: .none
            )
        }
        sparse.sharedExpert.update(
            modules: ModuleChildren.unflattened([
                (
                    "gate_proj",
                    makePackedNVFP4Linear(
                        inputDimensions: 16,
                        outputDimensions: 16
                    )
                ),
                (
                    "up_proj",
                    makePackedNVFP4Linear(
                        inputDimensions: 16,
                        outputDimensions: 16
                    )
                ),
                (
                    "down_proj",
                    makePackedNVFP4Linear(
                        inputDimensions: 16,
                        outputDimensions: 16
                    )
                ),
            ])
        )
        XCTAssertTrue(sparse.sharedExpert.gateProj is QuantizedLinear)
        XCTAssertTrue(sparse.sharedExpert.upProj is QuantizedLinear)
        XCTAssertTrue(sparse.sharedExpert.downProj is QuantizedLinear)

        let layers = try LagunaTextLoRAInjector.inject(
            into: model,
            rank: 2
        )
        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [
                TextSFTTokenizedExample(
                    inputTokenIds: [1, 2, 3],
                    labelTokenIds: [2, 3, 4],
                    lossMask: [0, 1, 1]
                ),
            ],
            config: TextLoRATrainingConfig(
                trainingSteps: 1,
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
            model(inputIDs)
        }

        XCTAssertEqual(report.steps, 1)
        XCTAssertNotNil(report.finalLoss)
        XCTAssertTrue(report.finalLoss?.isFinite == true)
        XCTAssertTrue(layers.values.contains {
            MLX.sum(MLX.abs($0.loraUp)).item(Float.self) > 0
        })
    }

    private func makePackedNVFP4Linear(
        inputDimensions: Int,
        outputDimensions: Int
    ) -> QuantizedLinear {
        QuantizedLinear(
            weight: MLXArray.zeros(
                [outputDimensions, inputDimensions * 4 / 32],
                dtype: .uint32
            ),
            bias: nil,
            scales: MLXArray.ones(
                [outputDimensions, inputDimensions / 16],
                dtype: .uint8
            ),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
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
            HFSafetensorsIndex.self,
            from: Data(contentsOf: rootURL.appending(path: "model.safetensors.index.json"))
        )
        let modelKeys = Set(LagunaCausalLM(
            config: config,
            quantizedSharedExperts: LagunaResources.hasQuantizedSharedExperts(index)
        ).parameters().flattened().map(\.0))
        let checkpointKeys = Set(index.weightMap.keys)
        let missingModelKeys = checkpointKeys.subtracting(modelKeys)
        let derivedRuntimeKeys = modelKeys.subtracting(checkpointKeys)

        XCTAssertFalse(checkpointKeys.isEmpty)
        XCTAssertTrue(missingModelKeys.isEmpty, "Missing checkpoint parameters: \(missingModelKeys.sorted())")
        let derivedFrequencyCount = config.layerTypes.indices.filter { index in
            config.ropeParameters(layerIndex: index).ropeType == "yarn"
        }.count
        XCTAssertEqual(derivedRuntimeKeys.count, derivedFrequencyCount)
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
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: rootURL.appending(path: "model.safetensors.index.json"))
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

        let model = LagunaCausalLM(
            config: config,
            quantizedSharedExperts: LagunaResources.hasQuantizedSharedExperts(index)
        )
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

    func testLastPositionOnlyMatchesFullForwardLastRow() throws {
        MLXRandom.seed(9)
        let model = LagunaCausalLM(config: try makeConfig())
        let tokens = MLXArray([1, 7, 4, 9]).reshaped(1, 4)
        let full = model(tokens)
        let last = model.forward(
            tokens,
            lastPositionOnly: true,
            terminalPrefillRowEnabled: true
        ).logits
        MLX.eval(full, last)

        let expected = full[0..., 3..., 0...]
        let maximumDifference = zip(
            expected.asArray(Float.self),
            last.asArray(Float.self)
        ).map { abs($0 - $1) }.max() ?? 0
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
        let forcedTerminalWithFinalCapture = model.forward(
            tokens,
            captureLayerIndices: [1],
            lastPositionOnly: true,
            terminalPrefillRowEnabled: true
        )
        MLX.eval(
            [baseline, captured.logits, forcedTerminalWithFinalCapture.logits]
                + Array(captured.capturedHiddenStates.values)
                + Array(forcedTerminalWithFinalCapture.capturedHiddenStates.values)
        )

        XCTAssertEqual(captured.capturedHiddenStates.keys.sorted(), [0, 1])
        XCTAssertEqual(captured.capturedHiddenStates[0]?.shape, [1, 3, 8])
        XCTAssertEqual(captured.capturedHiddenStates[1]?.shape, [1, 3, 8])
        XCTAssertEqual(
            forcedTerminalWithFinalCapture.capturedHiddenStates[1]?.shape,
            [1, 3, 8]
        )
        let maximumDifference = zip(
            baseline.asArray(Float.self),
            captured.logits.asArray(Float.self)
        ).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maximumDifference, 0.0001)
        let expectedLast = baseline[0..., 2..., 0...]
        let forcedDifference = MLX.max(
            MLX.abs(
                expectedLast.asType(.float32)
                    - forcedTerminalWithFinalCapture.logits.asType(.float32)
            )
        ).item(Float.self)
        XCTAssertLessThan(forcedDifference, 0.0001)
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
