import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore
@testable import MereRunLagunaModel
@testable import MereRunGemmaModel

final class LagunaModelTests: MereRunCoreTestCase {
    private func makeConfig(
        quantizedSharedExperts: Bool = false,
        numHiddenLayers: Int = 2
    ) throws -> LagunaConfig {
        let hiddenSize = quantizedSharedExperts ? 16 : 8
        let headDimension = quantizedSharedExperts ? 8 : 4
        let sharedExpertSize = quantizedSharedExperts ? 16 : 5
        let layerTypes = (0..<numHiddenLayers).map {
            $0.isMultiple(of: 2) ? "full_attention" : "sliding_attention"
        }
        let mlpLayerTypes = (0..<numHiddenLayers).map { $0 == 0 ? "dense" : "sparse" }
        var object: [String: Any] = [
            "model_type": "laguna",
            "vocab_size": 32,
            "hidden_size": hiddenSize,
            "intermediate_size": 16,
            "num_hidden_layers": numHiddenLayers,
            "num_attention_heads": 2,
            "num_attention_heads_per_layer": Array(repeating: 2, count: numHiddenLayers),
            "num_key_value_heads": 1,
            "head_dim": headDimension,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "attention_bias": false,
            "gating": "per-head",
            "layer_types": layerTypes,
            "sliding_window": 8,
            "mlp_layer_types": mlpLayerTypes,
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
        let inputTokenIds = (0..<40).map { ($0 % 8) + 1 }
        let labelTokenIds = Array(inputTokenIds.dropFirst()) + [1]
        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [
                TextSFTTokenizedExample(
                    inputTokenIds: inputTokenIds,
                    labelTokenIds: labelTokenIds,
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
            model(inputIDs)
        }

        XCTAssertEqual(report.steps, 2)
        XCTAssertEqual(report.layerCount, 8)
        XCTAssertNotNil(report.finalLoss)
        let updatedLayers = layers.values.filter {
            MLX.sum(MLX.abs($0.loraUp)).item(Float.self) > 0
        }
        XCTAssertFalse(updatedLayers.isEmpty)
    }

    func testNativeLagunaTrainerDisablesInferenceAsyncLadderDuringGradientTrace() throws {
        MLXRandom.seed(761)
        let model = LagunaCausalLM(config: try makeConfig(numHiddenLayers: 8))
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
        XCTAssertEqual(report.layerCount, 32)
        XCTAssertTrue(report.finalLoss?.isFinite == true)
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

        if LagunaMoEAccelerationPolicy.prefillExpertPairwiseScaleReuseEnabled {
            _ = model.prepareRuntimeAcceleration()
            let sparseLayers = model.model.layers.compactMap {
                $0.mlp as? LagunaSparseMoE
            }
            XCTAssertEqual(
                sparseLayers.filter {
                    $0.switchMLP.prefillPairwiseScaleReuseCertified
                }.count,
                sparseLayers.count,
                "Every loaded routed gate/up scale plane must pass the lossless pair certificate."
            )
            XCTAssertEqual(
                sparseLayers.filter {
                    $0.switchMLP.prefillDownPairwiseScaleReuseCertified
                }.count,
                sparseLayers.count,
                "Every loaded routed down scale plane must pass the lossless pair certificate."
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





}
