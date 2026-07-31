import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class InklingTests: MereRunCoreTestCase {
    func testConfigDecodesOfficialNestedTextShapeAndAffineQuantization() throws {
        let config = try JSONDecoder().decode(InklingConfig.self, from: Data(Self.configJSON.utf8))

        XCTAssertEqual(config.modelType, "inkling_mm_model")
        XCTAssertEqual(config.eosTokenIDs, [200_006])
        XCTAssertEqual(config.textConfig.hiddenSize, 8)
        XCTAssertEqual(config.textConfig.localLayerIDs, Set([0]))
        XCTAssertEqual(config.textConfig.denseIntermediateSize, 16)
        XCTAssertEqual(config.textConfig.moeIntermediateSize, 4)
        XCTAssertEqual(config.quantization?.bits, 2)
        XCTAssertEqual(config.quantization?.groupSize, 128)
        XCTAssertEqual(config.quantization?.mode, "affine")
        XCTAssertEqual(config.quantization?.scope, "routed_experts")
    }

    func testConfigPrefersMLXVLMExpertWidthAlias() throws {
        let compatible = Self.configJSON
            .replacingOccurrences(
                of: #""intermediate_size": 4,"#,
                with: #""intermediate_size": 16, "moe_intermediate_size": 4,"#
            )
        let config = try JSONDecoder().decode(InklingConfig.self, from: Data(compatible.utf8))

        XCTAssertEqual(config.textConfig.denseIntermediateSize, 16)
        XCTAssertEqual(config.textConfig.moeIntermediateSize, 4)
    }

    func testPromptRendersInklingReasoningAndRoleChannels() throws {
        let prompt = try InklingTokenizerAndTemplate.renderPrompt(
            messages: [
                ChatMessage(role: .system, content: "Be concise."),
                ChatMessage(role: .user, content: "Hello"),
            ],
            reasoningEffort: 0.9
        )

        XCTAssertEqual(
            prompt,
            "<|message_system|><|content_text|>Be concise.<|end_message|>"
                + "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_user|><|content_text|>Hello<|end_message|>"
                + "<|message_model|>"
        )
    }

    func testPromptUsesInklingToolDeclarationAndResolvesToolResultName() throws {
        let prompt = try InklingTokenizerAndTemplate.renderPrompt(
            messages: [
                ChatMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [
                        ChatMessageToolCall(
                            id: "call-1",
                            name: "read_file",
                            arguments: ["path": .string("a.txt")]
                        ),
                    ]
                ),
                ChatMessage(role: .tool, content: "hello", toolCallID: "call-1"),
            ],
            tools: [
                ToolDefinition(
                    name: "read_file",
                    description: "Read a file",
                    parameters: [
                        "path": ToolParameterProperty(type: "string", description: "File path"),
                    ],
                    required: ["path"]
                ),
            ],
            addGenerationPrompt: false
        )

        XCTAssertTrue(prompt.hasPrefix(
            #"<|message_system|>tool_declare<|content_xml|>[{"description":"Read a file","name":"read_file","parameters":{"properties":{"path":{"description":"File path","type":"string"}},"required":["path"],"type":"object"},"type":"function"}]<|end_message|>"#
        ))
        XCTAssertTrue(prompt.contains(
            #"<|message_model|>read_file<|content_invoke_tool_json|>{"args":{"path":"a.txt"},"name":"read_file"}<|end_message|>"#
        ))
        XCTAssertTrue(prompt.hasSuffix(
            "<|content_model_end_sampling|><|message_tool|>read_file<|content_text|>hello<|end_message|>"
        ))
    }

    func testOutputParserSeparatesReasoningVisibleTextAndToolInvocation() throws {
        let raw = "<|content_thinking|>Check the record.<|end_message|>"
            + "<|message_model|><|content_text|>Done.<|end_message|>"
            + "<|message_model|>write_file<|content_invoke_tool_json|>"
            + #"{"args":{"path":"x.txt"},"name":"write_file"}"#
            + "<|end_message|><|content_model_end_sampling|>"

        let parsed = InklingOutputParser.parse(raw)

        XCTAssertEqual(parsed.reasoning, "Check the record.")
        XCTAssertEqual(parsed.visible, "Done.")
        XCTAssertEqual(parsed.toolCalls, [ToolCall(name: "write_file", arguments: ["path": "x.txt"])])
    }

    func testTinyDenseModelRunsCachedPrefill() throws {
        let config = try JSONDecoder().decode(InklingConfig.self, from: Data(Self.configJSON.utf8))
        let model = InklingLanguageModel(config: config)
        let caches = [InklingLayerCache()]

        let output = model.forwardPrefill(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: caches)
        MLX.eval(output.logits)

        XCTAssertEqual(output.logits.shape, [1, 1, 32])
        XCTAssertEqual(caches[0].attention.offset, 3)
    }

    func testTinySparseModelRunsAffineTwoBitExpertPath() throws {
        let config = try JSONDecoder().decode(
            InklingConfig.self,
            from: Data(Self.sparseConfigJSON.utf8)
        )
        let model = InklingLanguageModel(config: config)

        let logits = model(MLXArray([Int32(1)]).reshaped(1, 1), cache: [InklingLayerCache()])
        MLX.eval(logits)

        XCTAssertEqual(logits.shape, [1, 1, 32])
    }

    func testTrainingLogitsMatchFullProjectionAtSelectedPositions() throws {
        MLXRandom.seed(74)
        let config = try JSONDecoder().decode(InklingConfig.self, from: Data(Self.configJSON.utf8))
        let model = InklingLanguageModel(config: config)
        let input = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(2, 3)
        let positions = MLXArray([Int32(1), Int32(3), Int32(5)])
        let full = model.trainingForward(input).reshaped(-1, config.textConfig.vocabSize)
        let expected = take(full, positions, axis: 0)
        let gathered = model.trainingLogits(
            inputIDs: input,
            flatTargetPositions: positions
        )
        MLX.eval(expected, gathered)

        XCTAssertEqual(gathered.shape, [3, config.textConfig.vocabSize])
        XCTAssertEqual(
            MLX.max(MLX.abs(expected - gathered)).item(Float.self),
            0,
            accuracy: 1e-6
        )
    }

    func testNativeTrainerUpdatesTinkerParityLoRA() throws {
        MLXRandom.seed(75)
        let config = try JSONDecoder().decode(InklingConfig.self, from: Data(Self.configJSON.utf8))
        let model = InklingLanguageModel(config: config)
        let layers = try InklingTextLoRAInjector.inject(into: model, rank: 2)
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

    func testNativeTrainerCrossesAffineTwoBitRoutedExperts() throws {
        MLXRandom.seed(76)
        let config = try JSONDecoder().decode(
            InklingConfig.self,
            from: Data(Self.sparseConfigJSON.utf8)
        )
        let model = InklingLanguageModel(config: config)
        let layers = try InklingTextLoRAInjector.inject(into: model, rank: 2)

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
            model.trainingForward(inputIDs)
        }

        XCTAssertEqual(report.steps, 1)
        XCTAssertTrue(report.finalLoss?.isFinite == true)
        let expertLayers = layers.values.compactMap {
            $0 as? InklingSharedOuterLoRASwitchLinear
        }
        XCTAssertEqual(expertLayers.count, 6)
        XCTAssertTrue(expertLayers.contains {
            MLX.sum(MLX.abs($0.loraUp)).item(Float.self) > 0
        })
    }

    func testDefaultInjectorCoversAttentionMLPExpertsAndUnembedding() throws {
        let config = try JSONDecoder().decode(
            InklingConfig.self,
            from: Data(Self.sparseConfigJSON.utf8)
        )
        let model = InklingLanguageModel(config: config)

        let layers = try InklingTextLoRAInjector.inject(into: model, rank: 2)
        let paths = Set(layers.keys)

        XCTAssertEqual(layers.count, 11)
        XCTAssertTrue(paths.contains("model.layers.0.self_attn.q_proj"))
        XCTAssertTrue(paths.contains("model.layers.0.mlp.switch_mlp.gate_proj"))
        XCTAssertTrue(paths.contains("model.layers.0.mlp.shared_experts.down_proj"))
        XCTAssertTrue(paths.contains("lm_head"))

        let routedGate = try XCTUnwrap(
            layers["model.layers.0.mlp.switch_mlp.gate_proj"]
                as? InklingSharedOuterLoRASwitchLinear
        )
        XCTAssertEqual(routedGate.sharedFactor, .input)
        XCTAssertEqual(routedGate.loraDown.shape, [2, 128])
        XCTAssertEqual(routedGate.loraUp.shape, [4, 128, 2])

        let routedDown = try XCTUnwrap(
            layers["model.layers.0.mlp.switch_mlp.down_proj"]
                as? InklingSharedOuterLoRASwitchLinear
        )
        XCTAssertEqual(routedDown.sharedFactor, .output)
        XCTAssertEqual(routedDown.loraDown.shape, [4, 2, 128])
        XCTAssertEqual(routedDown.loraUp.shape, [128, 2])
    }

    func testNativeAdapterRoundTripsIntoRuntimePaths() async throws {
        MLXRandom.seed(77)
        let config = try JSONDecoder().decode(InklingConfig.self, from: Data(Self.configJSON.utf8))
        let source = InklingLanguageModel(config: config)
        let sourceLayers = try InklingTextLoRAInjector.inject(
            into: source,
            rank: 2,
            alpha: 4
        )
        for layer in sourceLayers.values {
            layer.loraDown = MLXArray.ones(like: layer.loraDown) * 0.125
            layer.loraUp = MLXArray.ones(like: layer.loraUp) * 0.25
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling-adapter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let adapterURL = directory.appendingPathComponent("inkling.safetensors")
        try LoRASafetensorsWriter.save(
            loraLayers: sourceLayers,
            to: adapterURL,
            metadata: ["format": TextLoRATrainingManifest.inklingFormat]
        )

        MLXRandom.seed(77)
        let target = InklingLanguageModel(config: config)
        let input = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let baseline = target(input, cache: nil)
        MLX.eval(baseline)
        let report = try await InklingTextLoRAAdapter.apply(
            .local(path: adapterURL.path, scale: 1),
            to: target
        )
        let adapted = target(input, cache: nil)
        MLX.eval(adapted)

        XCTAssertEqual(report.matchedLayerCount, sourceLayers.count)
        XCTAssertEqual(report.injectedLayerCount, sourceLayers.count)
        XCTAssertGreaterThan(
            MLX.max(MLX.abs(adapted - baseline)).item(Float.self),
            0
        )
    }

    func testSparseModelQuantizesOnlyRoutedExperts() throws {
        let config = try JSONDecoder().decode(
            InklingConfig.self,
            from: Data(Self.sparseConfigJSON.utf8)
        )
        let moe = InklingSparseMoE(config: config.textConfig, quantization: config.quantization)

        XCTAssertNotNil(moe.switchMLP.gateProj.scales)
        XCTAssertEqual(moe.switchMLP.gateProj.weight.dtype, .uint32)
        XCTAssertNil(moe.sharedExperts.gateProj.scales)
        XCTAssertNotEqual(moe.sharedExperts.gateProj.weight.dtype, .uint32)
    }

    func testWeightMapperKeepsOnlyLanguageModelAndTransposesConv() {
        let tower = InklingResources.mapWeightKey("vision_tower.blocks.0.weight")
        XCTAssertTrue(tower.hasPrefix("_ignored."))
        XCTAssertEqual(
            InklingResources.mapWeightKey("language_model.model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        )

        let conv = MLXArray.zeros([8, 1, 4])
        let mapped = InklingResources.mapWeight(
            key: "model.layers.0.attn_sconv.conv.weight",
            value: conv
        )
        XCTAssertEqual(mapped.first?.1.shape, [8, 4, 1])
    }

    func testValidationRequiresEveryShardNamedByIndex() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling-validation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(name).path,
                contents: Data("{}".utf8)
            ))
        }
        let index = #"{"weight_map":{"language_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors","language_model.lm_head.weight":"model-00002-of-00002.safetensors"}}"#
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model.safetensors.index.json").path,
            contents: Data(index.utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model-00001-of-00002.safetensors").path,
            contents: Data()
        ))

        XCTAssertEqual(
            InklingResources.validate(rootURL: root).map(\.lastPathComponent),
            ["model-00002-of-00002.safetensors"]
        )
    }

    func testManagedRootManifestValidationUsesInklingLayout() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling-manifest-validation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try MereRunModelManifest.template(
            for: .inklingSmall,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(name).path,
                contents: Data("{}".utf8)
            ))
        }
        let index = #"{"weight_map":{"language_model.model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model.safetensors.index.json").path,
            contents: Data(index.utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model-00001-of-00001.safetensors").path,
            contents: Data()
        ))

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: InklingResources.modelID
        )
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertFalse(report.errors.contains { $0.contains("text_encoder") || $0.contains("tokenizer directory") })
    }

    private static let configJSON = #"""
    {
      "architectures": ["InklingForConditionalGeneration"],
      "model_type": "inkling_mm_model",
      "eos_token_id": 200006,
      "quantization": {"bits": 2, "group_size": 128, "mode": "affine", "scope": "routed_experts"},
      "text_config": {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "unpadded_vocab_size": 32,
        "rms_norm_eps": 0.000001,
        "use_embed_norm": true,
        "logits_mup_width_multiplier": 1.0,
        "model_max_length": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "swa_num_attention_heads": 2,
        "swa_num_key_value_heads": 1,
        "swa_head_dim": 4,
        "sliding_window_size": 8,
        "local_layer_ids": [0],
        "d_rel": 2,
        "rel_extent": 8,
        "log_scaling_n_floor": 16,
        "log_scaling_alpha": 0.1,
        "sconv_kernel_size": 2,
        "dense_mlp_idx": 1,
        "dense_intermediate_size": 16,
        "intermediate_size": 4,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "route_scale": 8.0
      }
    }
    """#

    private static let sparseConfigJSON = configJSON
        .replacingOccurrences(of: #""hidden_size": 8"#, with: #""hidden_size": 128"#)
        .replacingOccurrences(of: #""head_dim": 4"#, with: #""head_dim": 64"#)
        .replacingOccurrences(
            of: #""dense_intermediate_size": 16"#,
            with: #""dense_intermediate_size": 128"#
        )
        .replacingOccurrences(of: #""intermediate_size": 4"#, with: #""intermediate_size": 128"#)
        .replacingOccurrences(of: #""dense_mlp_idx": 1"#, with: #""dense_mlp_idx": 0"#)
}
