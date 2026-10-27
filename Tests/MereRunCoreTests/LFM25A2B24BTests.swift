import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class LFM25A2B24BTests: MereRunCoreTestCase {
    private func checkpointConfig() throws -> LFM2Config {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        return try JSONDecoder().decode(
            LFM2Config.self,
            from: Data(contentsOf: fixtures.appending(path: "LFM25A2B24B/config.json"))
        )
    }

    func testFP32A2BDecodeReclaimsUnusedCacheWithoutChangingOtherVariants() throws {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let source = try String(contentsOf: fixtures.appending(path: "LFM25A2B24B/config.json"), encoding: .utf8)
        let configured = source.replacingOccurrences(
            of: "\"model_type\"", with: "\"mererun_fp32_activations\":true,\"model_type\""
        )
        let fp32 = try JSONDecoder().decode(LFM2Config.self, from: Data(configured.utf8))
        let bf16 = try checkpointConfig()
        let gib = 1_024 * 1_024 * 1_024
        XCTAssertFalse(LFM2DecodeCachePolicy.shouldClear(
            config: bf16, activeMemory: 24 * gib, cacheMemory: 4 * gib, memoryLimit: 32 * gib
        ))
        XCTAssertFalse(LFM2DecodeCachePolicy.shouldClear(
            config: fp32, activeMemory: 31 * gib, cacheMemory: 128 * 1_024 * 1_024, memoryLimit: 32 * gib
        ))
        XCTAssertFalse(LFM2DecodeCachePolicy.shouldClear(
            config: fp32, activeMemory: 20 * gib, cacheMemory: gib / 2, memoryLimit: 32 * gib
        ))
        XCTAssertTrue(LFM2DecodeCachePolicy.shouldClear(
            config: fp32, activeMemory: 20 * gib, cacheMemory: gib, memoryLimit: 32 * gib
        ))
        XCTAssertTrue(LFM2DecodeCachePolicy.shouldClear(
            config: fp32, activeMemory: 29 * gib, cacheMemory: gib / 2, memoryLimit: 32 * gib
        ))
    }

    func testPinnedReleasedConfiguration() throws {
        let config = try checkpointConfig()
        XCTAssertEqual(config.modelType, "lfm2_moe")
        XCTAssertEqual(config.architectures, ["Lfm2MoeForCausalLM"])
        XCTAssertEqual(config.hiddenSize, 2_048)
        XCTAssertEqual(config.intermediateSize, 11_776)
        XCTAssertEqual(config.moeIntermediateSize, 1_536)
        XCTAssertEqual(config.numHiddenLayers, 40)
        XCTAssertEqual(config.numDenseLayers, 2)
        XCTAssertEqual(config.numExperts, 64)
        XCTAssertEqual(config.numExpertsPerTok, 4)
        XCTAssertEqual(config.routedScalingFactor, 1)
        XCTAssertEqual(config.vocabSize, 128_000)
        XCTAssertEqual(config.maxPositionEmbeddings, 131_072)
        XCTAssertEqual(config.ropeParameters?.ropeTheta, 10_000_000)
        XCTAssertEqual(config.fullAttentionLayerIndexes, Set(stride(from: 2, through: 38, by: 4)))
        XCTAssertEqual(config.eosTokenIds, [124_900])
        XCTAssertTrue(config.useExpertBias)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertNil(config.quantization)
        XCTAssertFalse(config.useFP32Activations)
        XCTAssertEqual(config.prefillChunkSize(promptTokens: 131_072), 512)
    }

    func testLocalConvertedA2BCheckpointKeepsItsTextOnlyFamily() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "Checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        var config = try String(contentsOf: fixtures.appending(path: "LFM25A2B24B/config.json"), encoding: .utf8)
        config = config.replacingOccurrences(of: "\"model_type\"", with:
            "\"quantization\":{\"bits\":8,\"group_size\":64,\"mode\":\"affine\"},\"model_type\"")
        try Data(config.utf8).write(to: root.appending(path: "config.json"))
        XCTAssertEqual(ModelFamilyIdentifier.textChatFamily(matching: root.path), .lfm2A2B)
        XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: root.path), .lfm2A2B)
        XCTAssertTrue(try checkpointConfig().isA2B24B)
    }

    func testContextBudgetUsesCheckpointLimitAndKeepsOperationalDefault() throws {
        let config = try checkpointConfig()
        XCTAssertEqual(try config.contextLength(requested: nil), 32_768)
        XCTAssertEqual(try config.contextLength(requested: 65_536, runtimeLimit: 131_072), 65_536)
        XCTAssertEqual(try config.contextLength(requested: 200_000, runtimeLimit: 131_072), 131_072)
        XCTAssertEqual(try config.contextLength(requested: 65_536), 32_768)
        XCTAssertThrowsError(try config.contextLength(requested: 0))
    }

    func testSixtyFourExpertCheckpointMappingRoundTripsNativeModel() throws {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        var text = try String(contentsOf: fixtures.appending(path: "LFM25A2B24B/config.json"), encoding: .utf8)
        // Keep all 40 layers, 64 experts, and the released hybrid schedule;
        // shrink matrix widths so this checks loader behavior without trained weights.
        for (field, value) in [
            ("vocab_size", 32), ("hidden_size", 16), ("intermediate_size", 32),
            ("moe_intermediate_size", 8), ("num_attention_heads", 4), ("num_key_value_heads", 2)
        ] {
            let pattern = "\"\(field)\": [0-9]+"
            text = text.replacingOccurrences(of: pattern, with: "\"\(field)\": \(value)", options: .regularExpression)
        }
        let config = try JSONDecoder().decode(LFM2Config.self, from: Data(text.utf8))
        let source = LFM2Model(config: config)
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in source.parameters().flattened() {
            if key.contains(".switch_mlp.") {
                for expert in 0..<config.numExperts {
                    var name = key.replacingOccurrences(of: ".switch_mlp.", with: ".experts.\(expert).")
                    for (native, original) in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")] {
                        name = name.replacingOccurrences(of: native, with: original)
                    }
                    checkpoint[name] = value[expert]
                }
            } else if key.hasSuffix(".conv.conv.weight") {
                checkpoint[key] = value.transposed(0, 2, 1)
            } else {
                checkpoint[key] = value
            }
        }
        XCTAssertEqual(checkpoint.count, 7_610)
        let restored = LFM2Model(config: config)
        let updates = try LFM2Resources.mapBF16MoEWeights(checkpoint, config: config)
        try restored.update(parameters: ModuleParameters.unflattened(updates), verify: .all)
        let input = MLXArray([Int32(1), 2]).reshaped([1, 2])
        let expected = source(input, cache: nil)
        let actual = restored(input, cache: nil)
        MLX.eval(expected, actual)
        XCTAssertLessThan(abs(expected - actual).max().item(Float.self), 0.00001)
    }

    func testManagedInstallPinsOriginalWeightsWithoutCompanionOrAutoDownload() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.a2b24BBF16ModelId))
        XCTAssertEqual(spec.upstreamRepoId, "LiquidAI/LFM2.5-24B-A2B")
        XCTAssertEqual(spec.upstreamRevision, "1769516ac588c4c558089c94f27d96b0dc67b183")
        XCTAssertEqual(spec.hubFallback?.patterns, LFM2Resources.snapshotPatterns)
        XCTAssertEqual(spec.hubFallback?.revision, spec.upstreamRevision)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatLFM2)
        XCTAssertEqual(spec.validationKind, .lfm2)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertEqual(spec.companionModelIDs, [])
        XCTAssertNil(LFM2Resources.dsparkModelID(for: spec.id, config: try checkpointConfig()))
        XCTAssertEqual(spec.apiProfile?.contextWindow, 131_072)
        XCTAssertEqual(spec.apiProfile?.thinkingLevels, [.high])
        XCTAssertTrue(LFM2Resources.managedModelIds.contains(spec.id))
        XCTAssertTrue(LFM2Resources.upstreamRepoIds.contains(try XCTUnwrap(spec.upstreamRepoId)))
        let manifest = MereRunModelManifest.template(for: .lfm25A2B24BBF16)
        XCTAssertEqual(manifest.engine, .lfm2)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.upstreamRepoId, "\(spec.upstreamRepoId!)@\(spec.upstreamRevision!)")
        let descriptor = try XCTUnwrap(ManagedModelCapabilityCatalog.descriptor(for: spec.id))
        XCTAssertEqual(descriptor.minimumUnifiedMemoryGB, 64)
        XCTAssertEqual(descriptor.recommendedUnifiedMemoryGB, 96)
    }

    func testReasoningDefaultsAndExplicitSamplingOverrides() throws {
        let request = ChatRequest(messages: [.init(role: .user, content: "hello")])
        for policy in [ChatDefaultPolicy.command, .openAI] {
            for id in [LFM2Resources.a2b24BBF16ModelId, LFM2Resources.a2b24BUpstreamRepoId] {
                let resolved = try ChatRequestResolver.resolve(request, modelID: id, sampling: .init(), policy: policy)
                XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: id), .lfm2A2B)
                XCTAssertEqual(resolved.temperature, 0.1)
                XCTAssertEqual(resolved.topP, 1)
                XCTAssertEqual(resolved.topK, 50)
                XCTAssertTrue(resolved.showThinking)
                let overridden = try ChatRequestResolver.resolve(
                    request, modelID: id,
                    sampling: .init(temperature: 0, topP: 0.8, topK: 12, thinking: false), policy: policy
                )
                XCTAssertEqual(overridden.temperature, 0)
                XCTAssertEqual(overridden.topP, 0.8)
                XCTAssertEqual(overridden.topK, 12)
                XCTAssertFalse(overridden.showThinking)
            }
        }
        let runtime = try NativeChatRuntime.command(modelID: LFM2Resources.a2b24BBF16ModelId, modelPath: "/fixture")
        guard case .textChatLFM2(_, let path) = runtime else {
            return XCTFail("24B-A2B must use the native LFM2 runtime")
        }
        XCTAssertEqual(path, "/fixture")
    }

    func testReleasedTemplateKeepsReasoningPrefixWhenDisplayIsDisabled() throws {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let template = try String(contentsOf: fixtures.appending(path: "LFM25A2B24B/chat_template.jinja"), encoding: .utf8)
        XCTAssertTrue(template.contains("<|im_start|>assistant\\n<think>"))
        let rendered = try LFM2TokenizerAndTemplate.renderPrompt(
            messages: [.init(role: .user, content: "hello")], tools: nil,
            addGenerationPrompt: true, includeThinking: false, generationPromptSuffix: "<think>"
        )
        XCTAssertTrue(rendered.hasSuffix("<|im_start|>assistant\n<think>"))
    }

    func testSigmoidRouterUsesBiasOnlyForSelectionAndAppliesScaling() throws {
        // Logits give probabilities [0.5, 0.75, 0.25]. Bias forces expert 2
        // into the route, but its mixture weight remains 0.25, not 1.25.
        let cases: [(Int, Bool, Float, Float)] = [
            (2, true, 1, 2.25), (2, false, 1, 2.25), (1, true, 0.25, 0.75),
            (3, true, 1.5, 2.75), (3, false, 1.5, 2.75),
        ]
        for (topK, normalize, selectedMass, selectedValue) in cases {
            let data = Data("""
                {"model_type":"lfm2_moe","vocab_size":8,"hidden_size":2,
                 "intermediate_size":2,"moe_intermediate_size":2,"num_hidden_layers":1,
                 "num_dense_layers":0,"num_experts":3,"num_experts_per_tok":\(topK),
                 "norm_topk_prob":\(normalize),"routed_scaling_factor":2,
                 "num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,
                 "norm_eps":0.00001,"conv_L_cache":3,"layer_types":["full_attention"],
                 "use_expert_bias":true}
                """.utf8)
            let config = try JSONDecoder().decode(LFM2Config.self, from: data)
            let layer = LFM2FeedForward(config: config, layerIndex: 0)
            try layer.update(parameters: ModuleParameters.unflattened([
                ("gate.weight", MLXArray([Float(0), 0, log(3), 0, -log(3), 0]).reshaped([3, 2])),
                ("expert_bias", MLXArray([Float(0), 0, 1])),
                ("switch_mlp.gate_proj.weight", MLXArray.ones([3, 2, 2])),
                ("switch_mlp.up_proj.weight", MLXArray.ones([3, 2, 2])),
                ("switch_mlp.down_proj.weight", stacked((1...3).map { MLXArray.ones([2, 2]) * Float($0) }))
            ]), verify: .all)
            let output = layer(MLXArray([Float(1), 0]).reshaped([1, 1, 2]))
            MLX.eval(output)
            let weighted = normalize ? selectedValue / (selectedMass + 1e-6) : selectedValue
            let expected = 2 / (1 + exp(Float(-1))) * weighted * 2
            for actual in output.asArray(Float.self) {
                XCTAssertEqual(actual, expected, accuracy: 0.00001)
            }
        }
    }
}
