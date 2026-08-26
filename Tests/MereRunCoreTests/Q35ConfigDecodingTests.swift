import Foundation
import XCTest
import MLX
import MLXRandom
@testable import MereRunCore

final class Q35ConfigDecodingTests: MereRunCoreTestCase {
    func testBonsaiMLXVisionPatchKernelKeepsPublishedLayout() throws {
        let weight = MLXArray.zeros([1_152, 2, 16, 16, 3])

        let mapped = try XCTUnwrap(
            Q35VisionTower.mapVisionWeight("vision_tower.patch_embed.proj.weight", weight).first
        )

        XCTAssertEqual(mapped.0, "visionTower.patch_embed.proj.weight")
        XCTAssertEqual(mapped.1.shape, [1_152, 2, 16, 16, 3])
    }

    func testPyTorchVisionPatchKernelConvertsToMLXLayout() throws {
        let weight = MLXArray.zeros([1_152, 3, 2, 16, 16])

        let mapped = try XCTUnwrap(
            Q35VisionTower.mapVisionWeight("vision_tower.patch_embed.proj.weight", weight).first
        )

        XCTAssertEqual(mapped.1.shape, [1_152, 2, 16, 16, 3])
    }

    func testBonsai27BDenseOneBitConfigDecodesPublishedShape() throws {
        var object = makeBaseConfig()
        var text = object["text_config"] as? [String: Any] ?? [:]
        text["model_type"] = "qwen3_5_text"
        text["hidden_size"] = 5_120
        text["num_hidden_layers"] = 64
        text["intermediate_size"] = 17_408
        text["num_attention_heads"] = 24
        text["num_key_value_heads"] = 4
        text["head_dim"] = 256
        text["layer_types"] = (0..<64).map { ($0 + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention" }
        text["linear_num_value_heads"] = 48
        text["linear_num_key_heads"] = 16
        text["linear_key_head_dim"] = 128
        text["linear_value_head_dim"] = 128
        text["max_position_embeddings"] = 262_144
        text["vocab_size"] = 248_320
        text.removeValue(forKey: "num_experts")
        text.removeValue(forKey: "num_experts_per_tok")
        text.removeValue(forKey: "shared_expert_intermediate_size")
        text.removeValue(forKey: "moe_intermediate_size")
        object["model_type"] = "qwen3_5"
        object["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        object["eos_token_id"] = 248_046
        object["quantization"] = ["group_size": 128, "bits": 1]
        object["text_config"] = text

        let config = try decodeConfig(object)

        XCTAssertEqual(config.quantization, Q35QuantizationConfig(groupSize: 128, bits: 1, mode: "affine"))
        XCTAssertFalse(config.textConfig.usesMoE)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 64)
        XCTAssertEqual(config.textConfig.hiddenSize, 5_120)
        XCTAssertEqual(config.textConfig.intermediateSize, 17_408)
        XCTAssertEqual(config.textConfig.maxPositionEmbeddings, Q35Resources.bonsai27B1BitContextLength)
        XCTAssertEqual(config.eosTokenIds, [248_046])
    }

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

    func testQ35Qwen3VLTargetSizeUsesUpstreamSpatialPixelBudget() throws {
        let target = try Q35Generator.qwen3VLTargetSize(
            originalWidth: 2_108,
            originalHeight: 1_094,
            patchSize: 16,
            spatialMergeSize: 2
        )

        XCTAssertEqual(target.width, 2_112)
        XCTAssertEqual(target.height, 1_088)
        XCTAssertEqual((target.width / 16) * (target.height / 16) / 4, 2_244)
    }

    func testQ38ImageResizeDoesNotCountDuplicatedTemporalPatch() throws {
        let target = try Q35Generator.qwen3VLTargetSize(
            originalWidth: 3_072,
            originalHeight: 3_072,
            patchSize: 16,
            spatialMergeSize: 2,
            minPixels: Q35Resources.q38TwentySevenBVisionMinPixels,
            maxPixels: Q35Resources.q38TwentySevenBVisionMaxPixels
        )

        XCTAssertEqual(target.width, 3_072)
        XCTAssertEqual(target.height, 3_072)
    }

    func testQ38ImageResizeUpscalesFromPublishedSpatialMinimum() throws {
        let target = try Q35Generator.qwen3VLTargetSize(
            originalWidth: 256,
            originalHeight: 128,
            patchSize: 16,
            spatialMergeSize: 2,
            minPixels: Q35Resources.q38TwentySevenBVisionMinPixels,
            maxPixels: Q35Resources.q38TwentySevenBVisionMaxPixels
        )

        XCTAssertEqual(target.width, 384)
        XCTAssertEqual(target.height, 192)
    }

    func testQ35ImageResizeUsesPythonTieToEvenRounding() throws {
        let target = try Q35Generator.qwen3VLTargetSize(
            originalWidth: 80,
            originalHeight: 80,
            patchSize: 16,
            spatialMergeSize: 2,
            minPixels: 1,
            maxPixels: 1_000_000
        )

        XCTAssertEqual(target.width, 64)
        XCTAssertEqual(target.height, 64)
    }

    func testQ35ImageResizeRejectsUpstreamUnsupportedAspectRatio() {
        XCTAssertThrowsError(
            try Q35Generator.qwen3VLTargetSize(
                originalWidth: 6_432,
                originalHeight: 32,
                patchSize: 16,
                spatialMergeSize: 2
            )
        )
    }

    func testQ35LinearAttentionConvWeightsKeepMLXLayout() {
        let mlxLayout = MLXArray.zeros([8, 4, 1])
        let normalizedMLX = Q35Generator.normalizedLinearAttentionConv1DWeight(mlxLayout)
        XCTAssertEqual(normalizedMLX.shape, [8, 4, 1])

        let pytorchDepthwiseLayout = MLXArray.zeros([8, 1, 4])
        let normalizedPyTorch = Q35Generator.normalizedLinearAttentionConv1DWeight(pytorchDepthwiseLayout)
        XCTAssertEqual(normalizedPyTorch.shape, [8, 4, 1])
    }

    func testQ35FusedPortableQuantizedLinearMatchesSeparateOutputs() {
        MLXRandom.seed(42)
        let lhs = PortableQuantizedLinear(
            weight: MLXRandom.uniform(-1.0 ..< 1.0, [4, 32]),
            bias: nil,
            groupSize: 32,
            bits: 4,
            mode: .affine
        )
        let rhs = PortableQuantizedLinear(
            weight: MLXRandom.uniform(-1.0 ..< 1.0, [6, 32]),
            bias: nil,
            groupSize: 32,
            bits: 4,
            mode: .affine
        )
        guard let fused = q35FusedPortableQuantizedLinear(lhs, rhs) else {
            XCTFail("Expected compatible quantized projections to fuse")
            return
        }

        let input = MLXRandom.uniform(-1.0 ..< 1.0, [2, 3, 32])
        let separate = MLX.concatenated([lhs(input), rhs(input)], axis: -1)
        let together = fused(input)
        MLX.eval(separate, together)

        XCTAssertEqual(together.shape, separate.shape)
        let maxDiff = MLX.max(MLX.abs((together - separate).asType(.float32))).item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-4)
    }

    #if os(macOS)
    func testQ35GatedDeltaMetalMatchesOpsReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Q35 gated-delta Metal parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }

        let batch = 1
        let sequence = 3
        let numKeyHeads = 2
        let numValueHeads = 4
        let keyHeadDim = 32
        let valueHeadDim = 4

        MLXRandom.seed(11)
        let q = MLXRandom.uniform(-0.3 ..< 0.3, [batch, sequence, numKeyHeads, keyHeadDim])
        let k = MLXRandom.uniform(-0.3 ..< 0.3, [batch, sequence, numKeyHeads, keyHeadDim])
        let v = MLXRandom.uniform(-0.3 ..< 0.3, [batch, sequence, numValueHeads, valueHeadDim])
        let a = MLXRandom.uniform(-0.1 ..< 0.1, [batch, sequence, numValueHeads])
        let b = MLXRandom.uniform(-0.1 ..< 0.1, [batch, sequence, numValueHeads])
        let aLog = MLXRandom.uniform(-0.2 ..< 0.2, [numValueHeads])
        let dtBias = MLXRandom.uniform(-0.2 ..< 0.2, [numValueHeads])
        let state = MLXRandom.uniform(-0.1 ..< 0.1, [batch, numValueHeads, valueHeadDim, keyHeadDim])

        let reference = q35GatedDeltaUpdateOps(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            numKeyHeads: numKeyHeads,
            numValueHeads: numValueHeads,
            valueHeadDim: valueHeadDim
        )
        let fast = try XCTUnwrap(q35GatedDeltaUpdateMetal(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            numKeyHeads: numKeyHeads,
            numValueHeads: numValueHeads,
            valueHeadDim: valueHeadDim
        ))

        let outputMaxDiff = MLX.max(MLX.abs(reference.0.asType(.float32) - fast.0.asType(.float32))).item(Float.self)
        let stateMaxDiff = MLX.max(MLX.abs(reference.1.asType(.float32) - fast.1.asType(.float32))).item(Float.self)
        XCTAssertEqual(outputMaxDiff, 0, accuracy: 1e-5)
        XCTAssertEqual(stateMaxDiff, 0, accuracy: 1e-5)
    }

    func testQ35GDNVerifyPreworkMetalIsBitExactAtClaimedWidths() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Q35 GDN prework Metal parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }

        let numKeyHeads = 16
        let numValueHeads = 48
        let keyHeadDim = 128
        let valueHeadDim = 128
        let convDim = 2 * numKeyHeads * keyHeadDim + numValueHeads * valueHeadDim

        for sequence in [3, 4, 5, 7, 9] {
            MLXRandom.seed(11)
            let qkv = MLXRandom.uniform(-0.5 ..< 0.5, [1, sequence, convDim]).asType(.bfloat16)
            let state = MLXRandom.uniform(-0.5 ..< 0.5, [1, 3, convDim]).asType(.bfloat16)
            let weight = MLXRandom.uniform(-0.2 ..< 0.2, [convDim, 4, 1]).asType(.bfloat16)
            let reference = q35GDNPreworkOps(
                qkv: qkv,
                convState: state,
                convWeight: weight,
                numKeyHeads: numKeyHeads,
                numValueHeads: numValueHeads,
                keyHeadDim: keyHeadDim,
                valueHeadDim: valueHeadDim
            )
            let fast = try XCTUnwrap(q35GDNPreworkMetal(
                qkv: qkv,
                convState: state,
                convWeight: weight,
                numKeyHeads: numKeyHeads,
                numValueHeads: numValueHeads,
                keyHeadDim: keyHeadDim,
                valueHeadDim: valueHeadDim
            ))

            for (name, expected, actual) in [
                ("q", reference.q, fast.q),
                ("k", reference.k, fast.k),
                ("v", reference.v, fast.v),
                ("convState", reference.convState, fast.convState),
            ] {
                MLX.eval(expected, actual)
                XCTAssertEqual(actual.shape, expected.shape, "\(name), S=\(sequence)")
                XCTAssertTrue(
                    MLX.allClose(expected, actual, rtol: 0, atol: 0).item(Bool.self),
                    "\(name) was not bit-exact at S=\(sequence)"
                )
            }
        }
    }

    func testQ35GDNVerifyPreworkBenchmarkWhenEnabled() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_TEST_Q35_GDN_BENCHMARK"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_Q35_GDN_BENCHMARK=1 to measure fused GDN prework.")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Q35 GDN prework benchmark requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }

        let numKeyHeads = 16
        let numValueHeads = 48
        let keyHeadDim = 128
        let valueHeadDim = 128
        let convDim = 2 * numKeyHeads * keyHeadDim + numValueHeads * valueHeadDim
        let iterations = 50
        let rmsNormWeight = MLXArray.ones([keyHeadDim], dtype: .bfloat16)

        for sequence in [4, 9] {
            MLXRandom.seed(17)
            let qkv = MLXRandom.uniform(-0.5 ..< 0.5, [1, sequence, convDim]).asType(.bfloat16)
            let state = MLXRandom.uniform(-0.5 ..< 0.5, [1, 3, convDim]).asType(.bfloat16)
            let weight = MLXRandom.uniform(-0.2 ..< 0.2, [convDim, 4, 1]).asType(.bfloat16)

            func reference() -> Q35GDNPreworkOutput {
                q35GDNPreworkOps(
                    qkv: qkv,
                    convState: state,
                    convWeight: weight,
                    numKeyHeads: numKeyHeads,
                    numValueHeads: numValueHeads,
                    keyHeadDim: keyHeadDim,
                    valueHeadDim: valueHeadDim,
                    rmsNormWeight: rmsNormWeight
                )
            }

            func fused() throws -> Q35GDNPreworkOutput {
                try XCTUnwrap(q35GDNPreworkMetal(
                    qkv: qkv,
                    convState: state,
                    convWeight: weight,
                    numKeyHeads: numKeyHeads,
                    numValueHeads: numValueHeads,
                    keyHeadDim: keyHeadDim,
                    valueHeadDim: valueHeadDim
                ))
            }

            func evaluate(_ output: Q35GDNPreworkOutput) {
                MLX.eval(output.q, output.k, output.v, output.convState)
            }

            for _ in 0..<3 {
                evaluate(reference())
                evaluate(try fused())
            }

            let referenceStart = Date()
            for _ in 0..<iterations {
                evaluate(reference())
            }
            let referenceSeconds = Date().timeIntervalSince(referenceStart)

            let fusedStart = Date()
            for _ in 0..<iterations {
                evaluate(try fused())
            }
            let fusedSeconds = Date().timeIntervalSince(fusedStart)
            let referenceMilliseconds = referenceSeconds * 1_000 / Double(iterations)
            let fusedMilliseconds = fusedSeconds * 1_000 / Double(iterations)
            let speedup = referenceMilliseconds / fusedMilliseconds
            print(String(
                format: "q35_gdn_prework_benchmark sequence=%d iterations=%d reference_ms=%.4f fused_ms=%.4f speedup=%.3fx",
                sequence,
                iterations,
                referenceMilliseconds,
                fusedMilliseconds,
                speedup
            ))
        }
    }

    func testQ35GDNVerifyPreworkRejectsDecodeWidth() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Q35 GDN prework eligibility requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        let qkv = MLXArray.zeros([1, 1, 10_240], dtype: .bfloat16)
        let state = MLXArray.zeros([1, 3, 10_240], dtype: .bfloat16)
        let weight = MLXArray.zeros([10_240, 4, 1], dtype: .bfloat16)

        XCTAssertNil(q35GDNPreworkMetal(
            qkv: qkv,
            convState: state,
            convWeight: weight,
            numKeyHeads: 16,
            numValueHeads: 48,
            keyHeadDim: 128,
            valueHeadDim: 128
        ))
    }
    #endif

    func testQ35RMSNormOffsetConversionSkipsLinearAttentionGateNorm() {
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.layers.0.input_layernorm.weight"))
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.layers.0.post_attention_layernorm.weight"))
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.layers.3.self_attn.q_norm.weight"))
        XCTAssertTrue(Q35Generator.isOffsetRMSNormWeight("model.norm.weight"))
        XCTAssertFalse(Q35Generator.isOffsetRMSNormWeight("model.layers.0.linear_attn.norm.weight"))
    }

    func testQ35OfficialCheckpointKeepsZeroCenteredRMSNormWeights() {
        let published = MLXArray([Float(-0.125), 0.25])
        let normalized = Q35Generator.normalizedRMSNormWeight(
            published,
            checkpointUsesZeroCenteredNorms: true
        )
        MLX.eval(normalized)

        XCTAssertEqual(normalized[0].item(Float.self), -0.125, accuracy: 1e-6)
        XCTAssertEqual(normalized[1].item(Float.self), 0.25, accuracy: 1e-6)
    }

    func testQ35ConvertedCheckpointConvertsDirectRMSNormScalesToOffsets() {
        let converted = MLXArray([Float(0.875), 1.25])
        let normalized = Q35Generator.normalizedRMSNormWeight(
            converted,
            checkpointUsesZeroCenteredNorms: false
        )
        MLX.eval(normalized)

        XCTAssertEqual(normalized[0].item(Float.self), -0.125, accuracy: 1e-6)
        XCTAssertEqual(normalized[1].item(Float.self), 0.25, accuracy: 1e-6)
    }

    func testQ35OfficialCheckpointLayoutDetectedFromEmbeddedMTP() {
        XCTAssertTrue(Q35Generator.checkpointUsesZeroCenteredRMSNorm(
            weightKeys: [
                "model.language_model.layers.0.input_layernorm.weight",
                "model.mtp.pre_fc_norm_hidden.weight",
            ],
            tensorShapes: [:]
        ))
    }

    func testQ35OfficialCheckpointLayoutDetectedFromPyTorchConv1D() {
        let key = "model.language_model.layers.0.linear_attn.conv1d.weight"
        XCTAssertTrue(Q35Generator.checkpointUsesZeroCenteredRMSNorm(
            weightKeys: [key],
            tensorShapes: [key: [12_288, 1, 4]]
        ))
        XCTAssertFalse(Q35Generator.checkpointUsesZeroCenteredRMSNorm(
            weightKeys: [key],
            tensorShapes: [key: [12_288, 4, 1]]
        ))
    }

    func testQ35ImageTokenExpansionPreservesCanonicalTemplateTokens() throws {
        let expanded = try Q35TokenizerAndTemplate.expandingImageTokenIds(
            [10, 248_056, 20, 248_056, 30],
            imageTokenId: 248_056,
            imageTokenCounts: [3, 2]
        )

        XCTAssertEqual(expanded, [10, 248_056, 248_056, 248_056, 20, 248_056, 248_056, 30])
    }

    func testQ35ImageTokenExpansionRejectsTemplateImageMismatch() {
        XCTAssertThrowsError(
            try Q35TokenizerAndTemplate.expandingImageTokenIds(
                [10, 20],
                imageTokenId: 248_056,
                imageTokenCounts: [3]
            )
        )
    }

    func testQ35TemplateMessagesPreserveReasoningAndToolCalls() throws {
        let rendered = Q35TokenizerAndTemplate.renderMessages([
            ChatMessage(
                role: .assistant,
                content: "I will inspect it.",
                reasoningContent: "Need the exact file.",
                toolCalls: [
                    ChatMessageToolCall(
                        id: "call_1",
                        name: "inspect_file",
                        arguments: ["path": .string("/tmp/input.png")]
                    ),
                ]
            ),
        ])
        let message = try XCTUnwrap(rendered.first)
        XCTAssertEqual(message["reasoning_content"] as? String, "Need the exact file.")
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: any Sendable]])
        let function = try XCTUnwrap(calls.first?["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "inspect_file")
    }

    func testQ38PinnedTokenizerRendersCanonicalVisionPromptWhenAvailable() throws {
        guard let rootPath = ProcessInfo.processInfo.environment["MERERUN_Q38_TOKENIZER_ROOT"] else {
            throw XCTSkip("Set MERERUN_Q38_TOKENIZER_ROOT to run pinned Qwen3.8 tokenizer parity.")
        }
        let template = try Q35TokenizerAndTemplate.load(
            from: URL(fileURLWithPath: rootPath),
            maxLengthOverride: Q35Resources.q38TwentySevenBContextLength
        )
        let tokens = try template.encodeForGeneration(
            messages: [ChatMessage(
                role: .user,
                content: "Read it.",
                imageUrl: "/tmp/page.png"
            )],
            addGenerationPrompt: true,
            includeThinking: true,
            maxLength: Q35Resources.q38TwentySevenBContextLength,
            imageTokenCounts: [3]
        )
        let decoded = template.decode(tokens: tokens)

        XCTAssertTrue(decoded.contains("Reasoning effort is set to xhigh."))
        XCTAssertTrue(decoded.contains(
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>Read it."
        ))
        XCTAssertTrue(decoded.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    func testQ38PinnedTokenizerRendersNativeReasoningEffortWhenAvailable() throws {
        guard let rootPath = ProcessInfo.processInfo.environment["MERERUN_Q38_TOKENIZER_ROOT"] else {
            throw XCTSkip("Set MERERUN_Q38_TOKENIZER_ROOT to run pinned Qwen3.8 tokenizer parity.")
        }
        let template = try Q35TokenizerAndTemplate.load(
            from: URL(fileURLWithPath: rootPath),
            maxLengthOverride: Q35Resources.q38TwentySevenBContextLength
        )
        let tokens = try template.encodeForGeneration(
            messages: [ChatMessage(role: .user, content: "Answer directly.")],
            addGenerationPrompt: true,
            includeThinking: true,
            reasoningEffort: "low",
            maxLength: Q35Resources.q38TwentySevenBContextLength
        )

        XCTAssertTrue(template.decode(tokens: tokens).contains("Reasoning effort is set to low."))
    }

    func testQ38ContinuousReasoningEffortMapsToNativeLevels() {
        XCTAssertNil(Q35Resources.q38ReasoningEffortLabel(for: nil))
        XCTAssertEqual(Q35Resources.q38ReasoningEffortLabel(for: 0), "low")
        XCTAssertEqual(Q35Resources.q38ReasoningEffortLabel(for: 0.2), "low")
        XCTAssertEqual(Q35Resources.q38ReasoningEffortLabel(for: 0.5), "medium")
        XCTAssertEqual(Q35Resources.q38ReasoningEffortLabel(for: 0.8), "xhigh")
        XCTAssertEqual(Q35Resources.q38ReasoningEffortLabel(for: 1), "xhigh")
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

    func testQ35MTPDraftLogitsSupportsDenseFeedForwardWeightLayout() throws {
        MLXRandom.seed(39)
        var configObject = makeTinyRuntimeConfig(layerTypes: ["full_attention"])
        var textConfig = configObject["text_config"] as? [String: Any] ?? [:]
        textConfig.removeValue(forKey: "num_experts")
        textConfig.removeValue(forKey: "num_experts_per_tok")
        textConfig.removeValue(forKey: "moe_intermediate_size")
        textConfig.removeValue(forKey: "shared_expert_intermediate_size")
        configObject["text_config"] = textConfig
        let config = try decodeConfig(configObject)
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

    func testQ35EmbeddedMTPWeightsSelectOnlyContainingShards() {
        let shards = Q35Generator.embeddedMTPShardFilenames(weightMap: [
            "model.language_model.layers.0.mlp.down_proj.weight": "model-00001.safetensors",
            "mtp.layers.0.mlp.down_proj.weight": "model-00018.safetensors",
            "mtp.norm.weight": "model-00018.safetensors",
            "mtp.fc.weight": "model-00017.safetensors",
        ])

        XCTAssertEqual(shards, ["model-00017.safetensors", "model-00018.safetensors"])
    }

    func testQ38MountedMTPComponentIsDiscovered() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "q38-mtp-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let mtpRoot = root.appendingPathComponent(Q35Resources.q38MTPComponentPath, isDirectory: true)
        try FileManager.default.createDirectory(at: mtpRoot, withIntermediateDirectories: true)
        try Data().write(to: mtpRoot.appendingPathComponent("model.safetensors"))

        let resources = try XCTUnwrap(
            Q35Generator.mtpResources(primary: Q35Resources(rootURL: root))
        )

        XCTAssertEqual(resources.rootURL.standardizedFileURL, mtpRoot.standardizedFileURL)
    }

    func testOrnithManagedMTPCompanionIsDiscovered() throws {
        let primaryRoot = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ornith-primary-\(UUID().uuidString)",
            isDirectory: true
        )
        let companionRoot = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ornith-mtp-\(UUID().uuidString)",
            isDirectory: true
        )
        defer {
            try? FileManager.default.removeItem(at: primaryRoot)
            try? FileManager.default.removeItem(at: companionRoot)
        }
        try FileManager.default.createDirectory(at: primaryRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: companionRoot, withIntermediateDirectories: true)
        let index = [
            "weight_map": [
                "mtp.norm.weight": Q35Resources.ornith35BMTPShardFilename,
            ],
        ]
        let indexData = try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
        try indexData.write(to: companionRoot.appendingPathComponent("model.safetensors.index.json"))
        try Data().write(
            to: companionRoot.appendingPathComponent(Q35Resources.ornith35BMTPShardFilename)
        )

        let resources = try XCTUnwrap(Q35Generator.mtpResources(
            primary: Q35Resources(rootURL: primaryRoot),
            companionRootURL: companionRoot
        ))

        XCTAssertEqual(resources.rootURL.standardizedFileURL, companionRoot.standardizedFileURL)
    }

    func testQ35PreparedFusedSwitchGLUReleasesSourceWeightsAndPreservesOutput() throws {
        MLXRandom.seed(43)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["full_attention"]))
        let switchMLP = Q35SwitchGLU(config: config)
        let input = MLXRandom.uniform(-0.2..<0.2, [1, 1, config.textConfig.hiddenSize])
        let indices = MLXArray([Int32(0)]).reshaped(1, 1, 1)
        let before = switchMLP.unsorted(input, indices: indices)
        MLX.eval(before)

        XCTAssertGreaterThan(switchMLP.sourceGateUpElementCount, 0)
        XCTAssertTrue(switchMLP.prepareFusedGateUpAndReleaseSources())
        XCTAssertTrue(switchMLP.hasPreparedFusedGateUp)
        XCTAssertEqual(switchMLP.sourceGateUpElementCount, 0)

        let after = switchMLP.unsorted(input, indices: indices)
        MLX.eval(after)
        let maximumError = MLX.max(MLX.abs(before - after)).item(Float.self)
        XCTAssertEqual(maximumError, 0, accuracy: 0.0001)
    }

    func testQ35StandaloneMTPWeightMappingAcceptsBareHeadKeys() {
        XCTAssertEqual(
            Q35Generator.mapMTPWeightKey("layers.0.self_attn.q_proj.weight", standalone: true),
            "layers.0.self_attn.q_proj.weight"
        )
        XCTAssertEqual(
            Q35Generator.mapMTPWeightKey("mtp.layers.0.self_attn.q_proj.weight", standalone: true),
            "layers.0.self_attn.q_proj.weight"
        )
        XCTAssertNil(Q35Generator.mapMTPWeightKey("lm_head.weight", standalone: true))
        XCTAssertNil(Q35Generator.mapMTPWeightKey("layers.0.self_attn.q_proj.weight"))
    }

    func testQ35MTPRMSNormWeightInventoryMatchesHeadArchitecture() {
        XCTAssertTrue(Q35Generator.isMTPRMSNormWeight("norm.weight"))
        XCTAssertTrue(Q35Generator.isMTPRMSNormWeight("pre_fc_norm_embedding.weight"))
        XCTAssertTrue(Q35Generator.isMTPRMSNormWeight("pre_fc_norm_hidden.weight"))
        XCTAssertTrue(Q35Generator.isMTPRMSNormWeight("layers.0.input_layernorm.weight"))
        XCTAssertTrue(Q35Generator.isMTPRMSNormWeight("layers.0.post_attention_layernorm.weight"))
        XCTAssertTrue(Q35Generator.isMTPRMSNormWeight("layers.0.self_attn.q_norm.weight"))
        XCTAssertTrue(Q35Generator.isMTPRMSNormWeight("layers.0.self_attn.k_norm.weight"))
        XCTAssertFalse(Q35Generator.isMTPRMSNormWeight("layers.0.self_attn.q_proj.weight"))
    }

    func testQ35OrnithMTPIndividualExpertsPackIntoDraftLayout() throws {
        var arrays: [String: MLXArray] = [
            "mtp.norm.weight": MLXArray.ones([4]),
        ]
        for expert in 0..<2 {
            arrays["mtp.layers.0.mlp.experts.\(expert).gate_proj.weight"] =
                MLXArray.ones([2, 4]) * Float(expert + 1)
            arrays["mtp.layers.0.mlp.experts.\(expert).up_proj.weight"] =
                MLXArray.ones([2, 4]) * Float(expert + 3)
            arrays["mtp.layers.0.mlp.experts.\(expert).down_proj.weight"] =
                MLXArray.ones([4, 2]) * Float(expert + 5)
        }

        let updates = try Q35Generator.mappedMTPUpdates(
            arrays: arrays,
            expertCount: 2,
            checkpointUsesZeroCenteredNorms: true
        )
        let convertedUpdates = try Q35Generator.mappedMTPUpdates(arrays: arrays, expertCount: 2)
        let mapped = Dictionary(uniqueKeysWithValues: updates)
        let converted = Dictionary(uniqueKeysWithValues: convertedUpdates)
        let gateUp = try XCTUnwrap(mapped["layers.0.mlp.experts.gate_up_proj"])
        let down = try XCTUnwrap(mapped["layers.0.mlp.experts.down_proj"])
        let norm = try XCTUnwrap(mapped["norm.weight"])
        let convertedNorm = try XCTUnwrap(converted["norm.weight"])
        MLX.eval(gateUp, down, norm, convertedNorm)

        XCTAssertEqual(gateUp.shape, [2, 4, 4])
        XCTAssertEqual(down.shape, [2, 4, 2])
        XCTAssertEqual(norm[0].item(Float.self), 1, accuracy: 0.0001)
        XCTAssertEqual(MLX.max(MLX.abs(convertedNorm)).item(Float.self), 0, accuracy: 0.0001)
        XCTAssertEqual(gateUp[0, 0, 0].item(Float.self), 1, accuracy: 0.0001)
        XCTAssertEqual(gateUp[0, 2, 0].item(Float.self), 3, accuracy: 0.0001)
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
        let session = Q35MTPDraftSession()
        let draftBlock = mtp.draftBlock(
            lastToken: 4,
            hidden: previousHidden,
            blockSize: 4,
            session: session,
            baseModel: model
        )
        MLX.eval(draftBlock.tokenIDs)
        let draftTokens = draftBlock.tokens

        XCTAssertEqual(draftTokens.count, 3)
        XCTAssertTrue(draftTokens.allSatisfy { $0 >= 0 && $0 < config.textConfig.vocabSize })
        XCTAssertEqual(session.committedHistoryCount, 1)
    }

    func testQ35MTPDraftSessionFlushesOnlyCommittedHistory() throws {
        MLXRandom.seed(41)
        let config = try decodeConfig(makeTinyRuntimeConfig(layerTypes: ["full_attention"]))
        let model = Q35Model(config: config)
        let mtp = Q35MTPModel(config: config)
        let session = Q35MTPDraftSession()
        let firstHidden = MLXRandom.uniform(-0.1..<0.1, [1, 1, config.textConfig.hiddenSize])

        _ = mtp.draftBlock(
            lastToken: 4,
            hidden: firstHidden,
            blockSize: 4,
            session: session,
            baseModel: model
        )
        XCTAssertEqual(session.committedHistoryCount, 1)

        let committedHidden = MLXRandom.uniform(-0.1..<0.1, [1, 1, config.textConfig.hiddenSize])
        session.recordCommittedTransitions(hiddenStates: committedHidden, nextTokens: [5])
        XCTAssertEqual(session.committedHistoryCount, 2)

        _ = mtp.draftBlock(
            lastToken: 6,
            hidden: committedHidden,
            blockSize: 2,
            session: session,
            baseModel: model
        )
        XCTAssertEqual(session.committedHistoryCount, 3)
    }

    func testQ35VerificationCachesRestoreAcceptedPrefixWithoutTargetReplay() throws {
        MLXRandom.seed(42)
        let config = try decodeConfig(makeTinyRuntimeConfig(
            layerTypes: ["linear_attention", "full_attention"]
        ))
        let model = Q35Model(config: config)
        let baseCaches = makeLayerCaches(config: config)
        let baseTokens = [1, 2, 3]
        let baseInput = MLXArray(baseTokens.map(Int32.init)).reshaped(1, baseTokens.count)
        let baseOutput = model.forward(baseInput, cache: baseCaches)
        MLX.eval(baseOutput.logits, baseOutput.hidden)

        let verificationTokens = [4, 5, 6, 7, 8, 9, 10, 11]
        for committedCount in 1..<verificationTokens.count {
            let candidateCaches = baseCaches.map { $0?.fork() }
            let verificationInput = MLXArray(verificationTokens.map(Int32.init))
                .reshaped(1, verificationTokens.count)
            let verification = model.forward(
                verificationInput,
                cache: candidateCaches,
                targetVerify: true
            )
            MLX.eval(verification.logits, verification.hidden)

            for cache in candidateCaches.compactMap({ $0 }) {
                XCTAssertTrue(cache.restoreVerificationPrefix(
                    totalTokens: verificationTokens.count,
                    tokenCount: committedCount
                ))
            }

            let continuation = MLXArray([Int32(12)]).reshaped(1, 1)
            let restoredOutput = model.forward(continuation, cache: candidateCaches)

            let referenceCaches = baseCaches.map { $0?.fork() }
            let committedInput = MLXArray(
                verificationTokens.prefix(committedCount).map(Int32.init)
            ).reshaped(1, committedCount)
            let committedOutput = model.forward(committedInput, cache: referenceCaches)
            MLX.eval(committedOutput.logits, committedOutput.hidden)
            let referenceOutput = model.forward(continuation, cache: referenceCaches)
            MLX.eval(restoredOutput.logits, referenceOutput.logits)

            let maximumError = MLX.max(MLX.abs(
                restoredOutput.logits.asType(.float32) - referenceOutput.logits.asType(.float32)
            )).item(Float.self)
            XCTAssertLessThan(maximumError, 0.0001)
        }
    }

    func testQ35MTPAdaptivePolicyStartsAtRollbackDepthAndStaysThereAfterFullAcceptance() {
        var policy = Q35MTPAdaptivePolicy(maxDraftDepth: 3)

        XCTAssertEqual(policy.draftDepth(offeredDepth: 3), 3)
        policy.record(acceptedDrafts: 3, drafted: 3)
        XCTAssertEqual(policy.draftDepth(offeredDepth: 3), 3)
    }

    func testQ35MTPAdaptivePolicyFallsBackAfterRepeatedRejection() {
        var policy = Q35MTPAdaptivePolicy(maxDraftDepth: 3)

        for _ in 0..<16 {
            let depth = policy.draftDepth(offeredDepth: 3)
            guard depth > 0 else { break }
            policy.record(acceptedDrafts: 0, drafted: depth)
        }

        XCTAssertEqual(policy.draftDepth(offeredDepth: 3), 0)
    }

    func testQ35CompactDraftRowsKeepPrefixControlsAndPadding() {
        let rows = MLXArray(
            (0..<Q35Model.compactDraftControlEnd).map(Float.init)
        ).reshaped(Q35Model.compactDraftControlEnd, 1)

        let compact = Q35Model.compactDraftRows(rows)
        MLX.eval(compact)

        XCTAssertEqual(compact.shape, [Q35Model.compactDraftPaddedCount, 1])
        XCTAssertEqual(
            compact[Q35Model.compactDraftPrefixCount - 1, 0].item(Float.self),
            Float(Q35Model.compactDraftPrefixCount - 1)
        )
        XCTAssertEqual(
            compact[Q35Model.compactDraftPrefixCount, 0].item(Float.self),
            Float(Q35Model.compactDraftControlStart)
        )
        XCTAssertEqual(
            compact[Q35Model.compactDraftRealCount - 1, 0].item(Float.self),
            Float(Q35Model.compactDraftControlEnd - 1)
        )
        XCTAssertEqual(compact[Q35Model.compactDraftRealCount, 0].item(Float.self), 0)
    }

    func testQ35CompactDraftSelectionMapsControlToken() {
        var values = Array(repeating: Float(0), count: Q35Model.compactDraftPaddedCount)
        values[Q35Model.compactDraftPrefixCount + 2] = 5
        let logits = MLXArray(values).reshaped(1, 1, values.count)

        let token = Q35Model.compactDraftTokenID(from: logits)
        MLX.eval(token)

        XCTAssertEqual(token.item(Int32.self), Int32(Q35Model.compactDraftControlStart + 2))
    }

    func testQ35CompactDraftSelectionUsesLowerIDForTie() {
        var values = Array(repeating: Float(0), count: Q35Model.compactDraftPaddedCount)
        values[7] = 5
        values[9] = 5
        let logits = MLXArray(values).reshaped(1, 1, values.count)

        let token = Q35Model.compactDraftTokenID(from: logits)
        MLX.eval(token)

        XCTAssertEqual(token.item(Int32.self), 7)
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

    func testQ35DenseMTPRequiresExplicitOptIn() {
        XCTAssertFalse(
            Q35Generator.shouldSpeculate(
                promptTokenCount: 32,
                maxContextTokens: 262_144,
                defaultMinimumPromptTokens: 0,
                enabledByDefault: false,
                environment: [:]
            )
        )
        XCTAssertTrue(
            Q35Generator.shouldSpeculate(
                promptTokenCount: 32,
                maxContextTokens: 262_144,
                defaultMinimumPromptTokens: 0,
                enabledByDefault: false,
                environment: ["MERERUN_Q35_MTP_SPECULATION": "1"]
            )
        )
    }

    func testOrnithMTPDefaultsToShortPromptSpeculationWithoutChangingQwen36Threshold() {
        XCTAssertEqual(
            Q35Generator.defaultMTPMinimumPromptTokens(
                modelId: Q35Resources.ornith35BMLX4BitModelId,
                usesMoE: true
            ),
            0
        )
        XCTAssertEqual(
            Q35Generator.defaultMTPMinimumPromptTokens(
                modelId: Q35Resources.q36NanoModelId,
                usesMoE: true
            ),
            6144
        )
        XCTAssertTrue(
            Q35Generator.shouldSpeculate(
                promptTokenCount: 32,
                maxContextTokens: 262_144,
                defaultMinimumPromptTokens: 0,
                enabledByDefault: true,
                environment: [:]
            )
        )
    }

    func testQ35MTPBlockSizeUsesEnvironmentClamp() {
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: [:]), 4)
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: ["MERERUN_Q35_MTP_BLOCK_SIZE": "1"]), 4)
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: ["MERERUN_Q35_MTP_BLOCK_SIZE": "6"]), 6)
        XCTAssertEqual(Q35Generator.mtpBlockSize(environment: ["MERERUN_Q35_MTP_BLOCK_SIZE": "32"]), 16)
        XCTAssertEqual(
            Q35Generator.mtpBlockSize(
                modelId: Q35Resources.q38TwentySevenB4BitModelId,
                environment: [:]
            ),
            8
        )
        XCTAssertEqual(
            Q35Generator.mtpBlockSize(
                modelId: Q35Resources.q38TwentySevenB4BitModelId,
                environment: ["MERERUN_Q35_MTP_BLOCK_SIZE": "16"]
            ),
            9
        )
    }

    func testQ35JSONModeAlwaysSelectsConstrainedSerialDecode() {
        XCTAssertEqual(
            Q35Generator.decodePath(
                jsonConstrained: true,
                continuousBatchingEnabled: true,
                mtpSpeculationEnabled: true
            ),
            .jsonConstrainedSerial
        )
        XCTAssertEqual(
            Q35Generator.decodePath(
                jsonConstrained: true,
                continuousBatchingEnabled: false,
                mtpSpeculationEnabled: false
            ),
            .jsonConstrainedSerial
        )
    }

    func testQ35UnconstrainedDecodePathKeepsAcceleratedModes() {
        XCTAssertEqual(
            Q35Generator.decodePath(
                jsonConstrained: false,
                continuousBatchingEnabled: true,
                mtpSpeculationEnabled: true
            ),
            .mtpSpeculativeSerial
        )
        XCTAssertEqual(
            Q35Generator.decodePath(
                jsonConstrained: false,
                continuousBatchingEnabled: true,
                mtpSpeculationEnabled: true,
                schedulerContended: true
            ),
            .continuousBatched
        )
        XCTAssertEqual(
            Q35Generator.decodePath(
                jsonConstrained: false,
                continuousBatchingEnabled: false,
                mtpSpeculationEnabled: true
            ),
            .mtpSpeculativeSerial
        )
        XCTAssertEqual(
            Q35Generator.decodePath(
                jsonConstrained: false,
                continuousBatchingEnabled: false,
                mtpSpeculationEnabled: false
            ),
            .pipelined
        )
    }

    func testQ38PrefillChunkUsesSafeDefaultAndShrinksWithLowLiveHeadroom() {
        let gibibyte = UInt64(1_024 * 1_024 * 1_024)
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.q38TwentySevenBModelId,
                availableMemory: 64 * gibibyte,
                environment: [:]
            ),
            1_024
        )
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.q38TwentySevenBModelId,
                availableMemory: 16 * gibibyte - 1,
                environment: [:]
            ),
            512
        )
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.defaultModelId,
                availableMemory: 1,
                environment: [:]
            ),
            1_024
        )
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.q38TwentySevenBModelId,
                availableMemory: nil,
                environment: [:]
            ),
            1_024
        )
    }

    func testQ38PrefillChunkShrinksUnderContentionAndHonorsOverride() {
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.q38TwentySevenBModelId,
                availableMemory: UInt64.max,
                activeRequestCount: 2,
                environment: [:]
            ),
            512
        )
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.q38TwentySevenBModelId,
                availableMemory: UInt64.max,
                activeRequestCount: 2,
                environment: ["MERERUN_Q35_PREFILL_CHUNK_TOKENS": "256"]
            ),
            256
        )
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.q38TwentySevenBModelId,
                availableMemory: 1,
                environment: ["MERERUN_Q35_PREFILL_CHUNK_TOKENS": "4096"]
            ),
            512
        )
        XCTAssertEqual(
            Q35Generator.prefillChunkSize(
                modelId: Q35Resources.q38TwentySevenBModelId,
                availableMemory: UInt64.max,
                environment: ["MERERUN_Q35_PREFILL_CHUNK_TOKENS": "4096"]
            ),
            4_096
        )
    }

    func testQ35MLXCacheClearRequiresReclaimablePressure() {
        let gib = 1_024 * 1_024 * 1_024
        XCTAssertFalse(Q35Generator.shouldClearMLXCache(
            activeMemory: 7 * gib,
            cacheMemory: 128 * 1_024 * 1_024,
            memoryLimit: 8 * gib
        ))
        XCTAssertFalse(Q35Generator.shouldClearMLXCache(
            activeMemory: 5 * gib,
            cacheMemory: 1 * gib,
            memoryLimit: 8 * gib
        ))
        XCTAssertTrue(Q35Generator.shouldClearMLXCache(
            activeMemory: 7 * gib,
            cacheMemory: 1 * gib,
            memoryLimit: 8 * gib
        ))
    }

    func testQ35ToolCallsSelectEarlyStoppablePipelinedDecode() {
        XCTAssertEqual(
            Q35Generator.decodePath(
                jsonConstrained: false,
                continuousBatchingEnabled: true,
                mtpSpeculationEnabled: true,
                stopAtCompletedToolCall: true
            ),
            .pipelined
        )
    }

    func testQ35TokenBudgetExhaustionReportsLength() {
        XCTAssertEqual(
            Q35Generator.finishReason(
                generatedTokenCount: 8,
                tokenBudget: 8,
                matchedStopSequence: false
            ),
            .length
        )
        XCTAssertEqual(
            Q35Generator.finishReason(
                generatedTokenCount: 7,
                tokenBudget: 8,
                matchedStopSequence: false
            ),
            .stop
        )
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

    func testOrnith15PublishedBF16ConfigurationDecodes() throws {
        var configObject = makeBaseConfig()
        configObject["model_type"] = "qwen3_5_moe"
        configObject["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        configObject["eos_token_id"] = [248_046, 248_044]
        configObject.removeValue(forKey: "vision_config")
        configObject.removeValue(forKey: "quantization")

        var textConfig = configObject["text_config"] as? [String: Any] ?? [:]
        textConfig["model_type"] = "qwen3_5_moe_text"
        textConfig["hidden_size"] = 2_048
        textConfig["num_hidden_layers"] = 40
        textConfig.removeValue(forKey: "intermediate_size")
        textConfig["shared_expert_intermediate_size"] = 512
        textConfig["moe_intermediate_size"] = 512
        textConfig["num_attention_heads"] = 16
        textConfig["num_key_value_heads"] = 2
        textConfig["head_dim"] = 256
        textConfig["num_experts"] = 256
        textConfig["num_experts_per_tok"] = 8
        textConfig["layer_types"] = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention"
        }
        textConfig.removeValue(forKey: "mlp_only_layers")
        textConfig["linear_num_value_heads"] = 32
        textConfig["linear_num_key_heads"] = 16
        textConfig["linear_key_head_dim"] = 128
        textConfig["linear_value_head_dim"] = 128
        textConfig["linear_conv_kernel_dim"] = 4
        textConfig["max_position_embeddings"] = 262_144
        textConfig["vocab_size"] = 248_320
        textConfig["eos_token_id"] = 248_044
        textConfig["rope_parameters"] = [
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000.0,
            "type": "default",
        ]
        configObject["text_config"] = textConfig

        let config = try decodeConfig(configObject)

        XCTAssertEqual(config.modelType, "qwen3_5_moe")
        XCTAssertNil(config.quantization)
        XCTAssertNil(config.visionConfig)
        XCTAssertEqual(config.eosTokenIds, [248_046, 248_044])
        XCTAssertEqual(config.textConfig.hiddenSize, 2_048)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 40)
        XCTAssertEqual(config.textConfig.moeIntermediateSize, 512)
        XCTAssertEqual(config.textConfig.sharedExpertIntermediateSize, 512)
        XCTAssertEqual(config.textConfig.numExperts, 256)
        XCTAssertEqual(config.textConfig.numExpertsPerTok, 8)
        XCTAssertTrue(config.textConfig.normTopKProb)
        XCTAssertEqual(config.textConfig.maxPositionEmbeddings, Q35Resources.ornith35BMLXContextLength)
    }

    func testQ38PublishedDenseVisionConfigurationDecodes() throws {
        var configObject = makeBaseConfig()
        configObject["model_type"] = "qwen3_5"
        configObject["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        configObject.removeValue(forKey: "eos_token_id")
        configObject["image_token_id"] = 248_056
        configObject["video_token_id"] = 248_057
        configObject["vision_start_token_id"] = 248_053
        configObject["vision_end_token_id"] = 248_054
        if var textConfig = configObject["text_config"] as? [String: Any] {
            textConfig["model_type"] = "qwen3_5_text"
            textConfig["hidden_size"] = 5120
            textConfig["num_hidden_layers"] = 64
            textConfig["intermediate_size"] = 17_408
            textConfig["num_attention_heads"] = 24
            textConfig["num_key_value_heads"] = 4
            textConfig["head_dim"] = 256
            textConfig["layer_types"] = (0..<64).map { ($0 + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention" }
            textConfig["linear_num_value_heads"] = 48
            textConfig["linear_num_key_heads"] = 16
            textConfig["linear_key_head_dim"] = 128
            textConfig["linear_value_head_dim"] = 128
            textConfig["max_position_embeddings"] = 262_144
            textConfig["vocab_size"] = 248_320
            textConfig["eos_token_id"] = 248_044
            textConfig["rope_parameters"] = [
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
            ]
            textConfig.removeValue(forKey: "num_experts")
            textConfig.removeValue(forKey: "num_experts_per_tok")
            textConfig.removeValue(forKey: "moe_intermediate_size")
            textConfig.removeValue(forKey: "shared_expert_intermediate_size")
            configObject["text_config"] = textConfig
        }
        if var visionConfig = configObject["vision_config"] as? [String: Any] {
            visionConfig["model_type"] = "qwen3_5"
            visionConfig["depth"] = 27
            visionConfig["hidden_act"] = "gelu_pytorch_tanh"
            visionConfig["hidden_size"] = 1152
            visionConfig["intermediate_size"] = 4304
            visionConfig["out_hidden_size"] = 5120
            visionConfig["patch_size"] = 16
            visionConfig["spatial_merge_size"] = 2
            visionConfig["num_position_embeddings"] = 2304
            configObject["vision_config"] = visionConfig
        }

        let config = try decodeConfig(configObject)

        XCTAssertEqual(config.modelType, "qwen3_5")
        XCTAssertEqual(config.eosTokenIds, [248_044])
        XCTAssertEqual(config.imageTokenId, 248_056)
        XCTAssertFalse(config.textConfig.usesMoE)
        XCTAssertEqual(config.textConfig.numExperts, 0)
        XCTAssertEqual(config.textConfig.numExpertsPerTok, 0)
        XCTAssertEqual(config.textConfig.hiddenSize, 5120)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 64)
        XCTAssertEqual(config.textConfig.intermediateSize, 17_408)
        XCTAssertEqual(config.textConfig.maxPositionEmbeddings, 262_144)
        XCTAssertEqual(config.visionConfig?.depth, 27)
        XCTAssertEqual(config.visionConfig?.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfig?.outHiddenSize, 5120)
        XCTAssertEqual(config.visionConfig?.patchSize, 16)
        XCTAssertEqual(config.visionConfig?.spatialMergeSize, 2)
    }

    func testQ38GenerationConfigDecodesAllPublishedStopTokens() throws {
        let data = Data(#"{"eos_token_id":[248046,248044]}"#.utf8)

        let config = try JSONDecoder().decode(Q35GenerationConfig.self, from: data)

        XCTAssertEqual(config.eosTokenIds, [248_046, 248_044])
    }

    func testQ38ResourceProfileUsesPublishedContextAndVisionBounds() throws {
        let profile = try XCTUnwrap(Q35Resources.profile(for: Q35Resources.q38TwentySevenBModelId))
        let bounds = Q35Resources.visionPixelBounds(forModelId: profile.modelId)

        XCTAssertEqual(profile.upstreamRepoId, Q35Resources.q38TwentySevenBUpstreamRepoId)
        XCTAssertEqual(profile.upstreamRevision, Q35Resources.q38TwentySevenBUpstreamRevision)
        XCTAssertEqual(Q35Resources.defaultContextLength(forModelId: profile.modelId), 262_144)
        XCTAssertEqual(bounds.minimum, 65_536)
        XCTAssertEqual(bounds.maximum, 16_777_216)
    }

    func testQ38FourBitResourceProfileKeepsPublishedRuntimeBounds() throws {
        let profile = try XCTUnwrap(
            Q35Resources.profile(for: Q35Resources.q38TwentySevenB4BitModelId)
        )
        let bounds = Q35Resources.visionPixelBounds(forModelId: profile.modelId)

        XCTAssertEqual(profile.upstreamRepoId, Q35Resources.q38TwentySevenB4BitUpstreamRepoId)
        XCTAssertEqual(profile.upstreamRevision, Q35Resources.q38TwentySevenB4BitUpstreamRevision)
        XCTAssertEqual(Q35Resources.defaultContextLength(forModelId: profile.modelId), 262_144)
        XCTAssertEqual(bounds.minimum, 65_536)
        XCTAssertEqual(bounds.maximum, 16_777_216)
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

    func testQ35ConfigUsesMLXLMNormTopKDefault() throws {
        var configObject = makeBaseConfig()
        var textConfig = configObject["text_config"] as? [String: Any] ?? [:]
        textConfig.removeValue(forKey: "norm_topk_prob")
        configObject["text_config"] = textConfig

        // Qwen3.5-family default is TRUE (HF transformers and mlx_lm both
        // renormalize top-k router scores when the key is absent). The older
        // Qwen3-MoE family defaulted to false; porting that default here
        // dampened every MoE block on checkpoints that omit the key.
        let defaulted = try decodeConfig(configObject)
        XCTAssertTrue(defaulted.textConfig.normTopKProb)

        textConfig["norm_topk_prob"] = false
        configObject["text_config"] = textConfig

        let explicit = try decodeConfig(configObject)
        XCTAssertFalse(explicit.textConfig.normTopKProb)
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
