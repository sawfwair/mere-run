import Foundation
import MLX
import Testing
import XCTest
@testable import MereRunCore

@Suite("DiffusionGemma", .serialized)
struct DiffusionGemmaTests {
    @Test("decodes dynamic mixed-bit OptiQ overrides without untyped dictionaries")
    func decodesQuantizationOverrides() throws {
        let data = Data(
            """
            {
              "model_type": "diffusion_gemma",
              "architectures": ["DiffusionGemmaForBlockDiffusion"],
              "canvas_length": 256,
              "eos_token_id": [1, 106],
              "text_config": {
                "model_type": "diffusion_gemma_text",
                "vocab_size": 262144,
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "num_hidden_layers": 1,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 0.000001,
                "max_position_embeddings": 262144,
                "pad_token_id": 0,
                "sliding_window": 1024,
                "layer_types": ["sliding_attention"],
                "final_logit_softcapping": 30.0,
                "num_experts": 128,
                "top_k_experts": 8,
                "rope_parameters": {
                  "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
                }
              },
              "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "model.decoder.embed_tokens": {"group_size": 64, "bits": 8}
              }
            }
            """.utf8
        )

        let config = try JSONDecoder().decode(DiffusionGemmaConfig.self, from: data)
        #expect(config.canvasLength == 256)
        #expect(config.textConfig.numExperts == 128)
        #expect(config.quantization.parameters(for: "model.decoder.embed_tokens").bits == 8)
        #expect(config.quantization.parameters(for: "model.decoder.layers.0.experts.gate_up_proj").bits == 4)
    }

    @Test("catalog pins the exact OptiQ revision and dedicated runtime")
    func catalogSpecIsPinnedAndAdditive() throws {
        let spec = try #require(ManagedModelCatalog.spec(for: DiffusionGemmaResources.modelID))
        #expect(spec.upstreamRepoId == DiffusionGemmaResources.upstreamModelID)
        #expect(spec.upstreamRevision == DiffusionGemmaResources.upstreamRevision)
        #expect(spec.validationKind == .diffusionGemma)
        #expect(spec.defaultRuntimeServingEngine == .textChatDiffusionGemma)
        #expect(spec.apiProfile?.inputModalities == [.text])
        #expect(spec.apiProfile?.supportsSeed == true)
        #expect(spec.estimatedDownloadBytes == 17_852_058_816)
    }

    @Test("masked drafts preserve revealed islands and mask unrevealed positions")
    func maskedDraftPreservesRevealedIslands() {
        let draft = DiffusionGemmaGenerator.maskedDraft(
            tokens: [10, 11, 12, 13],
            revealed: [true, false, true, false],
            decode: { $0.map(String.init).joined(separator: ",") }
        )

        #expect(draft == "10 [Mask] 12 [Mask]")
    }

    @Test("resource validation can separately require the vision sidecar")
    func resourceValidationSeparatesVision() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "diffusiongemma-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for relativePath in [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ] {
            FileManager.default.createFile(
                atPath: root.appending(path: relativePath).path,
                contents: Data()
            )
        }

        let resources = DiffusionGemmaResources(rootURL: root)
        #expect(resources.validate().isEmpty)
        #expect(resources.validate(requireVision: true) == [resources.visionWeightsURL])
    }

    @Test("hidden thinking removes Gemma channel markers")
    func hiddenThinkingRemovesChannelMarkers() {
        let decoded = "<|channel>thought\n<channel|>DiffusionGemma is running locally."

        #expect(
            DiffusionGemmaGenerator.cleanedResponse(decoded, showThinking: false)
                == "DiffusionGemma is running locally."
        )
    }
}

final class DiffusionGemmaKernelTests: MereRunCoreTestCase {
    func testSelectedLogitConfidenceMatchesFullSoftmax() {
        let logits = MLXArray([
            Float(0), 1, 2, 3,
            4, 3, 2, 1,
        ]).reshaped(1, 2, 4)
        let tokenIDs = MLXArray([Int32(3), 0]).reshaped(1, 2)
        let expected = takeAlong(
            softmax(logits, axis: -1),
            tokenIDs.expandedDimensions(axis: -1),
            axis: -1
        ).squeezed(axis: -1)
        let actual = DiffusionGemmaGenerator.tokenProbability(
            logits: logits,
            tokenIDs: tokenIDs
        )

        XCTAssertTrue(MLX.allClose(actual, expected, rtol: 1e-6, atol: 1e-6).item(Bool.self))
    }

    func testSortedExpertRoutingRestoresOriginalOrder() {
        XCTAssertFalse(DiffusionGemmaExpertRouting.shouldSort(routeCount: 63))
        XCTAssertTrue(DiffusionGemmaExpertRouting.shouldSort(routeCount: 64))

        let indices = MLXArray([Int32(3), 1, 3, 0, 2, 1, 0, 2]).reshaped(1, 2, 4)
        let plan = DiffusionGemmaExpertRouting.sortedPlan(indices: indices, topK: 4)
        MLX.eval(plan.order, plan.inverseOrder, plan.sortedIndices, plan.tokenOrder)

        XCTAssertEqual(plan.sortedIndices.asArray(Int32.self), [0, 0, 1, 1, 2, 2, 3, 3])
        XCTAssertEqual(
            take(plan.sortedIndices, plan.inverseOrder, axis: 0).asArray(Int32.self),
            indices.reshaped([indices.size]).asArray(Int32.self)
        )
    }
}
