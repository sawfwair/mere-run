import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class Qwen3VLEmbeddingTests: MereRunCoreTestCase {
    func testPromptMatchesQwenRetrievalChatFraming() {
        let prompt = Qwen3VLEmbeddingPromptBuilder.prompt(
            instruction: nil,
            text: "a white SUV",
            imageTokenCounts: []
        )

        XCTAssertEqual(
            prompt,
            "<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
                + "<|im_start|>user\na white SUV<|im_end|>\n"
                + "<|im_start|>assistant\n"
        )
    }

    func testPromptExpandsEveryImageSpanAndNormalizesInstruction() {
        let prompt = Qwen3VLEmbeddingPromptBuilder.prompt(
            instruction: "Retrieve visually similar vehicles",
            text: "rear view",
            imageTokenCounts: [2, 1]
        )

        XCTAssertTrue(prompt.contains("Retrieve visually similar vehicles."))
        XCTAssertTrue(
            prompt.contains(
                "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
                    + "<|vision_start|><|image_pad|><|vision_end|>rear view"
            )
        )
    }

    func testImageTokenRangesRemainSeparate() {
        XCTAssertEqual(
            Qwen3VLEmbeddingPromptBuilder.contiguousRanges(
                in: [7, 42, 42, 8, 42, 9],
                matching: 42
            ),
            [1..<3, 4..<5]
        )
    }

    func testOfficialAndMLXWeightKeysMapToSharedNativeEncoder() {
        XCTAssertEqual(
            Qwen3VLEmbeddingWeights.mapWeightKey(
                "model.language_model.layers.0.self_attn.q_proj.weight"
            ),
            "textEncoder.encoder.layers.0.self_attn.q_proj.weight"
        )
        XCTAssertEqual(
            Qwen3VLEmbeddingWeights.mapWeightKey(
                "model.visual.merger.linear_fc1.weight"
            ),
            "visionTower.patch_merger.mlp_0.weight"
        )
        XCTAssertEqual(
            Qwen3VLEmbeddingWeights.mapWeightKey(
                "vision_tower.deepstack_merger_list.0.linear_fc2.weight"
            ),
            "visionTower.deepstack_merger_list.0.mlp_2.weight"
        )
    }

    func testOfficialVisionPatchKernelMapsToMLXConv3DLayout() throws {
        let official = MLXArray.zeros([4, 3, 2, 2, 2])
        let mapped = try XCTUnwrap(
            Qwen3VLEmbeddingWeights.mapWeight(
                "model.visual.patch_embed.proj.weight",
                official
            ).first
        )

        XCTAssertEqual(mapped.0, "visionTower.patch_embed.proj.weight")
        XCTAssertEqual(mapped.1.shape, [4, 2, 2, 2, 3])
    }

    func testManagedCatalogPinsOfficialTwoBCheckpoint() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Qwen3VLEmbeddingCatalog.modelID)
        )
        let manifest = MereRunModelManifest.template(
            for: .visionEmbedQwen3VL2B,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(spec.category, .visionEmbed)
        XCTAssertEqual(spec.validationKind, .qwen3VLEmbedding)
        XCTAssertEqual(spec.hubFallback?.repoId, "Qwen/Qwen3-VL-Embedding-2B")
        XCTAssertEqual(spec.hubFallback?.revision.count, 40)
        XCTAssertEqual(spec.defaultCLICommands, ["vision embed"])
        XCTAssertEqual(manifest.supports, [.textEmbedding, .multimodalEmbedding])
        XCTAssertEqual(manifest.precision, .bf16)
    }
}
