import Foundation
import XCTest
@testable import MereRunCore

final class LightOnOCR3Tests: XCTestCase {
    private func configData(_ size: String) throws -> Data {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/LightOnOCR3")
        return try Data(contentsOf: directory.appendingPathComponent("\(size)-config.json"))
    }

    func testPublishedConfigsSelectTheirNativeArchitectures() throws {
        let oneBData = try configData("1B")
        let oneB = try JSONDecoder().decode(LightOnOCRConfig.self, from: oneBData)
        XCTAssertEqual(oneB.textConfig.hiddenSize, 1024)
        XCTAssertEqual(oneB.visionConfig.patchSize, 14)
        XCTAssertEqual(oneB.spatialMergeSize, 2)
        XCTAssertEqual(try JSONDecoder().decode(LightOnOCRResources.Architecture.self, from: oneBData).modelType, .mistral3)
        for size in ["0.8B", "4B"] {
            let data = try configData(size)
            XCTAssertEqual(try JSONDecoder().decode(LightOnOCRResources.Architecture.self, from: data).modelType, .qwen35)
            let config = try JSONDecoder().decode(Q35Config.self, from: data)
            XCTAssertFalse(config.textConfig.usesMoE)
            XCTAssertEqual(config.visionConfig?.patchSize, 16)
            XCTAssertEqual(config.imageTokenId, 248056)
            XCTAssertEqual(config.textConfig.ropeParameters.mropeSection, [11, 11, 10])
        }
        XCTAssertThrowsError(try JSONDecoder().decode(
            LightOnOCRResources.Architecture.self,
            from: Data(#"{"model_type":"unsupported"}"#.utf8)
        ))
    }

    func testPixtralPromptsMatchOfficialJinjaRendering() {
        let prefix = "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|>"
        let suffix = "<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertEqual(LightOnOCRGenerator.ocrPrompt(mode: .plain), prefix + suffix)
        XCTAssertEqual(LightOnOCRGenerator.ocrPrompt(mode: .grounding), prefix + "grounding\n" + suffix)
        XCTAssertEqual(LightOnOCRMode.plain.prompt, "")
        XCTAssertEqual(LightOnOCRMode.grounding.prompt, "grounding")
    }

    func testReleasedAndPriorTextPrefixesMapToActualEncoderKeys() {
        XCTAssertEqual(
            LightOnOCRGenerator.textWeightKey("language_model.model.layers.0.self_attn.q_norm.weight"),
            "encoder.layers.0.self_attn.q_norm.weight"
        )
        XCTAssertEqual(
            LightOnOCRGenerator.textWeightKey("language_model.embed_tokens.weight"),
            "encoder.embed_tokens.weight"
        )
        XCTAssertNil(LightOnOCRGenerator.textWeightKey("vision_encoder.patch_conv.weight"))
    }

    func testManagedPinsAndManifestEnginesAgree() throws {
        let oneB = try XCTUnwrap(ManagedModelCatalog.spec(for: "vision-ocr-lighton"))
        XCTAssertEqual(oneB.upstreamRepoId, "lightonai/LightOnOCR-3-1B")
        XCTAssertEqual(oneB.upstreamRevision, LightOnOCRResources.oneBRevision)
        XCTAssertEqual(oneB.hubFallback?.revision, LightOnOCRResources.oneBRevision)
        for id in [ModelResolver.ModelID.lightOnOCR3Small, .lightOnOCR3FourB] {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id.rawValue))
            let profile = try XCTUnwrap(Q35Resources.profile(for: id.rawValue))
            XCTAssertEqual(spec.upstreamRepoId, profile.upstreamRepoId)
            XCTAssertEqual(spec.upstreamRevision, profile.upstreamRevision)
            XCTAssertEqual(spec.upstreamRevision?.count, 40)
            let bounds = Q35Resources.visionPixelBounds(forModelId: id.rawValue)
            XCTAssertEqual(bounds.minimum, 65_536)
            XCTAssertEqual(bounds.maximum, 5_000_000)
            let manifest = MereRunModelManifest.template(for: id)
            XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
            XCTAssertEqual(manifest.supports, [.visionOCR])
            XCTAssertEqual(manifest.upstreamRepoId, "\(profile.upstreamRepoId)@\(profile.upstreamRevision)")
        }
    }
}
