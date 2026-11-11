import Foundation
import XCTest
@testable import MereRunCore

final class H3ImageIntegrationTests: XCTestCase {
    func testPinnedCatalogUsesSeparateImageEngineAndSixStepDefaults() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MiniMaxH3ImageResources.modelID))
        XCTAssertEqual(spec.upstreamRevision, "492fd8c4eca666f1085f57e30738c1e0d86b3439")
        XCTAssertEqual(spec.validationKind, .miniMaxH3Image)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertFalse(MiniMaxH3ImageResources.patterns.contains { $0.hasPrefix("transformer_diffusers") })
        let manifest = MereRunModelManifest.template(for: .miniMaxH3Image)
        XCTAssertEqual(try ImageGenerationBackend(manifest: manifest), .miniMaxH3Image)
        XCTAssertEqual(manifest.family, .h3Image)
        XCTAssertEqual(manifest.engine, .miniMaxH3Image)
        XCTAssertEqual(manifest.supports, [.txt2img, .referenceEdit])
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: URL(fileURLWithPath: "/tmp/h3-image.png"))
        let sampling = ImageGenerationSampling.resolve(options, manifest: manifest)
        XCTAssertEqual(sampling.steps, 6)
        XCTAssertEqual(sampling.guidanceScale, 1)
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        XCTAssertEqual(ModelFamilyIdentifier.imageGenerateFamily(manifest), "h3-image")
        XCTAssertEqual(ImageGenerationConditioning.inputMode(family: .h3Image,
            inputImage: URL(fileURLWithPath: "/tmp/ref.png"), referenceImages: []), "reference_image")
    }

    func testReferencesAndUnsupportedSamplingAreValidatedBeforeLoading() {
        let manifest = MereRunModelManifest.template(for: .miniMaxH3Image)
        var options = ImageGenerationOptions(prompt: "A camera", outputURL: URL(fileURLWithPath: "/tmp/h3-image.png"))
        options.inputImage = URL(fileURLWithPath: "/tmp/first.png")
        options.referenceImages = (0..<15).map { URL(fileURLWithPath: "/tmp/ref-\($0).png") }
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        options.referenceImages.append(URL(fileURLWithPath: "/tmp/seventeenth.png"))
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).contains { $0.code == "reference_count_invalid" })
        options.steps = 7
        options.guidanceScale = 2
        options.sigmaShift = 3
        options.strength = 0.5
        options.negativePrompt = "blur"
        options.width = Int.max
        options.height = Int.max
        let codes = Set(ImageGenerationPlan.issues(options, manifest: manifest).map(\.code))
        XCTAssertTrue(["steps_invalid", "cfg_invalid", "sigma_shift_unsupported", "strength_unsupported",
                       "negative_prompt_unsupported", "dimensions_invalid"].allSatisfy(codes.contains))
    }

    func testResourcesRequireImageVAEHeadsAndSafeCompleteShardPaths() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "h3-image-resources-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/H3Image")
        for directory in ["transformer", "text_encoder", "tokenizer", "vae"] {
            try FileManager.default.createDirectory(at: root.appending(path: directory), withIntermediateDirectories: true)
        }
        for (name, path) in [("transformer.json", "transformer/config.json"), ("vae.json", "vae/config.json"), ("pdd.json", "pdd_config.json")] {
            try FileManager.default.copyItem(at: fixture.appending(path: name), to: root.appending(path: path))
        }
        for path in ["LICENSE", "vae/diffusion_pytorch_model.safetensors", "pdd_fused_heads.safetensors",
                     "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "text_encoder/config.json",
                     "transformer/model.safetensors", "text_encoder/model.safetensors"] {
            try Data().write(to: root.appending(path: path))
        }
        for component in ["transformer", "text_encoder"] {
            try Data(#"{"weight_map":{"weight":"model.safetensors"}}"#.utf8)
                .write(to: root.appending(path: "\(component)/model.safetensors.index.json"))
        }
        let resources = MiniMaxH3ImageResources(rootURL: root)
        XCTAssertTrue(resources.validate().isEmpty)
        let q8Spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MiniMaxH3ImageResources.q8ModelID))
        XCTAssertEqual(q8Spec.missingPaths(in: root), [root.appending(path: MiniMaxH3ImageMLXFormat.filename)])
        XCTAssertEqual(MereRunModelManifest.template(for: .miniMaxH3ImageQ8).precision, .int8)
        XCTAssertTrue(try resources.configuration().isReleasedArchitecture)
        try MereRunModelManifest.template(for: .miniMaxH3Image).write(to: root)
        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: MiniMaxH3ImageResources.modelID)
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        try FileManager.default.removeItem(at: root.appending(path: "pdd_fused_heads.safetensors"))
        XCTAssertEqual(resources.validate(), [root.appending(path: "pdd_fused_heads.safetensors")])
        for path in ["../escape.safetensors", "/escape.safetensors", "nested/shard.safetensors", "..\\escape.safetensors"] {
            let data = try JSONEncoder().encode(["weight_map": ["weight": path]])
            try data.write(to: root.appending(path: "transformer/model.safetensors.index.json"))
            XCTAssertThrowsError(try resources.index("transformer"))
        }
    }
}
