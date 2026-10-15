import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class Flux3IntegrationTests: XCTestCase {
    func testVerifiedTextConfigurationUsesEightIntermediateLayers() throws {
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appending(path: "Fixtures/Flux3/text-encoder-config.json")
        let config = try JSONDecoder().decode(Qwen3VLEmbeddingRootConfig.self, from: Data(contentsOf: file)).textConfig
        XCTAssertEqual(config.numHiddenLayers, 36)
        XCTAssertEqual(config.hiddenSize * Flux3PromptPlan.outputLayers.count, 20480)
        XCTAssertLessThan(try XCTUnwrap(Flux3PromptPlan.outputLayers.last), config.numHiddenLayers)
        XCTAssertEqual(Flux3PromptPlan.presentation("A camera"),
                       "<|im_start|>user\nA camera<|im_end|>\n<|im_start|>assistant\n")
    }

    func testCatalogManifestAndSamplingKeepFlux3SeparateFromFlux2() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Flux3Resources.modelID))
        XCTAssertEqual(spec.upstreamRevision, "8454017f9ebaada30c998fccd9f67a80c7580f68")
        XCTAssertEqual(spec.validationKind, .flux3)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertFalse(Flux3Resources.patterns.contains { $0.hasPrefix("variants/") || $0.hasPrefix("transformer/") })
        let manifest = MereRunModelManifest.template(for: .flux3Klein)
        XCTAssertEqual(try ImageGenerationBackend(manifest: manifest), .flux3)
        XCTAssertEqual(manifest.engine, .flux3)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.supports, [.txt2img, .referenceEdit])
        let options = ImageGenerationOptions(prompt: "A brass camera", outputURL: URL(fileURLWithPath: "/tmp/flux3-test.png"))
        let sampling = ImageGenerationSampling.resolve(options, manifest: manifest)
        XCTAssertEqual(sampling.steps, 16)
        XCTAssertEqual(sampling.guidanceScale, 1)
        XCTAssertNil(sampling.sigmaShift)
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        XCTAssertEqual(ImageGenerationConditioning.inputMode(family: .flux3,
            inputImage: URL(fileURLWithPath: "/tmp/camera.png"), referenceImages: []), "reference_image")
        XCTAssertEqual(ImageGenerationConditioning.inputMode(family: .flux3,
            inputImage: nil, referenceImages: [URL(fileURLWithPath: "/tmp/camera.png")]), "reference_image")
    }

    func testInstalledRootUsesBuiltInScheduler() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "flux3-install-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appending(path: "text_encoder"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for path in ["config.json", "model.safetensors", "ae.safetensors", "text_encoder/config.json",
                     "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json", "text_encoder/model.safetensors"] {
            try Data("{}".utf8).write(to: root.appending(path: path))
        }
        try MereRunModelManifest.template(for: .flux3Klein).write(to: root)
        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Flux3Resources.modelID)
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertNil(report.manifest?.components?.scheduler)
    }

    func testUnsupportedOptionsAndEleventhReferenceFailBeforeModelLoad() {
        let manifest = MereRunModelManifest.template(for: .flux3Klein)
        var options = ImageGenerationOptions(prompt: "A brass camera", outputURL: URL(fileURLWithPath: "/tmp/flux3-test.png"))
        options.referenceImages = (0..<10).map { URL(fileURLWithPath: "/tmp/ref-\($0).png") }
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        options.inputImage = URL(fileURLWithPath: "/tmp/first.png")
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).contains { $0.code == "reference_count_invalid" })
        options.referenceImages = []
        options.guidanceScale = 3.5
        options.negativePrompt = "blurry"
        options.strength = 0.5
        options.sigmaShift = 3
        options.width = 1000
        let codes = Set(ImageGenerationPlan.issues(options, manifest: manifest).map(\.code))
        XCTAssertTrue(["cfg_invalid", "negative_prompt_unsupported", "strength_unsupported", "sigma_shift_unsupported", "dimensions_invalid"].allSatisfy(codes.contains))
    }

    func testResourceValidationRejectsMissingAndEscapingTextShards() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "flux3-resources-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appending(path: "text_encoder"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = Flux3Resources(rootURL: root)
        XCTAssertTrue(resources.validate().contains(root.appending(path: "model.safetensors")))
        let index = root.appending(path: "text_encoder/model.safetensors.index.json")
        try Data(#"{"weight_map":{"model.language_model.norm.weight":"../escape.safetensors"}}"#.utf8).write(to: index)
        XCTAssertEqual(resources.validate(), [index])
    }

    func testReferenceCanvasCropReflectPaddingAndOrientation() throws {
        let single = Flux3ImageIO.canvas(width: 1001, height: 667, multiple: false)
        XCTAssertEqual(single.fitWidth, 1001)
        XCTAssertEqual(single.width, 1008)
        XCTAssertEqual(single.height, 672)
        let large = Flux3ImageIO.canvas(width: 8000, height: 4000, multiple: false)
        XCTAssertLessThanOrEqual(large.width * large.height, 4 * 1024 * 1024)
        XCTAssertLessThanOrEqual(max(large.width, large.height), 4096)
        let multiple = Flux3ImageIO.canvas(width: 2000, height: 1000, multiple: true)
        XCTAssertLessThanOrEqual(multiple.fitWidth * multiple.fitHeight, 1024 * 1024)
        XCTAssertTrue(multiple.width.isMultiple(of: 16) && multiple.height.isMultiple(of: 16))
        let bytes: [UInt8] = (0..<6).flatMap { value -> [UInt8] in [UInt8(value), 0, 0, 255] }
        let image = try MediaImage(width: 3, height: 2, rgba8: bytes)
        let rotated = try Flux3ImageIO.oriented(image, orientation: 6)
        XCTAssertEqual(rotated.width, 2)
        XCTAssertEqual(rotated.height, 3)
        XCTAssertEqual(stride(from: 0, to: 24, by: 4).map { rotated.rgba8[$0] }, [3, 0, 4, 1, 5, 2])
        let padded = try Flux3ImageIO.paddedOrCropped(image, width: 5, height: 4, reflect: true)
        XCTAssertEqual(stride(from: 0, to: 20, by: 4).map { padded.rgba8[$0] }, [0, 0, 1, 2, 2])
    }
}
