import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class QwenImage21IntegrationTests: XCTestCase {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "ImageRuntimeTests/Fixtures/QwenImage21")
    }

    func testManagedModelPinsTermsAndImageDefaults() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-qwen-21"))
        XCTAssertEqual(spec.upstreamRevision, QwenImage21Resources.revision)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        let manifest = MereRunModelManifest.template(for: .qwenImage21)
        XCTAssertEqual(try ImageGenerationBackend(manifest: manifest), .qwenImage21)
        XCTAssertEqual(manifest.components?.tokenizer, .local(path: "processor"))
        XCTAssertEqual(manifest.defaults?.steps, 40)
        XCTAssertEqual(manifest.defaults?.cfg, 1)
        XCTAssertEqual(Set(manifest.supports ?? []), [.txt2img, .referenceEdit])
        let options = ImageGenerationOptions(prompt: "A teapot", outputURL: URL(fileURLWithPath: "/tmp/teapot.png"))
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        XCTAssertEqual(ImageGenerationSampling.resolve(options, manifest: manifest).steps, 40)
    }

    func testReferencesAndAlphaOutputAreValidatedBeforeLoading() {
        let manifest = MereRunModelManifest.template(for: .qwenImage21)
        var options = ImageGenerationOptions(prompt: "A teapot", outputURL: URL(fileURLWithPath: "/tmp/teapot.jpg"), width: 1000, steps: 1)
        options.referenceImages = Array(repeating: URL(fileURLWithPath: "/tmp/ref.png"), count: 11)
        let codes = Set(ImageGenerationPlan.issues(options, manifest: manifest).map(\.code))
        XCTAssertTrue(codes.isSuperset(of: ["reference_count_invalid", "dimensions_invalid", "steps_invalid", "output_format_invalid"]))
    }

    func testResourceAdmissionRejectsMalformedIndexAndMissingShards() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "qwen21-resources-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root.appending(path: "text_encoder"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let index = root.appending(path: "text_encoder/model.safetensors.index.json")
        try Data("{}".utf8).write(to: index)
        XCTAssertTrue(QwenImage21Resources(rootURL: root).validate().contains(index))
        try Data(#"{"weight_map":{"weight":"model-1.safetensors"}}"#.utf8).write(to: index)
        XCTAssertTrue(QwenImage21Resources(rootURL: root).validate().contains(root.appending(path: "text_encoder/model-1.safetensors")))
        try Data(#"{"weight_map":{"weight":"../outside.safetensors"}}"#.utf8).write(to: index)
        XCTAssertTrue(QwenImage21Resources(rootURL: root).validate().contains(index))
    }

    func testFlowScheduleMatchesIndependentDiffusersFixture() throws {
        struct Fixture: Decodable { let config: QwenImage21Scheduler; let sigmas: [Float] }
        let data = try Data(contentsOf: fixtures.appending(path: "scheduler.json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        let actual = try fixture.config.sigmas(steps: 40, tokenCount: 4096)
        XCTAssertEqual(actual.count, fixture.sigmas.count)
        for (value, expected) in zip(actual, fixture.sigmas) { XCTAssertEqual(value, expected, accuracy: 2e-6) }
        XCTAssertThrowsError(try fixture.config.sigmas(steps: 1, tokenCount: 4096))
    }

    func testRGBAResamplingMatchesPillowFixture() throws {
        struct Fixture: Decodable { let input: [UInt8]; let output: [UInt8] }
        let data = try Data(contentsOf: fixtures.appending(path: "resize.json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        let source = try MediaImage(width: 7, height: 5, rgba8: fixture.input)
        let resized = try QwenImage21ImageIO.resized(source, width: 4, height: 8)
        XCTAssertEqual(resized.rgba8, fixture.output)
    }
}
