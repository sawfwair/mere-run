import Foundation
import XCTest
@testable import MereRunCore

final class ManagedCosmos3SuperTests: XCTestCase {
    func testPinnedImageCatalogUsesConvertedArtifactAndNativeCommand() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Cosmos3Resources.superTextToImageMLXRepository))
        XCTAssertEqual(spec.id, ModelResolver.ModelID.cosmos3SuperTextToImage.rawValue)
        XCTAssertEqual(spec.category, .image)
        XCTAssertEqual(spec.validationKind, .cosmos3SuperTextToImage)
        XCTAssertEqual(spec.hubFallback?.revision, Cosmos3Resources.superTextToImageMLXRevision)
        XCTAssertEqual(spec.estimatedDownloadBytes, 37_032_673_013)
        XCTAssertEqual(spec.apiAvailability, .cliOnly)
        XCTAssertNil(spec.apiProfile)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.defaultCLICommands, ["video cosmos3 --mode text-to-image"])
        let manifest = MereRunModelManifest.template(for: .cosmos3SuperTextToImage)
        XCTAssertEqual(manifest.engine, .cosmos3Super)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.defaults?.steps, 4)
        XCTAssertEqual(manifest.defaults?.cfg, 1)
        XCTAssertEqual(manifest.supports, [.txt2img])
    }

    func testGenerationOnlyExportDoesNotRequireEdgeReasoner() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = Cosmos3Resources(rootURL: root)
        XCTAssertTrue(resources.validateSuperTextToImage().isEmpty)
        XCTAssertTrue(try resources.superTextToImageValidationIssues().isEmpty)
        XCTAssertFalse(resources.validateReasoner().isEmpty)
    }

    func testMissingIndexedShardIsReported() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let transformer = root.appendingPathComponent("transformer")
        try FileManager.default.removeItem(at: transformer.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        try Data(#"{"weight_map":{"layer.weight":"model-00001-of-00001.safetensors"}}"#.utf8).write(
            to: transformer.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        )
        XCTAssertEqual(Cosmos3Resources(rootURL: root).validateSuperTextToImage().map(\.lastPathComponent), [
            "model-00001-of-00001.safetensors"
        ])
    }

    func testRejectsNonDistilledOrWrongPrecisionExport() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let config = root.appendingPathComponent("transformer/config.json")
        let original = try String(contentsOf: config, encoding: .utf8)
        try original.replacingOccurrences(of: "\"bits\": 4", with: "\"bits\": 8")
            .write(to: config, atomically: true, encoding: .utf8)
        XCTAssertTrue(try Cosmos3Resources(rootURL: root).superTextToImageValidationIssues()
            .contains { $0.contains("INT4") })
        try original.write(to: config, atomically: true, encoding: .utf8)
        try Data(#"{"is_distilled":false}"#.utf8).write(to: root.appendingPathComponent("modular_model_index.json"))
        XCTAssertTrue(try Cosmos3Resources(rootURL: root).superTextToImageValidationIssues()
            .contains { $0.contains("four-step") })
    }

    private func makeFixture() throws -> URL {
        let source = try XCTUnwrap(Bundle.module.url(
            forResource: "Cosmos3SuperCatalog", withExtension: nil, subdirectory: "Fixtures"
        ))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.copyItem(at: source, to: root)
        for path in [
            "transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json", "text_tokenizer/tokenizer_config.json", "text_tokenizer/tokenizer.json",
        ] {
            let file = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: file)
        }
        return root
    }
}
