import Foundation
import XCTest
import MereRunContract
@testable import MereRunCore

final class NemotronBoltIntegrationTests: XCTestCase {
    func testCatalogPinsOriginalCheckpointWithoutAutoDownload() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: NemotronBoltResources.modelID))
        XCTAssertEqual(spec.category, .visionChat)
        XCTAssertEqual(spec.validationKind, .nemotronBolt)
        XCTAssertEqual(spec.hubFallback?.repoId, "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")
        XCTAssertEqual(spec.hubFallback?.revision, "96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6")
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, NemotronBoltResources.estimatedDownloadBytes)
        XCTAssertTrue(spec.companionModelIDs.isEmpty)
        let profile = try XCTUnwrap(spec.apiProfile)
        XCTAssertEqual(profile.inputModalities, [.text, .image])
        XCTAssertEqual(profile.outputModalities, [.text])
        XCTAssertTrue(profile.toolCall)
        XCTAssertTrue(profile.reasoning)
        XCTAssertFalse(profile.structuredOutput)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatNemotronH)
    }

    func testCommandAndResidentRuntimeSelectBoltRatherThanLightning() throws {
        XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: NemotronBoltResources.modelID), .nemotronBolt)
        XCTAssertEqual(ModelFamilyIdentifier.textChatFamily(matching: NemotronBoltResources.upstreamRepoID), .nemotronBolt)
        let runtime = NativeChatRuntime.make(engine: .textChatNemotronH, modelID: NemotronBoltResources.modelID, modelPath: nil)
        guard case .textChatNemotronBolt = runtime else { return XCTFail("Expected native Bolt generator") }
        let lightning = NativeChatRuntime.make(engine: .textChatNemotronH, modelID: NemotronHResources.modelID, modelPath: nil)
        guard case .textChatNemotronH = lightning else { return XCTFail("Expected existing Lightning generator") }
    }

    func testMissingShardCheckAndMalformedConfig() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let missing = NemotronBoltResources.missingTargetFiles(rootURL: root).map(\.lastPathComponent)
        XCTAssertTrue(missing.contains("config.json"))
        XCTAssertTrue(missing.contains("preprocessor_config.json"))
        XCTAssertFalse(NemotronBoltResources.validationMessages(rootURL: root).isEmpty)
    }
}

extension NemotronBoltIntegrationTests {
    func testImageGridUsesPinnedPythonTiesToEvenRounding() {
        let small = NemotronBoltImageProcessor.imagePatchGrid(width: 16, height: 32)
        XCTAssertEqual(small.width, 32)
        XCTAssertEqual(small.height, 32)
        let normal = NemotronBoltImageProcessor.imagePatchGrid(width: 768, height: 512)
        XCTAssertEqual(normal.width, 48)
        XCTAssertEqual(normal.height, 32)
    }
}

extension NemotronBoltIntegrationTests {
    func testManifestRetainsMixedPackingAndSourcePin() {
        let manifest = MereRunModelManifest.template(for: .nemotron35Bolt24GB)
        XCTAssertEqual(manifest.engine, .nemotronH)
        XCTAssertEqual(manifest.family, .nemotron)
        XCTAssertEqual(manifest.precision, .int2)
        XCTAssertEqual(manifest.quantization?.scheme, "ggml-iq2-xxs-q8-0-nvfp4-mixed")
        XCTAssertEqual(manifest.supports?.contains(.visionChat), true)
        XCTAssertEqual(manifest.supports?.contains(.videoUnderstanding), false)
        XCTAssertEqual(manifest.upstreamRepoId,
            "\(NemotronBoltResources.upstreamRepoID)@\(NemotronBoltResources.upstreamRevision)")
    }
}
