import AudioCore
import AudioSTT
import Foundation
import XCTest
@testable import MereRunCore

final class NemotronTranscribeRoutingTests: XCTestCase {
    func testManagedAndUpstreamSelectionRetainsBackendForBothTasks() throws {
        for id in [NemotronTranscribeResources.modelID, NemotronTranscribeResources.repository] {
            for task in [ASRTask.transcribe, .translate] {
                let route = try SpeechTranscriptionResolver.route(task: task, language: "Japanese", preferredBackend: .auto, modelOverride: id)
                XCTAssertEqual(route.decision.backend, .nemotron)
                XCTAssertEqual(route.modelOverride, id)
                XCTAssertEqual(route.decision.normalizedLanguageHint, "ja")
            }
        }
    }

    func testLocalCheckpointDetectionAndBackendMismatch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"architectures":["Nemotron3_5TranscribeForConditionalGeneration"],"model_type":"nemo_speechlm"}"#.utf8)
            .write(to: root.appendingPathComponent("config.json"))
        let route = try SpeechTranscriptionResolver.route(task: .transcribe, language: "zh", preferredBackend: .auto, modelOverride: root.path)
        XCTAssertEqual(route.decision.backend, .nemotron)
        XCTAssertThrowsError(try SpeechTranscriptionResolver.route(task: .transcribe, language: "en", preferredBackend: .qwen, modelOverride: root.path))
    }

    func testLanguageAndProviderValidationBeforeLoading() {
        XCTAssertThrowsError(try SpeechTranscriptionResolver.route(task: .transcribe, language: "Klingon", preferredBackend: .nemotron))
        XCTAssertThrowsError(try SpeechTranscriptionResolver.route(task: .translate, language: "en", preferredBackend: .nemotron,
            parakeetExecutionProvider: .coreML(artifactURL: URL(fileURLWithPath: "/tmp/unused"))))
        XCTAssertEqual(try NemotronTranscribeGenerator.validate(language: "auto"), nil)
    }

    func testManagedCatalogPinsOriginalBF16CheckpointAndLicense() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: NemotronTranscribeResources.modelID))
        XCTAssertEqual(spec.validationKind, .nemotronTranscribe)
        XCTAssertEqual(spec.upstreamRevision, "ef905575a106fdad904f3b0ce661dccad654607d")
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, 67_496_013_734)
        let manifest = MereRunModelManifest.template(for: .nemotronTranscribeASR)
        XCTAssertEqual(manifest.engine, .nemotronTranscribeASR)
        XCTAssertEqual(manifest.precision, .bf16)
    }

    func testDefaultRoutingStillPrefersParakeetAndQwen() throws {
        XCTAssertEqual(try SpeechTranscriptionResolver.route(task: .transcribe, language: nil, preferredBackend: .auto).decision.backend, .parakeet)
        XCTAssertEqual(try SpeechTranscriptionResolver.route(task: .translate, language: nil, preferredBackend: .auto).decision.backend, .qwen)
    }
}
