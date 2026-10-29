import AudioCore
import AudioSTT
import Foundation
import XCTest
@testable import MereRunCore

final class NemotronTranscribeRoutingTests: XCTestCase {
    func testManagedAndUpstreamSelectionRetainsBackendForBothTasks() throws {
        let audio = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try Data().write(to: audio)
        defer { try? FileManager.default.removeItem(at: audio) }
        for id in [NemotronTranscribeResources.modelID, NemotronTranscribeResources.repository,
                   NemotronTranscribeResources.bf16ModelID, NemotronTranscribeResources.quantizedRepository] {
            for task in [ASRTask.transcribe, .translate] {
                let route = try SpeechTranscriptionResolver.route(task: task, language: "Japanese", preferredBackend: .auto, modelOverride: id)
                XCTAssertEqual(route.decision.backend, .nemotron)
                XCTAssertEqual(route.modelOverride, id)
                XCTAssertEqual(route.decision.normalizedLanguageHint, "ja")
                let request = ASRRequest(audioURL: audio, language: "Japanese", task: task)
                let plan = try SpeechTranscriptionResolver.resolve(request: request, preferredBackend: .auto, modelOverride: id)
                XCTAssertEqual(plan.modelID, NemotronTranscribeResources.managedModelID(for: id))
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
        let audio = root.appendingPathComponent("audio.wav")
        try Data().write(to: audio)
        let request = ASRRequest(audioURL: audio)
        XCTAssertEqual(try SpeechTranscriptionResolver.resolve(request: request, preferredBackend: .auto, modelOverride: root.path).modelID,
                       NemotronTranscribeResources.bf16ModelID)
        try Data("{}".utf8).write(to: root.appendingPathComponent("expert-quantization.json"))
        XCTAssertEqual(try SpeechTranscriptionResolver.resolve(request: request, preferredBackend: .auto, modelOverride: root.path).modelID,
                       NemotronTranscribeResources.modelID)
    }

    func testLanguageAndProviderValidationBeforeLoading() {
        XCTAssertThrowsError(try SpeechTranscriptionResolver.route(task: .transcribe, language: "Klingon", preferredBackend: .nemotron))
        XCTAssertThrowsError(try SpeechTranscriptionResolver.route(task: .translate, language: "en", preferredBackend: .nemotron,
            parakeetExecutionProvider: .coreML(artifactURL: URL(fileURLWithPath: "/tmp/unused"))))
        XCTAssertEqual(try NemotronTranscribeGenerator.validate(language: "auto"), nil)
    }

    func testManagedCatalogDefaultsToPublishedINT4AndKeepsBF16Option() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: NemotronTranscribeResources.modelID))
        XCTAssertEqual(spec.validationKind, .nemotronTranscribe)
        XCTAssertEqual(spec.hubFallback?.repoId, NemotronTranscribeResources.quantizedRepository)
        XCTAssertEqual(spec.hubFallback?.revision, NemotronTranscribeResources.quantizedRevision)
        XCTAssertNotNil(NemotronTranscribeResources.quantizedRevision.range(of: "^[0-9a-f]{40}$", options: .regularExpression))
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, NemotronTranscribeResources.quantizedDownloadBytes)
        let manifest = MereRunModelManifest.template(for: .nemotronTranscribeASR)
        XCTAssertEqual(manifest.engine, .nemotronTranscribeASR)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        let bf16 = try XCTUnwrap(ManagedModelCatalog.spec(for: NemotronTranscribeResources.bf16ModelID))
        XCTAssertEqual(bf16.hubFallback?.repoId, NemotronTranscribeResources.repository)
        XCTAssertEqual(bf16.hubFallback?.revision, "ef905575a106fdad904f3b0ce661dccad654607d")
        XCTAssertEqual(bf16.estimatedDownloadBytes, 67_496_013_734)
        XCTAssertFalse(bf16.runtimeAutoDownloadAllowed)
        XCTAssertEqual(MereRunModelManifest.template(for: .nemotronTranscribeBF16ASR).precision, .bf16)
        XCTAssertEqual(ManagedModelCatalog.spec(for: NemotronTranscribeResources.repository)?.id, bf16.id)
        XCTAssertEqual(ManagedModelCatalog.spec(for: NemotronTranscribeResources.quantizedRepository)?.id, spec.id)
    }

    func testDefaultRoutingStillPrefersParakeetAndQwen() throws {
        XCTAssertEqual(try SpeechTranscriptionResolver.route(task: .transcribe, language: nil, preferredBackend: .auto).decision.backend, .parakeet)
        XCTAssertEqual(try SpeechTranscriptionResolver.route(task: .translate, language: nil, preferredBackend: .auto).decision.backend, .qwen)
    }
}
