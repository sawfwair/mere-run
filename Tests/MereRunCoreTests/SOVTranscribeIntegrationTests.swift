import AudioCore
@testable import AudioSTT
import Foundation
import MereRunCore
import XCTest

final class SOVTranscribeIntegrationTests: XCTestCase {
    func testAudioTokenTrimmingRetainsCompleteWindowsAndRoundsTheFinalTailUp() throws {
        func count(_ samples: Int, _ margin: Int?) throws -> Int {
            try SOVAudioTokenTrimming.retainedCount(sampleCount: samples, tokensPerWindow: 375, margin: margin)
        }
        XCTAssertEqual(try count(56_320, nil), 375)
        XCTAssertEqual(try count(56_320, 0), 44)
        XCTAssertEqual(try count(56_320, 12), 56)
        XCTAssertEqual(try count(56_320, 25), 69)
        XCTAssertEqual(try count(1, 0), 1)
        XCTAssertEqual(try count(1280, 0), 1)
        XCTAssertEqual(try count(1281, 0), 2)
        XCTAssertEqual(try count(480_000, 0), 375)
        XCTAssertEqual(try count(480_001, 0), 376)
        XCTAssertEqual(try count(480_001, 25), 401)
        XCTAssertEqual(try count(959_999, 25), 750)
        XCTAssertEqual(try count(28_800_000, 0), 22_500)
        XCTAssertEqual(try count(56_320, 375), 375)
        XCTAssertThrowsError(try count(56_320, -1))
        XCTAssertThrowsError(try count(56_320, 376))
        XCTAssertThrowsError(try count(0, 25))
    }

    func testPinnedTargetAndCompanionCatalogContracts() throws {
        let target = try XCTUnwrap(ManagedModelCatalog.spec(for: SOVTranscribeResources.repository))
        let draft = try XCTUnwrap(ManagedModelCatalog.spec(for: SOVTranscribeResources.draftRepository))
        XCTAssertNotNil(ManagedModelCapabilityCatalog.descriptor(for: target.id))
        XCTAssertNotNil(ManagedModelCapabilityCatalog.descriptor(for: draft.id))
        XCTAssertEqual(target.validationKind, .sovTranscribe)
        XCTAssertEqual(draft.validationKind, .sovTranscribeDFlash)
        XCTAssertFalse(target.runtimeAutoDownloadAllowed)
        XCTAssertFalse(draft.runtimeAutoDownloadAllowed)
        XCTAssertEqual(target.hubFallback?.revision, SOVTranscribeResources.revision)
        XCTAssertEqual(draft.hubFallback?.revision, SOVTranscribeResources.draftRevision)
        XCTAssertFalse(ManagedModelCatalog.allModelIDs.contains(SOVTranscribeResources.draftModelID))
        XCTAssertNil(ManagedModelCatalog.apiProfile(for: target.id))
    }

    func testLocalRootIsRecognizedAndMissingShardIsReported() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"sovereignai_transcribe"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        try Data(#"{"weight_map":{"model.audio_tower.conv1.weight":"model-00001-of-00001.safetensors"}}"#.utf8)
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        let route = try SpeechTranscriptionResolver.route(task: .transcribe, language: "Dutch", preferredBackend: .auto, modelOverride: root.path)
        XCTAssertEqual(route.decision.backend, .sov)
        XCTAssertEqual(route.modelOverride, root.path)
        let missing = SOVTranscribeResources.missingFiles(root: root, draft: false, fileManager: .default)
        XCTAssertTrue(missing.map(\.lastPathComponent).contains("model-00001-of-00001.safetensors"))
    }

    func testPromptAndWindowBoundaryMatchProcessorContract() throws {
        XCTAssertEqual(try SOVTranscribeGenerator.windowCount(sampleCount: 1), 1)
        XCTAssertEqual(try SOVTranscribeGenerator.windowCount(sampleCount: 480_000), 1)
        XCTAssertEqual(try SOVTranscribeGenerator.windowCount(sampleCount: 480_001), 2)
        XCTAssertThrowsError(try SOVTranscribeGenerator.windowCount(sampleCount: 0))
        XCTAssertThrowsError(try SOVTranscribeGenerator.windowCount(sampleCount: 28_800_001))
        let prompt = SOVTranscribeGenerator.prompt(instruction: nil, language: nil, audioTokens: 2)
        XCTAssertEqual(prompt, "<bos><|turn>user\nTranscribe the audio verbatim, outputting only spoken words in sequence.<|audio><|audio|><|audio|><audio|><turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        XCTAssertTrue(SOVTranscribeGenerator.prompt(instruction: "Use digits.", language: "fr", audioTokens: 1).contains("Use digits.<|audio>"))
        XCTAssertTrue(SOVTranscribeGenerator.prompt(instruction: nil, language: "de", audioTokens: 1).contains("German audio"))
    }

    func testRequestOptionsRoundTripAndLanguagePolicy() throws {
        let request = ASRRequest(audioURL: URL(fileURLWithPath: "/tmp/input.wav"),
                                 sov: SOVTranscribeOptions(instruction: "Keep fillers.", draftModel: "local-draft"))
        XCTAssertEqual(try JSONDecoder().decode(ASRRequest.self, from: JSONEncoder().encode(request)), request)
        XCTAssertEqual(try SOVTranscribeGenerator.validate(task: .transcribe, language: "Portuguese"), "pt")
        XCTAssertThrowsError(try SOVTranscribeGenerator.validate(task: .translate, language: "en"))
        XCTAssertThrowsError(try SOVTranscribeGenerator.validate(task: .transcribe, language: "Japanese"))
    }
}
