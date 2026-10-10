import AudioCore
import AudioSTT
import Foundation
import XCTest
@testable import MereRunCore

final class WhistleRoutingTests: XCTestCase {
    func testManagedAndUpstreamIDsSelectWhistle() throws {
        for id in ["speech-asr-whistle", "Cactus-Compute/whistle"] {
            let route = try SpeechTranscriptionResolver.route(
                task: .transcribe, language: "German", preferredBackend: .auto, modelOverride: id
            )
            XCTAssertEqual(route.decision.backend, .whistle)
            XCTAssertEqual(route.decision.normalizedLanguageHint, "de")
            XCTAssertEqual(route.modelOverride, id)
        }
    }

    func testWhistleRejectsUnsupportedTaskLanguageAndProviderBeforeLoading() {
        for task in [ASRTask.transcribe, .translate] {
            XCTAssertThrowsError(try SpeechTranscriptionResolver.route(
                task: task, language: "zh", preferredBackend: .whistle
            ))
        }
        XCTAssertThrowsError(try SpeechTranscriptionResolver.route(
            task: .translate, language: "en", preferredBackend: .whistle
        ))
        XCTAssertThrowsError(try SpeechTranscriptionResolver.route(
            task: .transcribe, language: "en", preferredBackend: .whistle,
            parakeetExecutionProvider: .coreML(artifactURL: URL(fileURLWithPath: "/tmp/unused"))
        ))
    }

    func testWhistleOptionsAreValidatedBeforeExecution() throws {
        let audio = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try Data().write(to: audio)
        defer { try? FileManager.default.removeItem(at: audio) }
        XCTAssertThrowsError(try SpeechTranscriptionResolver.resolve(
            request: ASRRequest(audioURL: audio, whistle: WhistleOptions(beamSize: 9)), preferredBackend: .whistle
        )) { error in
            XCTAssertEqual((error as? SpeechTranscriptionIssue)?.code, "invalid_whistle_options")
        }
    }

    func testWhistleDoesNotChangeAutomaticDefault() throws {
        let route = try SpeechTranscriptionResolver.route(task: .transcribe, language: nil, preferredBackend: .auto)
        XCTAssertEqual(route.decision.backend, .parakeet)
    }

    func testManagedInstallRequiresCompactVocabularyAndConfigOnly() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "speech-asr-whistle"))
        XCTAssertEqual(spec.validationKind, .whistle)
        XCTAssertEqual(spec.upstreamRevision, "ca5287601bef25af26dcf2e1b2bdc0843a7c19e5")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertEqual(Set(spec.missingPaths(in: root).map { $0.path.replacingOccurrences(of: root.path + "/", with: "") }),
                       ["config.json", "whistle.cact"])
        try FileManager.default.createDirectory(at: root.appendingPathComponent("checkpoints"), withIntermediateDirectories: true)
        for path in ["config.json", "whistle.cact"] {
            try Data("fixture".utf8).write(to: root.appendingPathComponent(path))
        }
        try MereRunModelManifest.template(for: .whistleASR).write(to: root)
        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: spec.id)
        XCTAssertTrue(report.isValid, "\(report.errors)")
    }
}
