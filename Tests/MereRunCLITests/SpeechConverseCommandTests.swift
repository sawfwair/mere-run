import MereRunCore
import XCTest
@testable import MereRunCLI

final class SpeechConverseCommandTests: XCTestCase {
    func testReplayParsesAndValidates() throws {
        let command = try SpeechConverse.parse([
            "--input", "user.wav", "-o", "reply.wav", "--seed", "42",
            "--reference-audio", "voice.wav", "--temperature", "0", "--max-frames", "3",
        ])
        XCTAssertEqual(command.model, HaruResources.modelID)
        XCTAssertEqual(command.seed, 42)
        XCTAssertEqual(command.referenceAudio, "voice.wav")
        XCTAssertEqual(command.temperature, 0)
        XCTAssertEqual(command.maxFrames, 3)
    }

    func testRequiresOneInputModeAndAFileReplayDestination() {
        XCTAssertThrowsError(try SpeechConverse.parse([]))
        XCTAssertThrowsError(try SpeechConverse.parse(["--input", "user.wav"]))
        XCTAssertThrowsError(try SpeechConverse.parse(["--input", "user.wav", "--live", "-o", "reply.wav"]))
        XCTAssertThrowsError(try SpeechConverse.parse(["--input", "user.wav", "-o", "reply.wav", "--max-frames", "0"]))
    }

    func testManagedModelIsPinnedAndResolvesToConversation() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: HaruResources.modelID))
        XCTAssertEqual(spec.hubFallback?.repoId, "higurashi-ai/Haru-4B")
        XCTAssertEqual(spec.hubFallback?.revision, "8d6e83bcd3701a76fc27a764965665da1df7da78")
        XCTAssertEqual(spec.defaultCLICommands, ["speech converse"])
        XCTAssertEqual(spec.apiAvailability, .cliOnly)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        let manifest = MereRunModelManifest.template(for: .haru4B)
        XCTAssertEqual(manifest.engine, .haru)
        XCTAssertEqual(manifest.supports, [.speechConversation])
        XCTAssertNotNil(CLIInferenceAdmissionClassifier.request(arguments: [
            "mere.run", "speech", "converse", "--model", "/local/haru", "--input", "user.wav", "--output", "reply.wav",
        ]))
    }
}
