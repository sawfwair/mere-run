import XCTest
@testable import MereRunCLI

final class SpeechSynthesizeCommandParsingTests: XCTestCase {
    func testSpeechSynthesizeDefaultsToStyleMode() throws {
        let cmd = try SpeechSynthesize.parse([
            "Hello from mere.run",
            "--output", "/tmp/out.wav",
        ])

        XCTAssertEqual(cmd.mode, .style)
        XCTAssertEqual(cmd.language, "auto")
        XCTAssertNil(cmd.profile)
        XCTAssertNil(cmd.refAudio)
        XCTAssertNil(cmd.refText)
        XCTAssertNil(cmd.saveProfile)
        XCTAssertFalse(cmd.stream)
        XCTAssertEqual(cmd.streamChunkTokens, 25)
    }

    func testSpeechSynthesizeParsesCloneModeOptions() throws {
        let cmd = try SpeechSynthesize.parse([
            "Clone this text",
            "--output", "/tmp/out.wav",
            "--mode", "clone",
            "--profile", "narrator",
            "--ref-audio", "/tmp/ref.wav",
            "--ref-text", "reference transcript",
            "--language", "en",
            "--save-profile", "my-clone",
        ])

        XCTAssertEqual(cmd.mode, .clone)
        XCTAssertEqual(cmd.profile, "narrator")
        XCTAssertEqual(cmd.refAudio, "/tmp/ref.wav")
        XCTAssertEqual(cmd.refText, "reference transcript")
        XCTAssertEqual(cmd.language, "en")
        XCTAssertEqual(cmd.saveProfile, "my-clone")
    }

    func testSpeechSynthesizeParsesStreamingOptions() throws {
        let cmd = try SpeechSynthesize.parse([
            "Stream this",
            "--output", "/tmp/out.wav",
            "--stream",
            "--stream-chunk-tokens", "30",
        ])

        XCTAssertTrue(cmd.stream)
        XCTAssertEqual(cmd.streamChunkTokens, 30)
    }

    func testSpeechSynthesizeRejectsLegacyHFCacheFlags() {
        XCTAssertThrowsError(
            try SpeechSynthesize.parse([
                "Hello",
                "--output", "/tmp/out.wav",
                "--hf-hub-cache", "/tmp/hf/hub",
            ])
        )
    }
}
