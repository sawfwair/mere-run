import XCTest
@testable import MereRunCLI

final class SpeechProfileCommandParsingTests: XCTestCase {
    func testSpeechProfileListParses() throws {
        _ = try SpeechProfileList.parse([])
    }

    func testSpeechProfileCreateParsesRequiredArguments() throws {
        let cmd = try SpeechProfileCreate.parse([
            "--name", "narrator",
            "--audio", "/tmp/ref.wav",
            "--text", "reference transcript",
            "--language", "en",
        ])

        XCTAssertEqual(cmd.name, "narrator")
        XCTAssertEqual(cmd.audio, "/tmp/ref.wav")
        XCTAssertEqual(cmd.text, "reference transcript")
        XCTAssertEqual(cmd.language, "en")
    }

    func testSpeechProfileCreateRequiresAudio() {
        XCTAssertThrowsError(
            try SpeechProfileCreate.parse([
                "--name", "narrator",
            ])
        )
    }

    func testSpeechProfileDeleteParsesID() throws {
        let cmd = try SpeechProfileDelete.parse([
            "--id", "8B346951-D74A-49FA-83D5-3AC4E4A23C24",
        ])
        XCTAssertEqual(cmd.id, "8B346951-D74A-49FA-83D5-3AC4E4A23C24")
    }

    func testSpeechExposesProfileEntrypoint() {
        let commandNames = Speech.configuration.subcommands.compactMap { $0.configuration.commandName }
        XCTAssertEqual(commandNames.sorted(), ["diarize", "listen", "profile", "synthesize", "transcribe"])
    }
}
