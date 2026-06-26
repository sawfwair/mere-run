import XCTest
@testable import MereRunCLI

final class SpeechTranscribeCommandParsingTests: XCTestCase {
    func testSpeechTranscribeParsesDefaults() throws {
        let cmd = try SpeechTranscribe.parse([
            "/tmp/input.wav",
        ])

        XCTAssertEqual(cmd.audio, "/tmp/input.wav")
        XCTAssertNil(cmd.model)
        XCTAssertEqual(cmd.backend, .auto)
        XCTAssertEqual(cmd.task, .transcribe)
        XCTAssertEqual(cmd.maxTokens, 448)
        XCTAssertFalse(cmd.stream)
        XCTAssertEqual(cmd.streamChunkMs, 200)
        XCTAssertEqual(cmd.streamDecodeMs, 500)
        XCTAssertTrue(cmd.timestamps)
    }

    func testSpeechTranscribeParsesBackendTaskAndModelOverride() throws {
        let cmd = try SpeechTranscribe.parse([
            "/tmp/input.wav",
            "--backend", "parakeet",
            "--task", "translate",
            "--model", "/tmp/custom-asr",
            "--language", "en",
            "--max-tokens", "256",
            "--no-timestamps",
        ])

        XCTAssertEqual(cmd.backend, .parakeet)
        XCTAssertEqual(cmd.task, .translate)
        XCTAssertEqual(cmd.model, "/tmp/custom-asr")
        XCTAssertEqual(cmd.language, "en")
        XCTAssertEqual(cmd.maxTokens, 256)
        XCTAssertFalse(cmd.timestamps)
    }

    func testSpeechTranscribeParsesStreamingOptions() throws {
        let cmd = try SpeechTranscribe.parse([
            "/tmp/input.wav",
            "--stream",
            "--stream-chunk-ms", "120",
            "--stream-decode-ms", "640",
        ])

        XCTAssertTrue(cmd.stream)
        XCTAssertEqual(cmd.streamChunkMs, 120)
        XCTAssertEqual(cmd.streamDecodeMs, 640)
    }

    func testSpeechTranscribeRejectsLegacyHFCacheFlags() {
        XCTAssertThrowsError(
            try SpeechTranscribe.parse([
                "/tmp/input.wav",
                "--hf-home", "/tmp/hf",
            ])
        )
    }

    func testMereRunCLIExposesRenamedEntrypoints() {
        let commandNames = Set(MereRunCLI.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set([
            "guide",
            "image",
            "text",
            "speech",
            "vision",
            "music",
            "sfx",
            "video",
            "model",
            "status",
            "api",
            "config",
            "open-webui",
            "plugin",
            "setup",
            "agent",
        ]))

        let speechNames = Set(Speech.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(speechNames, Set(["synthesize", "transcribe", "profile"]))

        let visionNames = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(visionNames, Set(["caption", "inspect", "ground", "segment", "track", "track-live", "ocr"]))

        let videoNames = Set(Video.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(videoNames, Set(["generate", "export-latents"]))

        let sfxNames = Set(SFX.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(sfxNames, Set(["ae", "clap", "condition", "generate", "video"]))
    }
}
