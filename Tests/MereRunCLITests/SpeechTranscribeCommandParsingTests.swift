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
        XCTAssertEqual(cmd.provider, .mlx)
        XCTAssertNil(cmd.coremlEncoder)
        XCTAssertEqual(cmd.task, .transcribe)
        XCTAssertEqual(cmd.maxTokens, 448)
        XCTAssertFalse(cmd.stream)
        XCTAssertEqual(cmd.streamChunkMs, 200)
        XCTAssertEqual(cmd.streamDecodeMs, 2_000)
        XCTAssertNil(cmd.inputFormat)
        XCTAssertNil(cmd.sampleRate)
        XCTAssertFalse(cmd.jsonl)
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

    func testSpeechTranscribeParsesExplicitCoreMLParakeetProvider() throws {
        let cmd = try SpeechTranscribe.parse([
            "/tmp/input.wav",
            "--backend", "parakeet",
            "--provider", "coreml",
            "--coreml-encoder", "/tmp/parakeet-coreml",
        ])

        XCTAssertEqual(cmd.backend, .parakeet)
        XCTAssertEqual(cmd.provider, .coreml)
        XCTAssertEqual(cmd.coremlEncoder, "/tmp/parakeet-coreml")
    }

    func testSpeechTranscribeRejectsIncompleteCoreMLProviderContract() {
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "/tmp/input.wav", "--backend", "parakeet", "--provider", "coreml",
        ]))
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "/tmp/input.wav", "--provider", "coreml",
            "--coreml-encoder", "/tmp/parakeet-coreml",
        ]))
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "/tmp/input.wav", "--coreml-encoder", "/tmp/parakeet-coreml",
        ]))
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "/tmp/input.wav", "--stream", "--backend", "parakeet", "--provider", "coreml",
            "--coreml-encoder", "/tmp/parakeet-coreml",
        ]))
    }

    func testSpeechTranscribeParsesRawPCMStandardInput() throws {
        let cmd = try SpeechTranscribe.parse([
            "-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "16000", "--jsonl",
        ])
        XCTAssertEqual(cmd.audio, "-")
        XCTAssertEqual(cmd.inputFormat, "pcm-s16le")
        XCTAssertEqual(cmd.sampleRate, 16_000)
        XCTAssertTrue(cmd.jsonl)
        XCTAssertNoThrow(try cmd.validate())
    }

    func testSpeechTranscribeAllowsParakeetRawPCMStreaming() throws {
        let cmd = try SpeechTranscribe.parse([
            "-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "16000",
            "--backend", "parakeet", "--jsonl",
        ])
        XCTAssertEqual(cmd.backend, .parakeet)
        XCTAssertNoThrow(try cmd.validate())
    }

    func testSpeechTranscribeRejectsParakeetStreamingTranslation() {
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "16000",
            "--backend", "parakeet", "--task", "translate", "--jsonl",
        ]))
    }

    func testSpeechTranscribeRejectsInvalidRawPCMContracts() throws {
        XCTAssertThrowsError(try SpeechTranscribe.parse(["-"]))
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "48000",
        ]))
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "/tmp/input.wav", "--stream", "--jsonl",
        ]))
    }

    func testSpeechListenParsesDefaultsAndDevice() throws {
        let defaults = try SpeechListen.parse([])
        XCTAssertEqual(defaults.decodeMs, 2_000)
        XCTAssertEqual(defaults.silenceMs, 900)
        let selected = try SpeechListen.parse(["--device", "input-uid", "--jsonl"])
        XCTAssertEqual(selected.device, "input-uid")
        XCTAssertTrue(selected.jsonl)
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
            "catalog",
            "image",
            "text",
            "speech",
            "vision",
            "audio",
            "music",
            "sfx",
            "video",
            "world",
            "geo",
            "graph",
            "executor",
            "relay",
            "run",
            "eval",
            "model",
            "adapter",
            "status",
            "gate",
            "api",
            "config",
            "open-webui",
            "plugin",
            "setup",
            "agent",
        ]))

        let speechNames = Set(Speech.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(speechNames, Set(["synthesize", "transcribe", "diarize", "listen", "profile"]))

        let visionNames = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(visionNames, Set([
            "caption", "inspect", "embed", "ground", "segment", "face", "track", "track-live", "pose", "flow", "ocr",
            "depth-video", "geometry", "geometry-multiview", "image-to-3d", "image-to-3d-multiview",
            "image-to-3d-trellis2", "serve",
        ]))

        let videoNames = Set(Video.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(
            videoNames,
            Set([
                "animate", "cosmos3", "dub-it", "generate", "session", "export-latents",
                "prepare-masks", "retake",
            ])
        )

        let sfxNames = Set(SFX.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(sfxNames, Set(["ae", "clap", "condition", "generate", "video"]))
    }
}
