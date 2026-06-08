import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class MusicAnalyzeCommandParsingTests: XCTestCase {
    func testMusicAnalyzeParsesDefaults() throws {
        let cmd = try MusicAnalyze.parse([
            "/tmp/song.mp3",
        ])

        XCTAssertEqual(cmd.audio, "/tmp/song.mp3")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.aceStep.rawValue)
        XCTAssertNil(cmd.checkpointsRoot)
        XCTAssertEqual(cmd.turboSubdirectory, "acestep-v15-turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "vae")
        XCTAssertNil(cmd.lmSubdirectory)
        XCTAssertNil(cmd.durationSeconds)
        XCTAssertEqual(cmd.maxNewTokens, 2048)
        XCTAssertEqual(cmd.lmTemperature, 0.3, accuracy: 0.0001)
        XCTAssertEqual(cmd.lmTopK, 0)
        XCTAssertEqual(cmd.lmTopP, 0.9, accuracy: 0.0001)
        XCTAssertFalse(cmd.includeRawLM)
        XCTAssertFalse(cmd.includeAudioCodes)
        XCTAssertFalse(cmd.quiet)
    }

    func testMusicAnalyzeParsesModelAndLMOverrides() throws {
        let cmd = try MusicAnalyze.parse([
            "/tmp/song.mp3",
            "--model", ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue,
            "--checkpoints-root", "/tmp/acestep",
            "--turbo-subdirectory", "acestep-v15-xl-turbo",
            "--vae-subdirectory", "vae-custom",
            "--lm-subdirectory", "acestep-5Hz-lm-4B",
            "--duration", "30",
            "--max-new-tokens", "512",
            "--lm-temperature", "0.2",
            "--lm-top-k", "32",
            "--lm-top-p", "0.75",
            "--include-raw-lm",
            "--include-audio-codes",
            "--quiet",
        ])

        XCTAssertEqual(cmd.model, ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue)
        XCTAssertEqual(cmd.checkpointsRoot, "/tmp/acestep")
        XCTAssertEqual(cmd.turboSubdirectory, "acestep-v15-xl-turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "vae-custom")
        XCTAssertEqual(cmd.lmSubdirectory, "acestep-5Hz-lm-4B")
        XCTAssertEqual(cmd.durationSeconds, 30)
        XCTAssertEqual(cmd.maxNewTokens, 512)
        XCTAssertEqual(cmd.lmTemperature, 0.2, accuracy: 0.0001)
        XCTAssertEqual(cmd.lmTopK, 32)
        XCTAssertEqual(cmd.lmTopP, 0.75, accuracy: 0.0001)
        XCTAssertTrue(cmd.includeRawLM)
        XCTAssertTrue(cmd.includeAudioCodes)
        XCTAssertTrue(cmd.quiet)
    }

    func testMusicAnalyzeOutputRoundTrips() throws {
        let output = MusicAnalyzeOutput(
            audio: "/tmp/song.mp3",
            model: ModelResolver.ModelID.aceStep.rawValue,
            checkpointsRoot: "/tmp/acestep",
            turboSubdirectory: "acestep-v15-turbo",
            lmSubdirectory: "acestep-5Hz-lm-1.7B",
            inputDurationSeconds: 180,
            analyzedDurationSeconds: 30,
            metadata: ACEStepMusicUnderstandingMetadata(
                caption: "bright source song",
                lyrics: "[Verse]\nhello",
                bpm: 125,
                durationSeconds: 30,
                keyscale: "D major",
                language: "en",
                timesignature: "4"
            ),
            rawLMOutput: nil,
            audioCodes: nil
        )

        let data = try JSONEncoder().encode(output)
        let decoded = try JSONDecoder().decode(MusicAnalyzeOutput.self, from: data)

        XCTAssertEqual(decoded, output)
    }

    func testMusicCommandExposesAnalyze() {
        let commandNames = Set(Music.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set(["analyze", "generate", "realtime"]))
    }
}
