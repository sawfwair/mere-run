import XCTest
import AudioCore
import MereRunCore
@testable import MereRunCLI

final class SpeechSynthesizeCommandParsingTests: XCTestCase {
    func testCLIAndAPIUseEquivalentDefaultAndExplicitSynthesisPlans() throws {
        let output = URL(fileURLWithPath: "/fixture/output.wav")
        let defaults = try SpeechSynthesize.parse(["hello", "--output", output.path])
        let apiDefaults = try APIServerContract.speechPlan(from: OpenAIAudioSpeechRequest(input: "hello"))
        XCTAssertEqual(try defaults.synthesisPlan(outputURL: output), try apiDefaults.synthesisPlan(outputURL: output))

        let description = "A deep, confident voice with crisp articulation. Use a friendly pace."
        let explicit = try SpeechSynthesize.parse([
            "hello", "--output", output.path, "--voice", description, "--temperature", "0.7"
        ])
        let apiExplicit = try APIServerContract.speechPlan(from: OpenAIAudioSpeechRequest(
            model: "tts-1", input: "hello", voice: "onyx", instructions: "Use a friendly pace.", temperature: 0.7
        ))
        XCTAssertEqual(try explicit.synthesisPlan(outputURL: output), try apiExplicit.synthesisPlan(outputURL: output))
        XCTAssertEqual(try apiExplicit.modelSelection().modelID, "speech-tts-qwen3-nano")
    }

    func testAPIInputTrimmingAndCLITextPreservationRemainExplicit() throws {
        let output = URL(fileURLWithPath: "/fixture/output.wav")
        let cli = try SpeechSynthesize.parse(["  hello  ", "--output", output.path])
        let api = try APIServerContract.speechPlan(from: OpenAIAudioSpeechRequest(input: "  hello  "))
        XCTAssertEqual(try cli.synthesisPlan(outputURL: output).request.text, "  hello  ")
        XCTAssertEqual(try api.synthesisPlan(outputURL: output).request.text, "hello")
    }

    func testQuietStreamingStillEmitsTokensWhenJSONProgressIsRequested() throws {
        let output = URL(fileURLWithPath: "/fixture/output.wav")
        let cli = try SpeechSynthesize.parse([
            "hello", "--output", output.path, "--stream", "--quiet", "--progress-json", "--stream-chunk-tokens", "17"
        ])
        let plan = try cli.synthesisPlan(outputURL: output)
        XCTAssertEqual(plan.streamingOptions?.emitTokenEvents, true)
        XCTAssertEqual(plan.streamingOptions?.chunkTokenInterval, 17)
        XCTAssertEqual(plan.exportPlan.options.format, .float32)
    }

    func testInvalidScalarsFailBeforeModelLookupAndOutputCreation() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            if FileManager.default.fileExists(atPath: directory.path) {
                try? FileManager.default.removeItem(at: directory)
            }
        }
        let output = directory.appendingPathComponent("output.wav")
        for temperature in ["nan", "inf", "-1", "2.1"] {
            let cli = try SpeechSynthesize.parse([
                "hello", "--output", output.path, "--model", "missing-speech-fixture", "--temperature=\(temperature)"
            ])
            do {
                try await cli.run()
                XCTFail("Invalid temperature was accepted")
            } catch SpeechSynthesisError.invalidInput(let field, _) {
                XCTAssertEqual(field, .temperature)
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        }
        let invalidStream = try SpeechSynthesize.parse([
            "hello", "--output", output.path, "--model", "missing-speech-fixture", "--stream", "--stream-chunk-tokens", "0"
        ])
        do { try await invalidStream.run(); XCTFail("Invalid stream interval was accepted") }
        catch SpeechSynthesisError.invalidInput(let field, _) { XCTAssertEqual(field, .streamChunkTokens) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

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
