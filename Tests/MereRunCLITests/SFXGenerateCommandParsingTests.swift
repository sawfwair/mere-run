import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class SFXGenerateCommandParsingTests: XCTestCase {
    func testSFXGenerateParsesDefaults() throws {
        let cmd = try SFXGenerate.parse([
            "door slam in a concrete stairwell",
        ])

        XCTAssertEqual(cmd.prompt, "door slam in a concrete stairwell")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshDFlow.rawValue)
        XCTAssertEqual(cmd.durationSeconds, 5.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.guidanceScale, 4.5, accuracy: 0.0001)
        XCTAssertNil(cmd.seed)
        XCTAssertEqual(try cmd.parseRenoiseSchedule(), [])
    }

    func testSFXGenerateParsesWooshOverrides() throws {
        let cmd = try SFXGenerate.parse([
            "metal wrench dropping onto concrete",
            "--model", "/tmp/woosh",
            "--duration", "7.5",
            "--steps", "4",
            "--cfg", "3.25",
            "--seed", "42",
            "--renoise", "0,0.5,0.5,0.3",
            "-o", "/tmp/wrench-clang.wav",
        ])

        XCTAssertEqual(cmd.model, "/tmp/woosh")
        XCTAssertEqual(cmd.durationSeconds, 7.5, accuracy: 0.0001)
        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.guidanceScale, 3.25, accuracy: 0.0001)
        XCTAssertEqual(cmd.seed, 42)
        XCTAssertEqual(try cmd.parseRenoiseSchedule(), [0, 0.5, 0.5, 0.3])
        XCTAssertEqual(cmd.output, "/tmp/wrench-clang.wav")
    }

    func testSFXGenerateRejectsWrongRenoiseCount() throws {
        let cmd = try SFXGenerate.parse([
            "glass break",
            "--steps", "4",
            "--renoise", "0.1,0.2",
        ])

        XCTAssertThrowsError(try cmd.parseRenoiseSchedule())
    }

    func testSFXAEEncodeParsesOptions() throws {
        let cmd = try SFXAEEncode.parse([
            "/tmp/input.wav",
            "--model", ModelResolver.ModelID.wooshFlow.rawValue,
            "-o", "/tmp/latents.npy",
            "--quiet",
        ])

        XCTAssertEqual(cmd.input, "/tmp/input.wav")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshFlow.rawValue)
        XCTAssertEqual(cmd.output, "/tmp/latents.npy")
        XCTAssertTrue(cmd.quiet)
    }

    func testSFXAEDecodeParsesOptions() throws {
        let cmd = try SFXAEDecode.parse([
            "/tmp/latents.npy",
            "--model", ModelResolver.ModelID.wooshDFlow.rawValue,
            "-o", "/tmp/output.wav",
        ])

        XCTAssertEqual(cmd.input, "/tmp/latents.npy")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshDFlow.rawValue)
        XCTAssertEqual(cmd.output, "/tmp/output.wav")
        XCTAssertFalse(cmd.quiet)
    }

    func testSFXConditionTextParsesOptions() throws {
        let cmd = try SFXConditionText.parse([
            "glass breaking",
            "--model", ModelResolver.ModelID.wooshFlow.rawValue,
            "-o", "/tmp/condition.safetensors",
            "--quiet",
        ])

        XCTAssertEqual(cmd.prompt, "glass breaking")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshFlow.rawValue)
        XCTAssertEqual(cmd.output, "/tmp/condition.safetensors")
        XCTAssertTrue(cmd.quiet)
    }

    func testSFXCLAPScoreParsesOptions() throws {
        let cmd = try SFXCLAPScoreCommand.parse([
            "glass breaking",
            "/tmp/glass.wav",
            "--model", ModelResolver.ModelID.wooshClap.rawValue,
            "--quiet",
        ])

        XCTAssertEqual(cmd.prompt, "glass breaking")
        XCTAssertEqual(cmd.audio, "/tmp/glass.wav")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshClap.rawValue)
        XCTAssertTrue(cmd.quiet)
    }

    func testSFXVideoGenerateParsesOptions() throws {
        let cmd = try SFXVideoGenerate.parse([
            "footsteps echoing in a hallway",
            "/tmp/synch.npy",
            "--model", ModelResolver.ModelID.wooshDVFlow8s.rawValue,
            "--synchformer-model", ModelResolver.ModelID.wooshSynchformer.rawValue,
            "--steps", "4",
            "--cfg", "3",
            "--renoise", "0,0.5,0.5,0.3",
            "--sync-batch-size", "2",
            "-o", "/tmp/video-sfx.wav",
            "--quiet",
        ])

        XCTAssertEqual(cmd.prompt, "footsteps echoing in a hallway")
        XCTAssertEqual(cmd.input, "/tmp/synch.npy")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshDVFlow8s.rawValue)
        XCTAssertEqual(cmd.synchformerModel, ModelResolver.ModelID.wooshSynchformer.rawValue)
        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.guidanceScale, 3)
        XCTAssertEqual(cmd.syncBatchSize, 2)
        XCTAssertEqual(try cmd.parseRenoiseSchedule(steps: 4), [0, 0.5, 0.5, 0.3])
        XCTAssertEqual(cmd.output, "/tmp/video-sfx.wav")
        XCTAssertTrue(cmd.quiet)
    }
}
