@testable import MereRunCLI
import XCTest

final class AudioGenerateCommandTests: XCTestCase {
    func testUsesOfficialLTX25Defaults() throws {
        let command = try AudioGenerate.parse(["a distant thunderstorm"])

        XCTAssertEqual(command.seed, 10)
        XCTAssertEqual(command.spatioTemporalBlocks, [28])
        XCTAssertNil(command.numFrames)
        XCTAssertTrue(command.autoDuration.isEmpty)
    }

    func testParsesNativeLTX25TextToAudioControls() throws {
        let command = try AudioGenerate.parse([
            "ocean surf with distant gulls",
            "--duration", "8",
            "--steps", "40",
            "--seed", "99",
            "--audio-cfg-guidance-scale", "8",
            "--audio-stg-guidance-scale", "1.5",
            "--audio-rescale", "0.6",
            "--audio-skip-step", "2",
            "--audio-stg-block", "27",
            "--audio-stg-block", "28",
            "--sigmas", "1,0.8,0.2,0",
        ])

        XCTAssertEqual(command.duration, 8)
        XCTAssertEqual(command.steps, 40)
        XCTAssertEqual(command.seed, 99)
        XCTAssertEqual(command.classifierFreeGuidanceScale, 8)
        XCTAssertEqual(command.spatioTemporalGuidanceScale, 1.5)
        XCTAssertEqual(command.guidanceRescale, 0.6)
        XCTAssertEqual(command.skipStep, 2)
        XCTAssertEqual(command.spatioTemporalBlocks, [27, 28])
        XCTAssertEqual(command.sigmaList, "1,0.8,0.2,0")
    }

    func testParsesPromptEnhancementAndAutoDuration() throws {
        let command = try AudioGenerate.parse([
            "a distant thunderstorm",
            "--auto-duration", "2", "12",
            "--enhance-prompt",
            "--prompt-enhancer-model", "text-chat-gemma4-12b-4bit",
            "--prompt-enhancer-model-root", "/tmp/gemma4",
        ])

        XCTAssertEqual(command.autoDuration, [2, 12])
        XCTAssertTrue(command.enhancePrompt)
        XCTAssertEqual(command.promptEnhancerModel, "text-chat-gemma4-12b-4bit")
        XCTAssertEqual(command.promptEnhancerModelRoot, "/tmp/gemma4")
    }
}
