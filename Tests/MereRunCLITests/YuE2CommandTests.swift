import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class YuE2CommandTests: XCTestCase {
    private func parse(_ options: [String] = []) throws -> MusicGenerate {
        try MusicGenerate.parse(["piano pop", "--model", "music-yue2", "--lyrics", "[Verse]\nhello"] + options)
    }

    func testNativeDefaultsAndScoreOptions() throws {
        let command = try parse()
        let plan = try command.resolvedYuE2Plan(explicitDurationSeconds: nil)
        XCTAssertTrue(command.isYuE2Request)
        XCTAssertEqual(plan.steps, 32)
        XCTAssertEqual(plan.seed, 831001)
        XCTAssertEqual(plan.planning, .full)
        XCTAssertEqual(plan.guidanceScale, 1)
        XCTAssertEqual(plan.semanticSampling.maximumTokens, 9000)
        XCTAssertEqual(plan.semanticSampling.minimumTokens, 200)
        let direct = try parse(["--score-mode", "off", "--duration", "2", "--semantic-temperature", "0"])
        let directPlan = try direct.resolvedYuE2Plan(explicitDurationSeconds: direct.durationSeconds)
        XCTAssertEqual(directPlan.guidanceScale, 1.01)
        XCTAssertEqual(directPlan.semanticSampling.maximumTokens, 50)
        XCTAssertEqual(directPlan.semanticSampling.minimumTokens, 50)
        XCTAssertEqual(directPlan.semanticSampling.temperature, 0)
    }

    func testDurationFloorAccountsForDecoderBoundary() throws {
        XCTAssertEqual(try MusicGenerate.yue2Frames(seconds: 2, minimum: false), 50)
        XCTAssertEqual(try MusicGenerate.yue2Frames(seconds: 2, minimum: true), 51)
        XCTAssertThrowsError(try MusicGenerate.yue2Frames(seconds: 360, minimum: true))
        XCTAssertThrowsError(try MusicGenerate.yue2Frames(seconds: .infinity, minimum: false))
        let command = try parse(["--minimum-duration", "2", "--max-frames", "51"])
        XCTAssertEqual(try command.resolvedYuE2Plan(explicitDurationSeconds: nil).semanticSampling.minimumTokens, 51)
    }

    func testIncompatibleInputsAreRejectedBeforeLoadingWeights() throws {
        let rejected = [
            ["--compose"], ["--performance-mode", "q8"], ["--sample-rate", "32000"],
            ["--source-audio", "/tmp/source.wav"], ["--quality", "draft"], ["--use-lm"],
            ["--adapter", "/tmp/adapter"], ["--score-mode", "off", "--abc-output", "/tmp/score.abc"],
            ["--max-frames", "0"], ["--max-frames", "9001"], ["--min-frames", "100", "--max-frames", "50"],
            ["--semantic-top-p", "0"], ["--semantic-temperature", "nan"], ["--steps", "0"],
            ["--no-recipe", "--recipe-output", "/tmp/recipe.json"], ["--instrumental"],
        ]
        for arguments in rejected {
            let command = try parse(arguments)
            XCTAssertThrowsError(try command.resolvedYuE2Plan(explicitDurationSeconds: nil), arguments.joined(separator: " "))
        }
    }

    func testGuideAndCapabilityExposeTheNativeContract() throws {
        let guide = try ModelGuideRegistry.guide(for: "music-yue2")
        let content = try GuideRegistry.content(for: guide.entry)
        XCTAssertTrue(content.contains("CC BY-NC 4.0"))
        XCTAssertTrue(content.contains("--abc-file"))
        XCTAssertTrue(content.contains("Checkpoint verification"))
    }
}
