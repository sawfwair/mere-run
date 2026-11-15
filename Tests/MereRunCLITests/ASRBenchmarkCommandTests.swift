import ArgumentParser
import XCTest
@testable import MereRunCLI

final class ASRBenchmarkCommandTests: XCTestCase {
    func testAllFiveFamiliesAcceptMultipleRecordingsAndResidentControls() throws {
        for backend in ["parakeet", "qwen", "whistle", "nemotron", "sov"] {
            let command = try ModelBenchmarkASR.parse(["short.wav", "long.wav", "--backend", backend,
                "--model-root", "/tmp/model", "--warmups", "2", "--repetitions", "5", "--json"])
            XCTAssertEqual(command.backend.rawValue, backend)
            XCTAssertEqual(command.audio, ["short.wav", "long.wav"])
            XCTAssertEqual(command.warmups, 2)
            XCTAssertEqual(command.repetitions, 5)
        }
        XCTAssertTrue(ModelBenchmark.configuration.subcommands.contains { $0.configuration.commandName == "asr" })
    }

    func testBackendSpecificOptionsCannotProduceMislabelledReceipts() throws {
        let base = ["audio.wav", "--model-root", "/tmp/model"]
        for flags in [["--backend", "auto"], ["--backend", "qwen", "--artifact", "/tmp/coreml"],
                      ["--backend", "parakeet", "--draft-model", "/tmp/draft"],
                      ["--backend", "whistle", "--experimental-dynamic-int8"],
                      ["--backend", "qwen", "--experimental-decoder", "q8"],
                      ["--backend", "sov", "--experimental-dynamic-int8", "--artifact", "/tmp/coreml"],
                      ["--backend", "sov", "--language", "en"], ["--backend", "qwen", "--beam-size", "1"],
                      ["--backend", "nemotron", "--word-timestamps"]] {
            XCTAssertThrowsError(try ModelBenchmarkASR.parse(base + flags), flags.description)
        }
        let command = try ModelBenchmarkASR.parse(base + ["--backend", "sov", "--draft-model", "/tmp/draft", "--experimental-dynamic-int8",
            "--experimental-decoder", "q8"])
        XCTAssertTrue(command.experimentalDynamicInt8)
        XCTAssertEqual(command.draftModel, "/tmp/draft")
        XCTAssertEqual(command.experimentalDecoder, .q8)
        let metal = try ModelBenchmarkASR.parse(base + ["--backend", "sov", "--experimental-decoder", "q8-metal"])
        XCTAssertEqual(metal.experimentalDecoder, .q8Metal)
    }

    func testInvalidMeasurementAndWhistleBoundsAreRejected() {
        let base = ["audio.wav", "--backend", "whistle", "--model-root", "/tmp/model"]
        for flags in [["--warmups", "-1"], ["--repetitions", "0"], ["--max-tokens", "0"],
                      ["--beam-size", "0"], ["--decoder-depth", "9"]] {
            XCTAssertThrowsError(try ModelBenchmarkASR.parse(base + flags), flags.description)
        }
    }
}
