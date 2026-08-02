@testable import MereRunCLI
import MereRunCore
import XCTest

final class AudioEnhanceCommandTests: XCTestCase {
    func testDefaultsUsePinnedAPBWEModel() throws {
        let command = try AudioEnhance.parse(["speech.wav"])
        XCTAssertEqual(command.audio, "speech.wav")
        XCTAssertEqual(command.model, ModelResolver.ModelID.apBWE16kTo48k.rawValue)
        XCTAssertEqual(command.dtype, "float32")
        XCTAssertNil(command.overlap)
        XCTAssertNoThrow(try command.validate())
    }

    func testParsesOutputAndComputeOptions() throws {
        let command = try AudioEnhance.parse([
            "speech.wav",
            "--output", "wideband.wav",
            "--dtype", "float16",
            "--overlap", "4",
        ])
        XCTAssertEqual(command.output, "wideband.wav")
        XCTAssertEqual(command.dtype, "float16")
        XCTAssertEqual(command.overlap, 4)
        XCTAssertNoThrow(try command.validate())
    }

    func testRejectsUnsupportedModelOverlapAndComputeType() {
        XCTAssertThrowsError(try AudioEnhance.parse([
            "speech.wav", "--model", "music-separate-bs-roformer-viperx-1297",
        ]))
        XCTAssertThrowsError(try AudioEnhance.parse(["speech.wav", "--overlap", "7"]))
        XCTAssertThrowsError(try AudioEnhance.parse(["speech.wav", "--dtype", "bfloat16"]))
    }

    func testAudioCommandOwnsEnhanceSubcommand() {
        XCTAssertEqual(Audio.configuration.subcommands.map { $0.configuration.commandName }, ["enhance"])
    }
}
