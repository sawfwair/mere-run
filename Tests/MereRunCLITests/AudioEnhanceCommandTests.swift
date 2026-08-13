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
        XCTAssertNil(command.inputRate)
        XCTAssertEqual(command.odeMethod, "midpoint")
        XCTAssertEqual(command.odeSteps, 4)
        XCTAssertNoThrow(try command.validate())
    }

    func testParsesUniverSRGeneralAudioOptions() throws {
        let command = try AudioEnhance.parse([
            "music.wav",
            "--model", ModelResolver.ModelID.univerSRAudio.rawValue,
            "--input-rate", "12000",
            "--ode-method", "rk4",
            "--ode-steps", "6",
            "--guidance-scale", "2",
            "--seed", "7",
            "--chunk-seconds", "5",
        ])
        XCTAssertEqual(command.inputRate, 12_000)
        XCTAssertEqual(command.odeMethod, "rk4")
        XCTAssertEqual(command.odeSteps, 6)
        XCTAssertEqual(command.guidanceScale, 2)
        XCTAssertEqual(command.seed, 7)
        XCTAssertEqual(command.chunkSeconds, 5)
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
        XCTAssertThrowsError(try AudioEnhance.parse([
            "speech.wav",
            "--model", ModelResolver.ModelID.univerSRAudio.rawValue,
            "--input-rate", "44100",
        ]))
        XCTAssertThrowsError(try AudioEnhance.parse([
            "speech.wav",
            "--model", ModelResolver.ModelID.univerSRAudio.rawValue,
            "--overlap", "2",
        ]))
    }

    func testAudioCommandOwnsEnhanceSubcommand() {
        XCTAssertEqual(Audio.configuration.subcommands.map { $0.configuration.commandName }, ["generate", "enhance"])
    }
}
