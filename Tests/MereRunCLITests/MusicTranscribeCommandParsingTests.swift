import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class MusicTranscribeCommandParsingTests: XCTestCase {
    func testDefaultsToMediumMIDI() throws {
        let command = try MusicTranscribe.parse(["song.mp3"])
        XCTAssertEqual(command.audio, "song.mp3")
        XCTAssertEqual(command.model, ModelResolver.ModelID.muScriptorMedium.rawValue)
        XCTAssertEqual(command.format, .midi)
        XCTAssertEqual(command.dtype, "bfloat16")
        XCTAssertFalse(command.sampling)
        XCTAssertEqual(command.chunkBatchSize, 4)
    }

    func testParsesStructuredOutputAndConditioning() throws {
        let command = try MusicTranscribe.parse([
            "song.wav",
            "--model", "music-muscriptor-small",
            "--format", "jsonl",
            "--output", "-",
            "--instruments", "voice,drums,bass",
            "--sampling",
            "--temperature", "0.8",
            "--strict-eos",
            "--chunk-batch-size", "2",
            "--dtype", "float32",
        ])
        XCTAssertEqual(command.model, ModelResolver.ModelID.muScriptorSmall.rawValue)
        XCTAssertEqual(command.format, .jsonl)
        XCTAssertEqual(command.output, "-")
        XCTAssertEqual(command.instruments, "voice,drums,bass")
        XCTAssertTrue(command.sampling)
        XCTAssertEqual(command.temperature, 0.8, accuracy: 1e-6)
        XCTAssertTrue(command.strictEOS)
        XCTAssertEqual(command.beamSize, 1)
        XCTAssertEqual(command.chunkBatchSize, 2)
        XCTAssertEqual(command.dtype, "float32")
    }

    func testParsesBeamSearch() throws {
        let command = try MusicTranscribe.parse([
            "song.wav",
            "--beam-size", "4",
        ])
        XCTAssertEqual(command.beamSize, 4)
        XCTAssertFalse(command.sampling)
    }

    func testListInstrumentsDoesNotRequireAudio() throws {
        let command = try MusicTranscribe.parse(["--list-instruments"])
        XCTAssertTrue(command.listInstruments)
        XCTAssertNil(command.audio)
    }

    func testRejectsInvalidChunkBatchSize() throws {
        var command = try MusicTranscribe.parse(["song.wav"])
        command.chunkBatchSize = 0
        XCTAssertThrowsError(try command.validate())
    }

}
