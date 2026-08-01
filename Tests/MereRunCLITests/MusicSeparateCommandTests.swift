@testable import MereRunCLI
import MereRunCore
import XCTest

final class MusicSeparateCommandTests: XCTestCase {
    func testDefaultsUsePinnedViperXModel() throws {
        let command = try MusicSeparate.parse(["mix.wav"])
        XCTAssertEqual(command.audio, "mix.wav")
        XCTAssertEqual(command.model, ModelResolver.ModelID.roFormerViperX1297.rawValue)
        XCTAssertEqual(command.overlap, 2)
        XCTAssertEqual(command.dtype, "float16")
        XCTAssertNoThrow(try command.validate())
    }

    func testRejectsInvalidOverlapAndComputeType() throws {
        XCTAssertThrowsError(try MusicSeparate.parse(["mix.wav", "--overlap", "11"]))
        XCTAssertThrowsError(try MusicSeparate.parse(["mix.wav", "--dtype", "bfloat16"]))
    }

    func testParsesPinnedFourStemModel() throws {
        let command = try MusicSeparate.parse([
            "mix.wav",
            "--model", ModelResolver.ModelID.roFormerFourStem.rawValue,
            "--overlap", "4",
        ])
        XCTAssertEqual(command.model, ModelResolver.ModelID.roFormerFourStem.rawValue)
        XCTAssertEqual(command.overlap, 4)
        XCTAssertNoThrow(try command.validate())
    }

    func testRejectsOverlapThatDoesNotDivideFourStemChunk() {
        XCTAssertThrowsError(try MusicSeparate.parse([
            "mix.wav",
            "--model", ModelResolver.ModelID.roFormerFourStem.rawValue,
            "--overlap", "8",
        ]))
    }
}
