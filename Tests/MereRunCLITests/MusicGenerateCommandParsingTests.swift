import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class MusicGenerateCommandParsingTests: XCTestCase {
    func testMusicGenerateParsesManagedDefaultModel() throws {
        let cmd = try MusicGenerate.parse([
            "warm synthwave groove",
        ])

        XCTAssertEqual(cmd.caption, "warm synthwave groove")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.aceStep.rawValue)
        XCTAssertNil(cmd.checkpointsRoot)
        XCTAssertEqual(cmd.turboSubdirectory, "acestep-v15-turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "vae")
        XCTAssertFalse(cmd.useLM)
        XCTAssertEqual(cmd.durationSeconds, 10.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.shift, 1.0, accuracy: 0.0001)
    }

    func testMusicGenerateParsesModelAndAdvancedOverrides() throws {
        let cmd = try MusicGenerate.parse([
            "club track",
            "--model", "/tmp/acestep",
            "--checkpoints-root", "/tmp/checkpoints",
            "--turbo-subdirectory", "turbo",
            "--vae-subdirectory", "custom-vae",
            "--lm-subdirectory", "custom-lm",
            "--text-subdirectory", "text-encoder",
            "--use-lm",
            "--duration", "18",
            "--steps", "12",
        ])

        XCTAssertEqual(cmd.model, "/tmp/acestep")
        XCTAssertEqual(cmd.checkpointsRoot, "/tmp/checkpoints")
        XCTAssertEqual(cmd.turboSubdirectory, "turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "custom-vae")
        XCTAssertEqual(cmd.lmSubdirectory, "custom-lm")
        XCTAssertEqual(cmd.textSubdirectory, "text-encoder")
        XCTAssertTrue(cmd.useLM)
        XCTAssertEqual(cmd.durationSeconds, 18.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.steps, 12)
    }
}
