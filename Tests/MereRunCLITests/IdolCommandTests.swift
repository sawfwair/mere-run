import MereRunCore
import XCTest
@testable import MereRunCLI

final class IdolCommandTests: XCTestCase {
    private let arguments = ["/tmp/object.png", "--model", "/tmp/idol", "--dino", "/tmp/dino", "--vae", "/tmp/vae", "-o", "/tmp/splats"]

    func testCommandRegistrationAndDefaults() throws {
        XCTAssertTrue(Vision.configuration.subcommands.contains { $0.configuration.commandName == "image-to-3d-idol" })
        let command = try VisionImageTo3DIdol.parse(arguments)
        XCTAssertEqual(try command.settings(), try IdolSamplingSettings())
        XCTAssertEqual(command.mode, "synthetic")
        XCTAssertFalse(command.alreadyFramed)
    }

    func testSamplingControlsAndRequiredCheckpoints() throws {
        let command = try VisionImageTo3DIdol.parse(arguments + ["--mode", "real", "--already-framed", "--steps", "12",
            "--seed", "99", "--guidance", "2.5", "--time-shift", "1.5", "--num-gaussians", "32768", "--dry-run", "--json"])
        XCTAssertEqual(try command.settings().steps, 12)
        XCTAssertEqual(command.numGaussians, 32768)
        XCTAssertTrue(command.dryRun)
        XCTAssertTrue(command.json)
        XCTAssertThrowsError(try VisionImageTo3DIdol.parse(["/tmp/object.png"]))
        for invalid in [["--steps", "0"], ["--mode", "other"], ["--num-gaussians", "32769"], ["--guidance", "nan"]] {
            XCTAssertThrowsError(try VisionImageTo3DIdol.parse(arguments + invalid))
        }
    }
}
