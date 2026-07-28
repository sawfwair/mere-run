import MereRunCore
import XCTest
@testable import MereRunCLI

final class Trellis2CommandTests: XCTestCase {
    func testBothPublicCommandPathsRegister() {
        let imageCommands = Set(Image.configuration.subcommands.map { $0.configuration.commandName })
        let visionCommands = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(imageCommands.contains("reconstruct-3d-trellis2"))
        XCTAssertTrue(visionCommands.contains("image-to-3d-trellis2"))
    }

    func testVisionCommandParsesNativeProductionControls() throws {
        let command = try VisionImageTo3DTrellis2.parse([
            "/tmp/chair.png",
            "--output", "/tmp/chair-trellis2",
            "--model", "image-3d-trellis2-4b",
            "--seed", "1234",
            "--max-tokens", "65536",
            "--already-framed",
            "--dry-run",
            "--json",
        ])
        XCTAssertEqual(command.input, "/tmp/chair.png")
        XCTAssertEqual(command.output, "/tmp/chair-trellis2")
        XCTAssertEqual(command.model, "image-3d-trellis2-4b")
        XCTAssertEqual(command.seed, 1_234)
        XCTAssertEqual(command.maxTokens, 65_536)
        XCTAssertTrue(command.alreadyFramed)
        XCTAssertTrue(command.dryRun)
        XCTAssertTrue(command.json)
    }

    func testImageCommandUsesMatchingDefaults() throws {
        let command = try ImageReconstruct3DTrellis2.parse(["/tmp/chair.png"])
        XCTAssertEqual(command.seed, 42)
        XCTAssertNil(command.textureSeed)
        XCTAssertEqual(command.maxTokens, 2_097_152)
        XCTAssertFalse(command.alreadyFramed)
        XCTAssertFalse(command.noRemesh)
        XCTAssertEqual(command.remeshBand, 1)
        XCTAssertEqual(command.sealRadius, 12)
        XCTAssertFalse(command.dryRun)
    }

    func testImageCommandParsesTextureAndRemeshControls() throws {
        let command = try ImageReconstruct3DTrellis2.parse([
            "/tmp/chair.png",
            "--texture-seed", "99",
            "--no-remesh",
            "--remesh-band", "1.5",
            "--seal-radius", "8",
        ])
        XCTAssertEqual(command.textureSeed, 99)
        XCTAssertTrue(command.noRemesh)
        XCTAssertEqual(command.remeshBand, 1.5)
        XCTAssertEqual(command.sealRadius, 8)
    }

    func testDefaultOutputUsesTrellis2SpecificDirectory() {
        let input = URL(fileURLWithPath: "/tmp/chair.asset.png")
        XCTAssertEqual(
            VisionImageTo3DTrellis2.resolveOutputURL(nil, inputURL: input).path,
            "/tmp/chair.asset-trellis2-3d"
        )
    }
}
