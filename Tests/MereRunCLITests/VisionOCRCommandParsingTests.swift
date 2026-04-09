import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class VisionOCRCommandParsingTests: XCTestCase {
    func testVisionOCRParsesManagedDefaultModel() throws {
        let cmd = try VisionOCR.parse([
            "/tmp/image.png",
        ])

        XCTAssertEqual(cmd.images, ["/tmp/image.png"])
        XCTAssertEqual(cmd.backend, .lighton)
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.lightOnOCR.rawValue)
        XCTAssertEqual(cmd.maxTokens, 4096)
        XCTAssertEqual(cmd.temperature, 0.2, accuracy: 0.0001)
        XCTAssertFalse(cmd.compare)
        XCTAssertFalse(cmd.quiet)
    }

    func testVisionOCRParsesOverrides() throws {
        let cmd = try VisionOCR.parse([
            "/tmp/one.png",
            "/tmp/two.png",
            "--backend", "glm",
            "--model", "/tmp/lighton",
            "--glmocr-cli", "/opt/bin/glmocr",
            "--glm-config", "/tmp/glm.yaml",
            "--output-dir", "/tmp/out",
            "--max-tokens", "2048",
            "--temperature", "0.4",
            "--compare",
            "--quiet",
        ])

        XCTAssertEqual(cmd.images, ["/tmp/one.png", "/tmp/two.png"])
        XCTAssertEqual(cmd.backend, .glm)
        XCTAssertEqual(cmd.model, "/tmp/lighton")
        XCTAssertEqual(cmd.glmocrCLI, "/opt/bin/glmocr")
        XCTAssertEqual(cmd.glmConfig, "/tmp/glm.yaml")
        XCTAssertEqual(cmd.outputDir, "/tmp/out")
        XCTAssertEqual(cmd.maxTokens, 2048)
        XCTAssertEqual(cmd.temperature, 0.4, accuracy: 0.0001)
        XCTAssertTrue(cmd.compare)
        XCTAssertTrue(cmd.quiet)
    }
}
