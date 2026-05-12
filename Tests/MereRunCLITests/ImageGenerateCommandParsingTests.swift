import XCTest
@testable import MereRunCLI

final class ImageGenerateCommandParsingTests: XCTestCase {
    func testParsesHiDreamReferenceOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "place this subject in a studio",
            "--model", "image-hidream-o1-dev",
            "--ref-image", "/tmp/ref1.png",
            "--ref-image", "/tmp/ref2.png",
            "--keep-original-aspect",
            "--width", "1024",
            "--height", "768",
        ])

        XCTAssertEqual(cmd.prompt, "place this subject in a studio")
        XCTAssertEqual(cmd.model, "image-hidream-o1-dev")
        XCTAssertEqual(cmd.referenceImages, ["/tmp/ref1.png", "/tmp/ref2.png"])
        XCTAssertTrue(cmd.keepOriginalAspect)
        XCTAssertEqual(cmd.width, 1024)
        XCTAssertEqual(cmd.height, 768)
        XCTAssertNil(cmd.steps)
        XCTAssertNil(cmd.cfgScale)
    }

    func testParsesExplicitHiDreamStepAndCFGOverrides() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a brass camera",
            "--model", "image-hidream-o1",
            "--steps", "4",
            "--cfg", "1.0",
        ])

        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.cfgScale, 1.0)
    }
}
