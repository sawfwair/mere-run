import Foundation
import XCTest
@testable import MereRunCLI

final class VisionCaptionCommandParsingTests: XCTestCase {
    func testVisionCaptionParsesDefaults() throws {
        let cmd = try VisionCaption.parse([
            "/tmp/image.png",
        ])

        XCTAssertEqual(cmd.images, ["/tmp/image.png"])
        XCTAssertNil(cmd.model)
        XCTAssertNil(cmd.outputDir)
        XCTAssertNil(cmd.prompt)
        XCTAssertNil(cmd.promptFile)
        XCTAssertEqual(cmd.focus, [])
        XCTAssertNil(cmd.triggerToken)
        XCTAssertEqual(cmd.maxTokens, 96)
        XCTAssertEqual(cmd.temperature, 0.2, accuracy: 0.0001)
        XCTAssertEqual(cmd.topP, 0.9, accuracy: 0.0001)
        XCTAssertEqual(try cmd.resolvedPromptInstruction(), VisionCaption.defaultPrompt)
    }

    func testVisionCaptionParsesDomainCaptioningOptions() throws {
        let cmd = try VisionCaption.parse([
            "/tmp/one.png",
            "/tmp/two.png",
            "--model", "/tmp/qwen-vl",
            "--output-dir", "/tmp/captions",
            "--prompt", "Write one compact training caption.",
            "--focus", "full card border", "printed title text", "visible gross-out gag",
            "--trigger-token", "gpkos13",
            "--max-tokens", "128",
            "--temperature", "0.1",
            "--top-p", "0.8",
        ])

        XCTAssertEqual(cmd.images, ["/tmp/one.png", "/tmp/two.png"])
        XCTAssertEqual(cmd.model, "/tmp/qwen-vl")
        XCTAssertEqual(cmd.outputDir, "/tmp/captions")
        XCTAssertEqual(cmd.prompt, "Write one compact training caption.")
        XCTAssertEqual(cmd.focus, ["full card border", "printed title text", "visible gross-out gag"])
        XCTAssertEqual(cmd.triggerToken, "gpkos13")
        XCTAssertEqual(cmd.maxTokens, 128)
        XCTAssertEqual(cmd.temperature, 0.1, accuracy: 0.0001)
        XCTAssertEqual(cmd.topP, 0.8, accuracy: 0.0001)

        let instruction = try cmd.resolvedPromptInstruction()
        XCTAssertTrue(instruction.contains("Write one compact training caption."))
        XCTAssertTrue(instruction.contains("full card border; printed title text; visible gross-out gag"))
        XCTAssertTrue(instruction.contains("gpkos13 will be added separately"))
    }

    func testVisionCaptionCombinesPromptFileWithAdditionalPromptAndFocus() throws {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-caption-prompt-\(UUID().uuidString).txt")
        try "Describe visible trading-card layout. Do not infer hidden details."
            .write(to: temp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: temp) }

        let cmd = try VisionCaption.parse([
            "/tmp/card.png",
            "--prompt-file", temp.path,
            "--prompt", "Use one sentence.",
            "--focus", "nameplate", "white border",
        ])

        let instruction = try cmd.resolvedPromptInstruction()
        XCTAssertTrue(instruction.contains("Describe visible trading-card layout. Do not infer hidden details."))
        XCTAssertTrue(instruction.contains("Additional instruction: Use one sentence."))
        XCTAssertTrue(instruction.contains("nameplate; white border"))
    }

    func testVisionCaptionPrefixesTriggerTokenDeterministically() {
        XCTAssertEqual(
            VisionCaption.captionOutput("a stained trading card character", triggerToken: "gpkos13"),
            "gpkos13 a stained trading card character"
        )
        XCTAssertEqual(
            VisionCaption.captionOutput("gpkos13 a stained trading card character", triggerToken: "gpkos13"),
            "gpkos13 a stained trading card character"
        )
        XCTAssertEqual(
            VisionCaption.captionOutput("  ", triggerToken: "gpkos13"),
            "gpkos13"
        )
    }
}
