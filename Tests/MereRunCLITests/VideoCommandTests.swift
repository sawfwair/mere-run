import Foundation
import XCTest
@testable import MereRunCLI

final class VideoCommandTests: XCTestCase {
    private var temporaryDirectories: [URL] = []

    override func tearDown() {
        for directory in temporaryDirectories {
            try? FileManager.default.removeItem(at: directory)
        }
        temporaryDirectories.removeAll()
        super.tearDown()
    }

    func testVideoCommandExposesGenerateAndExportLatents() {
        let commandNames = Set(Video.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set(["generate", "export-latents"]))
    }

    func testVideoGenerateParsesDefaults() throws {
        let cmd = try VideoGenerate.parse([
            "a cinematic drone flythrough"
        ])

        XCTAssertEqual(cmd.prompt, "a cinematic drone flythrough")
        XCTAssertEqual(cmd.variant, .distilled)
        XCTAssertEqual(cmd.width, 768)
        XCTAssertEqual(cmd.height, 512)
        XCTAssertEqual(cmd.numFrames, 65)
        XCTAssertNil(cmd.duration)
        XCTAssertEqual(cmd.fps, 24)
        XCTAssertEqual(cmd.imageStrength, 1.0)
        XCTAssertNil(cmd.endImage)
        XCTAssertEqual(cmd.endImageStrength, 1.0)
        XCTAssertNil(cmd.modelRoot)
    }

    func testVideoGenerateParsesStartAndEndKeyframes() throws {
        let cmd = try VideoGenerate.parse([
            "a flower opens from bud to bloom",
            "--image", "/tmp/start.png",
            "--image-strength", "0.85",
            "--end-image", "/tmp/end.png",
            "--end-image-strength", "0.7",
        ])

        XCTAssertEqual(cmd.image, "/tmp/start.png")
        XCTAssertEqual(cmd.imageStrength, 0.85, accuracy: 0.0001)
        XCTAssertEqual(cmd.endImage, "/tmp/end.png")
        XCTAssertEqual(cmd.endImageStrength, 0.7, accuracy: 0.0001)
    }

    func testVideoGenerateRejectsEndImageWithoutStartImage() async throws {
        let cmd = try VideoGenerate.parse([
            "a car drives from dawn into sunset",
            "--end-image", "/tmp/end.png",
        ])

        do {
            try await cmd.run()
            XCTFail("Expected --end-image without --image to fail validation.")
        } catch {
            let message = "\(error) \(error.localizedDescription)"
            XCTAssertTrue(message.contains("--end-image requires --image"))
        }
    }

    func testVideoGenerateParsesDurationOverride() throws {
        let cmd = try VideoGenerate.parse([
            "dialogue with background music",
            "--variant", "unified-av",
            "--duration", "15",
            "--fps", "24",
        ])

        XCTAssertEqual(cmd.variant, .unifiedAV)
        XCTAssertEqual(cmd.duration, 15)
        XCTAssertEqual(cmd.fps, 24)
    }

    func testNearestLTXFrameCountUsesClosestLegalFrameCount() {
        XCTAssertEqual(nearestLTXFrameCount(duration: 15, fps: 24), 361)
        XCTAssertEqual(nearestLTXFrameCount(duration: 5, fps: 24), 121)
        XCTAssertEqual(nearestLTXFrameCount(duration: 15, fps: 8), 121)
    }

    func testVideoExportLatentsParsesOverrides() throws {
        let cmd = try VideoExportLatents.parse([
            "a neon city skyline",
            "--model-root", "/tmp/ltx",
            "--output", "/tmp/out.safetensors",
            "--width", "1024",
            "--height", "576",
            "--num-frames", "33",
            "--seed", "7",
        ])

        XCTAssertEqual(cmd.prompt, "a neon city skyline")
        XCTAssertEqual(cmd.modelRoot, "/tmp/ltx")
        XCTAssertEqual(cmd.output, "/tmp/out.safetensors")
        XCTAssertEqual(cmd.width, 1024)
        XCTAssertEqual(cmd.height, 576)
        XCTAssertEqual(cmd.numFrames, 33)
        XCTAssertEqual(cmd.seed, 7)
    }

    func testValidateNativeModelRootRejectsMissingTokenizerDirectory() throws {
        let rootURL = try makeTempDirectory()
        try createFile(rootURL.appendingPathComponent("text_encoder/config.json"))
        try createFile(rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try createFile(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors"))
        try createFile(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors"))

        XCTAssertThrowsError(try validateNativeModelRoot(rootURL))
    }

    func testValidateNativeModelRootAcceptsExpectedLayout() throws {
        let rootURL = try makeTempDirectory()
        try createFile(rootURL.appendingPathComponent("text_encoder/config.json"))
        try createFile(rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try FileManager.default.createDirectory(
            at: rootURL.appendingPathComponent("tokenizer", isDirectory: true),
            withIntermediateDirectories: true
        )
        try createFile(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors"))
        try createFile(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors"))

        XCTAssertNoThrow(try validateNativeModelRoot(rootURL))
    }

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VideoCommandTests.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        temporaryDirectories.append(directory)
        return directory
    }

    private func createFile(_ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = Data("fixture".utf8)
        try data.write(to: url)
    }
}
