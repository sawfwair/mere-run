import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionSegmentCommandParsingTests: XCTestCase {
    private func writeMinimalSAM31Model(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: .visionSegmentSAM31, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)

        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        try Data().write(to: root.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer_config.json"))
    }

    override func tearDown() {
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        super.tearDown()
    }

    func testVisionSegmentParsesDefaults() throws {
        let cmd = try VisionSegment.parse([
            "/tmp/image.png",
            "--prompt", "a cat",
        ])

        XCTAssertEqual(cmd.image, "/tmp/image.png")
        XCTAssertEqual(cmd.prompt, ["a cat"])
        XCTAssertNil(cmd.model)
        XCTAssertNil(cmd.output)
        XCTAssertNil(cmd.jsonOutput)
        XCTAssertFalse(cmd.showBoxes)
        XCTAssertEqual(cmd.threshold, 0.3, accuracy: 0.0001)
        XCTAssertEqual(cmd.resolution, 1008)
    }

    func testVisionSegmentParsesOverridesAndMultiplePrompts() throws {
        let cmd = try VisionSegment.parse([
            "/tmp/image.png",
            "--prompt", "a person", "a phone",
            "--box", "1,2,3,4,person",
            "--point", "5,6,positive,person",
            "--model", "/tmp/sam31",
            "--output", "/tmp/out",
            "--json-output", "/tmp/out-meta",
            "--mask-output-dir", "/tmp/masks",
            "--show-boxes",
            "--multimask",
            "--threshold", "0.45",
            "--resolution", "504",
        ])

        XCTAssertEqual(cmd.prompt, ["a person", "a phone"])
        XCTAssertEqual(cmd.box, ["1,2,3,4,person"])
        XCTAssertEqual(cmd.point, ["5,6,positive,person"])
        XCTAssertEqual(cmd.model, "/tmp/sam31")
        XCTAssertEqual(cmd.output, "/tmp/out")
        XCTAssertEqual(cmd.jsonOutput, "/tmp/out-meta")
        XCTAssertEqual(cmd.maskOutputDir, "/tmp/masks")
        XCTAssertTrue(cmd.showBoxes)
        XCTAssertTrue(cmd.multimask)
        XCTAssertEqual(cmd.threshold, 0.45, accuracy: 0.0001)
        XCTAssertEqual(cmd.resolution, 504)
    }

    func testVisionSegmentParsesPromptSetFromTextBoxesAndPoints() throws {
        let cmd = try VisionSegment.parse([
            "/tmp/image.png",
            "--prompt", "a person",
            "--box", "10,20,30,40,person",
            "--point", "50,60,negative,person",
        ])

        let promptSet = try cmd.parsedPromptSet()
        XCTAssertEqual(promptSet.textPrompts, ["a person"])
        XCTAssertEqual(promptSet.boxPrompts.first?.label, "person")
        XCTAssertEqual(promptSet.pointPrompts.first?.isPositive, false)
    }

    func testVisionTrackParsesDefaults() throws {
        let cmd = try VisionTrack.parse([
            "/tmp/video.mp4",
            "--prompt", "a dog",
        ])

        XCTAssertEqual(cmd.video, "/tmp/video.mp4")
        XCTAssertEqual(cmd.prompt, ["a dog"])
        XCTAssertEqual(cmd.initFrame, 0)
        XCTAssertNil(cmd.endFrame)
        XCTAssertFalse(cmd.showBoxes)
    }

    func testVisionTrackResolvesDefaultOutputPaths() {
        let videoURL = URL(fileURLWithPath: "/tmp/photo.mov")
        XCTAssertEqual(
            VisionTrack.resolveOutputURL(nil, inputVideoURL: videoURL).path,
            "/tmp/photo_tracked.mp4"
        )
        XCTAssertEqual(
            VisionTrack.resolveJSONOutputURL(nil, inputVideoURL: videoURL).path,
            "/tmp/photo_tracked.json"
        )
    }

    func testVisionTrackLiveParsesDefaults() throws {
        let cmd = try VisionTrackLive.parse([
            "--output", "/tmp/live.mp4",
            "--prompt", "a person",
        ])

        XCTAssertEqual(cmd.output, "/tmp/live.mp4")
        XCTAssertEqual(cmd.prompt, ["a person"])
        XCTAssertEqual(cmd.camera, 0)
        XCTAssertEqual(cmd.durationSeconds, 10, accuracy: 0.0001)
        XCTAssertFalse(cmd.showBoxes)
    }

    func testResolveModelRootUsesManagedDefault() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("vision-segment-sam31", isDirectory: true)
        try writeMinimalSAM31Model(at: modelRoot)

        let resolved = try VisionSegment.resolveModelRoot(nil)
        XCTAssertEqual(resolved.modelID, "vision-segment-sam31")
        XCTAssertTrue(resolved.isManaged)
        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
    }

    func testResolveModelRootAcceptsExplicitLocalPath() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let localRoot = temp.appendingPathComponent("local-sam31", isDirectory: true)
        try FileManager.default.createDirectory(at: localRoot, withIntermediateDirectories: true)

        let resolved = try VisionSegment.resolveModelRoot(localRoot.path)
        XCTAssertFalse(resolved.isManaged)
        XCTAssertEqual(resolved.rootURL.standardizedFileURL, localRoot.standardizedFileURL)
    }

    func testResolveModelRootMissingManagedDefaultMentionsPull() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let emptyModelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyModelsRoot, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(emptyModelsRoot)

        XCTAssertThrowsError(try VisionSegment.resolveModelRoot(nil)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("mere.run model pull vision-segment-sam31"))
        }
    }

    func testVisionSegmentResolvesDefaultOutputPaths() {
        let imageURL = URL(fileURLWithPath: "/tmp/photo.jpg")
        XCTAssertEqual(
            VisionSegment.resolveAnnotatedOutputURL(nil, inputImageURL: imageURL).path,
            "/tmp/photo_segmented.jpg"
        )
        XCTAssertEqual(
            VisionSegment.resolveJSONOutputURL(nil, inputImageURL: imageURL).path,
            "/tmp/photo_segmented.json"
        )
    }

    func testVisionSubcommandsIncludeSegment() {
        let visionNames = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(visionNames, Set(["caption", "inspect", "segment", "track", "track-live", "ocr"]))
    }

    private func makeTempDir() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-cli-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
