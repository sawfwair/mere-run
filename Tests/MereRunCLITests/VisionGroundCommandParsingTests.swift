import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionGroundCommandParsingTests: XCTestCase {
    private func writeMinimalFalconModel(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: .visionGroundFalconPerception, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

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

    func testVisionGroundParsesDefaults() throws {
        let cmd = try VisionGround.parse([
            "/tmp/image.png",
            "--query", "a person",
        ])

        XCTAssertEqual(cmd.image, "/tmp/image.png")
        XCTAssertEqual(cmd.query, ["a person"])
        XCTAssertNil(cmd.model)
        XCTAssertNil(cmd.output)
        XCTAssertNil(cmd.jsonOutput)
        XCTAssertNil(cmd.maskOutputDir)
    }

    func testVisionGroundParsesPromptAliasAndOverrides() throws {
        let cmd = try VisionGround.parse([
            "/tmp/image.png",
            "--prompt", "a cat", "a dog",
            "--model", "/tmp/falcon",
            "--output", "/tmp/out",
            "--json-output", "/tmp/out-meta",
            "--mask-output-dir", "/tmp/masks",
        ])

        XCTAssertEqual(cmd.query, ["a cat", "a dog"])
        XCTAssertEqual(cmd.model, "/tmp/falcon")
        XCTAssertEqual(cmd.output, "/tmp/out")
        XCTAssertEqual(cmd.jsonOutput, "/tmp/out-meta")
        XCTAssertEqual(cmd.maskOutputDir, "/tmp/masks")
    }

    func testResolveModelRootUsesManagedDefault() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalFalconModel(at: modelRoot)

        let resolved = try VisionGround.resolveModelRoot(nil)
        XCTAssertEqual(resolved.modelID, "vision-ground-falcon-perception")
        XCTAssertTrue(resolved.isManaged)
        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
    }

    func testResolveModelRootAcceptsExplicitLocalPath() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let localRoot = temp.appendingPathComponent("local-falcon", isDirectory: true)
        try FileManager.default.createDirectory(at: localRoot, withIntermediateDirectories: true)

        let resolved = try VisionGround.resolveModelRoot(localRoot.path)
        XCTAssertFalse(resolved.isManaged)
        XCTAssertEqual(resolved.rootURL.standardizedFileURL, localRoot.standardizedFileURL)
    }

    func testResolveModelRootMissingManagedDefaultMentionsPull() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let emptyModelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyModelsRoot, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(emptyModelsRoot)

        XCTAssertThrowsError(try VisionGround.resolveModelRoot(nil)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("mere.run model pull vision-ground-falcon-perception"))
        }
    }

    func testVisionGroundResolvesDefaultOutputPaths() {
        let imageURL = URL(fileURLWithPath: "/tmp/photo.jpg")
        XCTAssertEqual(
            VisionGround.resolveAnnotatedOutputURL(nil, inputImageURL: imageURL).path,
            "/tmp/photo_grounded.jpg"
        )
        XCTAssertEqual(
            VisionGround.resolveJSONOutputURL(nil, inputImageURL: imageURL).path,
            "/tmp/photo_grounded.json"
        )
    }

    private func makeTempDir() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-cli-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
