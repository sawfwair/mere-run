import Foundation
import XCTest
@testable import MereRunCore

final class MereRunModelValidatorTests: MereRunCoreTestCase {

    private func writeMinimalValidModel(at root: URL, id: ModelResolver.ModelID) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        // Root marker
        try TestFileSystem.writeFile(root.appendingPathComponent("model_index.json"), contents: Data("{}".utf8))

        // Required components
        for component in ["transformer", "text_encoder", "vae"] {
            let dir = root.appendingPathComponent(component, isDirectory: true)
            try TestFileSystem.createDirectory(dir)
            try TestFileSystem.writeFile(dir.appendingPathComponent("config.json"), contents: Data("{}".utf8))
            try TestFileSystem.writeFile(dir.appendingPathComponent("model.safetensors"))
        }

        // Optional-but-expected components (warnings if missing)
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizerDir)
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))

        let schedulerDir = root.appendingPathComponent("scheduler", isDirectory: true)
        try TestFileSystem.createDirectory(schedulerDir)
        try TestFileSystem.writeFile(schedulerDir.appendingPathComponent("scheduler_config.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidQ35Model(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .q35, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidGemma4Model(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .gemma4, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidSAM31Model(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .visionSegmentSAM31, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizerDir)
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    func testValidModelPasses() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-klein-nano", isDirectory: true)
        try writeMinimalValidModel(at: root, id: .kleinNano)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-klein-nano")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
    }

    func testMissingWeightsFails() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-klein-nano", isDirectory: true)
        try writeMinimalValidModel(at: root, id: .kleinNano)

        // Remove transformer weights to trigger a validation error.
        try FileManager.default.removeItem(at: root.appendingPathComponent("transformer/model.safetensors"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-klein-nano")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains("No *.safetensors weights found in transformer/"))
    }

    func testManifestIdMismatchFails() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-klein-nano", isDirectory: true)
        try writeMinimalValidModel(at: root, id: .kleinNano)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-klein-max")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains("Manifest id mismatch: expected=image-klein-max found=image-klein-nano"))
    }

    func testQ35ChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-q35", isDirectory: true)
        try writeMinimalValidQ35Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-q35")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
    }

    func testGemma4ChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-gemma4", isDirectory: true)
        try writeMinimalValidGemma4Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-gemma4")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .gemma)
    }

    func testSAM31VisionSegmentationRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-segment-sam31", isDirectory: true)
        try writeMinimalValidSAM31Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-segment-sam31")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), Set([.visionSegmentation, .visionTracking]))
    }
}
