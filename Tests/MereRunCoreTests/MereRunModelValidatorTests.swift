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

    private func writeMinimalValidGemma4Model(at root: URL, id: ModelResolver.ModelID = .gemma4) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidHiDreamO1Model(at root: URL, id: ModelResolver.ModelID = .hidreamO1Dev) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("preprocessor_config.json"), contents: Data("{}".utf8))
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

    private func writeMinimalValidFalconModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .visionGroundFalconPerception, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizerDir)
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalValidPrivacyFilterModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .privacyFilter, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalValidQwen3ASRModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .qwen3ASR, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(
            root.appendingPathComponent("config.json"),
            contents: Data(#"{"quantization":{"bits":8,"group_size":64}}"#.utf8)
        )
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidQwen35AgentGGUFModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .qwen35Agent9B, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("Qwen3.5-9B-Q4_K_M.gguf"), contents: Data())
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

    func testHiDreamO1UnifiedRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-hidream-o1-dev", isDirectory: true)
        try writeMinimalValidHiDreamO1Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-hidream-o1-dev")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .hidream)
        XCTAssertEqual(report.manifest?.engine, .hidreamO1)
        XCTAssertEqual(report.manifest?.defaults?.steps, 28)
    }

    func testHiDreamO1FullUnifiedRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-hidream-o1", isDirectory: true)
        try writeMinimalValidHiDreamO1Model(at: root, id: .hidreamO1)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-hidream-o1")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .hidream)
        XCTAssertEqual(report.manifest?.engine, .hidreamO1)
        XCTAssertEqual(report.manifest?.defaults?.steps, 50)
        XCTAssertEqual(report.manifest?.defaults?.cfg, 5.0)
    }

    func testGemma4MaxChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-gemma4-max", isDirectory: true)
        try writeMinimalValidGemma4Model(at: root, id: .gemma4Max)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-gemma4-max")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.tier, .max)
        XCTAssertEqual(report.manifest?.family, .gemma)
    }

    func testGemma4NanoChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-gemma4-nano", isDirectory: true)
        try writeMinimalValidGemma4Model(at: root, id: .gemma4Nano)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-gemma4-nano")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.tier, .nano)
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

    func testFalconVisionGroundRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .falcon)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), Set([.visionGrounding, .visionDetection, .visionSegmentation]))
    }

    func testFalconValidationFailsWithoutTokenizer() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("tokenizer/tokenizer.json"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("tokenizer.json") })
    }

    func testFalconValidationFailsWithoutConfig() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("config.json"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("config.json") })
    }

    func testFalconValidationFailsWithoutWeights() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("model.safetensors"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("model.safetensors.index.json") })
    }

    func testPrivacyFilterRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-anonymize-privacy-filter", isDirectory: true)
        try writeMinimalValidPrivacyFilterModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-anonymize-privacy-filter")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .privacy)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), Set([.textAnonymization]))
    }

    func testQwen3ASRUsesHFQuantizationConfigWithoutManifestQuantization() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("speech-asr-qwen3", isDirectory: true)
        try writeMinimalValidQwen3ASRModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "speech-asr-qwen3")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
    }

    func testGGUFCodeModelDoesNotRequireManifestQuantization() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-agent-qwen35-9b", isDirectory: true)
        try writeMinimalValidQwen35AgentGGUFModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-agent-qwen35-9b")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertFalse(report.warnings.contains("Missing model root marker (expected model_index.json)."))
    }
}
