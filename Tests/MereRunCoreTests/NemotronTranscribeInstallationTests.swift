import Foundation
import XCTest
@testable import MereRunCore

final class NemotronTranscribeInstallationTests: MereRunCoreTestCase {
    func testLegacyBF16IsPreservedBeforeDefaultReplacement() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try legacyInstall(at: root)
        try NemotronTranscribeResources.preserveLegacyBF16Install(modelsRoot: root)
        let preserved = root.appendingPathComponent(NemotronTranscribeResources.bf16ModelID)
        XCTAssertEqual(try MereRunModelManifest.loadRequired(from: original).id, NemotronTranscribeResources.modelID)
        XCTAssertEqual(try MereRunModelManifest.loadRequired(from: preserved).id, NemotronTranscribeResources.bf16ModelID)
        let before = try FileManager.default.attributesOfItem(atPath: original.appendingPathComponent("model.safetensors").path)
        let after = try FileManager.default.attributesOfItem(atPath: preserved.appendingPathComponent("model.safetensors").path)
        XCTAssertEqual(try XCTUnwrap(before[.systemFileNumber] as? NSNumber).uint64Value,
                       try XCTUnwrap(after[.systemFileNumber] as? NSNumber).uint64Value)
        try FileManager.default.removeItem(at: original)
        XCTAssertEqual(try Data(contentsOf: preserved.appendingPathComponent("model.safetensors")), Data("BF16 weights".utf8))
        let resolver = ModelResolver(locations: ModelLocationSnapshot(primaryRoot: root))
        XCTAssertEqual(try resolver.resolve(.nemotronTranscribeBF16ASR).rootURL.standardizedFileURL, preserved.standardizedFileURL)
        XCTAssertNil(resolver.resolveIfPresent(.nemotronTranscribeASR))
    }

    func testPreservedCacheLinksSurviveRemovalOfLegacyRoot() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try legacyInstall(at: root)
        let weights = original.appendingPathComponent("model.safetensors")
        let cached = root.appendingPathComponent("cached-weights.safetensors")
        try FileManager.default.moveItem(at: weights, to: cached)
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: cached)
        try NemotronTranscribeResources.preserveLegacyBF16Install(modelsRoot: root)
        try FileManager.default.removeItem(at: original)
        let preserved = root.appendingPathComponent(NemotronTranscribeResources.bf16ModelID)
        XCTAssertEqual(try Data(contentsOf: preserved.appendingPathComponent("model.safetensors")), Data("BF16 weights".utf8))
    }

    func testExistingIncompleteBF16DestinationIsNotOverwritten() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try legacyInstall(at: root)
        let preserved = root.appendingPathComponent(NemotronTranscribeResources.bf16ModelID)
        try FileManager.default.createDirectory(at: preserved, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: preserved.appendingPathComponent("user-file"))
        XCTAssertThrowsError(try NemotronTranscribeResources.preserveLegacyBF16Install(modelsRoot: root))
        XCTAssertEqual(try Data(contentsOf: preserved.appendingPathComponent("user-file")), Data("keep".utf8))
        XCTAssertTrue(FileManager.default.fileExists(atPath: original.appendingPathComponent("model.safetensors").path))
    }

    func testInternalPayloadLinksFollowThePreservedTree() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try legacyInstall(at: root)
        let payload = original.appendingPathComponent("payload")
        try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
        let weights = original.appendingPathComponent("model.safetensors")
        let internalWeights = payload.appendingPathComponent("weights.safetensors")
        try FileManager.default.moveItem(at: weights, to: internalWeights)
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: internalWeights)
        try NemotronTranscribeResources.preserveLegacyBF16Install(modelsRoot: root)
        try FileManager.default.removeItem(at: original)
        let preserved = root.appendingPathComponent(NemotronTranscribeResources.bf16ModelID)
        XCTAssertEqual(try Data(contentsOf: preserved.appendingPathComponent("model.safetensors")), Data("BF16 weights".utf8))
    }

    func testDefaultRequiresAllPackedShardsAndDoesNotAcceptLegacyBF16() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try legacyInstall(at: root)
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: NemotronTranscribeResources.modelID))
        XCTAssertFalse(spec.missingPaths(in: original).isEmpty)
        for name in NemotronTranscribeResources.quantizedFiles {
            try Data("{}".utf8).write(to: original.appendingPathComponent(name))
        }
        XCTAssertTrue(spec.missingPaths(in: original).isEmpty)
        let last = NemotronTranscribeResources.quantizedFiles.last!
        try FileManager.default.removeItem(at: original.appendingPathComponent(last))
        XCTAssertTrue(spec.missingPaths(in: original).contains { $0.lastPathComponent == last })
        let bf16 = try XCTUnwrap(ManagedModelCatalog.spec(for: NemotronTranscribeResources.bf16ModelID))
        XCTAssertFalse(bf16.missingPaths(in: original).isEmpty)
    }

    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func legacyInstall(at root: URL) throws -> URL {
        let model = root.appendingPathComponent(NemotronTranscribeResources.modelID)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        var manifest = MereRunModelManifest.template(for: .nemotronTranscribeBF16ASR)
        manifest.id = NemotronTranscribeResources.modelID
        manifest.usageTermsAcknowledged = true
        try manifest.write(to: model)
        for name in NemotronTranscribeResources.files {
            try Data((name == "model.safetensors" ? "BF16 weights" : "{}").utf8).write(to: model.appendingPathComponent(name))
        }
        return model
    }
}
