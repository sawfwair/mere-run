#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import Foundation
import AudioCore
import MereRunCore
import XCTest
@testable import AudioSTT

final class SOVCoreMLManifestTests: XCTestCase {
    #if !canImport(CoreML)
    func testUnavailableProviderRejectsBeforeReadingArtifacts() {
        XCTAssertThrowsError(try SOVCoreMLEncoder(root: URL(fileURLWithPath: "/unavailable-sov-artifact"))) { error in
            XCTAssertEqual((error as? SpeechTranscriptionIssue)?.code, "unsupported_provider")
        }
    }
    #endif

    func testRejectsCorruptionAndUnlistedCompiledFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let compiled = root.appendingPathComponent("encoder.mlmodelc")
        try FileManager.default.createDirectory(at: compiled, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = compiled.appendingPathComponent("model.mil")
        let data = Data("compiled fixture".utf8)
        try data.write(to: file)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let manifest = SOVCoreMLEncoder.Manifest(
            schemaVersion: 1, sourceRepository: SOVTranscribeResources.repository, sourceRevision: SOVTranscribeResources.revision,
            inputShape: [1, 128, 1, 3000], outputShape: [1, 375, 2816], files: [.init(path: "encoder.mlmodelc/model.mil", sha256: digest)])
        XCTAssertNoThrow(try manifest.validate(root: root))
        try Data("extra".utf8).write(to: compiled.appendingPathComponent("unlisted"))
        XCTAssertThrowsError(try manifest.validate(root: root))
        try FileManager.default.removeItem(at: compiled.appendingPathComponent("unlisted"))
        try Data("corrupt".utf8).write(to: file)
        XCTAssertThrowsError(try manifest.validate(root: root))
    }

    func testRejectsWrongSourceAndTraversal() throws {
        let root = FileManager.default.temporaryDirectory
        for (revision, path) in [("wrong", "encoder.mlmodelc/model.mil"), (SOVTranscribeResources.revision, "encoder.mlmodelc/../outside")] {
            let manifest = SOVCoreMLEncoder.Manifest(schemaVersion: 1, sourceRepository: SOVTranscribeResources.repository,
                sourceRevision: revision, inputShape: [1, 128, 1, 3000], outputShape: [1, 375, 2816], files: [.init(path: path, sha256: "")])
            XCTAssertThrowsError(try manifest.validate(root: root))
        }
    }
}
