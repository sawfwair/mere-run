import Foundation
import XCTest
import MereRunModelKit

final class ModelMetadataTests: XCTestCase {
    func testManifestPersistsRevisionAndAcknowledgementWithoutCore() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = MereRunModelManifest(
            id: ManagedModelID.kleinNano.rawValue, family: .klein,
            sources: [.init(role: "weights", repository: "example/model", revision: "abc123", destinationPath: "transformer")],
            usageTermsAcknowledged: true, createdAt: Date(timeIntervalSince1970: 0)
        )
        try manifest.write(to: root)
        XCTAssertEqual(try MereRunModelManifest.loadRequired(from: root), manifest)
        let legacy = Data(#"{"schemaVersion":1,"id":"image-klein-nano"}"#.utf8)
        let decoded = try JSONDecoder().decode(MereRunModelManifest.self, from: legacy)
        XCTAssertEqual(decoded.id, manifest.id)
        XCTAssertNil(decoded.sources)
        XCTAssertNil(decoded.usageTermsAcknowledged)
    }

    func testArtifactPinFollowsSymlinkAndDetectsChangedBytes() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("snapshot")
        let link = root.appendingPathComponent("weights")
        try Data("abc".utf8).write(to: target)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        let pin = ModelArtifactPin(filename: "weights", byteCount: 3, sha256: digest.uppercased())
        XCTAssertEqual(try pin.verify(in: root), link)
        try Data("abd".utf8).write(to: target)
        XCTAssertThrowsError(try pin.verify(in: root)) {
            guard case .checksumMismatch(let path, let expected, _) = $0 as? ModelArtifactVerificationError else {
                return XCTFail("Expected checksum mismatch: \($0)")
            }
            XCTAssertEqual(path, link.path)
            XCTAssertEqual(expected, digest)
        }
        try Data("abcd".utf8).write(to: target)
        XCTAssertThrowsError(try pin.verify(in: root)) {
            XCTAssertEqual($0 as? ModelArtifactVerificationError, .sizeMismatch(path: link.path, expected: 3, actual: 4))
        }
    }
}
