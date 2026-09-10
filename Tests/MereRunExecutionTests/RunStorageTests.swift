import Foundation
import XCTest
import MereRunExecution

final class RunStorageTests: XCTestCase {
    func testLeaseExcludesOtherOwnersAndReleasesOnDeinit() throws {
        let directory = try root().appendingPathComponent("run")
        try RunDirectoryLease.createDirectory(directory)
        var owner: RunDirectoryLease? = try XCTUnwrap(RunDirectoryLease.acquire(in: directory, filename: ".run.lock"))
        XCTAssertNotNil(owner)
        XCTAssertNil(try RunDirectoryLease.acquire(in: directory, filename: ".run.lock"))
        owner = nil
        let successor = try XCTUnwrap(RunDirectoryLease.acquire(in: directory, filename: ".run.lock"))
        successor.release()
        successor.release()
        XCTAssertNotNil(try RunDirectoryLease.acquire(in: directory, filename: ".run.lock"))
    }

    func testDirectoryCreationPreservesExistingContentAndUsesPrivatePermissions() throws {
        let directory = try root()
        let file = directory.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: file)
        XCTAssertThrowsError(try RunDirectoryLease.createDirectory(directory))
        XCTAssertEqual(try Data(contentsOf: file), Data("keep".utf8))
        let run = directory.appendingPathComponent("run")
        try RunDirectoryLease.createDirectory(run)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: run.path)[.posixPermissions] as? Int, 0o700)
    }

    func testArtifactHashAndMetadataAbsenceDetectChanges() throws {
        let file = try root().appendingPathComponent("metadata.json")
        let absent = try RunFileSnapshot.capture(file)
        XCTAssertNil(absent.artifact)
        try Data("abc".utf8).write(to: file)
        XCTAssertFalse(try absent.isUnchanged())
        let original = try RunFileSnapshot.capture(file)
        XCTAssertEqual(original.artifact?.byteCount, 3)
        XCTAssertEqual(original.artifact?.sha256, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        XCTAssertTrue(try original.isUnchanged())
        try Data("abcd".utf8).write(to: file)
        XCTAssertFalse(try original.isUnchanged())
        try FileManager.default.removeItem(at: file)
        XCTAssertFalse(try original.isUnchanged())
        XCTAssertTrue(try absent.isUnchanged())
    }

    func testRecordCodecPreservesWireFormattingAndPrivatePermissions() throws {
        struct Fixture: Codable, Equatable {
            let createdAt: Date
            let state: RunState
        }
        let file = try root().appendingPathComponent("record.json")
        let value = Fixture(createdAt: Date(timeIntervalSince1970: 0), state: .interrupted)
        try RunRecordCodec.write(value, to: file)
        XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), """
        {
          "createdAt" : "1970-01-01T00:00:00Z",
          "state" : "interrupted"
        }
        """)
        XCTAssertEqual(try RunRecordCodec.decoder().decode(Fixture.self, from: Data(contentsOf: file)), value)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: file.path)[.posixPermissions] as? Int, 0o600)
    }

    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
