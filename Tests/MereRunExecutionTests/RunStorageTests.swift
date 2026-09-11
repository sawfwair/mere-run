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

    func testReplacementKeepsOldReaderAndPublishesCompletePrivateArtifact() throws {
        let file = try root().appendingPathComponent("record.json")
        try RunRecordCodec.write(["value": "original"], to: file)
        let original = try Data(contentsOf: file)
        let reader = try FileHandle(forReadingFrom: file)
        defer { try? reader.close() }
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: file.path)
        let updated = ["value": String(repeating: "replacement", count: 100_000)]
        let artifact = try RunRecordCodec.writeArtifact(updated, to: file)
        XCTAssertEqual(try reader.readToEnd(), original)
        XCTAssertEqual(try RunRecordCodec.decoder().decode([String: String].self, from: Data(contentsOf: file)), updated)
        XCTAssertEqual(try RunArtifact.read(file), artifact)
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        XCTAssertEqual(attributes[.posixPermissions] as? Int, 0o600)
#if canImport(Darwin)
        let volume = try file.resourceValues(forKeys: [.volumeSupportsFileProtectionKey])
        if volume.allValues[.volumeSupportsFileProtectionKey] as? Bool == true {
            XCTAssertEqual(attributes[.protectionKey] as? FileProtectionType, .completeUnlessOpen)
        }
#endif
    }

    func testFailedReplacementPreservesDestinationAndRemovesTemporaryFile() throws {
        let directory = try root()
        let destination = directory.appendingPathComponent("occupied")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        let keep = destination.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: keep)
        XCTAssertThrowsError(try RunRecordCodec.write(["value": "replacement"], to: destination))
        XCTAssertEqual(try Data(contentsOf: keep), Data("keep".utf8))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["occupied"])
    }

    func testUnreadableRecordExplainsRecoveryAndPreservesBytes() throws {
        let file = try root().appendingPathComponent("record.json")
        let original = Data("original".utf8)
        try original.write(to: file)
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: file.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path) }
        XCTAssertThrowsError(try RunRecordCodec.readData(at: file)) { error in
            XCTAssertEqual((error as? RunRecordReadIssue)?.url, file)
            XCTAssertTrue(error.localizedDescription.contains("Unlock the device or check the file permissions"))
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        XCTAssertEqual(try RunRecordCodec.readData(at: file), original)
        XCTAssertThrowsError(try RunRecordCodec.readData(at: file.appendingPathExtension("missing"))) { error in
            XCTAssertFalse(error is RunRecordReadIssue)
        }
    }

    private func root() throws -> URL {
        let base = ProcessInfo.processInfo.environment["MERERUN_TEST_STORAGE_ROOT"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) } ?? FileManager.default.temporaryDirectory
        let root = base.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
