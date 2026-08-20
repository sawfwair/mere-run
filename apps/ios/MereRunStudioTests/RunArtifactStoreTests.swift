import Foundation
import XCTest

final class RunArtifactStoreTests: XCTestCase {
    private enum ExpectedFailure: Error { case fetch }

    func testFailedRefreshPreservesLastVerifiedBundle() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("job-1", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let good = destination.appendingPathComponent("result.txt")
        try Data("last-good".utf8).write(to: good)

        do {
            _ = try await RunArtifactStore(runsRoot: root).refresh(jobID: "job-1") { staging in
                try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
                try Data("partial".utf8).write(to: staging.appendingPathComponent("result.txt"))
                throw ExpectedFailure.fetch
            }
            XCTFail("Expected refresh failure")
        } catch ExpectedFailure.fetch {}

        XCTAssertEqual(try String(contentsOf: good, encoding: .utf8), "last-good")
    }

    func testSuccessfulRefreshReplacesBundle() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("job-1", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: destination.appendingPathComponent("old.txt"))

        let files = try await RunArtifactStore(runsRoot: root).refresh(jobID: "job-1") { staging in
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
            try Data("new".utf8).write(to: staging.appendingPathComponent("result.txt"))
            return ["result.txt"]
        }

        XCTAssertEqual(files, [destination.appendingPathComponent("result.txt")])
        XCTAssertEqual(try String(contentsOf: files[0], encoding: .utf8), "new")
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.appendingPathComponent("old.txt").path))
    }

    private func temporaryRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("RunArtifactStoreTests-\(UUID().uuidString)", isDirectory: true)
    }
}
