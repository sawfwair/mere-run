import Foundation
@testable import MereRunCore
import XCTest

final class FileManagerSymlinkTraversalTests: XCTestCase {
    func testContentsListsDirectDirectory() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let entries = try FileManager.default.contentsOfDirectoryResolvingSymlinks(at: fixture.target)

        XCTAssertEqual(entries.map(\.lastPathComponent).sorted(), ["nested", "weights.safetensors"])
    }

    func testContentsListsSymlinkedComponentDirectory() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let entries = try FileManager.default.contentsOfDirectoryResolvingSymlinks(at: fixture.link)

        XCTAssertEqual(entries.map(\.lastPathComponent).sorted(), ["nested", "weights.safetensors"])
        XCTAssertTrue(entries.allSatisfy { FileManager.default.fileExists(atPath: $0.path) })
    }

    func testEnumeratorTraversesSymlinkedRoot() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let enumerator = try XCTUnwrap(FileManager.default.enumeratorResolvingSymlinks(at: fixture.link))
        let entries = enumerator.compactMap { ($0 as? URL)?.lastPathComponent }

        XCTAssertTrue(entries.contains("weights.safetensors"))
        XCTAssertTrue(entries.contains("config.json"))
    }

    func testBrokenDirectorySymlinkStillFails() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let link = root.appendingPathComponent("missing", isDirectory: true)
        try FileManager.default.createSymbolicLink(
            at: link,
            withDestinationURL: root.appendingPathComponent("not-created", isDirectory: true)
        )

        XCTAssertThrowsError(try FileManager.default.contentsOfDirectoryResolvingSymlinks(at: link))
    }

    private func makeFixture() throws -> (root: URL, target: URL, link: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let target = root.appendingPathComponent("snapshot/component", isDirectory: true)
        let install = root.appendingPathComponent("install", isDirectory: true)
        let link = install.appendingPathComponent("component", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try Data().write(to: target.appendingPathComponent("weights.safetensors"))
        let nested = target.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data().write(to: nested.appendingPathComponent("config.json"))
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        return (root, target, link)
    }
}
