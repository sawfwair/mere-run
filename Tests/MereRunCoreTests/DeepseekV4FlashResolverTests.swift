import Foundation
import XCTest
@testable import MereRunCore

final class DeepseekV4FlashResolverTests: XCTestCase {
    private var temporaryRoots: [URL] = []

    override func tearDown() {
        for root in temporaryRoots {
            try? FileManager.default.removeItem(at: root)
        }
        temporaryRoots.removeAll()
        super.tearDown()
    }

    func testPreferredGGUFAcceptsImatrixSymlink() throws {
        let modelDir = try makeTemporaryDirectory()
        let targetDir = try makeTemporaryDirectory()
        let target = targetDir.appendingPathComponent(DeepseekV4FlashResources.previousImatrixGGUFFile)
        XCTAssertTrue(FileManager.default.createFile(atPath: target.path, contents: Data("gguf".utf8)))
        let symlink = modelDir.appendingPathComponent(DeepseekV4FlashResources.previousImatrixGGUFFile)
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: target)

        let resolved = DeepseekV4FlashGenerator.preferredGGUF(in: modelDir)

        XCTAssertEqual(resolved?.standardizedFileURL, symlink.standardizedFileURL)
    }

    func testPreferredGGUFChoosesImatrixSymlinkOverPlainGGUF() throws {
        let modelDir = try makeTemporaryDirectory()
        let fallback = modelDir.appendingPathComponent("plain.gguf")
        XCTAssertTrue(FileManager.default.createFile(atPath: fallback.path, contents: Data("fallback".utf8)))
        let targetDir = try makeTemporaryDirectory()
        let target = targetDir.appendingPathComponent(DeepseekV4FlashResources.previousImatrixGGUFFile)
        XCTAssertTrue(FileManager.default.createFile(atPath: target.path, contents: Data("imatrix".utf8)))
        let symlink = modelDir.appendingPathComponent(DeepseekV4FlashResources.previousImatrixGGUFFile)
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: target)

        let resolved = DeepseekV4FlashGenerator.preferredGGUF(in: modelDir)

        XCTAssertEqual(resolved?.standardizedFileURL, symlink.standardizedFileURL)
    }

    func testBinaryLocateAcceptsPlatformSubdirectoryUnderOverride() throws {
        let root = try makeTemporaryDirectory()
        let platformDir = root.appendingPathComponent("linux-arm64", isDirectory: true)
        let server = platformDir.appendingPathComponent(DeepseekV4FlashBinary.Kind.server.rawValue)
        try makeExecutable(at: server)

        let resolved = try DeepseekV4FlashBinary.locate(
            .server,
            environment: [
                "MERERUN_DS4_BIN_DIR": root.path,
                "MERERUN_DS4_PLATFORM_DIR": "linux-arm64",
            ]
        )

        XCTAssertEqual(resolved.standardizedFileURL, server.standardizedFileURL)
    }

    func testBinaryLocatePreservesFlatOverrideDirectory() throws {
        let root = try makeTemporaryDirectory()
        let flatServer = root.appendingPathComponent(DeepseekV4FlashBinary.Kind.server.rawValue)
        let platformServer = root
            .appendingPathComponent("linux-arm64", isDirectory: true)
            .appendingPathComponent(DeepseekV4FlashBinary.Kind.server.rawValue)
        try makeExecutable(at: flatServer)
        try makeExecutable(at: platformServer)

        let resolved = try DeepseekV4FlashBinary.locate(
            .server,
            environment: [
                "MERERUN_DS4_BIN_DIR": root.path,
                "MERERUN_DS4_PLATFORM_DIR": "linux-arm64",
            ]
        )

        XCTAssertEqual(resolved.standardizedFileURL, flatServer.standardizedFileURL)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let root = try TestFileSystem.makeTempDir(prefix: "mererun-ds4-resolver-tests")
        temporaryRoots.append(root)
        return root
    }

    private func makeExecutable(at url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data("binary".utf8)))
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    }
}
