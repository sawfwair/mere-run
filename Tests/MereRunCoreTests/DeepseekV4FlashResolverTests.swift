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

    func testResourcePinsOfficial0731Artifact() {
        XCTAssertEqual(DeepseekV4FlashResources.defaultRepoId, "antirez/deepseek-v4-gguf")
        XCTAssertEqual(
            DeepseekV4FlashResources.defaultRevision,
            "1cd7b564460821938add0475a60b942c409295e0"
        )
        XCTAssertEqual(
            DeepseekV4FlashResources.imatrixGGUFFile,
            "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
        )
        XCTAssertEqual(DeepseekV4FlashResources.defaultGGUFByteCount, 86_720_111_488)
        XCTAssertEqual(
            DeepseekV4FlashResources.defaultGGUFSHA256,
            "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
        )
    }

    func testServerArgumentsCapContextPrefillAndDiskKV() {
        let arguments = DeepseekV4FlashGenerator.serverArguments(
            ggufURL: URL(fileURLWithPath: "/models/deepseek-v4-flash-0731.gguf"),
            port: 8_143,
            kvDirectory: URL(fileURLWithPath: "/models/kv-disk")
        )

        XCTAssertEqual(arguments, [
            "-m", "/models/deepseek-v4-flash-0731.gguf",
            "--host", "127.0.0.1",
            "--port", "8143",
            "--ctx", "32768",
            "--prefill-chunk", "1024",
            "--kv-disk-dir", "/models/kv-disk",
            "--kv-disk-space-mb", "8192",
        ])
    }

    func testPreferredInstalledGGUFChooses0731OverOlderImatrix() throws {
        let modelDir = try makeTemporaryDirectory()
        let aligned = modelDir.appendingPathComponent(
            DeepseekV4FlashResources.previousAlignedImatrixGGUFFile
        )
        let current = modelDir.appendingPathComponent(DeepseekV4FlashResources.imatrixGGUFFile)
        XCTAssertTrue(FileManager.default.createFile(atPath: aligned.path, contents: Data("old".utf8)))
        XCTAssertTrue(FileManager.default.createFile(atPath: current.path, contents: Data("0731".utf8)))

        let resolved = DeepseekV4FlashGenerator.preferredInstalledGGUF(in: modelDir)

        XCTAssertEqual(resolved?.standardizedFileURL, current.standardizedFileURL)
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
