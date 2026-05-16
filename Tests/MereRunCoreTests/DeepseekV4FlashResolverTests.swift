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

    private func makeTemporaryDirectory() throws -> URL {
        let root = try TestFileSystem.makeTempDir(prefix: "mererun-ds4-resolver-tests")
        temporaryRoots.append(root)
        return root
    }
}
