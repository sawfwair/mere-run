import Foundation
import XCTest
import MereRunMLXTestSupport
#if canImport(CryptoKit)
import CryptoKit
#endif

final class MLXTestLibrarySelectionTests: XCTestCase {
    #if canImport(CryptoKit)
    func testExplicitRebuildWinsOnlyWhileItsBinaryMatchesTheStamp() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let products = root.appendingPathComponent("build/Debug")
        let bundle = products.appendingPathComponent("Example.xctest")
        let vendor = root.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle", isDirectory: true)
        let current = products.appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
        let resources = current.appendingPathComponent("Contents/Resources")
        for path in [vendor, resources] {
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
        }
        let bytes = Data("rebuilt kernels".utf8)
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        try bytes.write(to: resources.appendingPathComponent("default.metallib"))
        try "metallib-sha256: \(digest)\n".write(to: resources.appendingPathComponent("default.metallib.version"), atomically: true, encoding: .utf8)
        let executable = bundle.appendingPathComponent("Contents/MacOS/Example")
        XCTAssertEqual(try MLXTestSupport.findCmlxBundle(
            executableURL: executable, arguments: [executable.path], fileManager: .default, workspaceRoots: [root]
        ), current)
        try Data("altered kernels".utf8).write(to: resources.appendingPathComponent("default.metallib"))
        XCTAssertEqual(try MLXTestSupport.findCmlxBundle(
            executableURL: executable, arguments: [executable.path], fileManager: .default, workspaceRoots: [root]
        ), vendor)
    }
    #endif

    func testPinnedVendorWinsOverRegeneratedAndPreviouslyInstalledLibraries() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let products = root.appendingPathComponent(".build/Products/Debug")
        let bundle = products.appendingPathComponent("Example.xctest")
        let vendor = root.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle", isDirectory: true)
        let current = products.appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
        let stale = bundle.appendingPathComponent("Contents/Resources/mlx-swift_Cmlx.bundle")
        for path in [vendor, current, stale] {
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
        }
        let selected = try MLXTestSupport.findCmlxBundle(
            executableURL: URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer/usr/bin/xctest"),
            arguments: ["xctest", bundle.path], fileManager: .default, workspaceRoots: [root]
        )
        XCTAssertEqual(selected, vendor)

        try FileManager.default.removeItem(at: vendor)
        let fallback = try MLXTestSupport.findCmlxBundle(
            executableURL: URL(fileURLWithPath: "/usr/bin/xctest"), arguments: ["xctest", bundle.path],
            fileManager: .default, workspaceRoots: [root]
        )
        XCTAssertEqual(fallback, current)
    }

    func testDirectTestExecutableUsesRepositoryKernelArtifact() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let products = root.appendingPathComponent("build/Debug")
        let current = products.appendingPathComponent("mlx-swift_Cmlx.bundle")
        let otherWorkspace = root.appendingPathComponent("other-workspace")
        try FileManager.default.createDirectory(at: current, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: otherWorkspace.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle"), withIntermediateDirectories: true
        )
        let executable = products.appendingPathComponent("Example.xctest/Contents/MacOS/Example")
        XCTAssertEqual(try MLXTestSupport.findCmlxBundle(
            executableURL: executable, arguments: [executable.path], fileManager: .default, workspaceRoots: [otherWorkspace]
        ), otherWorkspace.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle"))
    }

    func testColocatedLibraryUsesLinkedTestImageInsteadOfXcodeRunnerDirectory() {
        let runner = URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer/usr/bin/xctest")
        let bundle = URL(fileURLWithPath: "/tmp/products/Example.xctest")
        let directory = bundle.appendingPathComponent("Contents/MacOS", isDirectory: true)
        XCTAssertEqual(MLXTestSupport.linkedBinaryDirectory(executableURL: runner, arguments: [runner.path, bundle.path]), directory)
        let executable = directory.appendingPathComponent("Example")
        XCTAssertEqual(MLXTestSupport.linkedBinaryDirectory(executableURL: executable, arguments: [executable.path]), directory)
        let command = URL(fileURLWithPath: "/tmp/products/example-cli")
        XCTAssertEqual(MLXTestSupport.linkedBinaryDirectory(executableURL: command, arguments: [command.path]), command.deletingLastPathComponent())
    }
}
