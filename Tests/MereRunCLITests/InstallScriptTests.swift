import Foundation
import XCTest

final class InstallScriptTests: XCTestCase {
    private let modelSourceConfigFilename = "mererun-model-source-base-url.txt"

    func testInstallerIgnoresMissingModelSourceSidecar() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false)
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: fixture.destDirURL.appendingPathComponent("mere.run").path)
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent(modelSourceConfigFilename).path
            )
        )
        XCTAssertFalse(result.combinedOutput.contains("Cannot get the real path"))
    }

    func testInstallerCopiesPackagedModelSourceSidecarWhenPresent() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let sourceConfigURL = fixture.sourceDirURL.appendingPathComponent(modelSourceConfigFilename, isDirectory: false)
        try "https://models.example.com/\n".write(to: sourceConfigURL, atomically: true, encoding: .utf8)

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false)
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        let installedConfigURL = fixture.destDirURL.appendingPathComponent(modelSourceConfigFilename, isDirectory: false)
        XCTAssertEqual(
            try String(contentsOf: installedConfigURL, encoding: .utf8),
            "https://models.example.com/\n"
        )
    }

    private func makeInstallerFixture() throws -> InstallerFixture {
        let fileManager = FileManager.default
        let rootURL = fileManager.temporaryDirectory
            .appendingPathComponent("InstallScriptTests-\(UUID().uuidString)", isDirectory: true)
        let sourceDirURL = rootURL.appendingPathComponent("source", isDirectory: true)
        let destDirURL = rootURL.appendingPathComponent("dest", isDirectory: true)
        try fileManager.createDirectory(at: sourceDirURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: destDirURL, withIntermediateDirectories: true)

        let repoRootURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let installScriptURL = sourceDirURL.appendingPathComponent("install.sh", isDirectory: false)
        try fileManager.copyItem(
            at: repoRootURL.appendingPathComponent("scripts/install.sh", isDirectory: false),
            to: installScriptURL
        )
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: installScriptURL.path)

        let fakeBinaryURL = sourceDirURL.appendingPathComponent("mere.run", isDirectory: false)
        try """
        #!/usr/bin/env bash
        echo "mere.run test help"
        """.write(to: fakeBinaryURL, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeBinaryURL.path)

        return InstallerFixture(
            rootURL: rootURL,
            sourceDirURL: sourceDirURL,
            destDirURL: destDirURL,
            installScriptURL: installScriptURL
        )
    }

    private func runInstaller(scriptURL: URL, binDestURL: URL) throws -> InstallerRunResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptURL.path]
        var environment = ProcessInfo.processInfo.environment
        environment["MERERUN_INSTALL_BIN_DEST"] = binDestURL.path
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        process.environment = environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        return InstallerRunResult(
            status: process.terminationStatus,
            stdout: String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "",
            stderr: String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        )
    }
}

private struct InstallerFixture {
    let rootURL: URL
    let sourceDirURL: URL
    let destDirURL: URL
    let installScriptURL: URL
}

private struct InstallerRunResult {
    let status: Int32
    let stdout: String
    let stderr: String

    var combinedOutput: String {
        stdout + stderr
    }
}
