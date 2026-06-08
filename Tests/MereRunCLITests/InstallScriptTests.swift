import Foundation
import XCTest
#if os(Linux)
import Glibc
#endif

final class InstallScriptTests: XCTestCase {
    private let modelSourceConfigFilename = "mererun-model-source-base-url.txt"

    func testInstallerHelpDoesNotInstall() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false),
            arguments: ["--help"]
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(result.combinedOutput.contains("MERERUN_INSTALL_BIN_DEST"))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: fixture.destDirURL.appendingPathComponent("mere.run").path)
        )
    }

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

    func testInstallerCreatesCustomDestinationDirectoryWithoutSudo() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let nestedBinURL = fixture.rootURL
            .appendingPathComponent("custom", isDirectory: true)
            .appendingPathComponent("nested", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("mere.run", isDirectory: false)

        let result = try runInstaller(scriptURL: fixture.installScriptURL, binDestURL: nestedBinURL)

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(FileManager.default.fileExists(atPath: nestedBinURL.path))
        XCTAssertFalse(result.combinedOutput.contains("need sudo"))
    }

    func testInstallerUsesCLIPayloadSubdirectoryWhenPresent() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let cliPayloadURL = fixture.sourceDirURL.appendingPathComponent("CLI", isDirectory: true)
        try FileManager.default.createDirectory(at: cliPayloadURL, withIntermediateDirectories: true)
        try FileManager.default.moveItem(
            at: fixture.sourceDirURL.appendingPathComponent("mere.run", isDirectory: false),
            to: cliPayloadURL.appendingPathComponent("mere.run", isDirectory: false)
        )

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false)
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: fixture.destDirURL.appendingPathComponent("mere.run").path)
        )
        XCTAssertTrue(result.combinedOutput.contains("source: \(cliPayloadURL.path)"))
    }

    func testInstallerDoesNotCopyLegacyModelSourceSidecarWhenPresent() throws {
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
        XCTAssertFalse(FileManager.default.fileExists(atPath: installedConfigURL.path))
    }

    func testInstallerCopiesMlxBundleAndCompatibilityMetallibsWhenPresent() throws {
        #if os(Linux)
        throw XCTSkip("MLX Metal bundle staging is macOS-only.")
        #else
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let mlxResourcesURL = fixture.sourceDirURL
            .appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("Resources", isDirectory: true)
        try FileManager.default.createDirectory(at: mlxResourcesURL, withIntermediateDirectories: true)
        try "fake metallib".write(
            to: mlxResourcesURL.appendingPathComponent("default.metallib", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false)
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL
                    .appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
                    .path
            )
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent("Resources/default.metallib").path
            )
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent("Resources/mlx.metallib").path
            )
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent("mlx.metallib").path
            )
        )
        #endif
    }

    func testInstallerCopiesLinuxSharedLibrariesWhenPresent() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let sharedLibraryURL = fixture.sourceDirURL.appendingPathComponent("libmere_native.so", isDirectory: false)
        try "fake shared library".write(to: sharedLibraryURL, atomically: true, encoding: .utf8)

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false),
            extraEnvironment: ["MERERUN_INSTALL_PLATFORM": "Linux"]
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent("libmere_native.so").path
            )
        )
    }

    func testInstallerCopiesLinuxRuntimeBinaryWhenPresent() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let runtimeBinaryURL = fixture.sourceDirURL.appendingPathComponent("mere.run-bin", isDirectory: false)
        try "fake runtime binary".write(to: runtimeBinaryURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtimeBinaryURL.path)

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false),
            extraEnvironment: ["MERERUN_INSTALL_PLATFORM": "Linux"]
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent("mere.run-bin").path
            )
        )
    }

    func testInstallerCopiesDS4RuntimeWhenPresent() throws {
        let fixture = try makeInstallerFixture()
        defer { try? FileManager.default.removeItem(at: fixture.rootURL) }

        let ds4URL = fixture.sourceDirURL
            .appendingPathComponent("vendor", isDirectory: true)
            .appendingPathComponent("ds4", isDirectory: true)
        let metalURL = ds4URL.appendingPathComponent("metal", isDirectory: true)
        try FileManager.default.createDirectory(at: metalURL, withIntermediateDirectories: true)
        let serverURL = ds4URL.appendingPathComponent("ds4-server", isDirectory: false)
        try "fake ds4-server".write(to: serverURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: serverURL.path)
        try "fake shader".write(
            to: metalURL.appendingPathComponent("dense.metal", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )

        let result = try runInstaller(
            scriptURL: fixture.installScriptURL,
            binDestURL: fixture.destDirURL.appendingPathComponent("mere.run", isDirectory: false),
            extraEnvironment: ["MERERUN_INSTALL_DISABLE_DITTO": "1"]
        )

        XCTAssertEqual(result.status, 0, result.combinedOutput)
        XCTAssertTrue(result.combinedOutput.contains("installing DS4 inference binaries"))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent("vendor/ds4/ds4-server").path
            )
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.destDirURL.appendingPathComponent("vendor/ds4/metal/dense.metal").path
            )
        )
    }

    func testInstallLocalStagesDarwinFrameworksBundlesAndVendoredMlx() throws {
        #if os(Linux)
        throw XCTSkip("install-local Darwin runtime asset staging is macOS-only.")
        #else
        let fileManager = FileManager.default
        let rootURL = fileManager.temporaryDirectory
            .appendingPathComponent("InstallLocalTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? fileManager.removeItem(at: rootURL) }

        let repoURL = rootURL.appendingPathComponent("repo", isDirectory: true)
        let scriptsURL = repoURL.appendingPathComponent("scripts", isDirectory: true)
        let vendorMlxResourcesURL = repoURL
            .appendingPathComponent("vendor/mlx-swift_Cmlx.bundle/Contents/Resources", isDirectory: true)
        let buildURL = rootURL.appendingPathComponent("build", isDirectory: true)
        let fakeBinURL = rootURL.appendingPathComponent("bin", isDirectory: true)
        let destURL = rootURL.appendingPathComponent("dest/mere.run", isDirectory: false)
        try fileManager.createDirectory(at: scriptsURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: vendorMlxResourcesURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: buildURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: fakeBinURL, withIntermediateDirectories: true)

        let repoRootURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        for scriptName in ["install-local.sh", "install.sh"] {
            let scriptURL = scriptsURL.appendingPathComponent(scriptName, isDirectory: false)
            try fileManager.copyItem(
                at: repoRootURL.appendingPathComponent("scripts/\(scriptName)", isDirectory: false),
                to: scriptURL
            )
            try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: scriptURL.path)
        }

        let fakeCLIURL = buildURL.appendingPathComponent("mere.run", isDirectory: false)
        try """
        #!/usr/bin/env bash
        echo "mere.run local test"
        """.write(to: fakeCLIURL, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeCLIURL.path)

        try fileManager.createDirectory(
            at: buildURL.appendingPathComponent("magentart.framework", isDirectory: true),
            withIntermediateDirectories: true
        )
        try "fake framework".write(
            to: buildURL.appendingPathComponent("magentart.framework/magentart", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )
        try fileManager.createDirectory(
            at: buildURL.appendingPathComponent("MereRun_MereRunCLI.bundle", isDirectory: true),
            withIntermediateDirectories: true
        )
        try "fake guide".write(
            to: buildURL.appendingPathComponent("MereRun_MereRunCLI.bundle/guide.txt", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )
        try "fake metallib".write(
            to: vendorMlxResourcesURL.appendingPathComponent("default.metallib", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )

        let fakeSwiftURL = fakeBinURL.appendingPathComponent("swift", isDirectory: false)
        try """
        #!/usr/bin/env bash
        if [[ "$*" == *"--show-bin-path"* ]]; then
          printf '%s\\n' "\(buildURL.path)"
          exit 0
        fi
        exit 0
        """.write(to: fakeSwiftURL, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeSwiftURL.path)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            scriptsURL.appendingPathComponent("install-local.sh", isDirectory: false).path,
            "--no-build",
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["MERERUN_INSTALL_BIN_DEST"] = destURL.path
        environment["PATH"] = "\(fakeBinURL.path):/usr/bin:/bin:/usr/sbin:/sbin"
        process.environment = environment
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        try process.run()
        process.waitUntilExit()
        let output = String(data: outputPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

        XCTAssertEqual(process.terminationStatus, 0, output)
        XCTAssertTrue(fileManager.fileExists(atPath: destURL.path))
        XCTAssertTrue(fileManager.fileExists(atPath: destURL.deletingLastPathComponent().appendingPathComponent("magentart.framework/magentart").path))
        XCTAssertTrue(fileManager.fileExists(atPath: destURL.deletingLastPathComponent().appendingPathComponent("MereRun_MereRunCLI.bundle/guide.txt").path))
        XCTAssertTrue(fileManager.fileExists(atPath: destURL.deletingLastPathComponent().appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib").path))
        #endif
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

    private func runInstaller(
        scriptURL: URL,
        binDestURL: URL,
        arguments: [String] = [],
        extraEnvironment: [String: String] = [:]
    ) throws -> InstallerRunResult {
        #if os(Linux)
        return try runInstallerWithSystem(
            scriptURL: scriptURL,
            binDestURL: binDestURL,
            arguments: arguments,
            extraEnvironment: extraEnvironment
        )
        #else
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptURL.path] + arguments
        var environment = ProcessInfo.processInfo.environment
        environment["MERERUN_INSTALL_BIN_DEST"] = binDestURL.path
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        for (key, value) in extraEnvironment {
            environment[key] = value
        }
        process.environment = environment

        let captureRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("InstallScriptTests-output-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: captureRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: captureRoot) }
        let stdoutURL = captureRoot.appendingPathComponent("stdout.txt", isDirectory: false)
        let stderrURL = captureRoot.appendingPathComponent("stderr.txt", isDirectory: false)
        try Data().write(to: stdoutURL)
        try Data().write(to: stderrURL)
        let stdoutHandle = try FileHandle(forWritingTo: stdoutURL)
        let stderrHandle = try FileHandle(forWritingTo: stderrURL)
        process.standardOutput = stdoutHandle
        process.standardError = stderrHandle

        try process.run()
        process.waitUntilExit()
        try stdoutHandle.close()
        try stderrHandle.close()

        return InstallerRunResult(
            status: process.terminationStatus,
            stdout: String(data: try Data(contentsOf: stdoutURL), encoding: .utf8) ?? "",
            stderr: String(data: try Data(contentsOf: stderrURL), encoding: .utf8) ?? ""
        )
        #endif
    }

    #if os(Linux)
    private func runInstallerWithSystem(
        scriptURL: URL,
        binDestURL: URL,
        arguments: [String],
        extraEnvironment: [String: String]
    ) throws -> InstallerRunResult {
        let captureRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("InstallScriptTests-output-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: captureRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: captureRoot) }

        let stdoutURL = captureRoot.appendingPathComponent("stdout.txt", isDirectory: false)
        let stderrURL = captureRoot.appendingPathComponent("stderr.txt", isDirectory: false)
        let runnerURL = captureRoot.appendingPathComponent("run-installer.sh", isDirectory: false)

        var lines = [
            "set -e",
            "export MERERUN_INSTALL_BIN_DEST=\(shellQuote(binDestURL.path))",
            "export PATH='/usr/bin:/bin:/usr/sbin:/sbin'",
        ]
        for (key, value) in extraEnvironment.sorted(by: { $0.key < $1.key }) {
            lines.append("export \(key)=\(shellQuote(value))")
        }

        let renderedArguments = ([scriptURL.path] + arguments)
            .map(shellQuote)
            .joined(separator: " ")
        lines.append("exec /bin/bash \(renderedArguments)")
        try (lines.joined(separator: "\n") + "\n").write(to: runnerURL, atomically: true, encoding: .utf8)

        let command = [
            "/bin/bash",
            shellQuote(runnerURL.path),
            ">",
            shellQuote(stdoutURL.path),
            "2>",
            shellQuote(stderrURL.path),
        ].joined(separator: " ")

        let rawStatus = system(command)
        let status: Int32
        if rawStatus == -1 {
            status = -1
        } else {
            status = Int32((rawStatus >> 8) & 0xff)
        }

        return InstallerRunResult(
            status: status,
            stdout: String(data: try Data(contentsOf: stdoutURL), encoding: .utf8) ?? "",
            stderr: String(data: try Data(contentsOf: stderrURL), encoding: .utf8) ?? ""
        )
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
    #endif
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
