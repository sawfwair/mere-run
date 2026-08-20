@testable import MereRunApp
import XCTest

final class CLIBootstrapInstallerTests: XCTestCase {
    func testPreferredAutomaticInstallURLUsesFirstWritableCandidate() throws {
        let rootURL = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: rootURL) }

        let firstCandidateURL = rootURL.appendingPathComponent("first-bin", isDirectory: true)
        let secondCandidateURL = rootURL.appendingPathComponent("second-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: firstCandidateURL, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondCandidateURL, withIntermediateDirectories: true)

        let installURL = CLIBootstrapInstaller.preferredAutomaticInstallURL(
            installDirectoryCandidates: [
                firstCandidateURL,
                secondCandidateURL,
            ]
        )

        XCTAssertEqual(installURL, firstCandidateURL.appendingPathComponent("mere.run", isDirectory: false))
    }

    func testCopiesBundledPayloadToDestination() throws {
        let rootURL = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: rootURL) }

        let payloadURL = rootURL.appendingPathComponent("payload", isDirectory: true)
        let destinationURL = rootURL
            .appendingPathComponent("install", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("mere.run", isDirectory: false)

        try makePayload(at: payloadURL)

        try CLIBootstrapInstaller.installBundledCLI(from: payloadURL, to: destinationURL)

        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: destinationURL.path))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: destinationURL
                    .deletingLastPathComponent()
                    .appendingPathComponent("llama.framework", isDirectory: true)
                    .path
            )
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: destinationURL
                    .deletingLastPathComponent()
                    .appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
                    .path
            )
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: destinationURL
                    .deletingLastPathComponent()
                    .appendingPathComponent("Resources/default.metallib", isDirectory: false)
                    .path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: destinationURL
                    .deletingLastPathComponent()
                    .appendingPathComponent("mererun-model-source-base-url.txt", isDirectory: false)
                    .path
            )
        )
    }

    private func makePayload(at payloadURL: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: payloadURL, withIntermediateDirectories: true)

        let binaryURL = payloadURL.appendingPathComponent("mere.run", isDirectory: false)
        try "#!/usr/bin/env bash\n".write(to: binaryURL, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binaryURL.path)

        try fm.createDirectory(
            at: payloadURL.appendingPathComponent("llama.framework", isDirectory: true),
            withIntermediateDirectories: true
        )
        try fm.createDirectory(
            at: payloadURL.appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true),
            withIntermediateDirectories: true
        )
        try fm.createDirectory(
            at: payloadURL.appendingPathComponent("Resources", isDirectory: true),
            withIntermediateDirectories: true
        )
        try "fake".write(
            to: payloadURL.appendingPathComponent("Resources/default.metallib", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )
        try "https://models.example.com/\n".write(
            to: payloadURL.appendingPathComponent("mererun-model-source-base-url.txt", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )
    }

    private func temporaryDirectory() throws -> URL {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("CLIBootstrapInstallerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        return rootURL
    }
}
