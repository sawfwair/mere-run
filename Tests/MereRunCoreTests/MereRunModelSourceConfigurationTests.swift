import Foundation
import XCTest
@testable import MereRunCore

final class MereRunModelSourceConfigurationTests: XCTestCase {
    func testPublicBaseURLFallsBackToPackagedSidecarNearInvocationPath() throws {
        let fileManager = FileManager.default
        let realDir = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let installDir = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)

        try fileManager.createDirectory(at: realDir, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: installDir, withIntermediateDirectories: true)
        defer {
            try? fileManager.removeItem(at: realDir)
            try? fileManager.removeItem(at: installDir)
        }

        let configURL = installDir.appendingPathComponent(
            MereRunModelSourceConfiguration.packagedBaseURLFilename,
            isDirectory: false
        )
        try "https://example.com/\n".write(to: configURL, atomically: true, encoding: .utf8)
        let installCommandURL = installDir.appendingPathComponent("mere.run", isDirectory: false)
        try "".write(to: installCommandURL, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: installCommandURL.path)

        let resolved = MereRunModelSourceConfiguration.publicBaseURL(
            environment: [:],
            currentDirectoryPath: installDir.path,
            commandPath: "mere.run",
            pathEnvironment: installDir.path,
            processExecutablePath: realDir.appendingPathComponent("mere.run").path
        )

        XCTAssertEqual(resolved?.absoluteString, "https://example.com/")
    }

    func testHasAnyDownloadSourceTrueForExplicitConfiguration() {
        XCTAssertTrue(
            MereRunModelSourceConfiguration.hasAnyDownloadSource(
                environment: ["MERERUN_R2_ACCOUNT_ID": "abc123"]
            )
        )
    }

    func testHasAnyDownloadSourceFalseWithoutConfiguredSource() {
        XCTAssertFalse(
            MereRunModelSourceConfiguration.hasAnyDownloadSource(
                environment: [:]
            )
        )
    }
}
