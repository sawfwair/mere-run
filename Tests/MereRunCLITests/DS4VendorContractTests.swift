import Foundation
import XCTest

final class DS4VendorContractTests: XCTestCase {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testVendorPinMatchesRebuildScriptAndNotices() throws {
        let vendorRoot = repositoryRoot.appendingPathComponent("vendor/ds4", isDirectory: true)
        let version = try String(
            contentsOf: vendorRoot.appendingPathComponent("VERSION"),
            encoding: .utf8
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        let script = try String(
            contentsOf: repositoryRoot.appendingPathComponent("scripts/rebuild_ds4.sh"),
            encoding: .utf8
        )
        let readme = try String(
            contentsOf: vendorRoot.appendingPathComponent("README.md"),
            encoding: .utf8
        )
        let notices = try String(
            contentsOf: repositoryRoot.appendingPathComponent("THIRD_PARTY_NOTICES.md"),
            encoding: .utf8
        )

        XCTAssertNotNil(
            version.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression),
            "vendor/ds4/VERSION must contain one full upstream commit SHA."
        )
        XCTAssertTrue(script.contains(#"DS4_COMMIT="${DS4_COMMIT:-\#(version)}""#))
        XCTAssertTrue(script.contains("DS4_CODESIGN_IDENTITY"))
        XCTAssertTrue(readme.contains("commit `\(version)`"))
        XCTAssertTrue(readme.contains("DwarfStar inference binaries"))
        XCTAssertTrue(readme.contains("DS4_CODESIGN_IDENTITY=<certificate-fingerprint>"))
        XCTAssertTrue(notices.contains("pinned upstream commit: `\(version)`"))
    }

    func testVendorPayloadContainsRuntimeAndMetalAssets() {
        let vendorRoot = repositoryRoot.appendingPathComponent("vendor/ds4", isDirectory: true)
        let requiredPaths = [
            "ds4",
            "ds4-server",
            "ds4-bench",
            "LICENSE",
            "IRIS-LICENSE",
            "metal/dense.metal",
            "metal/dsv4_kv.metal",
            "metal/dsv41.metal",
            "metal/moe.metal",
            "metal/qwen4.metal",
            "metal/qwen4_vision.metal",
        ]

        for relativePath in requiredPaths {
            XCTAssertTrue(
                FileManager.default.fileExists(
                    atPath: vendorRoot.appendingPathComponent(relativePath).path
                ),
                "Missing required DwarfStar vendor asset: \(relativePath)"
            )
        }
    }

    #if os(macOS)
    func testVendoredExecutablesSupportMacOS15() throws {
        for name in ["ds4", "ds4-server", "ds4-bench"] {
            let binary = repositoryRoot.appendingPathComponent("vendor/ds4/\(name)")
            let output = try run(URL(fileURLWithPath: "/usr/bin/otool"), arguments: ["-l", binary.path])
            XCTAssertTrue(output.contains("minos 15.0"), "\(name) must support macOS 15")
        }
    }

    func testVendoredServerAcceptsManagedRuntimeOptions() throws {
        let binary = repositoryRoot.appendingPathComponent("vendor/ds4/ds4-server")
        let output = try run(binary, arguments: ["--help"])
        for option in ["--model", "--host", "--port", "--ctx", "--prefill-chunk",
                       "--kv-disk-dir", "--kv-disk-space-mb"] {
            XCTAssertTrue(output.contains(option), "Vendored server is missing \(option)")
        }
    }

    private func run(_ executable: URL, arguments: [String]) throws -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = output
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, "\(executable.lastPathComponent) failed")
        return String(decoding: data, as: UTF8.self)
    }
    #endif

}
