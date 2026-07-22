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
            "metal/dense.metal",
            "metal/dsv4_kv.metal",
            "metal/moe.metal",
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
}
