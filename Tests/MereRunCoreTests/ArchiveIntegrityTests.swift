import Foundation
import XCTest
@testable import MereRunCore

final class ArchiveIntegrityTests: XCTestCase {
    func testSHA256HexMatchesKnownDigest() throws {
        let url = try makeTempFile(contents: "hello")

        XCTAssertEqual(
            try ArchiveIntegrity.sha256Hex(of: url),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
    }

    func testVerifyAcceptsMatchingDigestCaseInsensitively() throws {
        let url = try makeTempFile(contents: "hello")

        XCTAssertNoThrow(
            try ArchiveIntegrity.verify(
                file: url,
                expectedSHA256: "2CF24DBA5FB0A30E26E83B2AC5B9E29E1B161E5C1FA7425E73043362938B9824"
            )
        )
    }

    func testVerifyRejectsMismatchedDigest() throws {
        let url = try makeTempFile(contents: "hello")
        let wrongDigest = String(repeating: "0", count: 64)

        XCTAssertThrowsError(
            try ArchiveIntegrity.verify(file: url, expectedSHA256: wrongDigest)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("SHA-256 mismatch"))
        }
    }

    func testRequiredSHA256RejectsMissingDigest() {
        XCTAssertThrowsError(
            try ArchiveIntegrity.requiredSHA256(nil, artifact: "models/example.tar.gz")
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("Missing required SHA-256 digest"))
            XCTAssertTrue(error.localizedDescription.contains("models/example.tar.gz"))
        }
    }

    func testManagedArchiveSourceResolvesDigestByRequestedKey() {
        let source = ManagedModelArchiveSource(
            key: "models/canonical.tar.gz",
            size: 123,
            packagedKey: "models/packaged.tar.gz",
            sha256: String(repeating: "a", count: 64),
            packagedSHA256: String(repeating: "b", count: 64)
        )

        XCTAssertEqual(source.expectedSHA256(for: "/models/canonical.tar.gz"), String(repeating: "a", count: 64))
        XCTAssertEqual(source.expectedSHA256(for: "models/packaged.tar.gz"), String(repeating: "b", count: 64))
        XCTAssertNil(source.expectedSHA256(for: "models/other.tar.gz"))
    }

    private func makeTempFile(contents: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
        try contents.write(to: url, atomically: true, encoding: .utf8)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }
}
