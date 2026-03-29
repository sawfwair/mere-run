import Foundation
import XCTest
@testable import MereRunCore

final class MereRunModelManifestTests: MereRunCoreTestCase {

    func testTemplateRoundTrip() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let manifest = MereRunModelManifest.template(for: .kleinNano, createdAt: Date(timeIntervalSince1970: 0))
        try manifest.write(to: temp)

        let loaded = try MereRunModelManifest.loadIfPresent(from: temp)
        XCTAssertNotNil(loaded)
        guard let loaded else { return }
        XCTAssertEqual(loaded.id, "image-klein-nano")
        XCTAssertEqual(loaded.engine, .flux2Klein)
        XCTAssertEqual(loaded.family, .klein)
        XCTAssertEqual(loaded.tier, .nano)
        XCTAssertEqual(loaded.variant, .distilled)
        XCTAssertEqual(loaded.precision, .int4)
        XCTAssertEqual(loaded.quantization?.bits, 4)
        XCTAssertEqual(loaded.quantization?.groupSize, 64)
        XCTAssertEqual(loaded.defaults?.steps, 4)
        XCTAssertEqual(loaded.supports?.contains(.referenceEdit), true)
    }

    func testWriteTemplateIfKnownOverwritesInvalidJSON() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        // Write a broken manifest file.
        let url = MereRunModelManifest.url(in: temp)
        try TestFileSystem.writeFile(url, contents: Data("{not-json".utf8))

        let written = try MereRunModelManifest.writeTemplateIfKnown(modelId: "image-zimage-max", to: temp, createdAt: Date(timeIntervalSince1970: 0))
        XCTAssertNotNil(written)
        guard let written else { return }
        XCTAssertEqual(written.id, "image-zimage-max")

        let loaded = try MereRunModelManifest.loadIfPresent(from: temp)
        XCTAssertNotNil(loaded)
        guard let loaded else { return }
        XCTAssertEqual(loaded.id, "image-zimage-max")
        XCTAssertEqual(loaded.engine, .zimageTurbo)
    }
}
