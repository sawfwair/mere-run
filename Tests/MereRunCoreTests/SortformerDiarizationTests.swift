import Foundation
import MLX
import XCTest

@testable import AudioSTT
@testable import AudioSortformer
@testable import MereRunCore

final class SortformerDiarizationTests: XCTestCase {
    func testManagedSortformerRootDoesNotRequireASRTextComponents() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        try MereRunModelManifest.template(
            for: .sortformerDiarization,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"))

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.sortformerDiarization.rawValue
        )

        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertEqual(report.manifest?.engine, .sortformer)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), [.speakerDiarization])
    }

    func testManagedSortformerSpecIsPinnedAndPubliclyPullable() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.sortformerDiarization.rawValue)
        )

        XCTAssertEqual(spec.category, .speechDiarization)
        XCTAssertEqual(spec.validationKind, .sortformer)
        XCTAssertEqual(spec.upstreamRevision, "e23e6404bd9859e93edbf94a740eb1c7fc58f12e")
        XCTAssertEqual(spec.hubFallback?.patterns, ["README.md", "config.json", "model.safetensors"])
        XCTAssertNil(spec.usageRestriction)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
    }
}
