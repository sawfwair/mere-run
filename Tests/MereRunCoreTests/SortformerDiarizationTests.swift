import Foundation
import MLX
import XCTest

@testable import AudioSTT
@testable import MereRunCore

final class SortformerDiarizationTests: XCTestCase {
    func testX86LinuxRuntimePromotionOnlyConvertsFloat16Weights() {
        let weights = [
            "half": MLX.ones([2], dtype: .float16),
            "float": MLX.ones([2], dtype: .float32),
            "integer": MLX.ones([2], dtype: .int32),
        ]

        let promoted = SortformerModel.runtimeCompatibleWeights(weights, promoteFloat16: true)

        XCTAssertEqual(promoted["half"]?.dtype, .float32)
        XCTAssertEqual(promoted["float"]?.dtype, .float32)
        XCTAssertEqual(promoted["integer"]?.dtype, .int32)
    }

    func testRTTMRendersStableAnonymousSpeakerLabels() {
        let output = DiarizationOutput(
            segments: [
                DiarizationSegment(start: 0, end: 1.25, speaker: 0),
                DiarizationSegment(start: 2.5, end: 4, speaker: 2),
            ],
            numSpeakers: 2,
            totalTime: 0.1
        )

        XCTAssertEqual(
            output.rttm(fileID: "meeting"),
            """
            SPEAKER meeting 1 0.000 1.250 <NA> <NA> speaker_0 <NA> <NA>
            SPEAKER meeting 1 2.500 1.500 <NA> <NA> speaker_2 <NA> <NA>
            """
        )
    }

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

    func testManagedSortformerSpecIsPinnedAndLicenseGated() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.sortformerDiarization.rawValue)
        )

        XCTAssertEqual(spec.category, .speechDiarization)
        XCTAssertEqual(spec.validationKind, .sortformer)
        XCTAssertEqual(spec.upstreamRevision, "e23e6404bd9859e93edbf94a740eb1c7fc58f12e")
        XCTAssertEqual(spec.hubFallback?.patterns, ["README.md", "config.json", "model.safetensors"])
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
    }
}
