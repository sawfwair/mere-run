import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class FaceAnalysisTests: XCTestCase {
    func testManagedBuffaloLModelUsesPinnedDetectorAndRecognizer() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: FaceAnalysisResources.modelID))
        XCTAssertEqual(spec.category, .visionFace)
        XCTAssertEqual(spec.validationKind, .insightFaceBuffaloL)
        XCTAssertEqual(spec.hubFallback?.repoId, "deepghs/insightface")
        XCTAssertEqual(spec.hubFallback?.revision, "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb")
        XCTAssertEqual(
            Set(spec.hubFallback?.patterns ?? []),
            [
                FaceAnalysisResources.detectorRelativePath,
                FaceAnalysisResources.recognizerRelativePath,
                "LICENSE*",
                "README.md",
            ]
        )
        XCTAssertEqual(
            spec.estimatedDownloadBytes,
            FaceAnalysisResources.detectorByteCount + FaceAnalysisResources.recognizerByteCount
        )
    }

    func testResourcesRequireBothExactCheckpointSizes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-face-resources-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = FaceAnalysisResources(rootURL: root)
        try createSparseFile(resources.detectorURL, byteCount: FaceAnalysisResources.detectorByteCount)
        XCTAssertEqual(resources.validate(), [resources.recognizerURL])
        try createSparseFile(resources.recognizerURL, byteCount: FaceAnalysisResources.recognizerByteCount)
        XCTAssertTrue(resources.validate().isEmpty)

        let handle = try FileHandle(forWritingTo: resources.detectorURL)
        try handle.truncate(atOffset: UInt64(FaceAnalysisResources.detectorByteCount - 1))
        try handle.close()
        XCTAssertEqual(resources.validate(), [resources.detectorURL])
    }

    func testEmbeddingNormalizationAndCosineSimilarity() throws {
        let normalized = FaceAnalysisMath.l2Normalized([3, 4])
        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.000_001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.000_001)
        XCTAssertEqual(
            try XCTUnwrap(FaceAnalysisMath.cosineSimilarity(normalized, normalized)),
            1,
            accuracy: 0.000_001
        )
        XCTAssertEqual(
            try XCTUnwrap(FaceAnalysisMath.cosineSimilarity([1, 0], [0, 1])),
            0,
            accuracy: 0.000_001
        )
        XCTAssertNil(FaceAnalysisMath.cosineSimilarity([1], [1, 2]))
    }

    func testDetectorInputUsesAspectFitAndInsightFaceNormalization() throws {
        let image = try MediaImage(
            width: 2,
            height: 1,
            rgba8: [255, 127, 0, 255, 255, 127, 0, 255]
        )
        let input = try FaceImageProcessing.detectorInput(from: image)
        XCTAssertEqual(input.scale, 320, accuracy: 0.000_001)
        XCTAssertEqual(input.values.count, 1 * 3 * 640 * 640)
        XCTAssertEqual(input.values[0], (255 - 127.5) / 128, accuracy: 0.000_001)
        XCTAssertEqual(input.values[640 * 640], (127 - 127.5) / 128, accuracy: 0.000_001)
        XCTAssertEqual(input.values[2 * 640 * 640], (0 - 127.5) / 128, accuracy: 0.000_001)
    }

    private func createSparseFile(_ url: URL, byteCount: Int64) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data().write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: UInt64(byteCount))
        try handle.close()
    }
}
