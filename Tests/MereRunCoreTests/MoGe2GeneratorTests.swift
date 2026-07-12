import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class MoGe2GeneratorTests: XCTestCase {
    func testPinnedModelProducesCompleteMetricArtifactBundle() async throws {
        guard let fixture = ProcessInfo.processInfo.environment["MERERUN_TEST_MOGE_ONNX"] else {
            throw XCTSkip("Set MERERUN_TEST_MOGE_ONNX to the pinned model.onnx fixture.")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("moge2-generator-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let imageURL = root.appendingPathComponent("gradient.png")
        let outputURL = root.appendingPathComponent("geometry", isDirectory: true)
        var pixels = [UInt8](repeating: 255, count: 28 * 28 * 4)
        for y in 0..<28 {
            for x in 0..<28 {
                let offset = (y * 28 + x) * 4
                pixels[offset] = UInt8(x * 9)
                pixels[offset + 1] = UInt8(y * 9)
                pixels[offset + 2] = UInt8((x + y) * 4)
            }
        }
        try MediaImageIO.writePNG(
            try MediaImage(width: 28, height: 28, rgba8: pixels),
            to: imageURL
        )

        let generator = MoGe2Generator()
        let result = try await generator.generate(
            imageURL: imageURL,
            outputDirectory: outputURL,
            model: fixture,
            configuration: MoGe2InferenceConfiguration(tokenCount: 1_200, maximumPointCount: 784)
        )
        await generator.unload()

        XCTAssertEqual(result.export.manifest.units, .meters)
        XCTAssertEqual(result.export.manifest.width, 28)
        XCTAssertEqual(result.export.manifest.height, 28)
        XCTAssertEqual(result.export.manifest.model.modelID, "vision-geometry-moge2-small")
        XCTAssertGreaterThan(result.export.manifest.depthStatistics.validPixelCount, 0)
        XCTAssertTrue(result.focalShift.focal.isFinite)
        XCTAssertGreaterThan(result.metricScale, 0)
        XCTAssertEqual(
            Set(result.export.manifest.artifacts.map(\.kind)),
            Set([
                .depthEXR, .depthPreview, .normalEXR, .normalPreview, .validityMask,
                .confidenceEXR, .pointCloud, .camera,
            ])
        )
        for artifact in result.export.manifest.artifacts {
            let url = outputURL.appendingPathComponent(artifact.relativePath)
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
            XCTAssertGreaterThan(artifact.byteCount, 0)
            XCTAssertEqual(artifact.sha256.count, 64)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.export.manifestURL.path))
    }
}
