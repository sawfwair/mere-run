import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class MoGe2GeneratorTests: XCTestCase {
    func testTokenGridPreservesNormalAspectRatiosWithinWorkloadLimit() throws {
        let landscape = try MoGe2TokenGrid.resolve(
            imageWidth: 1_920,
            imageHeight: 1_080,
            requestedTokenCount: 1_200
        )
        XCTAssertEqual(landscape.rows, 26)
        XCTAssertEqual(landscape.columns, 46)
        XCTAssertEqual(landscape.count, 1_196)

        let portrait = try MoGe2TokenGrid.resolve(
            imageWidth: 1_080,
            imageHeight: 1_920,
            requestedTokenCount: 1_200
        )
        XCTAssertEqual(portrait.rows, 46)
        XCTAssertEqual(portrait.columns, 26)
        XCTAssertEqual(portrait.count, 1_196)

        let maximumSquare = try MoGe2TokenGrid.resolve(
            imageWidth: 1_024,
            imageHeight: 1_024,
            requestedTokenCount: MoGe2TokenGrid.maximumTokenCount
        )
        XCTAssertEqual(maximumSquare.rows, 60)
        XCTAssertEqual(maximumSquare.columns, 60)
        XCTAssertEqual(maximumSquare.count, MoGe2TokenGrid.maximumTokenCount)
    }

    func testRejectsExtremeAspectTokenGridBeforeModelWork() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("moge2-extreme-aspect-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let imageURL = root.appendingPathComponent("extreme.png")
        try MediaImageIO.writePNG(
            try MediaImage(
                width: 16_384,
                height: 1,
                rgba8: [UInt8](repeating: 255, count: 16_384 * 4)
            ),
            to: imageURL
        )

        let generator = MoGe2Generator()
        do {
            _ = try await generator.generate(
                imageURL: imageURL,
                outputDirectory: root.appendingPathComponent("output", isDirectory: true),
                model: root.appendingPathComponent("missing-model.onnx").path,
                configuration: MoGe2InferenceConfiguration(
                    tokenCount: MoGe2InferenceConfiguration.maximumTokenCount
                )
            )
            XCTFail("Expected the derived token grid to exceed the native workload limit")
        } catch {
            XCTAssertEqual(
                error as? MoGe2TokenGridError,
                .tokenGridExceedsLimit(
                    rows: 1,
                    columns: 7_680,
                    actual: 7_680,
                    maximum: MoGe2TokenGrid.maximumTokenCount
                )
            )
        }
    }

    func testRejectsUnsafeTokenCountBeforeInputOrModelWork() async {
        let generator = MoGe2Generator()
        do {
            _ = try await generator.generate(
                imageURL: URL(fileURLWithPath: "/definitely/missing/frame.png"),
                outputDirectory: URL(fileURLWithPath: "/definitely/missing/output"),
                model: "/definitely/missing/model.onnx",
                configuration: MoGe2InferenceConfiguration(
                    tokenCount: MoGe2InferenceConfiguration.maximumTokenCount + 1
                )
            )
            XCTFail("Expected token count validation to fail")
        } catch {
            XCTAssertEqual(
                error as? MoGe2GeneratorError,
                .tokenCountOutOfRange(
                    actual: MoGe2InferenceConfiguration.maximumTokenCount + 1,
                    minimum: MoGe2InferenceConfiguration.minimumTokenCount,
                    maximum: MoGe2InferenceConfiguration.maximumTokenCount
                )
            )
        }
    }

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
