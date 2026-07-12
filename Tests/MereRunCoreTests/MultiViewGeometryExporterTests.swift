import Foundation
import MediaIO
@testable import MereRunCore
import XCTest

final class MultiViewGeometryExporterTests: XCTestCase {
    func testManifestPersistsCheckpointInputsRunControlsAndPreprocessingAfterInputsAreDeleted() throws {
        let fixture = try fixtureDirectories("multi-view-manifest")
        defer { fixture.cleanup() }
        let run = try fixtureRun(viewCount: 2, sourceDirectory: fixture.sources)
        let result = try MultiViewGeometryExporter.export(
            run: run,
            outputDirectory: fixture.output,
            configuration: try MultiViewGeometryExportConfiguration(
                confidencePercentile: 0,
                maximumPointCount: 100
            ),
            createdAt: Date(timeIntervalSince1970: 0)
        )

        try FileManager.default.removeItem(at: fixture.sources)
        let decoded = try JSONDecoder.iso8601.decode(
            MultiViewGeometryManifest.self,
            from: Data(contentsOf: result.manifestURL)
        )
        XCTAssertEqual(decoded.schemaVersion, 2)
        XCTAssertEqual(decoded.outputDirectory, fixture.output.path)
        XCTAssertEqual(decoded.checkpoint.repository, "depth-anything/DA3-SMALL")
        XCTAssertEqual(decoded.checkpoint.revision, String(repeating: "a", count: 40))
        XCTAssertEqual(decoded.checkpoint.sourceRepository, "ByteDance-Seed/Depth-Anything-3")
        XCTAssertEqual(decoded.checkpoint.sourceRevision, String(repeating: "b", count: 40))
        XCTAssertEqual(decoded.checkpoint.configurationSHA256, String(repeating: "d", count: 64))
        XCTAssertEqual(decoded.processResolution, 504)
        XCTAssertEqual(decoded.referenceViewStrategy, .middle)
        XCTAssertEqual(decoded.confidencePercentile, 0)
        XCTAssertEqual(decoded.maximumPointCount, 100)
        XCTAssertEqual(decoded.pointSamplingPolicy, "global-valid-row-major-stride-capped")
        XCTAssertEqual(decoded.views.count, 2)
        for (index, view) in decoded.views.enumerated() {
            XCTAssertEqual(view.sourcePath, fixture.sources.appendingPathComponent("view-\(index).bin").path)
            XCTAssertEqual(view.sourceByteCount, Int64(Data("source-\(index)".utf8).count))
            XCTAssertEqual(view.sourceSHA256.count, 64)
            XCTAssertEqual(view.preprocessing.processResolution, 504)
            XCTAssertEqual(view.preprocessing.batchCropLeft, index)
            XCTAssertEqual(view.preprocessing.processedWidth, 2)
            XCTAssertEqual(view.preprocessing.processedHeight, 2)
        }
        XCTAssertEqual(decoded.pointCount, 8)
        XCTAssertEqual(decoded.pointCloudRepresentation, "colored-points-not-mesh")
        XCTAssertFalse(decoded.threeDGaussianHandoff.containsGaussianParameters)
        XCTAssertTrue(decoded.artifacts.allSatisfy { $0.byteCount > 0 && $0.sha256.count == 64 })
        XCTAssertNotNil(decoded.artifacts.first { $0.kind == .pointCloudPLY })
        XCTAssertNotNil(decoded.artifacts.first { $0.kind == .pointCloudGLB })
        XCTAssertNotNil(decoded.artifacts.first { $0.kind == .transformsJSON })

        let transforms = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: fixture.output.appendingPathComponent("transforms.json"))
            ) as? [String: Any]
        )
        XCTAssertEqual(transforms["camera_model"] as? String, "OPENCV")
        XCTAssertEqual(transforms["ply_file_path"] as? String, "scene.ply")
        XCTAssertEqual((transforms["frames"] as? [[String: Any]])?.count, 2)
    }

    func testDeterministicPointLimitAndConfidenceThreshold() throws {
        let fixture = try fixtureDirectories("multi-view-limit")
        defer { fixture.cleanup() }
        let run = try fixtureRun(
            viewCount: 1,
            sourceDirectory: fixture.sources,
            depths: [[1, 2, 3, 4]],
            confidences: [[1, 2, 3, 4]]
        )
        let result = try MultiViewGeometryExporter.export(
            run: run,
            outputDirectory: fixture.output,
            configuration: try MultiViewGeometryExportConfiguration(
                confidencePercentile: 50,
                maximumPointCount: 1
            )
        )

        XCTAssertEqual(result.manifest.confidenceThreshold, 3)
        XCTAssertEqual(result.manifest.pointCount, 1)
        XCTAssertEqual(result.manifest.maximumPointCount, 1)
    }

    func testConfigurationRejectsInvalidPublicValuesWithoutPreconditionTrap() {
        XCTAssertThrowsError(try MultiViewGeometryExportConfiguration(confidencePercentile: .nan)) {
            guard case .invalidConfidencePercentile(let value) =
                $0 as? MultiViewGeometryExportConfigurationError else {
                return XCTFail("Unexpected error: \($0)")
            }
            XCTAssertTrue(value.isNaN)
        }
        XCTAssertThrowsError(try MultiViewGeometryExportConfiguration(confidencePercentile: 101))
        XCTAssertThrowsError(try MultiViewGeometryExportConfiguration(maximumPointCount: 0)) {
            XCTAssertEqual(
                $0 as? MultiViewGeometryExportConfigurationError,
                .invalidMaximumPointCount(0)
            )
        }
    }

    func testSixToTwoRerunAtomicallyRemovesStaleOwnedViewArtifacts() throws {
        let fixture = try fixtureDirectories("multi-view-stale")
        defer { fixture.cleanup() }
        let configuration = try MultiViewGeometryExportConfiguration(
            confidencePercentile: 0,
            maximumPointCount: 100
        )
        _ = try MultiViewGeometryExporter.export(
            run: fixtureRun(viewCount: 6, sourceDirectory: fixture.sources),
            outputDirectory: fixture.output,
            configuration: configuration
        )
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: fixture.output.appendingPathComponent("views/000005-depth.exr").path
        ))

        let second = try MultiViewGeometryExporter.export(
            run: fixtureRun(viewCount: 2, sourceDirectory: fixture.sources),
            outputDirectory: fixture.output,
            configuration: configuration
        )
        XCTAssertEqual(second.manifest.views.count, 2)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: fixture.output.appendingPathComponent("views/000002-depth.exr").path
        ))
        let viewFiles = try FileManager.default.contentsOfDirectory(
            at: fixture.output.appendingPathComponent("views"),
            includingPropertiesForKeys: nil
        )
        XCTAssertEqual(viewFiles.count, 8)
    }

    func testLateExportFailurePreservesPreviousSceneAndCleansStaging() throws {
        let fixture = try fixtureDirectories("multi-view-failure")
        defer { fixture.cleanup() }
        let configuration = try MultiViewGeometryExportConfiguration(
            confidencePercentile: 0,
            maximumPointCount: 100
        )
        let goodRun = try fixtureRun(viewCount: 2, sourceDirectory: fixture.sources)
        let first = try MultiViewGeometryExporter.export(
            run: goodRun,
            outputDirectory: fixture.output,
            configuration: configuration
        )
        let originalManifest = try Data(contentsOf: first.manifestURL)
        let badRun = try fixtureRun(
            viewCount: 2,
            sourceDirectory: fixture.sources,
            normalizedFocal: 0
        )

        XCTAssertThrowsError(try MultiViewGeometryExporter.export(
            run: badRun,
            outputDirectory: fixture.output,
            configuration: configuration
        )) {
            XCTAssertEqual(
                $0 as? GeometryError,
                .invalidIntrinsics("multi-view focal lengths must be positive and finite")
            )
        }
        XCTAssertEqual(try Data(contentsOf: first.manifestURL), originalManifest)
        let stagingPrefix = ".\(fixture.output.lastPathComponent).mere-run-staging-"
        let siblings = try FileManager.default.contentsOfDirectory(
            at: fixture.output.deletingLastPathComponent(),
            includingPropertiesForKeys: nil
        )
        XCTAssertFalse(siblings.contains { $0.lastPathComponent.hasPrefix(stagingPrefix) })
    }

    func testWorldProjectionSamplesDeclaredPixelCenters() throws {
        let fixture = try fixtureDirectories("multi-view-pixel-centers")
        defer { fixture.cleanup() }
        let run = try fixtureRun(viewCount: 1, sourceDirectory: fixture.sources)
        let view = try MultiViewGeometryView(result: try XCTUnwrap(run.views.first))

        XCTAssertEqual(MultiViewGeometryExporter.worldPoint(view: view, pixel: 0), [-0.25, -0.25, 1])
        XCTAssertEqual(MultiViewGeometryExporter.worldPoint(view: view, pixel: 3), [0.25, 0.25, 1])
    }

    private func fixtureRun(
        viewCount: Int,
        sourceDirectory: URL,
        depths: [[Float]]? = nil,
        confidences: [[Float]]? = nil,
        normalizedFocal: Double = 1
    ) throws -> DepthAnything3RunResult {
        try FileManager.default.createDirectory(at: sourceDirectory, withIntermediateDirectories: true)
        let image = try MediaImage(
            width: 2,
            height: 2,
            rgba8: [
                255, 0, 0, 255, 0, 255, 0, 255,
                0, 0, 255, 255, 255, 255, 255, 255,
            ]
        )
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: normalizedFocal,
            normalizedFY: normalizedFocal
        )
        let views = try (0..<viewCount).map { index in
            let source = sourceDirectory.appendingPathComponent("view-\(index).bin")
            if !FileManager.default.fileExists(atPath: source.path) {
                try Data("source-\(index)".utf8).write(to: source, options: .atomic)
            }
            let extrinsics = try GeometryCameraExtrinsics(
                rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                translation: [-Double(index), 0, 0]
            )
            return DepthAnything3ViewResult(
                index: index,
                sourceURL: source,
                inputIdentity: try DepthAnything3InputIdentity.capture(source),
                sourceImage: image,
                processedImage: image,
                preprocessingPlan: DepthAnything3PreprocessingPlan(
                    sourceWidth: 2,
                    sourceHeight: 2,
                    processResolution: 504,
                    boundaryWidth: 504,
                    boundaryHeight: 504,
                    divisibleWidth: 504,
                    divisibleHeight: 504,
                    batchCropLeft: index,
                    batchCropTop: 0,
                    processedWidth: 2,
                    processedHeight: 2
                ),
                depth: depths?[index] ?? [1, 1, 1, 1],
                confidence: confidences?[index] ?? [1, 1, 1, 1],
                intrinsics: intrinsics,
                extrinsics: extrinsics,
                predictedIntrinsics: intrinsics,
                predictedExtrinsics: extrinsics,
                suppliedCamera: nil
            )
        }
        return DepthAnything3RunResult(
            views: views,
            checkpoint: checkpoint(root: sourceDirectory),
            referenceViewStrategy: .middle,
            cameraSemantics: .predictedRelative,
            cameraScaleAlignment: "predicted-relative",
            depthScaleDivisor: 1,
            processResolution: 504,
            checkpointVerificationSeconds: 0.1,
            decodingSeconds: 0.2,
            preprocessingSeconds: 0.3,
            modelLoadSeconds: 0.4,
            inferenceSeconds: 0.5,
            postprocessingSeconds: 0.6
        )
    }

    private func checkpoint(root: URL) -> DepthAnything3Checkpoint {
        DepthAnything3Checkpoint(
            modelID: "vision-geometry-da3-small",
            repository: "depth-anything/DA3-SMALL",
            revision: String(repeating: "a", count: 40),
            sourceRepository: "ByteDance-Seed/Depth-Anything-3",
            sourceRevision: String(repeating: "b", count: 40),
            license: "Apache-2.0",
            rootURL: root,
            weightsURL: root.appendingPathComponent("model.safetensors"),
            configurationURL: root.appendingPathComponent("config.json"),
            weightsByteCount: 137_248_940,
            weightsSHA256: String(repeating: "c", count: 64),
            configurationByteCount: 1_202,
            configurationSHA256: String(repeating: "d", count: 64)
        )
    }

    private func fixtureDirectories(_ stem: String) throws -> FixtureDirectories {
        let parent = FileManager.default.temporaryDirectory.appendingPathComponent(
            "\(stem)-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        return FixtureDirectories(
            parent: parent,
            sources: parent.appendingPathComponent("sources", isDirectory: true),
            output: parent.appendingPathComponent("scene", isDirectory: true)
        )
    }
}

private struct FixtureDirectories {
    let parent: URL
    let sources: URL
    let output: URL

    func cleanup() {
        try? FileManager.default.removeItem(at: parent)
    }
}

private extension JSONDecoder {
    static var iso8601: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
