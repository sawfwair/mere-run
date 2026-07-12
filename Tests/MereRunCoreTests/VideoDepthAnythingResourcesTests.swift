import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class VideoDepthAnythingResourcesTests: XCTestCase {
    func testVariantsBindCanonicalModelSemanticsAndPins() {
        XCTAssertEqual(VideoDepthAnythingVariant.relative.modelID, "vision-depth-vda-small")
        XCTAssertEqual(VideoDepthAnythingVariant.relative.semantics, .affineRelative)
        XCTAssertEqual(
            VideoDepthAnythingVariant.relative.pin.artifacts[0].sha256,
            "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609"
        )
        XCTAssertEqual(VideoDepthAnythingVariant.metric.modelID, "vision-depth-vda-small-metric")
        XCTAssertEqual(VideoDepthAnythingVariant.metric.semantics, .metricMeters)
        XCTAssertEqual(
            VideoDepthAnythingVariant.metric.pin.artifacts[0].sha256,
            "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"
        )
    }

    func testConvertedPackagePinsAreCanonicalAndVariantSpecific() {
        let relative = VideoDepthAnythingVariant.relative.convertedPackagePins
        let metric = VideoDepthAnythingVariant.metric.convertedPackagePins
        XCTAssertEqual(relative.weights.byteCount, 116_362_340)
        XCTAssertEqual(
            relative.weights.sha256,
            "85c583474dcafda4d417776431343afcdfdfc97952d8ec00029d3452c55a05a2"
        )
        XCTAssertEqual(
            metric.weights.sha256,
            "0acf1e186750abddf5ae867a3a659ed67cd0c041e4e524e698a0dcb40195c779"
        )
        XCTAssertNotEqual(relative.configuration.sha256, metric.configuration.sha256)
        XCTAssertNotEqual(relative.sourceManifest.sha256, metric.sourceManifest.sha256)
        XCTAssertEqual(relative.license, metric.license)
    }

    func testRejectsSelfAttestedConvertedPackageEvenWhenItsOwnHashMatches() throws {
        let root = try selfAttestedConvertedFixture(variant: .relative)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try VideoDepthAnythingResources.inspectExplicit(root)) { error in
            guard case .invalidConvertedPackage(let detail) = error as? VideoDepthAnythingResourceError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("exact pinned conversion manifest"))
            XCTAssertFalse(detail.contains("could not decode"))
        }
    }

    func testRejectsOversizedSourceManifestBeforeJSONDecode() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vda-oversized-metadata-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(repeating: 0x7B, count: 4_096).write(to: root.appendingPathComponent("SOURCE.json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        try Data("license".utf8).write(to: root.appendingPathComponent("LICENSE"))
        try Data("weights".utf8).write(to: root.appendingPathComponent("model.safetensors"))

        XCTAssertThrowsError(try VideoDepthAnythingResources.inspectExplicit(root)) { error in
            guard case .invalidConvertedPackage(let detail) = error as? VideoDepthAnythingResourceError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("exact pinned conversion manifest"))
            XCTAssertFalse(detail.contains("could not decode"))
        }
    }

    func testInspectsExactConvertedPackagesWhenEnabled() throws {
        for (variant, variable) in [
            (VideoDepthAnythingVariant.relative, "MERERUN_TEST_VDA_CONVERTED_RELATIVE"),
            (.metric, "MERERUN_TEST_VDA_CONVERTED_METRIC"),
        ] {
            let path = ProcessInfo.processInfo.environment[variable] ?? ""
            if path.isEmpty || !FileManager.default.fileExists(atPath: path) { continue }
            let checkpoint = try VideoDepthAnythingResources.inspectExplicit(
                URL(fileURLWithPath: path)
            )
            XCTAssertEqual(checkpoint.variant, variant)
            XCTAssertEqual(checkpoint.format, .convertedSafetensors)
            XCTAssertEqual(checkpoint.weightsByteCount, variant.convertedPackagePins.weights.byteCount)
            XCTAssertEqual(checkpoint.weightsSHA256, variant.convertedPackagePins.weights.sha256)
            XCTAssertEqual(checkpoint.sourceSHA256, variant.pin.artifacts[0].sha256)

            let sourceRoot = URL(fileURLWithPath: path)
            let tamperedRoot = FileManager.default.temporaryDirectory
                .appendingPathComponent("vda-config-tamper-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: tamperedRoot, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tamperedRoot) }
            for filename in ["SOURCE.json", "LICENSE", "model.safetensors"] {
                try FileManager.default.createSymbolicLink(
                    at: tamperedRoot.appendingPathComponent(filename),
                    withDestinationURL: sourceRoot.appendingPathComponent(filename)
                )
            }
            var config = try Data(contentsOf: sourceRoot.appendingPathComponent("config.json"))
            config.append(0x7B)
            try config.write(to: tamperedRoot.appendingPathComponent("config.json"))
            XCTAssertThrowsError(try VideoDepthAnythingResources.inspectExplicit(tamperedRoot)) { error in
                guard case .invalidConvertedPackage(let detail) = error as? VideoDepthAnythingResourceError else {
                    return XCTFail("Unexpected error: \(error)")
                }
                XCTAssertTrue(detail.contains("config.json has"))
                XCTAssertFalse(detail.contains("could not decode"))
            }
        }
        let enabled = [
            "MERERUN_TEST_VDA_CONVERTED_RELATIVE",
            "MERERUN_TEST_VDA_CONVERTED_METRIC",
        ].contains { !(ProcessInfo.processInfo.environment[$0] ?? "").isEmpty }
        try XCTSkipIf(!enabled, "Set a MERERUN_TEST_VDA_CONVERTED_* package path")
    }

    func testInspectsAndLoadsPinnedPyTorchCheckpointWhenAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_PTH"] ?? ""
        try XCTSkipIf(path.isEmpty || !FileManager.default.fileExists(atPath: path))
        let checkpoint = try VideoDepthAnythingResources.inspectExplicit(URL(fileURLWithPath: path))
        XCTAssertEqual(checkpoint.variant, .relative)
        XCTAssertEqual(checkpoint.format, .pinnedPyTorch)
        XCTAssertEqual(checkpoint.weightsSHA256, VideoDepthAnythingVariant.relative.pin.artifacts[0].sha256)

        let model = try VideoDepthAnythingResources.loadModel(from: checkpoint)
        XCTAssertEqual(model.parameters().flattened().count, VideoDepthAnythingWeights.sourceTensorCount)
        MLX.Memory.clearCache()
    }

    func testInspectsManagedSymlinkToPinnedPyTorchCheckpointWhenAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_PTH"] ?? ""
        try XCTSkipIf(path.isEmpty || !FileManager.default.fileExists(atPath: path))
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vda-managed-symlink-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let managedURL = root.appendingPathComponent("video_depth_anything_vits.pth")
        try FileManager.default.createSymbolicLink(
            at: managedURL,
            withDestinationURL: URL(fileURLWithPath: path)
        )

        let checkpoint = try VideoDepthAnythingResources.inspectExplicit(managedURL)

        XCTAssertEqual(checkpoint.variant, .relative)
        XCTAssertEqual(
            checkpoint.weightsByteCount,
            VideoDepthAnythingVariant.relative.pin.artifacts[0].byteCount
        )
        XCTAssertEqual(checkpoint.weightsURL, managedURL.standardizedFileURL)
    }

    func testSplitsWindowDepthInFrameOrder() throws {
        let values = (0..<(VideoDepthAnythingWindowing.windowLength * 4)).map(Float.init)
        let frames = try VideoDepthAnythingGenerator.splitDepthWindow(values, width: 2, height: 2)
        XCTAssertEqual(frames.count, VideoDepthAnythingWindowing.windowLength)
        XCTAssertEqual(frames[0], [0, 1, 2, 3])
        XCTAssertEqual(frames[1], [4, 5, 6, 7])
        XCTAssertEqual(frames.last, [124, 125, 126, 127])
    }

    func testReviewArtifactHashesTheAssembledVideo() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vda-review-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let reviewURL = root.appendingPathComponent("depth-review.mp4")
        try Data("review-video".utf8).write(to: reviewURL)

        let artifact = try VideoDepthAnythingGenerator.reviewArtifact(url: reviewURL, root: root)
        XCTAssertEqual(artifact.kind, "depth-review-video")
        XCTAssertEqual(artifact.relativePath, "depth-review.mp4")
        XCTAssertEqual(artifact.mediaType, "video/mp4")
        XCTAssertEqual(artifact.byteCount, 12)
        XCTAssertEqual(artifact.sha256, try ModelArtifactPin.fileSHA256(reviewURL))
    }

    private func selfAttestedConvertedFixture(variant: VideoDepthAnythingVariant) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vda-converted-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let weightsURL = root.appendingPathComponent("model.safetensors")
        try Data("fixture".utf8).write(to: weightsURL)
        let outputSHA = try ModelArtifactPin.fileSHA256(weightsURL)
        let artifact = variant.pin.artifacts[0]
        let source = """
        {
            "conversion": {
              "converter": "convert_vda_small.py",
              "converterVersion": 1,
              "environment": {
                "python": "3.11.15",
                "numpy": "2.4.3",
                "torch": "2.13.0",
                "safetensors": "0.8.0"
              },
            "outputByteCount": 7,
            "outputFile": "model.safetensors",
            "outputSHA256": "\(outputSHA)",
            "tensorCount": 351,
            "scalarCount": 29080193
          },
          "license": "Apache-2.0",
          "modelID": "\(variant.modelID)",
          "source": {
            "byteCount": \(artifact.byteCount),
            "filename": "\(artifact.filename)",
            "repository": "\(variant.pin.repository)",
            "revision": "\(variant.pin.revision)",
            "sha256": "\(artifact.sha256)",
            "sourceCodeRepository": "\(variant.pin.sourceCodeRepository)",
            "sourceCodeRevision": "\(variant.pin.sourceCodeRevision)"
          }
        }
        """
        try Data(source.utf8).write(to: root.appendingPathComponent("SOURCE.json"))
        let config = """
        {
          "architecture": "video-depth-anything-small",
          "backbone": "dinov2-vits14",
          "depthSemantics": "\(variant.semantics.rawValue)",
          "featureChannels": 64,
          "intermediateLayers": [2, 5, 8, 11],
          "projectedChannels": [48, 96, 192, 384],
          "temporalAttentionBlocks": 2,
          "temporalAttentionHeads": 8,
          "temporalFrameCount": 32,
          "temporalOverlap": 10,
          "temporalTransformerBlocks": 1
        }
        """
        try Data(config.utf8).write(to: root.appendingPathComponent("config.json"))
        try Data("self-attested license".utf8).write(to: root.appendingPathComponent("LICENSE"))
        return root
    }
}
