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

    func testValidatesConvertedPackageProvenanceAndOutputChecksum() throws {
        let root = try convertedFixture(variant: .relative)
        defer { try? FileManager.default.removeItem(at: root) }

        let checkpoint = try VideoDepthAnythingResources.inspectExplicit(root)
        XCTAssertEqual(checkpoint.variant, .relative)
        XCTAssertEqual(checkpoint.format, .convertedSafetensors)
        XCTAssertEqual(checkpoint.weightsURL.lastPathComponent, "model.safetensors")
        XCTAssertEqual(checkpoint.sourceSHA256, VideoDepthAnythingVariant.relative.pin.artifacts[0].sha256)

        try Data("tampered".utf8).write(to: checkpoint.weightsURL)
        XCTAssertThrowsError(try VideoDepthAnythingResources.inspectExplicit(root))
    }

    func testRejectsConvertedPackageWithMismatchedSemantics() throws {
        let root = try convertedFixture(variant: .metric, configSemantics: DepthSemantics.affineRelative.rawValue)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try VideoDepthAnythingResources.inspectExplicit(root)) { error in
            XCTAssertEqual(
                error as? VideoDepthAnythingResourceError,
                .invalidConvertedPackage("config.json does not describe the production VDA-S graph")
            )
        }
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

    private func convertedFixture(
        variant: VideoDepthAnythingVariant,
        configSemantics: String? = nil
    ) throws -> URL {
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
            "converterSHA256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
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
            "sha256": "\(artifact.sha256)"
          }
        }
        """
        try Data(source.utf8).write(to: root.appendingPathComponent("SOURCE.json"))
        let config = """
        {
          "architecture": "video-depth-anything-small",
          "backbone": "dinov2-vits14",
          "depthSemantics": "\(configSemantics ?? variant.semantics.rawValue)",
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
        return root
    }
}
