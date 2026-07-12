import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionDepthVideoCommandTests: XCTestCase {
    func testParsesProductionOptions() throws {
        let command = try VisionDepthVideo.parse([
            "/tmp/shot.mp4",
            "--output", "/tmp/shot-depth",
            "--model", "vision-depth-vda-small-metric",
            "--input-size", "392",
            "--max-frames", "88",
            "--json",
        ])
        XCTAssertEqual(command.input, "/tmp/shot.mp4")
        XCTAssertEqual(command.output, "/tmp/shot-depth")
        XCTAssertEqual(command.model, "vision-depth-vda-small-metric")
        XCTAssertEqual(command.inputSize, 392)
        XCTAssertEqual(command.maxFrames, 88)
        XCTAssertTrue(command.json)
    }

    func testDefaultOutputAndPreflightPlanAreTruthful() {
        let input = URL(fileURLWithPath: "/tmp/plate.v001.mp4")
        let output = VisionDepthVideo.resolveOutputURL(nil, inputURL: input)
        XCTAssertEqual(output.path, "/tmp/plate.v001-depth")
        let checkpoint = VideoDepthAnythingCheckpoint(
            variant: .metric,
            format: .pinnedPyTorch,
            weightsURL: URL(fileURLWithPath: "/tmp/metric.pth"),
            weightsByteCount: 116_444_063,
            weightsSHA256: VideoDepthAnythingVariant.metric.pin.artifacts[0].sha256,
            sourceSHA256: VideoDepthAnythingVariant.metric.pin.artifacts[0].sha256
        )
        let plan = VisionDepthVideo.makePlan(
            inputURL: input,
            outputURL: output,
            inputSize: 518,
            maximumFrameCount: nil,
            checkpoint: checkpoint
        )
        XCTAssertEqual(plan.modelID, "vision-depth-vda-small-metric")
        XCTAssertEqual(plan.semantics, .metricMeters)
        XCTAssertEqual(plan.temporalWindowLength, 32)
        XCTAssertEqual(plan.temporalOverlap, 10)
        XCTAssertEqual(plan.frameStep, 22)
        XCTAssertTrue(plan.streamsFinalizedFrames)
        XCTAssertEqual(plan.retainedAlignmentFrameLimit, 8)
        XCTAssertEqual(plan.encoderMicroBatchSize, 4)
        XCTAssertEqual(plan.dptTailMicroBatchSize, 4)
        XCTAssertTrue(plan.checkpointVerified)
        XCTAssertFalse(plan.hasConfidence)
        XCTAssertFalse(plan.hasCameraIntrinsics)
        XCTAssertFalse(plan.hasPointCloud)
        XCTAssertTrue(plan.outputKinds.contains("depth-review-mp4"))
    }

    func testVisionRegistersDepthVideo() {
        let commandNames = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("depth-video"))
    }

    func testRunPayloadKeepsUnsupportedGeometryExplicitlyAbsent() {
        let checkpoint = VideoDepthAnythingCheckpoint(
            variant: .relative,
            format: .pinnedPyTorch,
            weightsURL: URL(fileURLWithPath: "/tmp/relative.pth"),
            weightsByteCount: 1,
            weightsSHA256: String(repeating: "a", count: 64),
            sourceSHA256: String(repeating: "a", count: 64)
        )
        let provenance = GeometryModelProvenance(
            modelID: checkpoint.variant.modelID,
            upstreamRepository: checkpoint.variant.pin.repository,
            upstreamRevision: checkpoint.variant.pin.revision,
            license: checkpoint.variant.pin.license
        )
        let manifest = DepthSequenceManifest(
            inputPath: "/tmp/shot.mp4",
            outputDirectory: "/tmp/shot-depth",
            width: 2,
            height: 2,
            fps: 24,
            semantics: .affineRelative,
            model: provenance,
            temporalWindowLength: 32,
            temporalOverlap: 10,
            frames: [DepthSequenceFrameManifest(
                index: 0,
                timeSeconds: 0,
                depthPath: "frames/000000-depth.exr",
                previewPath: "frames/000000-depth.png"
            )]
        )
        let review = VideoDepthReviewArtifact(
            relativePath: "depth-review.mp4",
            byteCount: 42,
            sha256: String(repeating: "b", count: 64)
        )
        let result = VideoDepthAnythingRunResult(
            export: DepthSequenceExportResult(
                manifest: manifest,
                manifestURL: URL(fileURLWithPath: "/tmp/shot-depth/depth-sequence-manifest.json")
            ),
            reviewVideo: review,
            checkpoint: checkpoint,
            sourceFPS: 24,
            windowCount: 1,
            checkpointVerificationSeconds: 0,
            frameExtractionSeconds: 0,
            modelLoadSeconds: 0,
            inferenceSeconds: 0,
            exportSeconds: 0
        )
        let payload = VisionDepthVideoRunPayload(result: result)
        XCTAssertEqual(payload.reviewVideo.sha256, String(repeating: "b", count: 64))
        XCTAssertFalse(payload.hasConfidence)
        XCTAssertFalse(payload.hasCameraIntrinsics)
        XCTAssertFalse(payload.hasPointCloud)
        XCTAssertTrue(payload.streamsFinalizedFrames)
        XCTAssertEqual(payload.retainedAlignmentFrameLimit, 8)
        XCTAssertEqual(payload.encoderMicroBatchSize, 4)
        XCTAssertEqual(payload.dptTailMicroBatchSize, 4)
    }
}
