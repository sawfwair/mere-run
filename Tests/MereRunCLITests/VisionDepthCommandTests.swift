import ArgumentParser
import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionDepthCommandTests: XCTestCase {
    func testParsesProductionOptions() throws {
        let command = try VisionDepth.parse([
            "/tmp/frame.png",
            "--output", "/tmp/depth",
            "--model", "/tmp/marigold",
            "--max-edge", "2048",
            "--checkpoint", "log-layered",
            "--json",
        ])
        XCTAssertEqual(command.input, "/tmp/frame.png")
        XCTAssertEqual(command.output, "/tmp/depth")
        XCTAssertEqual(command.model, "/tmp/marigold")
        XCTAssertEqual(command.maxEdge, 2_048)
        XCTAssertEqual(command.checkpoint, "log-layered")
        XCTAssertTrue(command.json)
        XCTAssertFalse(command.native)
    }

    func testDefaultsToThePaperCheckpoint() throws {
        XCTAssertEqual(try VisionDepth.resolveCheckpoint(nil), .logStage2)
        XCTAssertEqual(try VisionDepth.resolveCheckpoint(""), .logStage2)
    }

    func testCheckpointSelectionIsCaseInsensitiveAndRejectsUnknownNames() throws {
        XCTAssertEqual(try VisionDepth.resolveCheckpoint("Disparity-Base"), .disparityBase)
        XCTAssertThrowsError(try VisionDepth.resolveCheckpoint("log-stage3"))
    }

    func testDefaultOutputDirectoryIsDerivedFromTheInput() {
        let input = URL(fileURLWithPath: "/tmp/shot.001.png")
        XCTAssertEqual(
            VisionDepth.resolveOutputURL(nil, inputURL: input).path,
            "/tmp/shot.001-depth"
        )
        XCTAssertEqual(
            VisionDepth.resolveOutputURL("/tmp/elsewhere", inputURL: input).path,
            "/tmp/elsewhere"
        )
    }

    func testPlanReportsTheAlignedInferenceSize() {
        let input = URL(fileURLWithPath: "/tmp/shot.png")
        let plan = VisionDepth.makePlan(
            inputURL: input,
            outputURL: VisionDepth.resolveOutputURL(nil, inputURL: input),
            imageWidth: 1_920,
            imageHeight: 1_080,
            model: nil,
            configuration: MarigoldV2InferenceConfiguration(maximumEdge: 1_024)
        )
        XCTAssertEqual(plan.status, "planned")
        XCTAssertEqual(plan.imageWidth, 1_920)
        XCTAssertEqual(plan.inferenceWidth, 1_024)
        XCTAssertEqual(plan.inferenceHeight, 576)
        XCTAssertEqual(plan.checkpoint, "log-stage2")
        XCTAssertEqual(plan.parameterization, "log")
        XCTAssertFalse(plan.seeThrough)
        XCTAssertEqual(plan.semantics, "affine-relative")
        XCTAssertEqual(plan.model, "vision-depth-marigold-v2")
    }

    func testPlanKeepsTheNativeResolutionWhenRequested() {
        let input = URL(fileURLWithPath: "/tmp/shot.png")
        let plan = VisionDepth.makePlan(
            inputURL: input,
            outputURL: VisionDepth.resolveOutputURL(nil, inputURL: input),
            imageWidth: 1_920,
            imageHeight: 1_080,
            model: nil,
            configuration: MarigoldV2InferenceConfiguration(maximumEdge: nil)
        )
        XCTAssertEqual(plan.inferenceWidth, 1_920)
        XCTAssertEqual(plan.inferenceHeight, 1_088)
    }

    func testPlanDoesNotAdvertiseCameraOrPointCloudArtifacts() {
        let input = URL(fileURLWithPath: "/tmp/shot.png")
        let plan = VisionDepth.makePlan(
            inputURL: input,
            outputURL: VisionDepth.resolveOutputURL(nil, inputURL: input),
            imageWidth: 512,
            imageHeight: 512,
            model: nil,
            configuration: MarigoldV2InferenceConfiguration()
        )
        // Marigold recovers depth up to an unknown scale and shift and never
        // estimates intrinsics, so projecting its output would imply a camera.
        XCTAssertFalse(plan.outputKinds.contains("camera-json"))
        XCTAssertFalse(plan.outputKinds.contains("point-cloud-ply"))
        XCTAssertTrue(plan.outputKinds.contains("depth-exr"))
    }

    func testNativeAndMaxEdgeCannotBeCombined() async throws {
        var command = try VisionDepth.parse(["/tmp/frame.png", "--native", "--max-edge", "512"])
        do {
            try await command.run()
            XCTFail("Expected --native and --max-edge to be rejected together")
        } catch {
            XCTAssertTrue(error is ValidationError, "\(error)")
        }
    }

    func testMaxEdgeBelowThePatchGridIsRejected() async throws {
        var command = try VisionDepth.parse(["/tmp/frame.png", "--max-edge", "8"])
        do {
            try await command.run()
            XCTFail("Expected a max edge below the patch grid to be rejected")
        } catch {
            XCTAssertTrue(error is ValidationError, "\(error)")
        }
    }
}
