import Foundation
import XCTest
@testable import MereRunCore

final class Wan2WorldSessionTests: MereRunCoreTestCase {
    func testCameraControlsUseStableWorldAxesAndPromptSemantics() {
        let forward = Wan2WorldCameraControl.forward(meters: 1.25)
        XCTAssertEqual(forward.motion, .forward)
        XCTAssertEqual(forward.translationMeters, [0, 0, 1.25])
        XCTAssertTrue(forward.promptClause.contains("straight forward"))

        let left = Wan2WorldCameraControl.yawLeft(degrees: 30)
        XCTAssertEqual(left.motion, .yawLeft)
        XCTAssertEqual(left.rotationDegrees, [0, -30, 0])
        XCTAssertTrue(left.promptClause.contains("pivots left in place"))
    }

    func testCausalCameraTrajectoryHonorsRequestedMotionMagnitude() {
        let fifteen = Wan2DreamXARTrajectory.compile(
            control: .yawLeft(degrees: 15),
            pixelFrameCount: 9
        )
        let thirty = Wan2DreamXARTrajectory.compile(
            control: .yawLeft(degrees: 30),
            pixelFrameCount: 9
        )
        XCTAssertNotEqual(fifteen.viewMatrices, thirty.viewMatrices)

        let near = Wan2DreamXARTrajectory.compile(
            control: .forward(meters: 0.25),
            pixelFrameCount: 9
        )
        let far = Wan2DreamXARTrajectory.compile(
            control: .forward(meters: 1),
            pixelFrameCount: 9
        )
        XCTAssertNotEqual(near.viewMatrices, far.viewMatrices)
    }

    func testColdSessionCanResetToAnExplicitWorldFrame() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-session-model-\(UUID().uuidString)")
        let state = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-session-state-\(UUID().uuidString)")
        let source = state.appendingPathComponent("seed.png")
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: root),
            stateDirectory: state
        )

        var snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.phase, .cold)
        XCTAssertEqual(snapshot.transitionCount, 0)
        XCTAssertNil(snapshot.currentStateID)
        XCTAssertFalse(snapshot.keepsModelsWarm)
        XCTAssertFalse(snapshot.keepsTerminalLatent)

        try await session.reset(sourceImageURL: source)
        snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.currentFrameURL, source)
        XCTAssertNotNil(snapshot.currentStateID)
        XCTAssertEqual(snapshot.conditioningMode, .textAndFirstFrame)
    }

    func testTransitionRequestDefaultsToShortWorldClipGeometry() {
        let request = Wan2WorldTransitionRequest(
            prompt: "Keep the corridor coherent.",
            camera: .forward()
        )
        XCTAssertEqual(request.width, 512)
        XCTAssertEqual(request.height, 320)
        XCTAssertEqual(request.numFrames, 17)
        XCTAssertEqual(request.steps, 40)
        XCTAssertEqual(request.guidanceScale, 5)
        XCTAssertEqual(request.shift, 5)
        XCTAssertEqual(request.fps, 24)
    }

    func testCameraWeightsSelectProjectiveConditioningWithoutLoadingModels() async {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-camera-session-model-\(UUID().uuidString)")
        let state = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-camera-session-state-\(UUID().uuidString)")
        let cameraWeights = root.appendingPathComponent("camera_adapter.safetensors")
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: root),
            stateDirectory: state,
            cameraWeightsURL: cameraWeights
        )
        let snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.conditioningMode, .projectiveCameraLatents)
        XCTAssertFalse(snapshot.keepsModelsWarm)
    }

    func testCausalWeightsSelectPersistentConditioningWithoutLoadingModels() async {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-causal-session-model-\(UUID().uuidString)")
        let state = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-causal-session-state-\(UUID().uuidString)")
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: root),
            stateDirectory: state,
            causalWeightsURL: root.appendingPathComponent("causal.safetensors")
        )
        let snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.conditioningMode, .causalCameraLatents)
        XCTAssertFalse(snapshot.keepsModelsWarm)
        XCTAssertFalse(snapshot.keepsTerminalLatent)
    }
}
