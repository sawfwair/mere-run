import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class Wan2WorldSessionTests: MereRunCoreTestCase {
    func testDreamXWorldTrajectoryReturnsToOriginAcrossInverseRequests() {
        let forward = Wan2DreamXWorldTrajectory.compile(
            segments: [.init(action: "w")],
            pixelFrameCount: 9
        )
        let backward = Wan2DreamXWorldTrajectory.compile(
            segments: [.init(action: "s")],
            pixelFrameCount: 9,
            startingAt: forward.last!
        )
        XCTAssertGreaterThan(forward.last!.translationDistance(to: .identity), 0.3)
        XCTAssertEqual(backward.last!.translationDistance(to: .identity), 0, accuracy: 1e-5)

        let left = Wan2DreamXWorldTrajectory.compile(
            segments: [.init(action: "j")],
            pixelFrameCount: 9
        )
        let right = Wan2DreamXWorldTrajectory.compile(
            segments: [.init(action: "l")],
            pixelFrameCount: 9,
            startingAt: left.last!
        )
        XCTAssertGreaterThan(left.last!.yawDistanceDegrees(to: .identity), 7)
        XCTAssertEqual(right.last!.yawDistanceDegrees(to: .identity), 0, accuracy: 1e-4)
    }

    func testDreamXSceneMemoryRetrievesOnlyPaperThresholdRevisits() {
        var memory = Wan2DreamXSceneMemory(policy: .init(
            maximumFrameCount: 8,
            minimumFrameGap: 3,
            recyclingStrength: 0.08
        ))
        memory.record(
            cleanLatents: MLXArray([Float(1), 2, 3]).reshaped(1, 3, 1, 1),
            poses: [.identity, .identity, .identity],
            startingAt: 0
        )

        XCTAssertNil(memory.retrieve(for: .identity, targetFrameIndex: 2))
        let revisit = memory.retrieve(for: .identity, targetFrameIndex: 10)
        XCTAssertEqual(revisit?.metadata.memoryFrameIndex, 0)
        XCTAssertEqual(revisit?.metadata.temporalGap, 10)
        XCTAssertEqual(revisit?.metadata.viewOverlapScore ?? 0, 1, accuracy: 1e-6)

        var translated = Wan2DreamXWorldPose.identity.worldToCamera
        translated[3] = -0.2
        XCTAssertNil(memory.retrieve(
            for: Wan2DreamXWorldPose(worldToCamera: translated),
            targetFrameIndex: 10
        ))

        let yawed = Wan2DreamXWorldTrajectory.compile(
            segments: [.init(action: "j")],
            pixelFrameCount: 9
        ).last!
        XCTAssertNil(memory.retrieve(for: yawed, targetFrameIndex: 10))
    }

    func testDreamXSceneMemoryIsBoundedAndCheckpointRestorable() {
        var memory = Wan2DreamXSceneMemory(policy: .init(
            maximumFrameCount: 3,
            minimumFrameGap: 1
        ))
        memory.record(
            cleanLatents: MLXArray([Float(1), 2, 3]).reshaped(1, 3, 1, 1),
            poses: [.identity, .identity, .identity],
            startingAt: 0
        )
        let checkpoint = memory.checkpoint()
        memory.record(
            cleanLatents: MLXArray([Float(4), 5, 6]).reshaped(1, 3, 1, 1),
            poses: [.identity, .identity, .identity],
            startingAt: 3
        )
        XCTAssertEqual(memory.frameCount, 3)
        XCTAssertEqual(memory.retrieve(for: .identity, targetFrameIndex: 10)?.metadata.memoryFrameIndex, 3)

        memory.restore(checkpoint)
        XCTAssertEqual(memory.frameCount, 3)
        XCTAssertEqual(memory.retrieve(for: .identity, targetFrameIndex: 10)?.metadata.memoryFrameIndex, 0)
    }

    func testArtifactSequenceNeverReusesFilesAcrossLogicalSessionResets() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-world-artifacts-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data().write(to: root.appendingPathComponent("rollout-0001.mp4"))
        try Data().write(to: root.appendingPathComponent("state-0002.png"))
        try Data().write(to: root.appendingPathComponent("transition-0004.mp4"))
        var sequence = Wan2WorldArtifactSequence()

        XCTAssertEqual(sequence.next(in: root), 3)
        XCTAssertEqual(sequence.next(in: root), 5)
        XCTAssertEqual(sequence.next(in: root), 6)
    }

    func testCausalTransformerCheckpointRestoresCacheWindowAndGlobalPosition() {
        let state = Wan2CausalTransformerState(
            layerCount: 1,
            localAttentionFrames: 4,
            sinkFrames: 1
        )
        let cache = state.blocks[0].selfAttention
        _ = cache.update(
            key: MLXArray([Float(1), 2]).reshaped(1, 2, 1, 1),
            value: MLXArray([Float(3), 4]).reshaped(1, 2, 1, 1),
            currentStartToken: 0,
            spatialTokensPerFrame: 1
        )
        let checkpoint = state.checkpoint()

        _ = cache.update(
            key: MLXArray([Float(5), 6]).reshaped(1, 2, 1, 1),
            value: MLXArray([Float(7), 8]).reshaped(1, 2, 1, 1),
            currentStartToken: 2,
            spatialTokensPerFrame: 1
        )
        XCTAssertEqual(state.snapshot(spatialTokensPerFrame: 1).globalFrames, 4)

        state.restore(checkpoint)
        var restored = state.snapshot(spatialTokensPerFrame: 1)
        XCTAssertEqual(restored.globalFrames, 2)
        XCTAssertEqual(restored.cachedFrames, 2)

        let branch = cache.update(
            key: MLXArray([Float(9)]).reshaped(1, 1, 1, 1),
            value: MLXArray([Float(10)]).reshaped(1, 1, 1, 1),
            currentStartToken: 2,
            spatialTokensPerFrame: 1
        )
        eval(branch.key, branch.value)
        restored = state.snapshot(spatialTokensPerFrame: 1)
        XCTAssertEqual(restored.globalFrames, 3)
        XCTAssertEqual(branch.key.asArray(Float.self), [1, 2, 9])
        XCTAssertEqual(branch.value.asArray(Float.self), [3, 4, 10])
    }

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

    func testCausalCameraControlsUseUpstreamFixedModelSpaceSpeed() {
        let fifteen = Wan2DreamXARTrajectory.compile(
            control: .yawLeft(degrees: 15),
            pixelFrameCount: 9
        )
        let thirty = Wan2DreamXARTrajectory.compile(
            control: .yawLeft(degrees: 30),
            pixelFrameCount: 9
        )
        XCTAssertEqual(fifteen.viewMatrices, thirty.viewMatrices)

        let near = Wan2DreamXARTrajectory.compile(
            control: .forward(meters: 0.25),
            pixelFrameCount: 9
        )
        let far = Wan2DreamXARTrajectory.compile(
            control: .forward(meters: 1),
            pixelFrameCount: 9
        )
        XCTAssertEqual(near.viewMatrices, far.viewMatrices)
    }

    func testCausalCameraControlMatchesUpstreamFixedRateAtLastAlignedView() {
        let yaw = Wan2DreamXARTrajectory.compile(
            control: .yawLeft(degrees: 30),
            pixelFrameCount: 9
        )
        let yawLast = Array(yaw.viewMatrices.suffix(16))
        XCTAssertEqual(yawLast[0], cos(7.5 * .pi / 180), accuracy: 1e-5)
        XCTAssertEqual(abs(yawLast[2]), sin(7.5 * .pi / 180), accuracy: 1e-5)

        let forward = Wan2DreamXARTrajectory.compile(
            control: .forward(meters: 0.25),
            pixelFrameCount: 9
        )
        let forwardLast = Array(forward.viewMatrices.suffix(16))
        XCTAssertEqual(abs(forwardLast[11]), 0.375, accuracy: 1e-5)
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

    func testRolloutRequestUsesOfficialDreamXGeometryAndFrameSemantics() {
        let official = Wan2WorldRolloutRequest(
            prompt: "Keep the corridor coherent.",
            actionSequence: [.init(action: "w")]
        )
        XCTAssertEqual(official.width, 1_280)
        XCTAssertEqual(official.height, 704)
        XCTAssertEqual(official.latentFrameCount, 21)
        XCTAssertEqual(official.expectedPixelFrameCount, 81)
        XCTAssertEqual(official.speed, 1.5)
        XCTAssertEqual(official.fps, 16)

        let long = Wan2WorldRolloutRequest(
            prompt: "Keep the corridor coherent.",
            actionSequence: [.init(action: "w")],
            latentFrameCount: 63
        )
        XCTAssertEqual(long.expectedPixelFrameCount, 249)

        let maximum = Wan2WorldRolloutRequest(
            prompt: "Keep the corridor coherent.",
            actionSequence: [.init(action: "w")],
            latentFrameCount: Wan2CausalWorldGenerator.maximumRolloutLatentFrameCount
        )
        XCTAssertEqual(maximum.expectedPixelFrameCount, 1_005)
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
        XCTAssertEqual(snapshot.causalCheckpointCount, 0)
    }

    func testCausalCheckpointLocksAndRestoresExactSessionStateWithoutLoadingModels() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-causal-checkpoint-model-\(UUID().uuidString)")
        let state = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-causal-checkpoint-state-\(UUID().uuidString)")
        let source = state.appendingPathComponent("seed.png")
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: root),
            stateDirectory: state,
            causalWeightsURL: root.appendingPathComponent("causal.safetensors")
        )
        try await session.reset(sourceImageURL: source)
        let seededSnapshot = await session.snapshot()
        let lockedStateID = try XCTUnwrap(seededSnapshot.currentStateID)
        let checkpointID = UUID()
        let createdAt = Date(timeIntervalSince1970: 1_785_000_000)

        let receipt = try await session.createCausalCheckpoint(
            name: "Vesper gate",
            checkpointID: checkpointID,
            createdAt: createdAt
        )

        XCTAssertEqual(receipt.checkpointID, checkpointID)
        XCTAssertEqual(receipt.name, "Vesper gate")
        XCTAssertEqual(receipt.stateID, lockedStateID)
        XCTAssertEqual(receipt.currentFrameURL, source)
        XCTAssertEqual(receipt.generatedLatentFrameCount, 0)
        XCTAssertEqual(receipt.retainedLatentFrameCount, 0)
        let lockedSnapshot = await session.snapshot()
        let receipts = await session.causalCheckpointReceipts()
        XCTAssertEqual(lockedSnapshot.causalCheckpointCount, 1)
        XCTAssertEqual(receipts, [receipt])

        let restored = try await session.restoreCausalCheckpoint(checkpointID)
        let restoredSnapshot = await session.snapshot()
        XCTAssertEqual(restored, receipt)
        XCTAssertEqual(restoredSnapshot.currentStateID, lockedStateID)
        XCTAssertEqual(restoredSnapshot.currentFrameURL, source)

        let discarded = try await session.discardCausalCheckpoint(checkpointID)
        XCTAssertEqual(discarded, receipt)
        let discardedSnapshot = await session.snapshot()
        XCTAssertEqual(discardedSnapshot.causalCheckpointCount, 0)
    }
}
