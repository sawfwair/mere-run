import Foundation
import XCTest
@testable import MereRunCore

final class Wan2RealWorldSessionTests: MereRunCoreTestCase {
    func testCausalDreamXProducesPersistentWorldMoves() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let sourcePath = environment["MERERUN_WAN2_SOURCE_IMAGE"],
              let causalWeightsPath = environment["MERERUN_DREAMX_CAUSAL_WEIGHTS"],
              let outputPath = environment["MERERUN_DREAMX_CAUSAL_OUTPUT"] else {
            throw XCTSkip("Set the Wan2 resources, source image, DreamX causal weights, and output variables.")
        }
        let width = Int(environment["MERERUN_DREAMX_CAUSAL_WIDTH"] ?? "256") ?? 256
        let height = Int(environment["MERERUN_DREAMX_CAUSAL_HEIGHT"] ?? "160") ?? 160
        let output = URL(fileURLWithPath: outputPath)
        try? FileManager.default.removeItem(at: output)
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: URL(fileURLWithPath: root)),
            stateDirectory: output,
            causalWeightsURL: URL(fileURLWithPath: causalWeightsPath)
        )
        let prompt = "A continuous first-person exploration of the same empty yellow backrooms interior, with stained yellow walls, beige carpet, fluorescent ceiling panels, rigid architecture, stable lighting, and no people. Preserve the exact environment and spatial identity across every camera move."
        let started = Date()
        let progress: @Sendable (GenerationProgress) -> Void = { update in
            let elapsed = String(format: "%.1f", Date().timeIntervalSince(started))
            print("DreamX causal progress stage=\(update.stage.rawValue) step=\(update.stepIndex)/\(update.totalSteps) elapsed=\(elapsed)s")
        }
        let forward = try await session.transition(Wan2WorldTransitionRequest(
            prompt: prompt,
            camera: .forward(meters: 0.35),
            sourceImageURL: URL(fileURLWithPath: sourcePath),
            outputURL: output.appendingPathComponent("move-01-forward.mp4"),
            width: width,
            height: height,
            seed: 42,
            fps: 24
        ), progressHandler: progress)
        XCTAssertEqual(forward.conditioningMode, .causalCameraLatents)
        XCTAssertTrue(FileManager.default.fileExists(atPath: forward.outputURL.path))
        if environment["MERERUN_DREAMX_CAUSAL_MOVE_COUNT"] == "1" {
            let snapshot = await session.snapshot()
            XCTAssertEqual(snapshot.transitionCount, 1)
            XCTAssertTrue(snapshot.keepsModelsWarm)
            try await session.unload()
            return
        }
        let yaw = try await session.transition(Wan2WorldTransitionRequest(
            prompt: prompt,
            camera: .yawLeft(degrees: 15),
            outputURL: output.appendingPathComponent("move-02-yaw-left.mp4"),
            width: width,
            height: height,
            seed: 43,
            fps: 24
        ), progressHandler: progress)
        XCTAssertEqual(yaw.previousStateID, forward.stateID)
        XCTAssertTrue(FileManager.default.fileExists(atPath: yaw.outputURL.path))
        let snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.transitionCount, 2)
        XCTAssertTrue(snapshot.keepsModelsWarm)
        XCTAssertTrue(snapshot.keepsTerminalLatent)
        try await session.unload()
    }

    func testProjectiveCameraOfficialStyleQualityCandidate() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let sourcePath = environment["MERERUN_WAN2_SOURCE_IMAGE"],
              let cameraWeightsPath = environment["MERERUN_DREAMX_CAMERA_WEIGHTS"],
              let outputPath = environment["MERERUN_DREAMX_CAMERA_QUALITY_OUTPUT"] else {
            throw XCTSkip("Set the Wan2 model, source image, DreamX camera weights, and quality output variables.")
        }
        let output = URL(fileURLWithPath: outputPath)
        try? FileManager.default.removeItem(at: output)
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: URL(fileURLWithPath: root)),
            stateDirectory: output,
            cameraWeightsURL: URL(fileURLWithPath: cameraWeightsPath)
        )
        let receipt = try await session.transition(Wan2WorldTransitionRequest(
            prompt: "A continuous first-person camera view inside the exact same empty yellow backrooms corridor. The rigid walls, beige carpet, fluorescent ceiling panels, far doorway, lighting, geometry, and visual identity remain stable while the camera smoothly pivots left in place. No forward travel, no scene cut, no new room, no replacement environment.",
            camera: .yawLeft(degrees: 30),
            sourceImageURL: URL(fileURLWithPath: sourcePath),
            outputURL: output.appendingPathComponent("yaw-left-30-step.mp4"),
            width: 128,
            height: 128,
            numFrames: 17,
            steps: 30,
            guidanceScale: 5,
            shift: 5,
            seed: 42
        ))
        XCTAssertEqual(receipt.conditioningMode, .projectiveCameraLatents)
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.outputURL.path))
        try await session.unload()
    }

    func testProjectiveCameraLeftAndRightCandidatesFromSameSource() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let sourcePath = environment["MERERUN_WAN2_SOURCE_IMAGE"],
              let cameraWeightsPath = environment["MERERUN_DREAMX_CAMERA_WEIGHTS"],
              let outputPath = environment["MERERUN_DREAMX_CAMERA_DIRECTIONAL_OUTPUT"] else {
            throw XCTSkip("Set the Wan2 model, source image, DreamX camera weights, and directional output variables.")
        }
        let sourceURL = URL(fileURLWithPath: sourcePath)
        let output = URL(fileURLWithPath: outputPath)
        let width = Int(environment["MERERUN_DREAMX_CAMERA_WIDTH"] ?? "192") ?? 192
        let height = Int(environment["MERERUN_DREAMX_CAMERA_HEIGHT"] ?? "128") ?? 128
        let frames = Int(environment["MERERUN_DREAMX_CAMERA_FRAMES"] ?? "17") ?? 17
        let steps = Int(environment["MERERUN_DREAMX_CAMERA_STEPS"] ?? "8") ?? 8
        try? FileManager.default.removeItem(at: output)
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: URL(fileURLWithPath: root)),
            stateDirectory: output,
            cameraWeightsURL: URL(fileURLWithPath: cameraWeightsPath)
        )
        let prompt = "A first-person view inside the same yellow backrooms corridor, preserving its walls, carpet, fluorescent ceiling lights, geometry, and identity."
        let left = try await session.transition(Wan2WorldTransitionRequest(
            prompt: prompt,
            camera: .yawLeft(degrees: 30),
            sourceImageURL: sourceURL,
            outputURL: output.appendingPathComponent("yaw-left.mp4"),
            width: width,
            height: height,
            numFrames: frames,
            steps: steps,
            guidanceScale: 3,
            seed: 42
        ))
        try await session.reset(sourceImageURL: sourceURL)
        let right = try await session.transition(Wan2WorldTransitionRequest(
            prompt: prompt,
            camera: .yawRight(degrees: 30),
            sourceImageURL: sourceURL,
            outputURL: output.appendingPathComponent("yaw-right.mp4"),
            width: width,
            height: height,
            numFrames: frames,
            steps: steps,
            guidanceScale: 3,
            seed: 42
        ))
        XCTAssertEqual(left.conditioningMode, .projectiveCameraLatents)
        XCTAssertEqual(right.conditioningMode, .projectiveCameraLatents)
        XCTAssertTrue(FileManager.default.fileExists(atPath: left.outputURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: right.outputURL.path))
        try await session.unload()
    }

    func testProjectiveCameraTransitionUsesReleasedDreamXWeights() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let sourcePath = environment["MERERUN_WAN2_SOURCE_IMAGE"],
              let cameraWeightsPath = environment["MERERUN_DREAMX_CAMERA_WEIGHTS"],
              let outputPath = environment["MERERUN_DREAMX_CAMERA_SESSION_OUTPUT"] else {
            throw XCTSkip("Set the Wan2 model, source image, DreamX camera weights, and output variables.")
        }
        let output = URL(fileURLWithPath: outputPath)
        try? FileManager.default.removeItem(at: output)
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: URL(fileURLWithPath: root)),
            stateDirectory: output,
            cameraWeightsURL: URL(fileURLWithPath: cameraWeightsPath)
        )
        let receipt = try await session.transition(Wan2WorldTransitionRequest(
            prompt: "The same empty corridor remains coherent and rigid.",
            camera: .yawLeft(degrees: 15),
            sourceImageURL: URL(fileURLWithPath: sourcePath),
            width: 128,
            height: 128,
            numFrames: 5,
            steps: 1,
            guidanceScale: 1,
            seed: 17
        ))
        XCTAssertEqual(receipt.conditioningMode, .projectiveCameraLatents)
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.outputURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.terminalFrameURL.path))
        let snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.conditioningMode, .projectiveCameraLatents)
        XCTAssertTrue(snapshot.keepsModelsWarm)
        XCTAssertTrue(snapshot.keepsTerminalLatent)
        try await session.unload()
    }

    func testTwoTransitionsReuseWarmModelsAndTerminalLatent() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let sourcePath = environment["MERERUN_WAN2_SOURCE_IMAGE"],
              let outputPath = environment["MERERUN_WAN2_SESSION_OUTPUT"] else {
            throw XCTSkip("Set the Wan2 model, source image, and session output environment variables.")
        }
        let output = URL(fileURLWithPath: outputPath)
        try? FileManager.default.removeItem(at: output)
        let session = Wan2WorldSession(
            resources: Wan2Resources(rootURL: URL(fileURLWithPath: root)),
            stateDirectory: output
        )

        let first = try await session.transition(Wan2WorldTransitionRequest(
            prompt: "The same empty corridor remains coherent and rigid.",
            camera: .forward(meters: 0.25),
            sourceImageURL: URL(fileURLWithPath: sourcePath),
            width: 128,
            height: 128,
            numFrames: 5,
            steps: 1,
            seed: 7
        ))
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.outputURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.terminalFrameURL.path))
        XCTAssertNil(first.previousStateID)

        var snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.phase, .ready)
        XCTAssertEqual(snapshot.transitionCount, 1)
        XCTAssertTrue(snapshot.keepsModelsWarm)
        XCTAssertTrue(snapshot.keepsTerminalLatent)

        let second = try await session.transition(Wan2WorldTransitionRequest(
            prompt: "The same empty corridor remains coherent and rigid.",
            camera: .yawLeft(degrees: 15),
            width: 128,
            height: 128,
            numFrames: 5,
            steps: 1,
            seed: 8
        ))
        XCTAssertEqual(second.previousStateID, first.stateID)
        XCTAssertNotEqual(second.stateID, first.stateID)
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.outputURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.terminalFrameURL.path))

        snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.transitionCount, 2)
        XCTAssertEqual(snapshot.currentStateID, second.stateID)
        XCTAssertEqual(snapshot.currentFrameURL, second.terminalFrameURL)
        XCTAssertTrue(snapshot.keepsModelsWarm)
        XCTAssertTrue(snapshot.keepsTerminalLatent)

        try await session.unload()
        snapshot = await session.snapshot()
        XCTAssertEqual(snapshot.phase, .cold)
        XCTAssertFalse(snapshot.keepsModelsWarm)
        XCTAssertFalse(snapshot.keepsTerminalLatent)
    }
}
