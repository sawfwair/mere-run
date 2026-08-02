import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class WorldCommandTests: XCTestCase {
    func testWorldCommandExposesServe() {
        XCTAssertEqual(World.configuration.subcommands.map { $0.configuration.commandName }, ["serve"])
    }

    func testWorldServeParsesPersistentRuntimeOptions() throws {
        let command = try WorldServe.parse([
            "--host", "127.0.0.1",
            "--port", "9911",
            "--base-model", "/tmp/wan",
            "--model", "/tmp/dreamx",
            "--state-directory", "/tmp/world",
            "--scene-memory-strength", "0.2",
            "--scene-memory-max-frames", "144",
            "--scene-memory-minimum-gap", "6",
            "--scene-memory-max-yaw", "3",
            "--scene-memory-max-translation", "0.2",
            "--scene-memory-exact-yaw", "0.02",
            "--scene-memory-exact-translation", "0.002",
            "--prepare",
        ])
        XCTAssertEqual(command.port, 9_911)
        XCTAssertEqual(command.baseModel, "/tmp/wan")
        XCTAssertEqual(command.model, "/tmp/dreamx")
        XCTAssertEqual(command.stateDirectory, "/tmp/world")
        XCTAssertTrue(command.prepare)
        let memory = try command.resolvedSceneMemoryPolicy()
        XCTAssertEqual(memory.recyclingStrength, 0.2)
        XCTAssertEqual(memory.maximumFrameCount, 144)
        XCTAssertEqual(memory.minimumFrameGap, 6)
        XCTAssertEqual(memory.maximumYawDistanceDegrees, 3)
        XCTAssertEqual(memory.maximumTranslationDistance, 0.2)
        XCTAssertEqual(memory.exactRevisitMaximumYawDistanceDegrees, 0.02)
        XCTAssertEqual(memory.exactRevisitMaximumTranslationDistance, 0.002)
    }

    func testWorldServeDefaultsToNativeManagedModels() throws {
        let command = try WorldServe.parse([])
        XCTAssertEqual(command.baseModel, Wan2Resources.modelID)
        XCTAssertEqual(command.model, Wan2DreamXCausalResources.modelID)
        XCTAssertEqual(command.port, 8_791)
    }

    func testWorldServeSelectsCosmos3ActionBackend() throws {
        let command = try WorldServe.parse([
            "--backend", "cosmos3",
            "--model", Cosmos3Resources.modelID,
        ])
        XCTAssertEqual(command.backend, .cosmos3)
        XCTAssertEqual(command.model, Cosmos3Resources.modelID)
    }

    func testWorldServeReadsAPIKeyFromEnvironmentWithoutRequiringProcessArgument() throws {
        let command = try WorldServe.parse(["--host", "0.0.0.0"])

        XCTAssertEqual(
            command.resolvedAPIKey(environment: ["MERERUN_API_KEY": " world-secret "]),
            "world-secret"
        )
        XCTAssertNil(command.apiKey)
    }

    func testWorldServeCanDisableAndValidatesSceneMemory() throws {
        let disabled = try WorldServe.parse([
            "--disable-scene-memory",
            "--scene-memory-strength", "9",
        ])
        XCTAssertEqual(try disabled.resolvedSceneMemoryPolicy(), .disabled)

        let invalid = try WorldServe.parse(["--scene-memory-strength", "1.01"])
        XCTAssertThrowsError(try invalid.resolvedSceneMemoryPolicy())

        let invalidExact = try WorldServe.parse([
            "--scene-memory-max-yaw", "1",
            "--scene-memory-exact-yaw", "2",
        ])
        XCTAssertThrowsError(try invalidExact.resolvedSceneMemoryPolicy())
    }

    func testWorldTransitionPayloadDecodesDocumentedHTTPOverrides() throws {
        let data = Data(#"""
        {
          "prompt": "turn through the same station",
          "camera": {
            "motion": "yawRight",
            "translationMeters": [0, 0, 0],
            "rotationDegrees": [0, 15, 0]
          },
          "sourceImage": "/tmp/vesper.png",
          "output": "/tmp/vesper-right.mp4",
          "width": 320,
          "height": 176,
          "num_frames": 61,
          "steps": 30,
          "guidance_scale": 1.5,
          "shift": 3,
          "seed": 7,
          "fps": 30,
          "model_space_actions": [[0, 0, 0, 1, 0, 0, 0, 1, 0]]
        }
        """#.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let payload = try decoder.decode(WorldTransitionPayload.self, from: data)

        XCTAssertEqual(payload.prompt, "turn through the same station")
        XCTAssertEqual(payload.camera.motion, .yawRight)
        XCTAssertEqual(payload.camera.rotationDegrees, [0, 15, 0])
        XCTAssertEqual(payload.sourceImage, "/tmp/vesper.png")
        XCTAssertEqual(payload.output, "/tmp/vesper-right.mp4")
        XCTAssertEqual(payload.width, 320)
        XCTAssertEqual(payload.height, 176)
        XCTAssertEqual(payload.numFrames, 61)
        XCTAssertEqual(payload.steps, 30)
        XCTAssertEqual(payload.guidanceScale, 1.5)
        XCTAssertEqual(payload.shift, 3)
        XCTAssertEqual(payload.seed, 7)
        XCTAssertEqual(payload.fps, 30)
        XCTAssertEqual(payload.modelSpaceActions, [[0, 0, 0, 1, 0, 0, 0, 1, 0]])
    }

    func testWorldRolloutPayloadDecodesOfficialDreamXActionSequence() throws {
        let data = Data(#"""
        {
          "prompt": "move through the same coherent station",
          "action_seq": ["w", "wj", "wl"],
          "action_speed_list": [4, 6, 6],
          "source_image": "/tmp/vesper.png",
          "output": "/tmp/vesper-rollout.mp4",
          "width": 1280,
          "height": 704,
          "num_output_frames": 63,
          "speed": 1.5,
          "seed": 7,
          "fps": 16
        }
        """#.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let payload = try decoder.decode(WorldRolloutPayload.self, from: data)
        let request = try payload.request()

        XCTAssertEqual(payload.actionSeq, ["w", "wj", "wl"])
        XCTAssertEqual(payload.actionSpeedList, [4, 6, 6])
        XCTAssertEqual(request.actionSequence.map(\.action), ["w", "wj", "wl"])
        XCTAssertEqual(request.actionSequence.map(\.weight), [4, 6, 6])
        XCTAssertEqual(request.latentFrameCount, 63)
        XCTAssertEqual(request.expectedPixelFrameCount, 249)
        XCTAssertEqual(request.width, 1_280)
        XCTAssertEqual(request.height, 704)
        XCTAssertEqual(request.speed, 1.5)
        XCTAssertEqual(request.fps, 16)
    }

    func testWorldRolloutPayloadRejectsMismatchedWeightsAndOpposingControls() throws {
        let mismatched = WorldRolloutPayload(
            prompt: "same station",
            actionSeq: ["w", "j"],
            actionSpeedList: [1],
            sourceImage: nil,
            output: nil,
            width: nil,
            height: nil,
            numOutputFrames: nil,
            speed: nil,
            seed: nil,
            fps: nil
        )
        XCTAssertThrowsError(try mismatched.request()) { error in
            XCTAssertEqual(
                error as? WorldRolloutValidationError,
                .mismatchedActionWeights(actions: 2, weights: 1)
            )
        }

        let contradictory = WorldRolloutPayload(
            prompt: "same station",
            actionSeq: ["ws"],
            actionSpeedList: nil,
            sourceImage: nil,
            output: nil,
            width: nil,
            height: nil,
            numOutputFrames: nil,
            speed: nil,
            seed: nil,
            fps: nil
        )
        XCTAssertThrowsError(try contradictory.request()) { error in
            XCTAssertEqual(error as? WorldRolloutValidationError, .contradictoryAction("ws"))
        }

        let oversized = WorldRolloutPayload(
            prompt: "same station",
            actionSeq: ["w"],
            actionSpeedList: nil,
            sourceImage: nil,
            output: nil,
            width: 1_312,
            height: 704,
            numOutputFrames: 246,
            speed: nil,
            seed: nil,
            fps: nil
        )
        XCTAssertThrowsError(try oversized.request()) { error in
            XCTAssertEqual(
                error as? WorldRolloutValidationError,
                .resolutionExceedsMaximum(width: 1_312, height: 704)
            )
        }

        let tooLong = WorldRolloutPayload(
            prompt: "same station",
            actionSeq: ["w"],
            actionSpeedList: nil,
            sourceImage: nil,
            output: nil,
            width: nil,
            height: nil,
            numOutputFrames: 255,
            speed: nil,
            seed: nil,
            fps: nil
        )
        XCTAssertThrowsError(try tooLong.request()) { error in
            XCTAssertEqual(error as? WorldRolloutValidationError, .latentFrameCountExceedsMaximum(255))
        }
    }
}
