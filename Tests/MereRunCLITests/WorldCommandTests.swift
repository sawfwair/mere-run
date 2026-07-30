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
            "--prepare",
        ])
        XCTAssertEqual(command.port, 9_911)
        XCTAssertEqual(command.baseModel, "/tmp/wan")
        XCTAssertEqual(command.model, "/tmp/dreamx")
        XCTAssertEqual(command.stateDirectory, "/tmp/world")
        XCTAssertTrue(command.prepare)
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
}
