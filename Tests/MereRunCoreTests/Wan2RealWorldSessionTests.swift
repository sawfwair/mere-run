import Foundation
import XCTest
@testable import MereRunCore

final class Wan2RealWorldSessionTests: MereRunCoreTestCase {
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
