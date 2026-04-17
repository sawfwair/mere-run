import Foundation
import XCTest
@testable import MereRunCore

final class FalconPerceptionIntegrationTests: MereRunCoreTestCase {
    func testGroundingRoundTripWhenLocalModelIsAvailable() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelRoot = env["MERERUN_FALCON_PERCEPTION_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_MODEL_ROOT to run Falcon Perception integration coverage.")
        }
        guard let imagePath = env["MERERUN_FALCON_PERCEPTION_TEST_IMAGE"], !imagePath.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_TEST_IMAGE to run Falcon Perception integration coverage.")
        }

        let modelURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path),
              FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("Falcon integration assets are missing on disk.")
        }

        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let annotatedURL = temp.appendingPathComponent("grounded.png")
        let jsonURL = temp.appendingPathComponent("grounded.json")
        let masksURL = temp.appendingPathComponent("masks", isDirectory: true)

        let grounder = try FalconPerceptionGrounder(
            modelRootURL: modelURL,
            expectedModelID: "vision-ground-falcon-perception"
        )
        let result = try grounder.ground(
            imageURL: imageURL,
            queries: [env["MERERUN_FALCON_PERCEPTION_TEST_QUERY"] ?? "person"],
            annotatedImageURL: annotatedURL,
            jsonOutputURL: jsonURL,
            maskOutputDirectoryURL: masksURL
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: result.annotatedImageURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.jsonOutputURL.path))
        XCTAssertEqual(result.metadata.modelID, result.modelID)
    }
}
