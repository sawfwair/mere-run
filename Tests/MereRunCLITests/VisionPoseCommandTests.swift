import ArgumentParser
import Foundation
import XCTest
@testable import MereRunCLI

final class VisionPoseCommandTests: XCTestCase {
    func testParsesPoseOptions() throws {
        let command = try VisionPose.parse([
            "/tmp/person.png",
            "--json-output", "/tmp/pose",
            "--no-face",
            "--max-hands", "1",
            "--minimum-confidence", "0.25",
            "--json",
        ])

        XCTAssertEqual(command.image, "/tmp/person.png")
        XCTAssertEqual(command.jsonOutput, "/tmp/pose")
        XCTAssertFalse(command.noBody)
        XCTAssertFalse(command.noHands)
        XCTAssertTrue(command.noFace)
        XCTAssertEqual(command.maxHands, 1)
        XCTAssertEqual(command.minimumConfidence, 0.25, accuracy: 0.0001)
        XCTAssertTrue(command.json)
    }

    func testDefaultOutputPath() {
        let input = URL(fileURLWithPath: "/tmp/person.frame.png")
        XCTAssertEqual(
            VisionPose.resolveJSONOutputURL(nil, inputImageURL: input).path,
            "/tmp/person.frame_pose.json"
        )
        XCTAssertEqual(
            VisionPose.resolveJSONOutputURL("/tmp/custom", inputImageURL: input).path,
            "/tmp/custom.json"
        )
    }

    func testRejectsEmptyDetectorSelection() throws {
        XCTAssertThrowsError(try VisionPose.parse([
            "/tmp/person.png", "--no-body", "--no-hands", "--no-face",
        ]))
    }
}
