import Foundation
import XCTest
@testable import MereRunCLI

final class VisionFlowCommandTests: XCTestCase {
    func testParsesFlowOptions() throws {
        let command = try VisionFlow.parse([
            "/tmp/a.png", "/tmp/b.png", "--output", "/tmp/flow", "--json-output", "/tmp/meta",
            "--accuracy", "very-high", "--json",
        ])
        XCTAssertEqual(command.from, "/tmp/a.png")
        XCTAssertEqual(command.to, "/tmp/b.png")
        XCTAssertEqual(command.output, "/tmp/flow")
        XCTAssertEqual(command.jsonOutput, "/tmp/meta")
        XCTAssertEqual(command.accuracy, .veryHigh)
        XCTAssertTrue(command.json)
    }

    func testResolvesDefaultOutputs() {
        let from = URL(fileURLWithPath: "/tmp/a.png")
        let to = URL(fileURLWithPath: "/tmp/b.png")
        let flow = VisionFlow.resolveFlowOutputURL(nil, fromURL: from, toURL: to)
        XCTAssertEqual(flow.path, "/tmp/a_to_b_flow.flo")
        XCTAssertEqual(VisionFlow.resolveJSONOutputURL(nil, flowOutputURL: flow).path, "/tmp/a_to_b_flow.json")
    }
}
