import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionGeometryCommandTests: XCTestCase {
    func testParsesProductionOptions() throws {
        let command = try VisionGeometry.parse([
            "/tmp/frame.png",
            "--output", "/tmp/geometry",
            "--model", "/tmp/model.onnx",
            "--resolution-level", "5",
            "--token-count", "1800",
            "--max-points", "250000",
            "--json",
        ])
        XCTAssertEqual(command.input, "/tmp/frame.png")
        XCTAssertEqual(command.output, "/tmp/geometry")
        XCTAssertEqual(command.model, "/tmp/model.onnx")
        XCTAssertEqual(command.resolutionLevel, 5)
        XCTAssertEqual(command.tokenCount, 1_800)
        XCTAssertEqual(command.maxPoints, 250_000)
        XCTAssertTrue(command.json)
    }

    func testDefaultOutputAndPlanAreDeterministic() {
        let input = URL(fileURLWithPath: "/tmp/shot.001.png")
        let output = VisionGeometry.resolveOutputURL(nil, inputURL: input)
        XCTAssertEqual(output.path, "/tmp/shot.001-geometry")
        let plan = VisionGeometry.makePlan(
            inputURL: input,
            outputURL: output,
            imageWidth: 1920,
            imageHeight: 1080,
            model: nil,
            configuration: MoGe2InferenceConfiguration(resolutionLevel: 0)
        )
        XCTAssertEqual(plan.tokenCount, 1_200)
        XCTAssertEqual(plan.tokenRows, 26)
        XCTAssertEqual(plan.tokenColumns, 46)
        XCTAssertEqual(plan.networkHeight, 364)
        XCTAssertEqual(plan.networkWidth, 644)
        XCTAssertTrue(plan.outputKinds.contains("camera-json"))
    }
}
