import ArgumentParser
import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class TextClassifyCommandTests: XCTestCase {
    func testCommandParsesManagedModelAndPreflight() throws {
        let command = try TextClassify.parse([
            "--input", "request.json", "--model", GLiNERCatalog.modelID,
            "--output", "result.json", "--preflight", "--pretty"
        ])
        XCTAssertEqual(command.input, "request.json")
        XCTAssertEqual(command.model, GLiNERCatalog.modelID)
        XCTAssertEqual(command.output, "result.json")
        XCTAssertTrue(command.preflight)
        XCTAssertTrue(command.pretty)
    }

    func testRequestDefaultsAndRejectsCorruptSchema() throws {
        let data = Data(#"{"text":"Please refund the order.","tasks":[{"name":"intent","labels":["refund","other"]}]}"#.utf8)
        let request = try JSONDecoder().decode(GLiNERClassificationRequest.self, from: data)
        try request.validate()
        XCTAssertFalse(request.tasks[0].multiLabel)
        XCTAssertEqual(request.tasks[0].threshold, 0.5)

        let invalid = GLiNERClassificationRequest(text: "A", tasks: [
            GLiNERClassificationTask(name: "intent", labels: ["refund", "refund"])
        ])
        XCTAssertThrowsError(try invalid.validate())
    }
}
