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
        let longBatch = try TextClassify.parse(["--input", "requests.json", "--long", "--batch"])
        XCTAssertTrue(longBatch.long)
        XCTAssertTrue(longBatch.batch)
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

    func testExtractionCommandAndAPIShapes() throws {
        let command = try TextExtract.parse(["--input", "request.json", "--model", GLiNERCatalog.modelID,
                                             "--long", "--batch", "--preflight"])
        XCTAssertTrue(command.long)
        XCTAssertTrue(command.batch)
        XCTAssertTrue(command.preflight)
        let single = Data(#"{"model":"text-classify-gliner25-decide","text":"Alice joined Acme.","entities":[{"name":"person"}]}"#.utf8)
        let request = try JSONDecoder().decode(GLiNERExtractionAPIRequest.self, from: single)
        try request.validate()
        XCTAssertFalse(request.batch)
        XCTAssertEqual(request.requests.count, 1)
        let batch = Data(#"{"model":"text-classify-gliner25-decide","long":true,"requests":[{"text":"Alice joined Acme.","entities":[{"name":"person"}]}]}"#.utf8)
        let batchRequest = try JSONDecoder().decode(GLiNERExtractionAPIRequest.self, from: batch)
        try batchRequest.validate()
        XCTAssertTrue(batchRequest.batch)
        XCTAssertTrue(batchRequest.long)
        XCTAssertEqual(batchRequest.requests.count, 1)
    }
}
