import Foundation
import StudioKit
import XCTest

final class StudioExtractionsTests: XCTestCase {
    func testExampleRoundTripsAndCommandUsesExtraction() throws {
        let document = StudioExtractionDocument.example
        XCTAssertTrue(document.problems.isEmpty)
        let data = try document.requestJSON()
        let restored = try StudioExtractionDocument.importing(data)
        XCTAssertEqual(restored.text, document.text)
        XCTAssertEqual(restored.entities.map(\.name), ["person", "organization", "location"])
        XCTAssertEqual(restored.relations.map(\.name), ["works_for", "located_in"])
        XCTAssertEqual(restored.structures.first?.fields.map(\.name), ["person", "organization", "location"])

        var draft = CommandDraft()
        draft.inputPath = "/tmp/extraction.json"
        draft.model = "text-classify-gliner25-decide"
        draft.outputPath = "/tmp/result.json"
        draft.preflight = true
        XCTAssertEqual(CommandArguments.textExtract(draft), [
            "text", "extract", "--input", "/tmp/extraction.json", "--model", "text-classify-gliner25-decide",
            "--output", "/tmp/result.json", "--preflight"
        ])
        XCTAssertEqual(CommandTemplateID.textExtract.studioTask, .textExtract)
    }

    func testDuplicateSchemaNamesAreRejected() {
        var document = StudioExtractionDocument.example
        document.relations[0].name = "person"
        XCTAssertFalse(document.problems.isEmpty)
    }

    func testLongPreflightPlanDecodes() throws {
        let data = Data("""
        [{"model":"text-classify-gliner25-decide","inputTokens":48,"wordCount":9,"schemaNames":["entities"]},
         {"model":"text-classify-gliner25-decide","inputTokens":39,"wordCount":8,"schemaNames":["entities"]}]
        """.utf8)
        let plans = try JSONDecoder().decode([StudioExtractionPlan].self, from: data)
        XCTAssertEqual(plans.map(\.inputTokens), [48, 39])
    }

    func testJointSchemaImportsAndKeepsClassificationTasks() throws {
        let data = Data("""
        {"text":"Alice joined Acme.","entities":[{"name":"person"}],
         "classifications":[{"name":"tone","labels":["positive","negative"],
           "multi_label":true,"threshold":0.4}]}
        """.utf8)
        let document = try StudioExtractionDocument.importing(data)
        XCTAssertTrue(document.problems.isEmpty)
        XCTAssertEqual(document.classifications.first?.name, "tone")
        XCTAssertTrue(document.classifications.first?.multiLabel == true)
        let exported = try document.requestJSON()
        let again = try StudioExtractionDocument.importing(exported)
        XCTAssertEqual(again.classifications.first?.labels.map(\.name), ["positive", "negative"])
        XCTAssertEqual(again.classifications.first?.threshold, 0.4)
    }
}
