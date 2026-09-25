@testable import StudioKit
import XCTest

final class StudioClassificationsTests: XCTestCase {
    func testExampleBuildsTheCLIRequestAndRoundTripsDescriptions() throws {
        let document = StudioClassificationDocument.example
        XCTAssertTrue(document.problems.isEmpty)
        let data = try document.requestJSON()
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let tasks = try XCTUnwrap(json["tasks"] as? [[String: Any]])

        XCTAssertEqual(json["text"] as? String, document.text)
        XCTAssertEqual(tasks.map { $0["name"] as? String }, ["department", "intent"])
        XCTAssertEqual(tasks[0]["labels"] as? [String], ["billing", "technical support", "sales"])
        XCTAssertEqual((tasks[0]["descriptions"] as? [String: String])?["billing"], "Payments, charges, and refunds")
        XCTAssertEqual(tasks[1]["multi_label"] as? Bool, true)
        XCTAssertEqual(tasks[1]["threshold"] as? Double, 0.5)

        let imported = try StudioClassificationDocument.importing(data)
        XCTAssertEqual(imported.text, document.text)
        XCTAssertEqual(imported.tasks.map(\.name), document.tasks.map(\.name))
        XCTAssertEqual(imported.tasks[0].labels.map(\.detail), document.tasks[0].labels.map(\.detail))
    }

    func testImportedDefaultsAndOptionalPromptArePreserved() throws {
        let data = Data("""
        {"text":"A late parcel", "tasks":[{"name":"topic", "labels":["shipping", "billing"],
          "prompt":"Choose the topic", "descriptions":{"shipping":"Delivery problems"}}]}
        """.utf8)
        let document = try StudioClassificationDocument.importing(data)
        XCTAssertFalse(document.tasks[0].multiLabel)
        XCTAssertEqual(document.tasks[0].threshold, 0.5)
        XCTAssertEqual(document.tasks[0].prompt, "Choose the topic")
        XCTAssertEqual(document.tasks[0].labels[0].detail, "Delivery problems")
        XCTAssertTrue(document.problems.isEmpty)
    }

    func testInvalidLabelsAndThresholdBlockTheEditor() {
        var document = StudioClassificationDocument.example
        document.tasks[0].labels[1].name = "billing"
        document.tasks[0].threshold = 1.2
        document.tasks[1].prompt = "[SEP_TEXT]"
        XCTAssertTrue(document.problems.contains("Task 1 labels must be unique."))
        XCTAssertTrue(document.problems.contains("Task 1 threshold must be between 0 and 1."))
        XCTAssertTrue(document.problems.contains("Task 2 prompt contains a reserved marker."))
    }

    func testResultAndPreflightDecodeFromOutputAndStderr() throws {
        let result = """
        {"model":"text-classify-gliner25-decide", "runtime":"mlx", "inputTokens":81,
          "heads":{"topic":{"labels":["shipping"],
            "probabilities":{"shipping":0.82,"billing":0.18}}}}
        """
        let decoded = try XCTUnwrap(StudioClassificationOutput(outputText: result + "\n\nSTDERR\nLoading model"))
        guard case .result(let answers) = decoded else { return XCTFail("Expected classifications") }
        XCTAssertEqual(answers.heads["topic"]?.labels, ["shipping"])
        XCTAssertEqual(answers.heads["topic"]?.probabilities["billing"], 0.18)

        let preflight = Data("""
        {"model":"text-classify-gliner25-decide", "inputTokens":81,
          "labelCount":2, "taskNames":["topic"]}
        """.utf8)
        guard case .fit(let plan) = StudioClassificationOutput(data: preflight) else {
            return XCTFail("Expected fit plan")
        }
        XCTAssertEqual(plan.inputTokens, 81)
        XCTAssertEqual(plan.taskNames, ["topic"])
    }

    func testClassificationTemplateUsesTheFormRequest() {
        var draft = CommandDraft()
        draft.inputPath = "/tmp/classification.json"
        draft.model = "text-classify-gliner25-decide"
        draft.outputPath = "/tmp/result.json"
        draft.force = true
        draft.preflight = true
        XCTAssertEqual(CommandArguments.textClassify(draft), [
            "text", "classify", "--input", "/tmp/classification.json",
            "--model", "text-classify-gliner25-decide", "--output", "/tmp/result.json",
            "--pretty", "--preflight",
        ])
        XCTAssertEqual(CommandTemplateID.textClassify.studioTask, .textClassify)
    }
}
