@testable import StudioKit
import XCTest

final class StudioDecisionsTests: XCTestCase {
    /// The example encodes to the handbook's request: bare-string criteria, and no criteria for
    /// yes or no, whose CLI type is `noul`.
    func testTheExampleEncodesToTheHandbookRequest() throws {
        let json = try JSONSerialization.jsonObject(with: StudioDecisionDocument.example.requestJSON()) as? [String: Any]
        let questions = try XCTUnwrap(json?["questions"] as? [[String: Any]])

        XCTAssertEqual(json?["state"] as? String, "I was charged twice for my subscription. Please refund the duplicate charge.")
        XCTAssertEqual(questions.map { $0["id"] as? String }, ["department", "urgency", "refund"])
        XCTAssertEqual(questions.map { $0["type"] as? String }, ["choice", "score", "noul"])
        XCTAssertEqual(questions[0]["criteria"] as? [String], ["billing", "technical support", "sales"])
        XCTAssertEqual(questions[1]["criteria"] as? [String], ["routine", "soon", "immediate"])
        XCTAssertNil(questions[2]["criteria"])
        XCTAssertEqual(questions[0]["instructions"] as? String, "Which department should handle this request?")
    }

    func testOptionsWithDescriptionsEncodeAsObjectsAndYesNoWordingAsTrueFalse() throws {
        let document = StudioDecisionDocument(text: "t", questions: [
            .init(kind: .choice, prompt: "Which?", options: [.init(label: "billing", detail: "Invoices"), .init(label: "sales")]),
            .init(kind: .yesNo, prompt: "Asks for a refund.", options: [.init(label: "true", detail: "wants money back")]),
        ])
        let json = try JSONSerialization.jsonObject(with: document.requestJSON()) as? [String: Any]
        let questions = try XCTUnwrap(json?["questions"] as? [[String: Any]])
        let choice = try XCTUnwrap(questions[0]["criteria"] as? [Any])

        XCTAssertEqual((choice[0] as? [String: String])?["description"], "Invoices")
        XCTAssertEqual(choice[1] as? String, "sales")
        XCTAssertEqual(questions[1]["criteria"] as? [[String: String]], [["label": "true", "description": "wants money back"]])
    }

    func testImportingReadsStringAndObjectCriteriaAndKeepsIds() throws {
        let data = Data("""
        {"state": "Refund please", "questions": [
          {"id": "dept", "type": "choice", "instructions": "Which department?",
           "criteria": ["billing", {"label": "sales", "description": "New purchases"}]},
          {"id": "refund", "type": "noul", "instructions": "Asks for a refund."}
        ]}
        """.utf8)

        let document = try StudioDecisionDocument.importing(data)

        XCTAssertEqual(document.text, "Refund please")
        XCTAssertEqual(document.questions.map(\.kind), [.choice, .yesNo])
        XCTAssertEqual(document.questions[0].options.map(\.label), ["billing", "sales"])
        XCTAssertEqual(document.questions[0].options[1].detail, "New purchases")
        XCTAssertEqual(document.resolvedKeys, ["dept", "refund"])
        XCTAssertThrowsError(try StudioDecisionDocument.importing(Data(#"{"state":"x","questions":[{"id":"a","type":"rank","instructions":"?"}]}"#.utf8)))
    }

    func testIdsDeriveFromTheQuestionAndStayUnique() {
        let document = StudioDecisionDocument(text: "t", questions: [
            .init(kind: .yesNo, prompt: "Is this urgent?"),
            .init(kind: .yesNo, prompt: "Is this urgent?"),
            .init(kind: .yesNo, prompt: "!!!"),
        ])
        XCTAssertEqual(document.resolvedKeys, ["is-this-urgent", "is-this-urgent-2", "question-3"])
    }

    func testProblemsAreTheCLIsChecksInThePagesWords() {
        XCTAssertEqual(StudioDecisionDocument().problems, ["Add the text to judge.", "Add a question."])
        XCTAssertTrue(StudioDecisionDocument.example.problems.isEmpty)

        let document = StudioDecisionDocument(text: "t", questions: [
            .init(kind: .choice, prompt: "", options: [.init(label: "a"), .init(label: "a")]),
            .init(kind: .score, prompt: "How much?", options: [.init(label: "low")]),
        ])
        XCTAssertEqual(document.problems, [
            "Question 1 needs a question.",
            "Question 1 repeats an option.",
            "Question 2 needs at least two levels.",
        ])
    }

    /// A new question's blank options are unfinished, not repeated.
    func testBlankOptionsAreNotRepeats() {
        let document = StudioDecisionDocument(text: "t", questions: [
            .blank(.choice),
            .init(kind: .choice, prompt: "Which?", options: [.init(label: "a"), .init(label: "b"), .init(label: "")]),
            .init(kind: .score, prompt: "How much?", options: [.init(label: "mid"), .init(label: "mid")]),
            .init(kind: .yesNo, prompt: "Holds.", key: String(repeating: "k", count: 257)),
        ])
        XCTAssertEqual(document.problems, [
            "Question 1 needs a question.",
            "Question 1 needs at least two options.",
            "Question 2 has an empty option.",
            "Question 4's ID is longer than 256 bytes.",
        ])
    }

    func testTokenBudgetsAndTrimmedWordingRoundTrip() throws {
        let data = Data("""
        {"state": "x", "max_tokens": 256, "head_max_tokens": 96, "questions": [
          {"id": "q", "type": "noul", "instructions": "Holds.", "criteria": [{"label": "true", "description": "it does"}]}
        ]}
        """.utf8)
        var document = try StudioDecisionDocument.importing(data)
        XCTAssertEqual(document.maxTokens, 256)
        XCTAssertEqual(document.headMaxTokens, 96)

        document.questions[0].options[0].detail = "  it does \n"
        let json = try JSONSerialization.jsonObject(with: document.requestJSON()) as? [String: Any]
        XCTAssertEqual(json?["max_tokens"] as? Int, 256)
        XCTAssertEqual(json?["head_max_tokens"] as? Int, 96)
        let questions = try XCTUnwrap(json?["questions"] as? [[String: Any]])
        XCTAssertEqual(questions[0]["criteria"] as? [[String: String]], [["label": "true", "description": "it does"]])
        let example = try JSONSerialization.jsonObject(with: StudioDecisionDocument.example.requestJSON()) as? [String: Any]
        XCTAssertNil(example?["max_tokens"])
    }

    /// A run without an output file printed its document; stderr follows it in the captured text.
    func testOutputReadsPrintedDocumentsPastStderr() throws {
        let plan = #"{"model": "text-decide-laya", "maxTokens": 512, "headMaxTokens": 192, "questions": []}"#
        let printed = "{\n  \"model\": \"text-decide-laya\",\n  \"answers\": {},\n  \"plan\": \(plan)\n}\n\nSTDERR\nLoading text-decide-laya…"

        XCTAssertEqual(StudioDecisionOutput(outputText: printed)?.result?.answers, [:])
        XCTAssertEqual(StudioDecisionOutput(outputText: plan)?.plan.maxTokens, 512)
        XCTAssertNil(StudioDecisionOutput(outputText: plan)?.result)
        XCTAssertNil(StudioDecisionOutput(outputText: "error: no such model"))
    }

    /// A result as `text decide` writes it (`LayaDecisionResponse`), and a `--preflight` plan.
    func testDecodesTheResultAndThePreflightPlan() throws {
        let plan = """
        {"model": "text-decide-laya", "maxTokens": 512, "headMaxTokens": 192, "questions": [
          {"id": "department", "inputTokens": 40, "stateTokens": 18, "stateTokensDropped": 0,
           "instructionTokensDropped": 0, "optionTokensDropped": [0, 0, 0], "optionCount": 3},
          {"id": "refund", "inputTokens": 512, "stateTokens": 400, "stateTokensDropped": 120,
           "instructionTokensDropped": 0, "optionTokensDropped": [0, 2], "optionCount": 2}
        ]}
        """
        let result = """
        {"model": "text-decide-laya", "runtime": "mlx", "inputTokens": 552, "outputTokens": 0, "plan": \(plan),
         "answers": {
           "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, "technical support": 0.06, "sales": 0.03},
             "confidence": 0.78, "actProbability": 0.2, "rawTemperature": 0.4, "appliedTemperature": 0.5, "temperatureClamped": true},
           "refund": {"type": "noul", "noul": 0.94, "probabilities": {"false": 0.06, "true": 0.94},
             "confidence": 0.94, "actProbability": 0.7, "rawTemperature": 1.1, "appliedTemperature": 1.1, "temperatureClamped": false}
         }}
        """

        let decoded = try JSONDecoder().decode(StudioDecisionResult.self, from: Data(result.utf8))
        XCTAssertEqual(decoded.answers["department"]?.choice, "billing")
        XCTAssertEqual(decoded.answers["department"]?.temperatureClamped, true)
        XCTAssertEqual(decoded.answers["refund"]?.noul, 0.94)
        XCTAssertNil(decoded.plan.question("department")?.truncationNote)
        XCTAssertEqual(
            decoded.plan.question("refund")?.truncationNote,
            "Cut 120 tokens from the end of the text, 2 from the options to fit."
        )

        let preflight = try JSONDecoder().decode(StudioDecisionPlan.self, from: Data(plan.utf8))
        XCTAssertEqual(preflight.maxTokens, 512)
        XCTAssertTrue(try XCTUnwrap(preflight.question("refund")).wasTruncated)
        XCTAssertThrowsError(try JSONDecoder().decode(StudioDecisionResult.self, from: Data(plan.utf8)))
    }
}
