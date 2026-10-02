import Foundation
import XCTest
import MLX
@testable import MereRunCore

/// Opt-in comparison against independently generated checkpoint outputs.
/// Calls the runtime directly; it does not establish CLI admission behavior.
final class ClefFlashCheckpointParityTests: MereRunCoreTestCase {
    private struct Reference: Decodable {
        let revision: String
        let reference_sha256: String
        let cases: [String: Probe]
        struct Probe: Decodable {
            let input_tokens: Int
            let fields: [Field]
            let ids: [Int]
            let answers: [String: Answer]
        }
        struct Field: Decodable {
            let id: String
            let question_span: [Int]
            let option_spans: [[Int]]
            let option_ids: [String]
        }
        struct Answer: Decodable {
            let type: String
            let choice: String?
            let noul: Double?
            let score: Double?
            let probabilities: [String: Double]?
        }
    }

    func testInstalledCheckpointMatchesIndependentReferenceWhenProvided() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let checkpoint = environment["MERERUN_TEST_CLEF_FLASH_ROOT"],
              let casesPath = environment["MERERUN_TEST_CLEF_PARITY_DIR"] else {
            throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu, MERERUN_TEST_CLEF_FLASH_ROOT, and MERERUN_TEST_CLEF_PARITY_DIR.")
        }
        let root = URL(fileURLWithPath: checkpoint)
        let directory = URL(fileURLWithPath: casesPath)
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: directory.appending(path: "reference.json")))
        XCTAssertEqual(reference.revision, ClefCatalog.flashRevision)
        XCTAssertEqual(reference.reference_sha256, "5b381f596a2507885a5a736bec7caef3200fb882f91ef2fea08a6bfa8f1877d2")
        XCTAssertTrue(ClefCatalog.validate(root: root).isEmpty)
        var outputs: [String: ClefDecisionResponse] = [:]
        for name in ["text", "image", "video", "resize"] {
            let expected = try XCTUnwrap(reference.cases[name])
            let request = try ClefDecisionRequest.decode(Data(contentsOf: directory.appending(path: "\(name).json")))
            let operation = try ClefDecisionOperation(root: root, modelID: ClefCatalog.flashModelID)
            defer { operation.unload() }
            let plan = try operation.prepare(request)
            let processor = try ClefResources(root: root).configuration().processor
            let media = try ClefPreparedMedia.prepare(request, processor: processor)
            let tokens = try ClefTokenizer.load(root: root).sequence(request, modelID: ClefCatalog.flashModelID, mediaText: media.text)
            XCTAssertEqual(tokens.ids, expected.ids, name)
            XCTAssertEqual(plan.inputTokens, expected.input_tokens, name)
            XCTAssertEqual(plan.questions.map(\.id), expected.fields.map(\.id), name)
            for (field, target) in zip(plan.questions, expected.fields) {
                XCTAssertEqual(field.questionSpan, target.question_span, name)
                XCTAssertEqual(field.optionSpans, target.option_spans, name)
                XCTAssertEqual(field.optionIDs, target.option_ids, name)
            }
            let result = try operation.predict(request)
            XCTAssertEqual(result.usage.input_tokens, expected.input_tokens, name)
            XCTAssertEqual(result.usage.output_tokens, 0, name)
            XCTAssertEqual(Set(result.answers.keys), Set(expected.answers.keys), name)
            for (id, target) in expected.answers {
                let answer = try XCTUnwrap(result.answers[id])
                XCTAssertEqual(answer.type, target.type, "\(name)/\(id)")
                XCTAssertEqual(answer.choice, target.choice, "\(name)/\(id)")
                if let noul = target.noul {
                    XCTAssertEqual(try XCTUnwrap(answer.noul), noul, accuracy: 0.01, "\(name)/\(id)")
                }
                if let score = target.score {
                    XCTAssertEqual(try XCTUnwrap(answer.score), score, accuracy: 0.02, "\(name)/\(id)")
                }
                if let probabilities = target.probabilities {
                    let actual = try XCTUnwrap(answer.probabilities)
                    XCTAssertEqual(Set(actual.keys), Set(probabilities.keys))
                    for (option, value) in probabilities {
                        XCTAssertEqual(try XCTUnwrap(actual[option]), value, accuracy: 0.01, "\(name)/\(id)/\(option)")
                    }
                }
            }
            outputs[name] = result
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(outputs).write(to: directory.appending(path: "native-runtime.json"), options: .atomic)
    }
}
