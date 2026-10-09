import Foundation
import XCTest
@testable import MereRunCore

/// Original trained weights against independently exported, single-question PyTorch answers.
final class D1ReleasedCheckpointTests: MereRunCoreTestCase {
    private struct Reference: Decodable {
        let family: String
        let media: String
        let answers: [String: Answer]
        let usage: Usage
        struct Usage: Decodable { let input_tokens: Int; let output_tokens: Int }
        struct Answer: Decodable {
            let type: String
            let noul: Double?
            let choice: String?
            let score: Double?
            let confidence: Double?
            let probabilities: [String: Double]?
        }
    }
    func testOriginalReleasedCheckpointProbabilities() throws {
        guard let directory = ProcessInfo.processInfo.environment["MERERUN_TEST_D1_RELEASED_CHECKPOINTS"] else {
            throw XCTSkip("Set MERERUN_TEST_D1_RELEASED_CHECKPOINTS to a directory with pinned causal/ and omni/ checkpoints.")
        }
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil)).appending(path: "D1/released")
        for family in ["omni", "causal"] {
            let root = URL(fileURLWithPath: directory).appending(path: family)
            let operation = try D1DecisionOperation(root: root, modelID: family)
            defer { operation.unload() }
            let tolerance = family == "omni" ? 0.0001 : 0.02
            for media in family == "omni" ? ["text", "image", "audio"] : ["text", "image"] {
                let reference = try JSONDecoder().decode(Reference.self,
                    from: Data(contentsOf: fixture.appending(path: "reference-\(family)-\(media).json")))
                XCTAssertEqual(reference.family, family)
                XCTAssertEqual(reference.media, media)
                let input = try D1DecisionRequest.decode(Data(contentsOf: fixture.appending(path: "\(media).json")))
                let request = D1DecisionRequest(state: input.state, questions: input.questions,
                    images: input.images.map { fixture.appending(path: $0).path },
                    audio: input.audio.map { fixture.appending(path: $0).path }, maxTokens: input.maxTokens)
                let plan = try operation.prepare(request)
                let result = try operation.predict(request)
                XCTAssertEqual(result.usage.input_tokens, reference.usage.input_tokens)
                XCTAssertEqual(result.usage.input_tokens, plan.inputTokens)
                XCTAssertEqual(result.usage.output_tokens, reference.usage.output_tokens)
                XCTAssertEqual(Set(result.answers.keys), Set(reference.answers.keys))
                for (name, expected) in reference.answers {
                    let actual = try XCTUnwrap(result.answers[name])
                    let context = "\(family)/\(media)/\(name)"
                    XCTAssertEqual(actual.type, expected.type, context)
                    XCTAssertEqual(actual.choice, expected.choice, context)
                    for (value, target) in [(actual.noul, expected.noul), (actual.score, expected.score),
                                            (actual.confidence, expected.confidence)] {
                        if let target { XCTAssertEqual(try XCTUnwrap(value), target, accuracy: tolerance, context) }
                    }
                    if let probabilities = expected.probabilities {
                        let values = try XCTUnwrap(actual.probabilities)
                        XCTAssertEqual(Set(values.keys), Set(probabilities.keys), context)
                        for (option, target) in probabilities {
                            XCTAssertEqual(try XCTUnwrap(values[option]), target, accuracy: tolerance, "\(context)/\(option)")
                        }
                    }
                }
            }
        }
    }
}
