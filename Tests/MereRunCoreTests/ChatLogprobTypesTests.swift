import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ChatLogprobTypesTests: XCTestCase {
    func testOpenAILogprobsExposeRawAndPolicyWithoutReasoningText() throws {
        let visible = ChatTokenLogprob(
            tokenID: 7,
            token: "answer",
            region: .visible,
            rawLogprob: -0.2,
            policyLogprob: -0.1,
            rawEntropy: 1.2,
            policyEntropy: 0.8,
            rawTop1Top2Margin: 0.4,
            policyTop1Top2Margin: 0.7,
            topLogprobs: [
                ChatTopLogprob(
                    tokenID: 7,
                    token: "answer",
                    rawLogprob: -0.2,
                    policyLogprob: -0.1
                ),
            ]
        )
        let reasoning = ChatTokenLogprob(
            tokenID: 8,
            token: nil,
            region: .reasoning,
            rawLogprob: -1,
            policyLogprob: -0.9,
            rawEntropy: 2,
            policyEntropy: 1.8,
            rawTop1Top2Margin: 0.1,
            policyTop1Top2Margin: 0.2
        )
        let diagnostics = ChatLogprobDiagnostics(
            capture: .top(1),
            measuredTokens: [visible, reasoning],
            captureSeconds: 0.03
        )
        let encoded = try JSONEncoder().encode(OpenAIChatLogprobs(diagnostics))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])

        XCTAssertEqual(content.count, 1)
        XCTAssertEqual(content[0]["token"] as? String, "answer")
        XCTAssertEqual(content[0]["raw_logprob"] as? Double, -0.2)
        XCTAssertEqual(content[0]["policy_logprob"] as? Double, -0.1)
        XCTAssertEqual((object["mere_summary"] as? [String: Any])?["tokenCount"] as? Int, 2)
    }

}
