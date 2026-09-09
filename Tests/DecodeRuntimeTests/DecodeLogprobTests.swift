import Foundation
import MLX
import MereRunDecode
import MereRunMLXTestSupport
import XCTest

final class DecodeLogprobTests: MLXTestCase {
    func testMeasurementSeparatesRawModelAndFilteredPolicy() {
        let measurement = tokenLogprobMeasurement(
            logits: MLXArray([Float(0), 0, 0, 0]), selectedToken: 2,
            config: GenerationConfig(temperature: 1, topP: 1, repetitionPenalty: nil, bannedTokens: [0, 1]),
            previousTokens: [], topLogprobs: 2
        )
        XCTAssertEqual(measurement.rawLogprob, log(0.25), accuracy: 0.00001)
        XCTAssertEqual(measurement.policyLogprob, log(0.5), accuracy: 0.00001)
        XCTAssertEqual(measurement.rawEntropy, log(4), accuracy: 0.00001)
        XCTAssertEqual(measurement.policyEntropy, log(2), accuracy: 0.00001)
        XCTAssertEqual(Set(measurement.topLogprobs.map(\.tokenID)), [2, 3])
    }

    func testDiagnosticsKeepSummaryAndRedactedTokensThroughCodableRoundTrip() throws {
        let reasoning = ChatTokenLogprob(
            tokenID: 7, region: .reasoning, rawLogprob: -2, policyLogprob: -1,
            rawEntropy: 3, policyEntropy: 2, rawTop1Top2Margin: 0.2, policyTop1Top2Margin: 0.4
        )
        let visible = ChatTokenLogprob(
            tokenID: 8, token: "answer", region: .visible, rawLogprob: -1, policyLogprob: -0.5,
            rawEntropy: 2, policyEntropy: 1, rawTop1Top2Margin: 0.4, policyTop1Top2Margin: 0.8
        )
        let diagnostics = ChatLogprobDiagnostics(
            capture: .tokens, measuredTokens: [reasoning, visible], captureSeconds: 0.25
        )
        let restored = try JSONDecoder().decode(
            ChatLogprobDiagnostics.self, from: JSONEncoder().encode(diagnostics)
        )
        XCTAssertEqual(restored, diagnostics)
        XCTAssertEqual(restored.source.rawValue, "final_target")
        XCTAssertEqual(restored.summary.tokenCount, 2)
        XCTAssertEqual(restored.summary.meanRawLogprob, -1.5)
        XCTAssertEqual(restored.summary.minimumRawLogprob, -2)
        XCTAssertNil(restored.tokens?.first?.token)
        XCTAssertEqual(restored.tokens?.last?.token, "answer")

        let summary = ChatLogprobDiagnostics(
            capture: .summary, measuredTokens: [reasoning, visible], captureSeconds: 0.25
        )
        XCTAssertNil(summary.tokens)
        XCTAssertEqual(summary.summary, diagnostics.summary)
    }

    func testEmptySummaryContainsFiniteZeroMeasurements() {
        let summary = ChatLogprobSummary(tokens: [])
        XCTAssertEqual(summary.tokenCount, 0)
        XCTAssertEqual(summary.meanRawLogprob, 0)
        XCTAssertEqual(summary.minimumPolicyLogprob, 0)
        XCTAssertEqual(summary.meanRawEntropy, 0)
        XCTAssertEqual(summary.meanPolicyEntropy, 0)
    }

    func testCaptureClampsTopCandidateCount() {
        XCTAssertEqual(ChatLogprobCapture.top(0).topLogprobs, 1)
        XCTAssertEqual(ChatLogprobCapture.top(5).topLogprobs, 5)
        XCTAssertEqual(ChatLogprobCapture.top(100).topLogprobs, 20)
        XCTAssertFalse(ChatLogprobCapture.summary.includesTokens)
        XCTAssertTrue(ChatLogprobCapture.tokens.includesTokens)
    }

    func testPolicyProbabilitiesPreserveExactTopPBoundaryWithTies() {
        let probabilities: [Float] = [0.09, 0.09, 0.09, 0.73]
        let logits = MLXArray(probabilities.map(log))
        let config = GenerationConfig(
            maxTokens: 1,
            temperature: 1,
            topK: 0,
            topP: 0.8,
            repetitionPenalty: nil,
            topPPrefilter: 0
        )

        let policy = samplingProbabilities(
            logits: logits,
            config: config,
            previousTokens: []
        )
        MLX.eval(policy)
        let nonzero = policy.asArray(Float.self).filter { $0 > 0 }

        XCTAssertEqual(nonzero.count, 2)
        XCTAssertEqual(policy.sum().item(Float.self), 1, accuracy: 0.0001)
    }
}
