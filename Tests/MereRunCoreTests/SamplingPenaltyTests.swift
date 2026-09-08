import MLX
import XCTest
@testable import MereRunCore

final class SamplingPenaltyTests: MereRunCoreTestCase {
    func testPresenceCountsOnceAndFrequencyCountsOccurrencesExcludingPrompt() {
        let logits = MLXArray([Float(5), 4, 3, 2])
        let config = GenerationConfig(
            repetitionPenalty: nil, presencePenalty: 1.5, frequencyPenalty: 0.5,
            penaltyPromptTokenCount: 2
        )
        let history = MLXArray([Int32(0), 0, 1, 1, 2])
        let result = applyingSamplingPenalties(logits, config: config, history: history)
        XCTAssertEqual(result.asArray(Float.self), [5, 1.5, 1, 2])
        XCTAssertEqual(logits.asArray(Float.self), [5, 4, 3, 2])
    }

    func testRepetitionIsSignAwareAndDoesNotCompoundDuplicateTokensOrMutateLogits() {
        let logits = MLXArray([Float(4), -2, 3, 0])
        let config = GenerationConfig(repetitionPenalty: 2, repetitionContextSize: Int.max)
        let history = MLXArray([Int32(0), 0, 1])
        let first = applyingSamplingPenalties(logits, config: config, history: history)
        let second = applyingSamplingPenalties(logits, config: config, history: history)
        XCTAssertEqual(first.asArray(Float.self), [2, -4, 3, 0])
        XCTAssertEqual(second.asArray(Float.self), [2, -4, 3, 0])
        XCTAssertEqual(logits.asArray(Float.self), [4, -2, 3, 0])
    }

    func testPenaltiesPrecedeTopKAndMatchGreedyAndPolicySamplers() {
        let logits = MLXArray([Float(4), 3, 2])
        let config = GenerationConfig(
            temperature: 0, topK: 1, topP: 1, repetitionPenalty: nil, presencePenalty: 2
        )
        let history = [0]
        XCTAssertEqual(sampleToken(logits: logits, config: config, previousTokens: history), 1)
        XCTAssertEqual(greedySampleTokenArray(logits: logits, config: config, previousTokens: history).item(Int.self), 1)
        XCTAssertEqual(sampledTokenArray(
            logits: logits, config: config,
            previousTokenIndices: MLXArray(history.map(Int32.init)), banMask: nil
        ).item(Int.self), 1)
        var stochastic = config
        stochastic.temperature = 0.6
        XCTAssertEqual(samplingProbabilities(
            logits: logits, config: stochastic, previousTokens: history
        ).asArray(Float.self), [0, 1, 0])
        XCTAssertEqual(sampleToken(logits: logits, config: stochastic, previousTokens: history), 1)
    }

    func testPipelinedDecodePenalizesGeneratedTokensAndResetsBetweenRequests() throws {
        let config = GenerationConfig(
            temperature: 0, topP: 1, repetitionPenalty: nil,
            presencePenalty: 2, penaltyPromptTokenCount: 2
        )
        for _ in 0..<2 {
            let request = AutoregressiveDecodeRequest(
                initialLogits: MLXArray([Float(4), 3.5, 3]).reshaped(1, 1, 3),
                generationConfig: config, eosTokens: [], tokenBudget: 3, historySeedTokens: [0, 0]
            )
            let result = try AutoregressiveDecodeEngine.decode(request, stepForward: { _ in
                MLXArray([Float(4), 3.5, 3]).reshaped(1, 1, 3)
            })
            XCTAssertEqual(result.generatedTokens, [0, 1, 2])
        }
    }

    func testEmptyPromptHistoryAndNegativePresencePenalty() {
        let config = GenerationConfig(repetitionPenalty: nil, presencePenalty: -1)
        let empty = repetitionHistoryArray(promptTokens: [], config: config)
        XCTAssertNil(empty)
        let history = appendingRepetitionHistory(empty, token: MLXArray(Int32(1)), config: config)
        let result = applyingSamplingPenalties(MLXArray([Float(1), 1]), config: config, history: history)
        XCTAssertEqual(result.asArray(Float.self), [1, 2])
    }
}
