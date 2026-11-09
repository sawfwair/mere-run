import Foundation
import MLX
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunDecode

final class SamplingDistributionTests: MLXTestCase {
    func testUnfilteredBF16MatchesCategoricalFloatPromotion() {
        let logits = MLXArray([Float(1.234), -0.9, 2.17, 0.32]).asType(.bfloat16)
        let config = GenerationConfig(temperature: 0.7, topP: 1, repetitionPenalty: nil)
        let probabilities = samplingDistribution(logits: logits, config: config, previousTokenIndices: nil, banMask: nil)
        let expected = softmax((logits / config.temperature).asType(.float32))
        XCTAssertEqual(probabilities.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(probabilities.sum().item(Float.self), 1, accuracy: 1e-6)
    }

    func testRecommendedProfilesAndPenaltiesMatchIndependentFullDistribution() {
        let logits = MLXArray((0..<64).map { Float($0) / 16 }).asType(.bfloat16)
        let history = [1, 2, 60, 60, 62]
        for thinking in [false, true] {
            let config = GenerationConfig(
                temperature: thinking ? 1 : 0.7, topK: 20, topP: thinking ? 0.95 : 0.8,
                repetitionPenalty: 1.1, repetitionContextSize: 4,
                presencePenalty: thinking ? 0 : 1.5, frequencyPenalty: 0.25,
                penaltyPromptTokenCount: 2, bannedTokens: [63]
            )
            let probabilities = samplingDistribution(
                logits: logits, config: config,
                previousTokenIndices: MLXArray(history.map(Int32.init)),
                banMask: tokenBanMask(vocabularySize: 64, dtype: logits.dtype, tokens: config.bannedTokens)
            )
            let reference = samplingProbabilities(logits: logits, config: config, previousTokens: history)
            XCTAssertLessThan((probabilities - reference).abs().max().item(Float.self), 1e-6)
            XCTAssertEqual(probabilities[63].item(Float.self), 0)
            XCTAssertEqual(probabilities.sum().item(Float.self), 1, accuracy: 1e-6)
            for _ in 0..<16 {
                let token = sampledTokenArray(
                    logits: logits, config: config, previousTokenIndices: MLXArray(history.map(Int32.init)),
                    banMask: tokenBanMask(vocabularySize: 64, dtype: logits.dtype, tokens: config.bannedTokens)
                ).item(Int.self)
                XCTAssertGreaterThan(probabilities[token].item(Float.self), 0)
            }
        }
    }

    func testPrefilterAndMinPMatchSamplerSupport() {
        let logits = MLXArray((0..<1024).map { Float($0) / 1024 })
        for limit in [0, 32] {
            let config = GenerationConfig(
                temperature: 1, topP: 0.95, minP: 0.8, repetitionPenalty: nil, topPPrefilter: limit
            )
            let probabilities = samplingDistribution(logits: logits, config: config, previousTokenIndices: nil, banMask: nil)
            XCTAssertEqual(probabilities.sum().item(Float.self), 1, accuracy: 1e-6)
            for _ in 0..<32 {
                let token = sampledTokenArray(logits: logits, config: config, previousTokenIndices: nil, banMask: nil)
                    .item(Int.self)
                XCTAssertGreaterThan(probabilities[token].item(Float.self), 0)
            }
            if limit > 0 { XCTAssertEqual(probabilities[0..<992].sum().item(Float.self), 0) }
        }
    }
}
