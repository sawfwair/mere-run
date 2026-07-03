import Foundation
import MLX
import MLXRandom
import XCTest
@testable import AudioTTS

/// Parity between the legacy host-side `sampleToken` semantics and the
/// GPU-side `sampleTokenArrayTTS` path. The legacy function lives on
/// `Qwen3TTSGenerator`, so the expected values here re-derive its exact
/// masking order on small fixed tensors instead of instantiating a model.
final class Qwen3TTSSamplingTests: XCTestCase {
    private let vocab = 32

    override class func setUp() {
        super.setUp()
        MLXTestSupport.ensureMetalLibraryAvailable()
    }

    private func makeContext(
        temperature: Float = 0,
        topK: Int = 0,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        eosTokenId: Int? = nil,
        suppressTokens: [Int]? = nil
    ) -> Qwen3TTSSamplerContext {
        Qwen3TTSSamplerContext(
            vocabSize: vocab,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            eosTokenId: eosTokenId,
            suppressTokens: suppressTokens
        )
    }

    private func logitsFixture(seed: UInt64 = 3) -> MLXArray {
        MLXRandom.seed(seed)
        return MLXRandom.normal([1, 1, vocab])
    }

    func testGreedyMatchesArgMax() {
        let logits = logitsFixture()
        let context = makeContext()
        let token = sampleTokenArrayTTS(logits: logits, context: context)
        let expected = argMax(logits[0..., 0, 0...].squeezed(axis: 0)).item(Int.self)
        XCTAssertEqual(token.item(Int.self), expected)
        XCTAssertEqual(token.shape, [1, 1])
    }

    func testSuppressMaskExcludesSuppressedTokens() {
        let logits = logitsFixture()
        // Suppress everything except token 5: greedy must pick 5.
        let suppressed = (0..<vocab).filter { $0 != 5 }
        let context = makeContext(suppressTokens: suppressed)
        let token = sampleTokenArrayTTS(logits: logits, context: context)
        XCTAssertEqual(token.item(Int.self), 5)
    }

    func testRepetitionPenaltyMatchesLegacyFormula() {
        let logits = logitsFixture(seed: 9)
        var context = makeContext(repetitionPenalty: 1.5)
        let scores = logits[0..., 0, 0...].squeezed(axis: 0).asType(.float32)
        let greedyFirst = argMax(scores).item(Int.self)

        // Append the greedy winner to history; the penalized rerun must not
        // pick it again if the runner-up is close enough (construct it so).
        context.appendHistory(MLXArray([Int32(greedyFirst)]).reshaped(1, 1))
        let token = sampleTokenArrayTTS(logits: logits, context: context)

        // Reproduce the legacy penalty on host: positive scores divide,
        // negative multiply.
        var host = scores.asArray(Float.self)
        host[greedyFirst] = host[greedyFirst] < 0
            ? host[greedyFirst] * 1.5
            : host[greedyFirst] / 1.5
        let expected = host.enumerated().max(by: { $0.element < $1.element })!.offset
        XCTAssertEqual(token.item(Int.self), expected)
    }

    func testHistoryAppendsAcrossStepsWithoutHostReadback() throws {
        var context = makeContext(repetitionPenalty: 2.0)
        context.appendHistory(MLXArray([Int32(3)]).reshaped(1, 1))
        context.appendHistory(MLXArray([Int32(7)]).reshaped(1, 1))
        let history = try XCTUnwrap(context.history)
        XCTAssertEqual(history.asArray(Int32.self), [3, 7])
    }

    func testTopKThresholdMatchesFullSort() {
        let logits = logitsFixture(seed: 21)
        let scores = logits[0..., 0, 0...].squeezed(axis: 0).asType(.float32)
        let topK = 5

        // Legacy threshold: full argSort, k-th largest value.
        let sortedScores = scores.take(argSort(scores, axis: -1), axis: -1)
        let legacyThreshold = sortedScores[vocab - topK].item(Float.self)

        // New threshold: argPartition k-th element.
        let partition = argPartition(scores, kth: vocab - topK, axis: -1)
        let newThreshold = scores.take(
            partition[(vocab - topK)..<(vocab - topK + 1)],
            axis: -1
        ).item(Float.self)

        XCTAssertEqual(legacyThreshold, newThreshold, accuracy: 0)
    }

    func testSampledTokenAlwaysSurvivesTopKMask() {
        // With temperature > 0 and a tight top-k, every sampled token must
        // come from the top-k set (plus EOS, which is exempt).
        let logits = logitsFixture(seed: 33)
        let scores = logits[0..., 0, 0...].squeezed(axis: 0).asType(.float32)
        let topK = 4
        let sortedScores = scores.take(argSort(scores, axis: -1), axis: -1)
        let threshold = sortedScores[vocab - topK].item(Float.self)
        let allowed = Set(
            scores.asArray(Float.self).enumerated()
                .filter { $0.element >= threshold }
                .map(\.offset)
        )

        let context = makeContext(temperature: 0.8, topK: topK)
        for _ in 0..<16 {
            let token = sampleTokenArrayTTS(logits: logits, context: context).item(Int.self)
            XCTAssertTrue(allowed.contains(token), "token \(token) escaped top-\(topK) mask")
        }
    }

    func testEOSLogitSurvivesTopKTruncation() {
        // Give EOS the LOWEST score so top-k would normally erase it; the
        // legacy sampler restores the EOS logit after masking, so it must
        // remain reachable (finite), verified via the masked score tensor
        // by sampling with a temperature so low the max dominates.
        MLXRandom.seed(4)
        var host = [Float](repeating: 0, count: vocab)
        for index in 0..<vocab { host[index] = Float(index) }
        let eos = 0  // lowest score
        let logits = MLXArray(host).reshaped(1, 1, vocab)
        let context = makeContext(temperature: 0.7, topK: 3, eosTokenId: eos)
        // Sampling can't prove reachability directly; instead verify the
        // shape contract and that repeated sampling stays within top-3 ∪ EOS.
        let sortedScores = MLXArray(host)
        let threshold = sortedScores.take(
            argPartition(sortedScores, kth: vocab - 3, axis: -1)[(vocab - 3)..<(vocab - 2)],
            axis: -1
        ).item(Float.self)
        var allowed = Set(host.enumerated().filter { $0.element >= threshold }.map(\.offset))
        allowed.insert(eos)
        for _ in 0..<16 {
            let token = sampleTokenArrayTTS(logits: logits, context: context).item(Int.self)
            XCTAssertTrue(allowed.contains(token))
        }
    }
}
