import Foundation
import XCTest
import MLX
import MLXRandom
@testable import MereRunCore

final class SamplingPrefilterTests: MereRunCoreTestCase {
    /// The prefiltered top-p sampler must only ever emit tokens from the true
    /// top-p nucleus. Build a vocabulary-sized logit vector whose entire
    /// nucleus is a handful of known indices and check many draws.
    func testPrefilteredTopPSamplesStayInsideNucleus() {
        let vocabulary = 8_192
        let favored = [17, 917, 4_242, 8_000]

        var values = [Float](repeating: -20, count: vocabulary)
        for (rank, index) in favored.enumerated() {
            values[index] = 10 - Float(rank)
        }
        let logits = MLXArray(values)
        let config = GenerationConfig(temperature: 0.8, topK: 0, topP: 0.9)

        MLXRandom.seed(41)
        var seen = Set<Int>()
        for _ in 0..<48 {
            let token = sampledTokenArray(
                logits: logits,
                config: config,
                previousTokenIndices: nil,
                banMask: nil
            )
            MLX.eval(token)
            seen.insert(Int(token.item(Int32.self)))
        }

        XCTAssertTrue(
            seen.isSubset(of: Set(favored)),
            "prefiltered top-p sampled outside the nucleus: \(seen.subtracting(favored))"
        )
        XCTAssertTrue(seen.contains(favored[0]), "the dominant token should appear across 48 draws")
    }

    /// Greedy short-circuit is unaffected by the prefilter path.
    func testGreedyPicksArgmax() {
        let vocabulary = 4_096
        var values = [Float](repeating: 0, count: vocabulary)
        values[123] = 42
        let logits = MLXArray(values)
        let config = GenerationConfig(temperature: 0, topK: 0, topP: 1)

        let token = sampledTokenArray(
            logits: logits,
            config: config,
            previousTokenIndices: nil,
            banMask: nil
        )
        MLX.eval(token)
        XCTAssertEqual(Int(token.item(Int32.self)), 123)
    }

    func testMinPFiltersRelativeToMostLikelyTokenAndRenormalizes() {
        let probabilities: [Float] = [0.60, 0.20, 0.10, 0.061, 0.039]
        let logits = MLXArray(probabilities.map(log))
        let config = GenerationConfig(
            temperature: 1,
            topK: 0,
            topP: 1,
            minP: 0.1,
            repetitionPenalty: nil
        )

        let filtered = samplingProbabilities(
            logits: logits,
            config: config,
            previousTokens: []
        )
        MLX.eval(filtered)
        let values = filtered.asArray(Float.self)

        XCTAssertEqual(values[0], 0.60 / 0.961, accuracy: 0.0001)
        XCTAssertEqual(values[1], 0.20 / 0.961, accuracy: 0.0001)
        XCTAssertEqual(values[2], 0.10 / 0.961, accuracy: 0.0001)
        XCTAssertEqual(values[3], 0.061 / 0.961, accuracy: 0.0001)
        XCTAssertEqual(values[4], 0, accuracy: 0.0001)
        XCTAssertEqual(filtered.sum().item(Float.self), 1, accuracy: 0.0001)
    }

    func testMinPComposesWithTopPWithoutReintroducingFilteredTokens() {
        let probabilities: [Float] = [0.55, 0.25, 0.10, 0.06, 0.04]
        let logits = MLXArray(probabilities.map(log))
        let config = GenerationConfig(
            temperature: 1,
            topK: 0,
            topP: 0.95,
            minP: 0.2,
            repetitionPenalty: nil,
            topPPrefilter: 0
        )

        let filtered = samplingProbabilities(
            logits: logits,
            config: config,
            previousTokens: []
        )
        MLX.eval(filtered)
        let values = filtered.asArray(Float.self)

        XCTAssertGreaterThan(values[0], 0)
        XCTAssertGreaterThan(values[1], 0)
        XCTAssertEqual(values[2], 0, accuracy: 0.0001)
        XCTAssertEqual(values[3], 0, accuracy: 0.0001)
        XCTAssertEqual(values[4], 0, accuracy: 0.0001)
        XCTAssertEqual(filtered.sum().item(Float.self), 1, accuracy: 0.0001)
    }

    func testMinPDoesNotChangeGreedySampling() {
        let logits = MLXArray([Float(0.1), 0.3, 4.0, 1.0])
        let config = GenerationConfig(
            temperature: 0,
            topK: 1,
            topP: 0.1,
            minP: 1,
            repetitionPenalty: nil
        )

        XCTAssertEqual(
            sampleToken(logits: logits, config: config, previousTokens: []),
            2
        )
        XCTAssertEqual(
            sampledTokenArray(
                logits: logits,
                config: config,
                previousTokenIndices: nil,
                banMask: nil
            ).item(Int.self),
            2
        )
    }

    func testGPUAndProbabilitySamplersRespectSameMinPSupport() {
        let logits = MLXArray([Float(4), 2, 1, 0, -1])
        let config = GenerationConfig(
            temperature: 1,
            topK: 0,
            topP: 1,
            minP: 0.1,
            repetitionPenalty: nil
        )
        let probabilities = samplingProbabilities(
            logits: logits,
            config: config,
            previousTokens: []
        )
        MLX.eval(probabilities)
        let allowed = Set(probabilities.asArray(Float.self).enumerated().compactMap {
            $0.element > 0 ? $0.offset : nil
        })

        MLXRandom.seed(97)
        var sampled = Set<Int>()
        for _ in 0..<64 {
            sampled.insert(sampledTokenArray(
                logits: logits,
                config: config,
                previousTokenIndices: nil,
                banMask: nil
            ).item(Int.self))
        }

        XCTAssertFalse(allowed.isEmpty)
        XCTAssertTrue(sampled.isSubset(of: allowed))
    }

    func testCategoricalLogitsKeepFilteredTokensAtZeroMass() {
        let logits = categoricalLogits(
            probabilities: MLXArray([Float(0.75), 0, 0.25, 0])
        )
        MLX.eval(logits)
        let values = logits.asArray(Float.self)

        XCTAssertEqual(values[0], log(0.75), accuracy: 0.0001)
        XCTAssertTrue(values[1].isInfinite && values[1] < 0)
        XCTAssertEqual(values[2], log(0.25), accuracy: 0.0001)
        XCTAssertTrue(values[3].isInfinite && values[3] < 0)
    }

    func testTopKPartitionMatchesExactSortedThreshold() {
        let logits = MLXArray([Float(0.1), 7, -2, 3, 9, 2, 8, 1])

        let filtered = applyingTopK(logits, topK: 3)
        MLX.eval(filtered)
        let values = filtered.asArray(Float.self)

        XCTAssertEqual(values[1], 7)
        XCTAssertEqual(values[4], 9)
        XCTAssertEqual(values[6], 8)
        XCTAssertTrue(values[0].isInfinite && values[0] < 0)
        XCTAssertTrue(values[3].isInfinite && values[3] < 0)
    }

    func testBFloat16TopKPartitionPreservesSortedThresholdAndTies() {
        let logits = MLXArray([
            Float(0.992), 1.0001, 1.0002, 0.9999, -4, 0.5, 1.25, 3
        ]).asType(.bfloat16)
        let topK = 3

        let partitioned = applyingTopK(logits, topK: topK)
        let sortedIndices = argSort(logits, axis: -1)
        let sortedLogits = logits.take(sortedIndices, axis: -1)
        let threshold = sortedLogits[sortedLogits.dim(-1) - topK]
        let reference = MLX.where(
            logits .< threshold,
            MLXArray(-Float.infinity),
            logits
        )
        MLX.eval(partitioned, reference)

        XCTAssertEqual(
            partitioned.asType(.float32).asArray(Float.self),
            reference.asType(.float32).asArray(Float.self)
        )
    }

    func testPerRequestExactTopPBypassesCandidatePrefilter() {
        let vocabulary = 1_024
        let diffuseCandidates = 600
        let logits = MLXArray(
            [Float](repeating: 0, count: diffuseCandidates)
                + [Float](repeating: -100, count: vocabulary - diffuseCandidates)
        )
        let partitioned = argPartition(logits, kth: vocabulary - 256, axis: -1)
        let prefilteredSet = Set(
            partitioned[(vocabulary - 256)...].asArray(Int32.self).map(Int.init)
        )
        let exactConfig = GenerationConfig(
            temperature: 1,
            topK: 0,
            topP: 0.9,
            repetitionPenalty: nil,
            topPPrefilter: 0
        )

        MLXRandom.seed(73)
        var sampledOutsidePrefilter = false
        for _ in 0..<64 {
            let token = sampledTokenArray(
                logits: logits,
                config: exactConfig,
                previousTokenIndices: nil,
                banMask: nil
            ).item(Int.self)
            sampledOutsidePrefilter = sampledOutsidePrefilter || !prefilteredSet.contains(token)
        }

        XCTAssertTrue(sampledOutsidePrefilter)
    }
}
