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
}
