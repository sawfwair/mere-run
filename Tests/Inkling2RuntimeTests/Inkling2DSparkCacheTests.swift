import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunInkling2Model

final class Inkling2DSparkCacheTests: MLXTestCase {
    func testAdaptiveDepthUsesReachedPositionsAndLeavesOverridesFixed() {
        var low = Inkling2DSparkDepth(maximum: 7, fixedDepth: nil)
        for _ in 0..<4 { low.observe(proposed: 3, accepted: 0) }
        XCTAssertEqual(low.depth, 2)
        for _ in 0..<4 { low.observe(proposed: 2, accepted: 0) }
        XCTAssertEqual(low.depth, 1)

        var high = Inkling2DSparkDepth(maximum: 4, fixedDepth: nil)
        for _ in 0..<4 { high.observe(proposed: 3, accepted: 3) }
        XCTAssertEqual(high.depth, 4)
        for _ in 0..<4 { high.observe(proposed: 4, accepted: 4) }
        XCTAssertEqual(high.depth, 4)

        var fixed = Inkling2DSparkDepth(maximum: 7, fixedDepth: 5)
        for _ in 0..<8 { fixed.observe(proposed: 5, accepted: 0) }
        XCTAssertEqual(fixed.depth, 5)
    }

    func testSampledVerificationUsesFirstRejectionOrTargetBonus() {
        let proposed = MLXArray([Int32(0), 1])
        let draft = MLXArray([Float(0.8), 0.1, 0.1, 0.2, 0.7, 0.1]).reshaped(2, 3)
        let target = MLXArray([Float(0.4), 0.3, 0.3, 0, 0.5, 0.5, 0.1, 0.2, 0.7]).reshaped(3, 3)
        let cases: [([Float], Int, [Float])] = [
            ([0.75, 0], 0, [0, 0.2, 0.2]),
            ([0.25, 0.9], 1, [0, 0, 0.4]),
            ([0.25, 0.5], 2, [0.1, 0.2, 0.7])
        ]
        for (draws, expected, mass) in cases {
            let (accepted, weights) = Inkling2DSparkVerification.checkSampled(
                proposals: proposed, target: target, draft: draft, uniforms: MLXArray(draws)
            )
            XCTAssertEqual(accepted.item(Int.self), expected)
            for (actual, expected) in zip(weights.asArray(Float.self), mass) {
                XCTAssertEqual(actual, expected, accuracy: 1e-6)
            }
        }
    }

    func testLookupRequiresTwoAgreeingContinuations() throws {
        let pattern = Array(1...24)
        let suffix = Array(pattern.prefix(8))
        let history = pattern + pattern + pattern + suffix
        let match = try XCTUnwrap(Inkling2DSparkLookup.match(in: history, limit: 16))
        XCTAssertEqual(Array(history[match]), Array(9...24))
        XCTAssertNil(Inkling2DSparkLookup.match(in: pattern + suffix, limit: 16))
        XCTAssertNil(Inkling2DSparkLookup.match(in: history, limit: 7))
        let different = pattern + suffix + Array(30...45) + suffix
        XCTAssertNil(Inkling2DSparkLookup.match(in: different, limit: 16))
    }

    func testLookupCorrectionMatchesOneHotDraft() {
        let proposals = MLXArray([Int32(1), 0])
        let target = MLXArray([Float(0.3), 0.6, 0.1, 0.3, 0.5, 0.2, 0.1, 0.3, 0.6]).reshaped(3, 3)
        let draft = MLXArray([Float(0), 1, 0, 1, 0, 0]).reshaped(2, 3)
        let draws: [[Float]] = [[0.8, 0.2], [0.5, 0.9], [0.5, 0.2]]
        for draw in draws {
            let uniforms = MLXArray(draw)
            let lookup = Inkling2DSparkVerification.checkDeterministic(proposals: proposals, target: target, uniforms: uniforms)
            let sampled = Inkling2DSparkVerification.checkSampled(proposals: proposals, target: target, draft: draft, uniforms: uniforms)
            XCTAssertEqual(lookup.0.item(Int32.self), sampled.0.item(Int32.self))
            XCTAssertEqual(lookup.1.asArray(Float.self), sampled.1.asArray(Float.self))
        }
    }

    func testLastDraftLayerKeepsFullAttentionContextForProposalRows() throws {
        let config = Inkling2DSparkConfiguration(
            hiddenSize: 64, numHiddenLayers: 1, numAttentionHeads: 2, numKeyValueHeads: 1, headDim: 32,
            intermediateSize: 96, rmsNormEps: 1e-6, ropeTheta: 10_000, localLayerIds: [0], slidingWindowSize: 8,
            blockSize: 8, markovRank: 16, markovHeadType: "rnn_hidden", targetLayerIds: [0], qBias: false, oBias: false
        )
        func values(_ shape: [Int], stride: Int) -> MLXArray {
            MLXArray((0..<shape.reduce(1, *)).map { Float(($0 * stride) % 107 - 53) / 256 })
                .reshaped(shape).asType(.bfloat16)
        }
        var weights: [String: MLXArray] = [:]
        for (stride, (name, shape)) in [
            ("self_attn.q_proj", [64, 64]), ("self_attn.k_proj", [32, 64]), ("self_attn.v_proj", [32, 64]),
            ("self_attn.k_proj_ctx", [32, 64]), ("self_attn.v_proj_ctx", [32, 64]), ("self_attn.o_proj", [64, 64]),
            ("mlp.gate_proj", [96, 64]), ("mlp.up_proj", [96, 64]), ("mlp.down_proj", [64, 96]),
        ].enumerated() {
            weights["layers.0.\(name).weight"] = values(shape, stride: 2 * stride + 3)
        }
        for name in ["self_attn.q_norm", "self_attn.k_norm", "self_attn.k_norm_ctx"] {
            weights["layers.0.\(name).weight"] = MLXArray.ones([32], dtype: .bfloat16)
        }
        for name in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            weights["layers.0.\(name).weight"] = MLXArray.ones([64], dtype: .bfloat16)
        }
        let layer = try Inkling2DSparkLayer(weights: .init(arrays: weights), index: 0, config: config)
        var cache = Inkling2DSparkLayerCache()
        layer.append(values([1, 6, 64], stride: 29), offset: 11, cache: &cache)
        let input = values([1, 8, 64], stride: 31)
        let full = layer.forward(input, offset: 17, cache: cache)
        for count in [1, 3, 7] {
            let rows = 1..<(count + 1)
            let proposals = layer.forward(input, offset: 17, cache: cache, queries: rows)
            XCTAssertEqual(proposals.shape, [1, count, 64])
            XCTAssertTrue(allClose(proposals, full[0..., rows, 0...], rtol: 0.005, atol: 0.005).item(Bool.self))
        }
    }

    func testPartialVerificationCommitsAttentionAndConvolutionPrefix() throws {
        let cache = Inkling2Cache(layerCount: 2)
        let layers = cache.layers
        let prompt = MLXArray([Float(1), 2, 3, 4])
        let verification = MLXArray([Float(5), 6, 7, 8])
        let correction = MLXArray([Float(9)])
        for (index, layer) in layers.enumerated() {
            let tokens = prompt.reshaped(1, 1, 4, 1)
            _ = layer.update(keys: tokens, values: tokens * 10, window: index == 0 ? 4 : nil)
            _ = layer.context(prompt.reshaped(1, 4, 1), index: 0, kernel: 3)
        }

        cache.beginVerification()
        for (index, layer) in layers.enumerated() {
            let tokens = verification.reshaped(1, 1, 4, 1)
            _ = layer.update(keys: tokens, values: tokens * 10, window: index == 0 ? 4 : nil)
            _ = layer.context(verification.reshaped(1, 4, 1), index: 0, kernel: 3)
        }
        cache.commitVerification(2)

        let local = layers[0], global = layers[1]
        XCTAssertEqual(cache.offset, 6)
        XCTAssertEqual(global.offset, 6)
        XCTAssertEqual(try XCTUnwrap(local.keys).asArray(Float.self), [4, 5, 6])
        XCTAssertEqual(try XCTUnwrap(local.values).asArray(Float.self), [40, 50, 60])
        XCTAssertEqual(try XCTUnwrap(global.keys).asArray(Float.self), [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(local.context(correction.reshaped(1, 1, 1), index: 0, kernel: 3).asArray(Float.self), [5, 6, 9])
        XCTAssertEqual(global.context(correction.reshaped(1, 1, 1), index: 0, kernel: 3).asArray(Float.self), [5, 6, 9])
    }
}
