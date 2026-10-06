import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunKolibriModel

final class KolibriRuntimeTests: MLXTestCase {
    func configuration(quantization: String = "{}") throws -> KolibriConfiguration {
        try JSONDecoder().decode(KolibriConfiguration.self, from: Data("""
        {"model_type":"kolibri1","hidden_size":64,"num_hidden_layers":2,
         "num_attention_heads":6,"num_key_value_heads":2,"head_dim":16,
         "max_position_embeddings":128,"rms_norm_eps":0.000001,"vocab_size":97,
         "rope_theta":10000,"num_experts":7,"num_experts_per_tok":3,
         "moe_intermediate_size":64,"shared_expert_intermediate_size":64,
         "norm_topk_prob":false,"sliding_window":5,
         "layer_types":["sliding_attention","full_attention"],"eos_token_id":96,
         "mererun_quantization":\(quantization)}
        """.utf8))
    }

    func testRouterSelectsWithLogitBiasButWeightsUnbiasedSigmoid() throws {
        let router = KolibriRouter(config: try configuration())
        let logits: [Float] = [-2, -1, 0, 1, 2, 3, 4]
        try router.update(parameters: .unflattened([
            "weight": concatenated([MLXArray(logits).reshaped(7, 1), MLXArray.zeros([7, 63])], axis: 1),
            "e_score_correction_bias": MLXArray([Float(10), 10, 10, 0, 0, 0, 0])
        ]), verify: [.allModelKeysSet, .shapeMismatch])
        let input = MLXArray([Float(1)] + [Float](repeating: 0, count: 63)).reshaped(1, 1, 64)
        let result = router(input)
        let ids = result.indices.asArray(Int32.self)
        let weights = result.weights.asArray(Float.self)
        XCTAssertEqual(Set(ids), Set([0, 1, 2]))
        for (id, weight) in zip(ids, weights) {
            XCTAssertEqual(weight, 1 / (1 + exp(-logits[Int(id)])), accuracy: 1e-6)
        }
        XCTAssertNotEqual(weights.reduce(0, +), 1, accuracy: 0.01)
    }

    func testSlidingCacheRetainsBoundedHistoryAndAbsoluteOffset() {
        let cache = KolibriCache(window: 5)
        let first = MLXArray((0..<8).map(Float.init)).reshaped(1, 1, 8, 1)
        let result = cache.update(keys: first, values: first)
        XCTAssertEqual(result.0.dim(2), 8)
        XCTAssertEqual(cache.keys?.asArray(Float.self), [4, 5, 6, 7])
        let second = MLXArray([Float(8), 9]).reshaped(1, 1, 2, 1)
        let next = cache.update(keys: second, values: second)
        XCTAssertEqual(next.0.asArray(Float.self), [4, 5, 6, 7, 8, 9])
        XCTAssertEqual(cache.keys?.asArray(Float.self), [6, 7, 8, 9])
        XCTAssertEqual(cache.offset, 10)
    }

    func testFullAttentionHasNoRoPE() throws {
        let config = try configuration()
        XCTAssertNil(KolibriAttention(config: config, index: 1).rope)
        XCTAssertNotNil(KolibriAttention(config: config, index: 0).rope)
    }

    func testQuantizationRejectsRouterAndUnknownProjectionPolicies() throws {
        let config = try configuration(quantization: #"{"model.layers.0.mlp.gate":{"bits":2,"group_size":64}}"#)
        XCTAssertThrowsError(try config.validate())
    }

    func testIndependentTorchFixtureAndChunkedPrefillParity() throws {
        struct Reference: Decodable {
            let tokens: [Int32]
            let logits: [Float]
        }
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
        let config = try JSONDecoder().decode(KolibriConfiguration.self,
                                              from: Data(contentsOf: folder.appendingPathComponent("config.json")))
        let reference = try JSONDecoder().decode(Reference.self,
                                                 from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
        let model = try KolibriCausalLM(config: config)
        let weights = try loadArrays(url: folder.appendingPathComponent("weights.safetensors"))
        try model.update(parameters: .unflattened(weights), verify: [.allModelKeysSet, .shapeMismatch])
        let tokens = MLXArray(reference.tokens).reshaped(1, -1)
        let expected = MLXArray(reference.logits).reshaped(1, reference.tokens.count, config.vocabSize)
        let full = model(tokens)
        XCTAssertLessThan(abs(full - expected).max().item(Float.self), 3e-5)
        for chunkSize in [1, 3, 8] {
            let cache = model.makeCache()
            var parts: [MLXArray] = []
            for start in stride(from: 0, to: reference.tokens.count, by: chunkSize) {
                let end = min(start + chunkSize, reference.tokens.count)
                parts.append(model(tokens[0..., start..<end], cache: cache))
            }
            let chunked = concatenated(parts, axis: 1)
            XCTAssertLessThan(abs(chunked - full).max().item(Float.self), 3e-5, "chunk size \(chunkSize)")
        }
    }

    func testPackedTwoBitExpertSelectionMatchesDequantizedBanks() throws {
        let config = try configuration()
        let projection = KolibriProjection(input: 64, output: 16, experts: config.numExperts,
                                           policy: .init(bits: 2, groupSize: 64))
        let values: [Float] = (0..<7168).map { index in
            let numerator = (index * 17) % 101 - 50
            return Float(numerator) / 100
        }
        let weights = MLXArray(values).reshaped(7, 16, 64).asType(.bfloat16)
        let (packed, scales, optionalBiases) = quantized(weights, groupSize: 64, bits: 2)
        let biases = try XCTUnwrap(optionalBiases)
        try projection.update(parameters: .unflattened(["weight": packed, "scales": scales, "biases": biases]),
                              verify: [.allModelKeysSet, .shapeMismatch])
        let input = MLXArray((0..<128).map { Float($0 % 11) / 11 }).reshaped(1, 2, 1, 64).asType(.bfloat16)
        let ids = MLXArray([Int32(1), 5, 2, 4, 0, 6]).reshaped(1, 2, 3)
        let actual = projection(input, indices: ids)
        let dense = dequantized(packed, scales: scales, biases: biases, groupSize: 64, bits: 2)
        let expected = matmul(input.expandedDimensions(axis: -2), dense.take(ids, axis: 0).swappedAxes(-1, -2))
            .squeezed(axis: -2)
        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 0.04)
    }

}
