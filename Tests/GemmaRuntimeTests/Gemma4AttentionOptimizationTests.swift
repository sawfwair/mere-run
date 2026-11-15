import Foundation
import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunGemmaModel

final class Gemma4AttentionOptimizationTests: MLXTestCase {
    func testDirectPartialRoPEPreservesHalfLayoutAndUnrotatedDimensions() {
        MLXRandom.seed(743)
        for dim in [64, 512] {
            for partial: Float in [0, 0.25, 0.5, 1] {
                let rope = Gemma4ProportionalRoPE(dims: dim, base: 1_000_000, partialRotaryFactor: partial)
                for dtype: DType in [.float32, .bfloat16, .float16] {
                    let input = MLXRandom.normal([2, 17, 3, dim + 8]).asType(dtype).transposed(0, 2, 1, 3)
                    for offset in [0, 779, 1024, 65_537] {
                        rope.directRotation = false
                        let expected = rope(input, offset: offset)
                        rope.directRotation = true
                        let actual = rope(input, offset: offset)
                        XCTAssertEqual(actual.shape, input.shape)
                        XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0,
                                       "dim=\(dim), partial=\(partial), dtype=\(dtype), offset=\(offset)")
                    }
                }
            }
        }
    }

    func testMaskCacheKeysPositionsWindowAndDType() {
        let masks = Gemma4AttentionMaskCache()
        var builds = 0
        let keys: [Gemma4AttentionMaskCache.Key] = [
            .init(queryLength: 16, queryOffset: 1024, keyLength: 1040, window: 1024, dtype: .bfloat16),
            .init(queryLength: 16, queryOffset: 1040, keyLength: 1040, window: 1024, dtype: .bfloat16),
            .init(queryLength: 16, queryOffset: 1024, keyLength: 1040, window: nil, dtype: .bfloat16),
            .init(queryLength: 16, queryOffset: 1024, keyLength: 1040, window: 1024, dtype: .float32),
            .init(queryLength: 1, queryOffset: 1024, keyLength: 1025, window: 1024, dtype: .bfloat16)
        ]
        for key in keys {
            let first = masks.resolve(key) { builds += 1; return .array(MLXArray(builds)) }
            let second = masks.resolve(key) { XCTFail("Identical layer positions must reuse their mask"); return .none }
            XCTAssertTrue(first.mask === second.mask)
        }
        XCTAssertEqual(builds, keys.count)
    }

    func testSharedMasksAndDirectRoPEPreserveChunkedAndSharedKVForward() throws {
        let configData = Data(#"""
        {"model_type":"gemma4_text","max_position_embeddings":262144,"rms_norm_eps":0.000001,"hidden_size":32,"num_hidden_layers":4,"intermediate_size":64,"num_attention_heads":2,
         "num_key_value_heads":1,"head_dim":16,"global_head_dim":32,"vocab_size":64,
         "hidden_size_per_layer_input":0,"sliding_window":8,"num_kv_shared_layers":2,
         "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
         "attention_k_eq_v":true,"rope_parameters":{
           "sliding_attention":{"rope_type":"default","rope_theta":10000},
           "full_attention":{"rope_type":"proportional","rope_theta":1000000,"partial_rotary_factor":0.25}}}
        """#.utf8)
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: configData)
        let model = Gemma4TextCausalLM(config: config)
        let referenceCache = model.makeAttentionCache(), optimizedCache = model.makeAttentionCache()
        for count in [7, 5, 1, 16] {
            let ids = MLXArray((0..<count).map { Int32($0 % 64) }).reshaped(1, count)
            let input = model.inputEmbeddings(for: ids)
            model.setExperimentalAttention(false)
            let expected = model.languageModel.forwardDetailed(embeddings: input, inputIds: ids, cache: referenceCache).hidden
            eval(expected)
            model.setExperimentalAttention(true)
            let actual = model.languageModel.forwardDetailed(embeddings: input, inputIds: ids, cache: optimizedCache).hidden
            XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0)
            XCTAssertEqual(referenceCache.map(\.offset), optimizedCache.map(\.offset))
        }
        // Content-dependent masks bypass reuse, even when positions match.
        let ids = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let input = model.inputEmbeddings(for: ids)
        for mask in [MLXArray([Int32(1), 1, 1, 0]).reshaped(1, 4), MLXArray([Int32(1), 1, 1, 1]).reshaped(1, 4)] {
            model.setExperimentalAttention(false)
            let expected = model.languageModel.forwardDetailed(embeddings: input, inputIds: ids, attentionMask: mask).hidden
            eval(expected)
            model.setExperimentalAttention(true)
            let actual = model.languageModel.forwardDetailed(embeddings: input, inputIds: ids, attentionMask: mask).hidden
            XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0)
        }
    }
}
