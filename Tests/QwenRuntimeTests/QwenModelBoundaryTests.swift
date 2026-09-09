import Foundation
import MLX
import MLXRandom
import MereRunKVCache
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel

final class QwenModelBoundaryTests: MLXTestCase {
    func testDenseAndHybridPrefillMatchSerialDecode() throws {
        for hybrid in [false, true] {
            MLXRandom.seed(913)
            let config = try configuration(hybrid: hybrid)
            let model = Q35Model(config: config)
            let tokens = [1, 7, 4, 9, 2]
            let prefill = model.forward(input(tokens), cache: caches(config))
            let serialCaches = caches(config)
            let serial = tokens.map { token -> MLXArray in
                let output = model.forward(input([token]), cache: serialCaches)
                MLX.eval(output.logits)
                return output.logits
            }
            assertClose(prefill.logits, MLX.concatenated(serial, axis: 1))
        }
    }

    func testHybridPrefixForkAndVerificationRollbackMatchAcceptedSerialTokens() throws {
        MLXRandom.seed(914)
        let config = try configuration(hybrid: true)
        let model = Q35Model(config: config)
        let prefixCaches = caches(config)
        MLX.eval(model.forward(input([1, 7, 4]), cache: prefixCaches).logits)
        let candidate = prefixCaches.map { $0?.fork() }
        let reference = prefixCaches.map { $0?.fork() }
        let proposals = [9, 2, 6, 3]
        MLX.eval(model.forward(input(proposals), cache: candidate, targetVerify: true).logits)
        for cache in candidate.compactMap({ $0 }) {
            XCTAssertTrue(cache.restoreVerificationPrefix(totalTokens: proposals.count, tokenCount: 2))
        }
        for token in proposals.prefix(2) {
            MLX.eval(model.forward(input([token]), cache: reference).logits)
        }
        assertClose(
            model.forward(input([5]), cache: candidate).logits,
            model.forward(input([5]), cache: reference).logits
        )
        let fresh = caches(config)
        MLX.eval(model.forward(input([1, 7, 4]), cache: fresh).logits)
        assertClose(
            model.forward(input([8]), cache: prefixCaches).logits,
            model.forward(input([8]), cache: fresh).logits
        )
    }

    func testDraftHistoryForkMatchesColdHistoryWithoutMutatingPrefix() throws {
        MLXRandom.seed(915)
        let config = try configuration(hybrid: false)
        let model = Q35Model(config: config)
        let draft = Q35MTPModel(config: config)
        let tokens = [1, 7, 4, 9, 2]
        let output = model.forward(input(tokens), cache: caches(config))
        MLX.eval(output.logits, output.hidden)
        let prefix = Q35MTPDraftSession(promptTokens: tokens, promptHidden: output.hidden)
        let cold = Q35MTPDraftSession(promptTokens: tokens, promptHidden: output.hidden)
        let hidden = output.hidden[0..., (-1)..., 0...]
        let expected = draft.draftBlock(lastToken: 5, hidden: hidden, blockSize: 4,
                                       session: cold, baseModel: model)
        let actual = draft.draftBlock(lastToken: 5, hidden: hidden, blockSize: 4,
                                     session: prefix.fork(), baseModel: model)
        XCTAssertEqual(actual.tokens, expected.tokens)
        XCTAssertEqual(prefix.committedHistoryCount, tokens.count - 1)
        XCTAssertEqual(prefix.pendingHistoryCount, tokens.count - 1)
    }

    private func input(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
    }

    private func caches(_ config: Q35Config) -> [Q35LayerCache?] {
        config.textConfig.layerTypes.map {
            $0 == "full_attention" ? .full(KVCacheSimple()) : .linear(Q35LinearCache())
        }
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let error = (actual - expected).abs().max().item(Float.self)
        XCTAssertLessThan(error, 0.0001, file: file, line: line)
    }

    private func configuration(hybrid: Bool) throws -> Q35Config {
        let firstLayer = hybrid ? "linear_attention" : "full_attention"
        return try JSONDecoder().decode(Q35Config.self, from: Data("""
        {"model_type":"qwen3_5","text_config":{
          "model_type":"qwen3_5_text","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
          "layer_types":["\(firstLayer)","full_attention"],"mtp_num_hidden_layers":1,
          "linear_num_key_heads":1,"linear_num_value_heads":2,
          "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
          "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,
          "eos_token_id":63,"vocab_size":64,"max_position_embeddings":4096,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1}}}
        """.utf8))
    }
}
