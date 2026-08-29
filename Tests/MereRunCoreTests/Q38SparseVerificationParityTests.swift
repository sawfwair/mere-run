import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q38SparseVerificationParityTests: MereRunCoreTestCase {
    func testFlashNextBF16ReadoutMatchesSerialRows() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Readout arithmetic parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        let model = Q35Model(config: try configuration())
        MLXRandom.seed(97)
        let weight = MLXRandom.normal([8_192, 2_560]).asType(.bfloat16)
        model.update(modules: ModuleChildren.unflattened([("lm_head", Linear(weight: weight, bias: nil))]))
        for width in [2, 3, 4, 7, 9] {
            let hidden = MLXRandom.normal([1, width, 2_560]).asType(.bfloat16)
            let actual = model.logits(from: hidden)
            let expected = MLX.concatenated((0..<width).map { row in
                MLX.matmul(hidden[0..., row..<(row + 1), 0...], weight.T)
            }, axis: 1)
            XCTAssertEqual(
                (actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0,
                "BF16 readout changed serial arithmetic at width=\(width)"
            )
        }
        let prefill = MLXRandom.normal([1, 16, 2_560]).asType(.bfloat16)
        XCTAssertEqual(
            (model.logits(from: prefill) - MLX.matmul(prefill, weight.T)).abs().max().item(Float.self), 0
        )
    }

    func testIndexerFloat32ScoresMatchSerialRows() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Indexer arithmetic parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(94)
        for blocks in [2_004, 8_065, 8_192, 16_185, 16_384, 32_425, 32_768] {
            let keys = MLXRandom.normal([1, 1, blocks, 128]).asType(.bfloat16).asType(.float32)
            for width in [2, 3, 4, 7, 9] {
                let queries = MLXRandom.normal([1, 4, width, 128]).asType(.bfloat16).asType(.float32)
                let actual = Q38QSAIndexer.scoreHeads(queries: queries, keys: keys, preserveSerialRows: true)
                let expected = MLX.concatenated((0..<width).map { row in
                    MLX.matmul(queries[0..., 0..., row..<(row + 1), 0...], keys.swappedAxes(-1, -2))
                }, axis: 2)
                XCTAssertEqual(
                    (actual - expected).abs().max().item(Float.self), 0,
                    "Indexer score arithmetic changed at blocks=\(blocks), width=\(width)"
                )
            }
        }
    }

    func testIndexerScorePrefillKeepsNativeMatmul() {
        MLXRandom.seed(96)
        let queries = MLXRandom.normal([1, 4, 16, 128]).asType(.bfloat16)
        let keys = MLXRandom.normal([1, 1, 600, 128]).asType(.bfloat16)
        let expected = MLX.matmul(queries.asType(.float32), keys.asType(.float32).swappedAxes(-1, -2))
        let actual = Q38QSAIndexer.scoreHeads(queries: queries, keys: keys, preserveSerialRows: false)
        XCTAssertEqual((actual - expected).abs().max().item(Float.self), 0)
    }

    func testTiedBlockSelectionMatchesSerialWithMaskedFutureBlocks() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Sparse selection parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        for prefix in [2_053, 32_260, 64_740, 64_743, 129_700, 129_703] {
            for width in [2, 4, 7] {
                let blocks = (prefix + width) / 4
                for period in [1, 17] {
                    let scores = MLXArray((0..<blocks).map { Float($0 % period) })
                        .reshaped(1, 1, blocks)
                    let actual = Q38SparseAttention.select(
                        scores: MLX.broadcast(scores, to: [1, width, blocks]),
                        offsets: [prefix], budget: 2_048, ratio: 4
                    )
                    for row in 0..<width {
                        let serialBlocks = (prefix + row + 1) / 4
                        let expected = Q38SparseAttention.select(
                            scores: scores[0..., 0..., 0..<serialBlocks],
                            offsets: [prefix + row], budget: 2_048, ratio: 4
                        )
                        XCTAssertEqual(
                            actual.indices[0..., row..<(row + 1), 0...].asArray(Int32.self),
                            expected.indices.asArray(Int32.self),
                            "Tied QSA selection changed at prefix=\(prefix), width=\(width), row=\(row), period=\(period)"
                        )
                    }
                }
            }
        }
    }

    func testSparseAttentionReductionMatchesSerialRowsAcrossLongContexts() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Sparse attention arithmetic parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(95)
        for promptTokens in [32_260, 64_740, 129_700] {
            let historyLength = promptTokens + 12
            let keys = MLXRandom.normal([1, 2, historyLength, 256]).asType(.bfloat16)
            let values = MLXRandom.normal(keys.shape).asType(.bfloat16)
            for width in [2, 3, 4, 7, 9] {
                let queries = MLXRandom.normal([1, 24, width, 256]).asType(.bfloat16)
                let selection = Q38SparseAttention.select(
                    scores: MLXRandom.normal([1, width, historyLength / 4]),
                    offsets: [promptTokens], budget: 2_048, ratio: 4
                )
                let actual = Q38SparseAttention.attend(
                    queries: queries, keys: keys, values: values,
                    indices: selection.indices, valid: selection.valid, scale: 1 / 16
                )
                let expected = MLX.concatenated((0..<width).map { row in
                    Q38SparseAttention.attend(
                        queries: queries[0..., 0..., row..<(row + 1), 0...], keys: keys, values: values,
                        indices: selection.indices[0..., row..<(row + 1), 0...],
                        valid: selection.valid[0..., row..<(row + 1), 0...], scale: 1 / 16
                    )
                }, axis: 2)
                XCTAssertEqual(
                    (actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0,
                    "Sparse attention arithmetic changed at prompt=\(promptTokens), width=\(width)"
                )
            }
        }
    }

    func testIndexerAttentionMatchesSerialAcrossFutureBlockBoundaries() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Long-history indexer parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(114)
        let indexer = Q38QSAIndexer(config: try configuration())
        indexer.update(parameters: indexer.parameters().mapValues { $0.asType(.bfloat16) })
        for prefix in [32_260, 64_740, 64_743, 129_700, 129_703] {
            let rawKeys = MLXRandom.normal([1, 1, prefix, 128]).asType(.bfloat16)
            let positions = Q38QSAIndexer.positionRows(batch: 1, count: prefix, offsets: [0], positionIds: nil)
            let base = Q38QSACache()
            _ = base.updateIndexer(keys: rawKeys, positions: positions)
            for width in [2, 4, 7] {
                let hidden = MLXRandom.normal([1, width, 2_560]).asType(.bfloat16)
                let queries = MLXRandom.normal([1, 24, width, 256]).asType(.bfloat16)
                let keys = MLXRandom.normal([1, 2, prefix + width, 256]).asType(.bfloat16)
                let values = MLXRandom.normal(keys.shape).asType(.bfloat16)
                MLX.eval(hidden, queries, keys, values, rawKeys)
                let blockCache = try XCTUnwrap(base.fork() as? Q38QSACache)
                let actual = try XCTUnwrap(indexer.attention(
                    hidden: hidden, queries: queries, keys: keys, values: values, offsets: [prefix],
                    positionIds: nil, cache: blockCache, scale: 1 / 16
                ))
                let serialCache = try XCTUnwrap(base.fork() as? Q38QSACache)
                let rows = try (0..<width).map { row in
                    try XCTUnwrap(indexer.attention(
                        hidden: hidden[0..., row..<(row + 1), 0...],
                        queries: queries[0..., 0..., row..<(row + 1), 0...],
                        keys: keys[0..., 0..., 0..<(prefix + row + 1), 0...],
                        values: values[0..., 0..., 0..<(prefix + row + 1), 0...],
                        offsets: [prefix + row], positionIds: nil, cache: serialCache, scale: 1 / 16
                    ))
                }
                let expected = MLX.concatenated(rows, axis: 2)
                XCTAssertEqual(
                    (actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0,
                    "Future QSA keys changed a causal query at prefix=\(prefix), width=\(width)"
                )
            }
        }
    }

    private func configuration() throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","tie_word_embeddings":false,"text_config":{
          "model_type":"qwen4_exp","hidden_size":2560,"intermediate_size":16,
          "num_hidden_layers":0,"num_attention_heads":24,"num_key_value_heads":2,
          "head_dim":256,"layer_types":[],"hc_count":4,"hc_lowrank":4,
          "linear_num_value_heads":1,"linear_num_key_heads":1,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":2,
          "max_position_embeddings":262144,"attention_bias":false,"attention_dropout":0,
          "attn_output_gate":true,"output_gate_type":"sigmoid",
          "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,
          "indexer_budget":2048,"indexer_compress_ratio":4,
          "ple_layer_ids":[],"vocab_size":8192,"eos_token_id":8191,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,
            "mrope_interleaved":true,"mrope_section":[11,11,10]}
        }}
        """#.utf8))
    }
}
