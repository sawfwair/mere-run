import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

/// Uses synthetic weights with checkpoint-sized projections, not model assets.
final class Q38LayerVerificationParityTests: MereRunCoreTestCase {
    func testAlignedFlashNextQ4MatchesSerialRows() throws {
        try requireGPU()
        MLXRandom.seed(120)
        for (inputSize, outputSize) in [(2_560, 48), (2_560, 6_144), (10_240, 320), (6_144, 2_560)] {
            let dense = MLXRandom.normal([outputSize, inputSize]).asType(.bfloat16)
            let (weight, scales, biases) = MLX.quantized(dense, groupSize: 64, bits: 4)
            let layer = PortableQuantizedLinear(
                weight: weight, bias: nil, scales: scales, biases: biases,
                groupSize: 64, bits: 4, mode: .affine
            )
            for count in [2, 3, 4] {
                let input = MLXRandom.normal([1, count, inputSize]).asType(.bfloat16)
                let serial = (0..<count).map { row in
                    let result = layer(input[0..., row..<(row + 1), 0...])
                    MLX.eval(result)
                    return result
                }
                assertExact(layer(input), MLX.concatenated(serial, axis: 1),
                            "Q4 M=\(count), K=\(inputSize), N=\(outputSize)")
            }
        }
    }

    func testMixedHyperConnectionMatchesSerialRows() throws {
        try requireGPU()
        MLXRandom.seed(121)
        let layer = Q38GatedResidual(config: try configuration())
        installMixedWeights(layer)
        let input = MLXRandom.normal([1, 4, 10_240]).asType(.bfloat16)
        let mixed = layer.mix(input)
        var serialMixed: [MLXArray] = []
        var serialInjection: [MLXArray] = []
        var serialCombined: [MLXArray] = []
        for row in 0..<4 {
            let piece = input[0..., row..<(row + 1), 0...]
            let result = layer.mix(piece)
            let combined = layer.combine(piece)
            MLX.eval(result.mixed, result.injectionWeights, combined)
            serialMixed.append(result.mixed)
            serialInjection.append(result.injectionWeights)
            serialCombined.append(combined)
        }
        assertExact(mixed.mixed, MLX.concatenated(serialMixed, axis: 1), "hyper mix")
        assertExact(mixed.injectionWeights, MLX.concatenated(serialInjection, axis: 1), "hyper injection")
        assertExact(layer.combine(input), MLX.concatenated(serialCombined, axis: 1), "hyper combine")
    }

    func testMixedExpertBlockMatchesSerialRows() throws {
        try requireGPU()
        MLXRandom.seed(122)
        let layer = Q35FeedForward(config: try configuration())
        installMixedWeights(layer)
        let input = MLXRandom.normal([1, 4, 2_560]).asType(.bfloat16)
        let actual = layer(input)
        let serial = (0..<4).map { row in
            let result = layer(input[0..., row..<(row + 1), 0...])
            MLX.eval(result)
            return result
        }
        assertExact(actual, MLX.concatenated(serial, axis: 1), "mixed experts")
    }

    func testMixedPLEMatchesSerialRows() throws {
        try requireGPU()
        MLXRandom.seed(124)
        let layer = Q38PLELayer(config: try configuration(), pleLayerIndex: 0)
        installMixedWeights(layer)
        // Sixteen n-gram heads use the primes 5 through 61 (496 total rows).
        let dense = (MLXRandom.normal([496, 160]) * 0.1).asType(.bfloat16)
        let (weight, scales, biases) = MLX.quantized(dense, groupSize: 32, bits: 4)
        layer.pleEmbedding.installShards([
            PreQuantizedEmbedding(weight: weight, scales: scales, biases: biases, groupSize: 32, bits: 4),
        ])
        let base = Q35LinearCache()
        let prefix = MLXRandom.normal([1, 5, 10_240]).asType(.bfloat16)
        let primed = layer(prefix, inputIds: MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5), cache: base)
        MLX.eval(primed)
        let input = MLXRandom.normal([1, 4, 10_240]).asType(.bfloat16)
        let tokens = MLXArray([Int32(6), 7, 31, 8]).reshaped(1, 4)
        let candidate = base.fork()
        let reference = base.fork()
        let actual = layer(input, inputIds: tokens, cache: candidate, targetVerify: true)
        MLX.eval(actual)
        let serial = (0..<4).map { row in
            let result = layer(input[0..., row..<(row + 1), 0...],
                               inputIds: tokens[0..., row..<(row + 1)], cache: reference)
            MLX.eval(result)
            return result
        }
        assertExact(actual, MLX.concatenated(serial, axis: 1), "mixed PLE")
    }

    func testMixedLinearDecoderBlockMatchesSerialRows() throws {
        try qualifyDecoder(layerType: .linear, prefixCount: 31)
    }

    func testMixedLinearStateRemainsSerialExactAcrossRollbackRounds() throws {
        try requireGPU()
        MLXRandom.seed(125)
        let layer = Q35DecoderLayer(config: try configuration(), layerIndex: 0)
        installMixedWeights(layer)
        var committed = Q35LinearCache()
        let reference = Q35LinearCache()
        for round in 0..<64 {
            let input = MLXRandom.normal([1, 4, 10_240]).asType(.bfloat16)
            let candidate = committed.fork()
            let actual = layer(input, fullMask: .none, cache: .linear(candidate), targetVerify: true)
            MLX.eval(actual)
            let kept = round % 4 + 1
            let serial = (0..<kept).map { row in
                let output = layer(input[0..., row..<(row + 1), 0...],
                                   fullMask: .none, cache: .linear(reference))
                MLX.eval(output)
                return output
            }
            XCTAssertTrue(candidate.restoreVerificationPrefix(tokenCount: kept))
            assertExact(actual[0..., 0..<kept, 0...], MLX.concatenated(serial, axis: 1),
                        "linear rollback output, round=\(round)")
            assertExact(try XCTUnwrap(candidate.convState), try XCTUnwrap(reference.convState),
                        "linear convolution state, round=\(round)")
            assertExact(try XCTUnwrap(candidate.recurrentState), try XCTUnwrap(reference.recurrentState),
                        "linear recurrent state, round=\(round)")
            committed = candidate
        }
    }

    func testMixedSparseDecoderBlockMatchesSerialRows() throws {
        for prefix in [2_053, 64_740, 64_743, 129_700, 129_703] {
            try qualifyDecoder(layerType: .full, prefixCount: prefix)
        }
    }

    func testMixedDenseAndQSABoundaryDecoderBlocksMatchSerialRows() throws {
        for prefix in [31, 64, 1_024, 2_047, 2_048] {
            try qualifyDecoder(layerType: .full, prefixCount: prefix)
        }
    }

    private func qualifyDecoder(layerType: Q35AttentionLayerType, prefixCount: Int) throws {
        try requireGPU()
        MLXRandom.seed(123)
        let layer = Q35DecoderLayer(config: try configuration(), layerIndex: 0, layerTypeOverride: layerType)
        installMixedWeights(layer)
        let base: Q35LayerCache
        if layerType == .full {
            // Synthetic history isolates long-position arithmetic without
            // processing a checkpoint-sized prefill through the fixture.
            let cache = Q38QSACache()
            let keys = MLXRandom.normal([1, 2, prefixCount, 256]).asType(.bfloat16)
            let values = MLXRandom.normal(keys.shape).asType(.bfloat16)
            let indexKeys = MLXRandom.normal([1, 1, prefixCount, 128]).asType(.bfloat16)
            let positions = Q38QSAIndexer.positionRows(batch: 1, count: prefixCount, offsets: [0], positionIds: nil)
            _ = cache.update(keys: keys, values: values)
            _ = cache.updateIndexer(keys: indexKeys, positions: positions)
            MLX.eval(keys, values, indexKeys, positions)
            base = .full(cache)
        } else {
            base = .linear(Q35LinearCache())
            let prefix = MLXRandom.normal([1, prefixCount, 10_240]).asType(.bfloat16)
            MLX.eval(layer(prefix, fullMask: .causal, cache: base))
        }
        for width in [2, 3, 4] {
            let input = MLXRandom.normal([1, width, 10_240]).asType(.bfloat16)
            let candidate = base.fork()
            let reference = base.fork()
            let actual = layer(input, fullMask: .causal, cache: candidate, targetVerify: true)
            MLX.eval(actual)
            let serial = (0..<width).map { row in
                let result = layer(input[0..., row..<(row + 1), 0...], fullMask: .none, cache: reference)
                MLX.eval(result)
                return result
            }
            assertExact(actual, MLX.concatenated(serial, axis: 1),
                        "\(layerType.rawValue) decoder, prefix=\(prefixCount), width=\(width)")
        }
    }

    private func installMixedWeights(_ module: Module) {
        module.update(parameters: module.parameters().mapValues { $0.asType(.bfloat16) })
        var replacements: [(String, Module)] = []
        for (key, child) in module.leafModules().flattened() {
            if let expert = child as? Q35SwitchLinear {
                let (weight, scales, biases) = MLX.quantized(expert.weight, groupSize: 128, bits: 2)
                replacements.append((key, Q35SwitchLinear(
                    weight: weight, scales: scales, biases: biases, bias: nil, groupSize: 128, bits: 2
                )))
            } else if let linear = child as? Linear,
                      key != "gate", key != "shared_expert_gate",
                      !key.hasSuffix("mlp.gate"), !key.hasSuffix("mlp.shared_expert_gate"),
                      !key.hasSuffix("index_qk_proj") {
                let (weight, scales, biases) = MLX.quantized(linear.weight, groupSize: 64, bits: 4)
                replacements.append((key, PortableQuantizedLinear(
                    weight: weight, bias: nil, scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine
                )))
            }
        }
        module.update(modules: ModuleChildren.unflattened(replacements))
        MLX.eval(module.parameters().flattened().map(\.1))
    }

    private func requireGPU() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu,
                          "Mixed decoder parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
    }

    private func assertExact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        let difference = (actual.asType(.float32) - expected.asType(.float32)).abs()
        XCTAssertEqual(difference.max().item(Float.self), 0, "\(label) changed serial arithmetic")
    }

    private func configuration() throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","text_config":{
          "model_type":"qwen4_exp_text","hidden_size":2560,"intermediate_size":640,
          "num_hidden_layers":1,"layer_types":["linear_attention"],
          "num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
          "num_experts":16,"num_experts_per_tok":10,"norm_topk_prob":true,
          "moe_intermediate_size":640,"shared_expert_intermediate_size":640,
          "linear_num_key_heads":16,"linear_num_value_heads":48,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
          "hc_count":4,"hc_lowrank":320,"output_gate_type":"sigmoid",
          "ple_layer_ids":[],"ple_embed_dim":2560,"ple_conv_kernel_size":4,
          "ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":5,"seed":1234,
          "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,
          "indexer_budget":2048,"indexer_compress_ratio":4,
          "max_position_embeddings":262144,"vocab_size":32,"eos_token_id":31,"rms_norm_eps":0.000001,
          "attention_bias":false,"attention_dropout":0,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,
            "mrope_interleaved":true,"mrope_section":[11,11,10]}}}
        """#.utf8))
    }
}
