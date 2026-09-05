import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q35VerificationTransferTests: MereRunCoreTestCase {
    func testQwen27BProjectionsMatchSerialThroughWidthNine() throws {
        try requireGPU()
        MLXRandom.seed(612)
        for (inputSize, outputSize) in [(5_120, 17_408), (17_408, 5_120), (5_120, 96), (6_144, 5_120)] {
            let dense = (MLXRandom.normal([outputSize, inputSize]) * 0.01).asType(.bfloat16)
            let (weight, scales, biases) = MLX.quantized(dense, groupSize: 64, bits: 4)
            let layer = PortableQuantizedLinear(weight: weight, bias: nil, scales: scales,
                                                biases: biases, groupSize: 64, bits: 4, mode: .affine)
            for width in [4, 8, 9] {
                let input = MLXRandom.normal([1, width, inputSize]).asType(.bfloat16)
                let expected = MLX.concatenated((0..<width).map { row in
                    layer(input[0..., row..<(row + 1), 0...])
                }, axis: 1)
                assertExact(layer(input), expected, "Qwen K=\(inputSize), N=\(outputSize), M=\(width)")
            }
        }
    }

    func testOrnithQ4ExpertVerificationMatchesSerialBeforeAndAfterFusion() throws {
        try requireGPU()
        MLXRandom.seed(613)
        let config = try ornithConfiguration()
        let layer = Q35FeedForward(config: config)
        installWeights(layer)
        let switchMLP = try XCTUnwrap(layer.switchMLP)
        for prepared in Q35FusedSwitchGLUPolicy.enabled ? [false, true] : [false] {
            if prepared {
                XCTAssertTrue(layer.prepareFusedSwitchGLU())
                XCTAssertEqual(switchMLP.sourceGateUpElementCount, 0)
            }
            for width in [2, 4, 8, 9] {
                let input = MLXRandom.normal([1, width, 2_048]).asType(.bfloat16)
                let expected = MLX.concatenated((0..<width).map { row in
                    layer(input[0..., row..<(row + 1), 0...])
                }, axis: 1)
                assertExact(layer(input, targetVerify: true), expected,
                            "Ornith M=\(width), prepared=\(prepared)")
            }
        }
    }

    func testOrnithLinearVerificationRestoresEveryAcceptedPrefix() throws {
        try requireGPU()
        MLXRandom.seed(614)
        let layer = Q35LinearAttention(config: try ornithConfiguration())
        installWeights(layer)
        let base = Q35LinearCache()
        MLX.eval(layer(MLXRandom.normal([1, 7, 2_048]).asType(.bfloat16), cache: base))
        for width in [4, 8, 9] {
            let input = MLXRandom.normal([1, width, 2_048]).asType(.bfloat16)
            for kept in 1...width {
                let candidate = base.fork()
                let reference = base.fork()
                let actual = layer(input, cache: candidate, targetVerify: true)
                MLX.eval(actual)
                let expected = MLX.concatenated((0..<kept).map { row in
                    let output = layer(input[0..., row..<(row + 1), 0...], cache: reference)
                    MLX.eval(output)
                    return output
                }, axis: 1)
                XCTAssertTrue(candidate.restoreVerificationPrefix(tokenCount: kept))
                assertExact(actual[0..., 0..<kept, 0...], expected, "Ornith output M=\(width), kept=\(kept)")
                assertExact(try XCTUnwrap(candidate.convState), try XCTUnwrap(reference.convState), "conv")
                assertExact(try XCTUnwrap(candidate.recurrentState), try XCTUnwrap(reference.recurrentState), "GDN")
            }
        }
    }

    private func installWeights(_ module: Module) {
        module.update(parameters: module.parameters().mapValues { $0.asType(.bfloat16) })
        var replacements: [(String, Module)] = []
        for (key, child) in module.leafModules().flattened() {
            if let expert = child as? Q35SwitchLinear {
                let shape = expert.weight.shape
                let dense = (MLXRandom.normal([shape[0], shape[1], shape[2] * 8]) * 0.01).asType(.bfloat16)
                let (weight, scales, biases) = MLX.quantized(dense, groupSize: 64, bits: 4)
                MLX.eval(weight, scales)
                replacements.append((key, Q35SwitchLinear(weight: weight, scales: scales, biases: biases,
                                                          bias: nil, groupSize: 64, bits: 4)))
            } else if let linear = child as? Linear {
                let bits = key == "gate" || key == "shared_expert_gate" ? 8 : 4
                let (weight, scales, biases) = MLX.quantized(linear.weight, groupSize: 64, bits: bits)
                replacements.append((key, PortableQuantizedLinear(
                    weight: weight, bias: nil, scales: scales, biases: biases,
                    groupSize: 64, bits: bits, mode: .affine
                )))
            }
        }
        module.update(modules: ModuleChildren.unflattened(replacements))
        MLX.eval(module.parameters().flattened().map(\.1))
    }

    private func assertExact(_ actual: MLXArray, _ expected: MLXArray, _ message: String) {
        XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0, message)
    }

    private func requireGPU() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Qwen transfer parity requires the GPU.")
    }

    private func ornithConfiguration() throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen3_5_moe","quantization":{"bits":4,"group_size":64},"text_config":{
          "model_type":"qwen3_5_moe_text","hidden_size":2048,"intermediate_size":512,"num_hidden_layers":1,
          "num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,
          "num_experts":256,"num_experts_per_tok":8,"norm_topk_prob":true,
          "moe_intermediate_size":512,"shared_expert_intermediate_size":512,
          "layer_types":["linear_attention"],"linear_num_key_heads":16,"linear_num_value_heads":32,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
          "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,"eos_token_id":63,"vocab_size":64,"max_position_embeddings":262144,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25}}}
        """#.utf8))
    }
}
