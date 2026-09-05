import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q35FullAttentionVerificationTests: MereRunCoreTestCase {
    func testOrnithFullAttentionMatchesEachSerialWindow() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Full-attention parity requires the GPU.")
        MLXRandom.seed(615)
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen3_5_moe","quantization":{"bits":4,"group_size":64},"text_config":{
          "model_type":"qwen3_5_moe_text","hidden_size":2048,"intermediate_size":512,"num_hidden_layers":1,
          "num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,
          "linear_num_key_heads":16,"linear_num_value_heads":32,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
          "layer_types":["full_attention"],"attention_bias":false,"attention_dropout":0,
          "attn_output_gate":true,"eos_token_id":63,"vocab_size":64,
          "max_position_embeddings":262144,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25}}}
        """#.utf8))
        let layer = Q35FullAttention(config: config)
        layer.update(parameters: layer.parameters().mapValues { $0.asType(.bfloat16) })
        var replacements: [(String, Module)] = []
        for (name, child) in layer.leafModules().flattened() {
            guard let linear = child as? Linear else { continue }
            let (weight, scales, biases) = MLX.quantized(linear.weight, groupSize: 64, bits: 4)
            replacements.append((name, PortableQuantizedLinear(
                weight: weight, bias: nil, scales: scales, biases: biases, groupSize: 64, bits: 4, mode: .affine
            )))
        }
        layer.update(modules: ModuleChildren.unflattened(replacements))
        MLX.eval(layer.parameters().flattened().map(\.1))
        for prefix in [0, 7, 55, 127, 1_021, 4_093, 8_190, 16_381, 32_766, 65_533] {
            let base = KVCacheSimple()
            if prefix > 0 {
                let keys = MLXRandom.normal([1, 2, prefix, 256]).asType(.bfloat16)
                let values = MLXRandom.normal(keys.shape).asType(.bfloat16)
                _ = base.update(keys: keys, values: values)
                MLX.eval(keys, values)
            }
            for width in [2, 4, 5, 6, 8, 9] {
                let candidate = base.fork()
                let reference = base.fork()
                let input = MLXRandom.normal([1, width, 2_048]).asType(.bfloat16)
                let actual = layer(input, mask: .causal, cache: candidate, targetVerify: true)
                let expected = MLX.concatenated((0..<width).map { row in
                    let output = layer(input[0..., row..<(row + 1), 0...], mask: .none, cache: reference)
                    MLX.eval(output)
                    return output
                }, axis: 1)
                let error = (actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self)
                XCTAssertEqual(error, 0, "prefix=\(prefix), width=\(width)")
            }
        }
    }
}
