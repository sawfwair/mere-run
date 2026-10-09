import Foundation
import MereRunTensor
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunQwenModel

final class PPLXEmbedV2QuantizationTests: MLXTestCase {
    func configuration() throws -> Q35Config {
        let json = #"""
        {"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text",
        "vocab_size":64,"hidden_size":64,"intermediate_size":128,
        "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
        "linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":16,
        "linear_value_head_dim":16,"linear_conv_kernel_dim":4,
        "layer_types":["linear_attention","full_attention"],"is_causal":false,
        "max_position_embeddings":262144,"rms_norm_eps":0.000001,
        "attention_bias":false,"attention_dropout":0.0,"attn_output_gate":true,
        "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.5,
        "rope_type":"default","mrope_interleaved":true,"mrope_section":[2,2,4]}}}
        """#
        return try JSONDecoder().decode(Q35Config.self, from: Data(json.utf8))
    }

    func contract(_ modules: [String: Int]) throws -> PPLXEmbedV2QuantizationConfig {
        struct Parameters: Encodable { let bits: Int; let group_size = 32; let mode = "affine" }
        struct Contract: Encodable {
            let bits = 4; let group_size = 32; let mode = "affine"; let modules: [String: Parameters]
        }
        return try JSONDecoder().decode(PPLXEmbedV2QuantizationConfig.self,
            from: JSONEncoder().encode(Contract(modules: modules.mapValues { Parameters(bits: $0) })))
    }

    func packed(_ encoder: PPLXEmbedV2Encoder, modules: [String: Int]) -> ([String: MLXArray], [String: MLXArray]) {
        var arrays = Dictionary(uniqueKeysWithValues: encoder.parameters().flattened().filter { encoder.checkpointParameterNames.contains($0.0) })
        var ordinary = arrays
        for (path, bits) in modules {
            let (weight, scales, biases) = MLX.quantized(arrays[path + ".weight"]!, groupSize: 32, bits: bits)
            arrays[path + ".weight"] = weight
            arrays[path + ".scales"] = scales
            arrays[path + ".biases"] = biases
            ordinary[path + ".weight"] = MLX.dequantized(weight, scales: scales, biases: biases, groupSize: 32, bits: bits)
        }
        MLX.eval(Array(arrays.values) + Array(ordinary.values))
        return (arrays, ordinary)
    }

    func testMixedPackedEncoderMatchesExplicitDequantization() throws {
        let config = try configuration()
        let encoder = PPLXEmbedV2Encoder(config: config)
        let modules = ["embed_tokens": 8, "layers.0.linear_attn.in_proj_qkv": 4,
                       "layers.0.linear_attn.in_proj_z": 4, "layers.0.linear_attn.out_proj": 4,
                       "layers.1.self_attn.q_proj": 4, "layers.1.self_attn.k_proj": 4,
                       "layers.1.self_attn.v_proj": 4, "layers.1.self_attn.o_proj": 4,
                       "layers.0.mlp.up_proj": 4, "layers.1.mlp.down_proj": 8]
        let (arrays, ordinary) = packed(encoder, modules: modules)
        let reference = PPLXEmbedV2Encoder(config: config)
        try reference.update(parameters: ModuleParameters.unflattened(ordinary), verify: [.noUnusedKeys, .shapeMismatch])
        try encoder.installQuantizedWeights(arrays, config: contract(modules))
        let ids = MLXArray([Int32(2), 5, 7, 11, 3], [1, 5])
        let actual = encoder(inputIDs: ids), expected = reference(inputIDs: ids)
        let error = MLX.max(MLX.abs(actual - expected)).item(Float.self)
        XCTAssertLessThan(error, 0.0001)
        let leaves = Dictionary(uniqueKeysWithValues: encoder.leafModules().flattened())
        XCTAssertTrue(leaves["embed_tokens"] is PreQuantizedEmbedding)
        XCTAssertTrue(leaves["layers.1.self_attn.q_proj"] is QuantizedLinear)
        XCTAssertFalse(leaves["layers.0.linear_attn.in_proj_a"] is QuantizedLinear)
    }

    func testPackedContractRejectsMissingMetadataGeometryAndPrecisionChanges() throws {
        let encoder = PPLXEmbedV2Encoder(config: try configuration())
        let modules = ["embed_tokens": 8]
        let (arrays, _) = packed(encoder, modules: modules)
        var missing = arrays; missing.removeValue(forKey: "embed_tokens.scales")
        XCTAssertThrowsError(try encoder.installQuantizedWeights(missing, config: contract(modules)))
        XCTAssertThrowsError(try encoder.installQuantizedWeights(arrays, config: contract(["embed_tokens": 4])))
        var precision = arrays; precision["norm.weight"] = arrays["norm.weight"]!.asType(.bfloat16)
        XCTAssertThrowsError(try encoder.installQuantizedWeights(precision, config: contract(modules)))
        for path in ["layers.0.linear_attn.in_proj_a", "layers.0.linear_attn.in_proj_b", "norm", "visual.patch_embed", "contextual_projection"] {
            XCTAssertThrowsError(try contract([path: 4]).validate())
        }
    }
}
