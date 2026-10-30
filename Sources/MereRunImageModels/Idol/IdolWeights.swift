import Foundation
import MLX

/// Exact publisher tensor namespaces. Only transformer linear projections run in FP16;
/// embeddings, learned rotary maps, normalization and output heads remain FP32.
struct IdolWeights {
    let arrays: [String: MLXArray]

    init(_ arrays: [String: MLXArray], shapes: [String: [Int]]) throws {
        guard Set(arrays.keys) == Set(shapes.keys) else {
            let missing = Set(shapes.keys).subtracting(arrays.keys).sorted()
            let extra = Set(arrays.keys).subtracting(shapes.keys).sorted()
            throw IdolError.invalidWeights("Tensor inventory mismatch; missing \(missing), unexpected \(extra).")
        }
        for (name, shape) in shapes {
            guard arrays[name]?.shape == shape, arrays[name]?.dtype == .float32 else {
                throw IdolError.invalidWeights("Expected original FP32 tensor \(name) with shape \(shape).")
            }
        }
        self.arrays = Dictionary(uniqueKeysWithValues: arrays.map { name, value in
            let block = name.hasPrefix("blocks.") || name.hasPrefix("noise_refiner.") || name.hasPrefix("context_refiner.")
            let projection = (name.hasSuffix(".weight") || name.hasSuffix(".bias")) && !name.contains(".norm")
            return (name, block && projection ? value.asType(.float16) : value)
        })
    }

    subscript(_ key: String) -> MLXArray { arrays[key]! }

    func linear(_ input: MLXArray, _ name: String, half: Bool = false) -> MLXArray {
        let dtype: DType = half ? .float16 : .float32
        var result = matmul(input.asType(dtype), self[name + ".weight"].asType(dtype).T)
        if let bias = arrays[name + ".bias"] { result = result + bias.asType(dtype) }
        return result
    }
}

struct IdolWeightSchema {
    var shapes: [String: [Int]] = [:]
    mutating func linear(_ name: String, _ input: Int, _ output: Int, bias: Bool = true) {
        shapes[name + ".weight"] = [output, input]
        if bias { shapes[name + ".bias"] = [output] }
    }
    mutating func norm(_ name: String, _ width: Int) {
        shapes[name + ".weight"] = [width]
        shapes[name + ".bias"] = [width]
    }
    mutating func scalarEmbed(_ name: String, _ width: Int) {
        linear(name + ".mlp.0", 256, width)
        linear(name + ".mlp.2", width, width)
    }
    mutating func ffn(_ name: String, _ width: Int, _ ratio: Float) {
        let hidden = Int(Float(width) * ratio)
        linear(name + ".mlp.0", width, hidden)
        linear(name + ".mlp.2", hidden, width)
    }
    mutating func repo(_ name: String, _ width: Int, _ heads: Int) {
        norm(name + ".norm", width)
        linear(name + ".gate_map", width, width / 8, bias: false)
        linear(name + ".content_map", width, width / 8, bias: false)
        linear(name + ".final_map", width / 8, 3 * heads, bias: false)
        let dim = width / heads, first = dim / 6
        shapes[name + ".freqs_0"] = [first]
        shapes[name + ".freqs_1"] = [first]
        shapes[name + ".freqs_2"] = [dim / 2 - 2 * first]
    }
    mutating func attention(_ name: String, _ width: Int, _ heads: Int, context: Int? = nil) {
        if let context {
            linear(name + ".to_q", width, width)
            linear(name + ".to_kv", context, 2 * width)
        } else { linear(name + ".to_qkv", width, 3 * width) }
        linear(name + ".to_out", width, width)
        shapes[name + ".q_rms_norm.gamma"] = [heads, width / heads]
        shapes[name + ".k_rms_norm.gamma"] = [heads, width / heads]
    }
    static func denoiser(_ config: IdolConfiguration.Denoiser) -> [String: [Int]] {
        var schema = Self()
        let width = config.model_channels, heads = config.num_heads
        schema.scalarEmbed("t_embedder", width)
        schema.linear("adaLN_modulation.1", width, 6 * width)
        schema.linear("input_layer", config.in_channels, width)
        schema.linear("cond_embedder", config.cond_channels, width)
        schema.linear("cond_embedder2", config.cond2_channels, width)
        schema.shapes["pos_pe"] = [1, config.q_token_length, 3]
        schema.shapes["shift_table"] = [1, 2, width]
        schema.linear("out_layer", width, config.in_channels)
        schema.linear("cam_out_layer", width, config.cam_channels)
        for index in 0..<config.num_refiner_blocks {
            schema.linear("cam_refiner.mlp.\(2 * index)", index == 0 ? config.cam_channels : width, width)
        }
        for (blocks, rotary, count, modulation) in [
            ("blocks", "repo_layers", config.num_blocks, true),
            ("noise_refiner", "noise_repo_layers", config.num_refiner_blocks, true),
            ("context_refiner", "context_repo_layers", config.num_refiner_blocks, false)
        ] {
            for index in 0..<count {
                let prefix = "\(blocks).\(index)"
                schema.repo("\(rotary).\(index)", width, heads)
                schema.linear(prefix + ".attn.qkv", width, 3 * width)
                schema.linear(prefix + ".attn.out", width, width)
                schema.shapes[prefix + ".attn.q_norm.gamma"] = [heads, width / heads]
                schema.shapes[prefix + ".attn.k_norm.gamma"] = [heads, width / heads]
                schema.ffn(prefix + ".mlp", width, config.mlp_ratio)
                if modulation { schema.shapes[prefix + ".shift_table"] = [1, 6 * width] }
                else {
                    schema.norm(prefix + ".norm1", width)
                    schema.norm(prefix + ".norm2", width)
                }
            }
        }
        return schema.shapes
    }
    static func decoder(_ config: IdolConfiguration.Decoder, gaussian: Bool) -> [String: [Int]] {
        var schema = Self()
        let width = config.model_channels, heads = config.num_heads
        schema.linear("in_proj", 3, width)
        schema.linear("input_layer", width, width)
        schema.linear("out_proj", width, gaussian ? 15 * config.representation_config!.num_gaussians : 8)
        if gaussian {
            schema.shapes["points_offset_perturbation"] = [config.representation_config!.num_gaussians, 3]
            schema.shapes["base_offset_scale"] = []
        } else {
            schema.scalarEmbed("l_embedder", width)
            schema.linear("adaLN_modulation.1", width, 6 * width)
        }
        for index in 0..<config.num_blocks {
            let prefix = "blocks.\(index)"
            schema.attention(prefix + ".cross_attn", width, heads, context: config.cond_channels)
            schema.ffn(prefix + ".mlp", width, config.mlp_ratio)
            if gaussian {
                schema.attention(prefix + ".self_attn", width, heads)
                schema.norm(prefix + ".norm2", width)
            }
        }
        return schema.shapes
    }
}
