import Foundation
import MLX
import MLXNN

extension Q35Generator {
    /// The v2 sidecar stores direct norm scales and affine Q4/group-32
    /// projections. The draft shares the target's embedding and output head.
    static func loadBonsai25MTPWeights(into draft: Q35MTPModel, from url: URL) throws {
        let source = try MLX.loadArrays(url: url)
        let weights = try bonsai25MTPWeights(source, expectedParameters: draft.parameters().flattened())
        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            weights, to: draft, groupSize: 32, bits: 4, applySVDResiduals: false
        )
    }

    static func bonsai25MTPWeights(
        _ source: [String: MLXArray],
        expectedParameters: [(String, MLXArray)]
    ) throws -> [String: MLXArray] {
        let modules = [
            "nextn.eh_proj": "fc",
            "nextn.enorm": "pre_fc_norm_embedding",
            "nextn.hnorm": "pre_fc_norm_hidden",
            "nextn.shared_head_norm": "norm",
            "attn_norm": "layers.0.input_layernorm",
            "post_attention_norm": "layers.0.post_attention_layernorm",
            "attn_q": "layers.0.self_attn.q_proj",
            "attn_k": "layers.0.self_attn.k_proj",
            "attn_v": "layers.0.self_attn.v_proj",
            "attn_output": "layers.0.self_attn.o_proj",
            "attn_q_norm": "layers.0.self_attn.q_norm",
            "attn_k_norm": "layers.0.self_attn.k_norm",
            "ffn_gate": "layers.0.mlp.gate_proj",
            "ffn_up": "layers.0.mlp.up_proj",
            "ffn_down": "layers.0.mlp.down_proj",
        ]
        let expected = Dictionary(uniqueKeysWithValues: expectedParameters)
        var mapped: [String: MLXArray] = [:]
        for (module, destination) in modules {
            guard let shape = expected[destination + ".weight"]?.shape else {
                throw Q35Error.generationFailed("Bonsai 2.5 MTP v2 has an incompatible draft module: \(destination).")
            }
            let fields: [(String, DType, [Int])]
            if shape.count == 1 {
                fields = [("weight", .float32, shape)]
            } else {
                let groups = [shape[0], shape[1] / 32]
                fields = [
                    ("weight", .uint32, [shape[0], shape[1] / 8]),
                    ("scales", .float16, groups),
                    ("biases", .float16, groups),
                ]
            }
            for (field, dtype, fieldShape) in fields {
                let key = module + "." + field
                guard let value = source[key], value.shape == fieldShape, value.dtype == dtype else {
                    throw Q35Error.generationFailed("Bonsai 2.5 MTP v2 has a missing or incompatible tensor: \(key).")
                }
                mapped[destination + "." + field] = dtype == .float32 ? value.asType(.float16) : value
            }
        }
        guard mapped.count == source.count else {
            throw Q35Error.generationFailed("Bonsai 2.5 MTP v2 contains unexpected tensors.")
        }
        return mapped
    }
}
