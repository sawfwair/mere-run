import MLX

/// Shapes from the released checkpoint headers at c14071b549f4fcab10909f3c4acc6a2edb366947.
package enum UnlimitedOCRCheckpoint {
    package static var shapes: [String: [Int]] {
        var result = global
        for layer in 0..<12 {
            let prefix = "model.layers.\(layer)."
            for (name, shape) in decoder { result[prefix + name] = shape }
            for (name, shape) in layer == 0 ? dense : moe {
                if name.contains("{expert}") {
                    for expert in 0..<64 { result[prefix + name.replacingOccurrences(of: "{expert}", with: String(expert))] = shape }
                } else { result[prefix + name] = shape }
            }
            for (name, shape) in sam {
                let global = [2, 5, 8, 11].contains(layer)
                if name.hasPrefix("global:") && !global || name.hasPrefix("local:") && global { continue }
                let suffix = name.split(separator: ":").last.map(String.init) ?? name
                result["model.sam_model.blocks.\(layer)." + suffix] = shape
            }
        }
        for layer in 0..<24 {
            for (name, shape) in clip { result["model.vision_model.transformer.layers.\(layer)." + name] = shape }
        }
        return result
    }

    package static func validate(_ tensors: [String: MLXArray]) throws {
        let expected = shapes
        guard Set(tensors.keys) == Set(expected.keys) else {
            let missing = Set(expected.keys).subtracting(tensors.keys).sorted().prefix(4)
            let extra = Set(tensors.keys).subtracting(expected.keys).sorted().prefix(4)
            throw UnlimitedOCRError.checkpoint("missing \(missing); unexpected \(extra)")
        }
        for (name, shape) in expected {
            guard let tensor = tensors[name], tensor.shape == shape,
                  tensor.dtype == (name.hasSuffix("position_ids") ? .int64 : .bfloat16) else {
                throw UnlimitedOCRError.checkpoint("shape or dtype mismatch for \(name); expected \(shape)")
            }
        }
    }

    private static let global: [String: [Int]] = [
        "lm_head.weight": [129280, 1280],
        "model.embed_tokens.weight": [129280, 1280],
        "model.image_newline": [1280],
        "model.norm.weight": [1280],
        "model.projector.layers.bias": [1280],
        "model.projector.layers.weight": [1280, 2048],
        "model.sam_model.neck.0.weight": [256, 768, 1, 1],
        "model.sam_model.neck.1.bias": [256],
        "model.sam_model.neck.1.weight": [256],
        "model.sam_model.neck.2.weight": [256, 256, 3, 3],
        "model.sam_model.neck.3.bias": [256],
        "model.sam_model.neck.3.weight": [256],
        "model.sam_model.net_2.weight": [512, 256, 3, 3],
        "model.sam_model.net_3.weight": [1024, 512, 3, 3],
        "model.sam_model.patch_embed.proj.bias": [768],
        "model.sam_model.patch_embed.proj.weight": [768, 3, 16, 16],
        "model.sam_model.pos_embed": [1, 64, 64, 768],
        "model.view_seperator": [1280],
        "model.vision_model.embeddings.class_embedding": [1024],
        "model.vision_model.embeddings.patch_embedding.weight": [1024, 3, 14, 14],
        "model.vision_model.embeddings.position_embedding.weight": [257, 1024],
        "model.vision_model.embeddings.position_ids": [1, 257],
        "model.vision_model.pre_layrnorm.bias": [1024],
        "model.vision_model.pre_layrnorm.weight": [1024],
    ]

    private static let decoder: [String: [Int]] = [
        "input_layernorm.weight": [1280],
        "post_attention_layernorm.weight": [1280],
        "self_attn.dash_router.head_cls": [128],
        "self_attn.dash_router.k_norm.weight": [128],
        "self_attn.dash_router.k_proj.weight": [128, 1280],
        "self_attn.dash_router.q_norm.weight": [128],
        "self_attn.dash_router.q_proj.weight": [128, 1280],
        "self_attn.indexer.k_norm.weight": [128],
        "self_attn.indexer.k_proj.weight": [128, 1280],
        "self_attn.indexer.q_norm.weight": [128],
        "self_attn.indexer.q_proj.weight": [128, 1280],
        "self_attn.k_proj.weight": [1280, 1280],
        "self_attn.o_proj.weight": [1280, 1280],
        "self_attn.q_proj.weight": [1280, 1280],
        "self_attn.v_proj.weight": [1280, 1280],
    ]

    private static let dense: [String: [Int]] = [
        "mlp.down_proj.weight": [1280, 6848],
        "mlp.gate_proj.weight": [6848, 1280],
        "mlp.up_proj.weight": [6848, 1280],
    ]

    private static let moe: [String: [Int]] = [
        "mlp.experts.{expert}.down_proj.weight": [1280, 896],
        "mlp.experts.{expert}.gate_proj.weight": [896, 1280],
        "mlp.experts.{expert}.up_proj.weight": [896, 1280],
        "mlp.gate.weight": [64, 1280],
        "mlp.shared_experts.down_proj.weight": [1280, 1792],
        "mlp.shared_experts.gate_proj.weight": [1792, 1280],
        "mlp.shared_experts.up_proj.weight": [1792, 1280],
    ]

    private static let sam: [String: [Int]] = [
        "attn.proj.bias": [768],
        "attn.proj.weight": [768, 768],
        "attn.qkv.bias": [2304],
        "attn.qkv.weight": [2304, 768],
        "global:attn.rel_pos_h": [127, 64],
        "global:attn.rel_pos_w": [127, 64],
        "local:attn.rel_pos_h": [27, 64],
        "local:attn.rel_pos_w": [27, 64],
        "mlp.lin1.bias": [3072],
        "mlp.lin1.weight": [3072, 768],
        "mlp.lin2.bias": [768],
        "mlp.lin2.weight": [768, 3072],
        "norm1.bias": [768],
        "norm1.weight": [768],
        "norm2.bias": [768],
        "norm2.weight": [768],
    ]

    private static let clip: [String: [Int]] = [
        "layer_norm1.bias": [1024],
        "layer_norm1.weight": [1024],
        "layer_norm2.bias": [1024],
        "layer_norm2.weight": [1024],
        "mlp.fc1.bias": [4096],
        "mlp.fc1.weight": [4096, 1024],
        "mlp.fc2.bias": [1024],
        "mlp.fc2.weight": [1024, 4096],
        "self_attn.out_proj.bias": [1024],
        "self_attn.out_proj.weight": [1024, 1024],
        "self_attn.qkv_proj.bias": [3072],
        "self_attn.qkv_proj.weight": [3072, 1024],
    ]
}
