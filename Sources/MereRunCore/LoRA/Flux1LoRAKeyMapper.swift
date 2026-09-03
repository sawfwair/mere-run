import Foundation
import MLX

enum Flux1LoRAKeyMapper {
    static func map(
        baseKey: String,
        down: MLXArray,
        up: MLXArray
    ) -> [String: (down: MLXArray, up: MLXArray)] {
        let key = normalize(baseKey)
        if let global = globalKey(key) {
            return [global: (down, up)]
        }
        if let mapped = bflDoubleBlock(key, down: down, up: up) {
            return mapped
        }
        return [standardKey(key): (down, up)]
    }

    private static func normalize(_ key: String) -> String {
        var result = key
        for prefix in ["base_model.model.", "diffusion_model.", "lora_unet_", "transformer.", "model."]
        where result.hasPrefix(prefix) {
            result = String(result.dropFirst(prefix.count))
            break
        }
        if result.hasSuffix(".weight") {
            result = String(result.dropLast(".weight".count))
        }
        return result
    }

    private static func standardKey(_ key: String) -> String {
        key
            .replacingOccurrences(of: ".ff.net.0.proj", with: ".ff.input.proj")
            .replacingOccurrences(of: ".ff.net.2", with: ".ff.output")
            .replacingOccurrences(of: ".ff_context.net.0.proj", with: ".ff_context.input.proj")
            .replacingOccurrences(of: ".ff_context.net.2", with: ".ff_context.output")
    }

    private static func globalKey(_ key: String) -> String? {
        let mappings = [
            "img_in": "x_embedder",
            "txt_in": "context_embedder",
            "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
            "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
            "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
            "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
            "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
            "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
            "final_layer.adaLN_modulation.1": "norm_out.linear",
            "final_layer.linear": "proj_out",
        ]
        return mappings[key]
    }

    private static func bflDoubleBlock(
        _ key: String,
        down: MLXArray,
        up: MLXArray
    ) -> [String: (down: MLXArray, up: MLXArray)]? {
        guard let parsed = parseBlock(key, prefix: "double_blocks_")
                ?? parseDottedBlock(key, prefix: "double_blocks") else {
            return nil
        }
        let prefix = "transformer_blocks.\(parsed.index)"
        switch parsed.remainder {
        case "img_attn_qkv":
            return splitQKV(down: down, up: up, prefix: "\(prefix).attn.to_")
        case "txt_attn_qkv":
            return splitQKV(down: down, up: up, prefix: "\(prefix).attn.add_")
        case "img_attn_proj":
            return ["\(prefix).attn.to_out.0": (down, up)]
        case "txt_attn_proj":
            return ["\(prefix).attn.to_add_out": (down, up)]
        case "img_mlp_0":
            return ["\(prefix).ff.input.proj": (down, up)]
        case "img_mlp_2":
            return ["\(prefix).ff.output": (down, up)]
        case "txt_mlp_0":
            return ["\(prefix).ff_context.input.proj": (down, up)]
        case "txt_mlp_2":
            return ["\(prefix).ff_context.output": (down, up)]
        default:
            return [:]
        }
    }

    private static func parseBlock(_ key: String, prefix: String) -> (index: Int, remainder: String)? {
        guard key.hasPrefix(prefix) else { return nil }
        let tail = key.dropFirst(prefix.count)
        guard let separator = tail.firstIndex(of: "_"),
              let index = Int(tail[..<separator]) else { return nil }
        return (index, String(tail[tail.index(after: separator)...]))
    }

    private static func parseDottedBlock(_ key: String, prefix: String) -> (index: Int, remainder: String)? {
        let components = key.split(separator: ".")
        guard components.count >= 3,
              components[0] == prefix,
              let index = Int(components[1]) else { return nil }
        return (index, components.dropFirst(2).joined(separator: "_"))
    }

    private static func splitQKV(
        down: MLXArray,
        up: MLXArray,
        prefix: String
    ) -> [String: (down: MLXArray, up: MLXArray)] {
        guard down.ndim == 2, up.ndim == 2, up.dim(0) % 3 == 0 else { return [:] }
        let outputSize = up.dim(0) / 3
        let rank = up.dim(1)
        let upParts = (0..<3).map { up[$0 * outputSize..<($0 + 1) * outputSize, 0...] }
        let downParts: [MLXArray]
        if down.dim(0) == rank {
            downParts = [down, down, down]
        } else if down.dim(0) == rank * 3 {
            downParts = (0..<3).map { down[$0 * rank..<($0 + 1) * rank, 0...] }
        } else {
            return [:]
        }
        return Dictionary(uniqueKeysWithValues: zip(["q", "k", "v"], zip(downParts, upParts)).map {
            ("\(prefix)\($0.0)", (down: $0.1.0, up: $0.1.1))
        })
    }
}
