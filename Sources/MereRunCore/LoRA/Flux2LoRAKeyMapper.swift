import Foundation
import MLX

public enum Flux2LoRAKeyMapper {

    public static func map(
        baseKey: String,
        down: MLXArray,
        up: MLXArray
    ) -> [String: (down: MLXArray, up: MLXArray)] {
        let normalized = normalizeBaseKey(baseKey)

        // Try global/embedder layer mapping first
        if let global = mapGlobalKey(normalized) {
            return [global: (down: down, up: up)]
        }

        if let mapped = mapBFLKey(normalized, down: down, up: up) {
            return mapped
        }

        if let standard = mapStandardKey(normalized) {
            return [standard: (down: down, up: up)]
        }

        return [:]
    }

    private static func mapGlobalKey(_ key: String) -> String? {
        let globalMappings: [String: String] = [
            // BFL/PEFT patterns → MereRun module paths
            "img_in": "x_embedder",
            "txt_in": "context_embedder",
            "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
            "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
            "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
            "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
            "single_stream_modulation.lin": "single_stream_modulation.linear",
            "final_layer.linear": "proj_out",

            // Direct mflux patterns (architecture difference: mflux has linear_1/2 directly)
            "time_guidance_embed.linear_1": "time_guidance_embed.timestep_embedder.linear_1",
            "time_guidance_embed.linear_2": "time_guidance_embed.timestep_embedder.linear_2",
        ]

        // Direct matches
        if let mapped = globalMappings[key] {
            return mapped
        }

        // Identity matches (already correct)
        let identityKeys = [
            "x_embedder", "context_embedder",
            "double_stream_modulation_img.linear",
            "double_stream_modulation_txt.linear",
            "single_stream_modulation.linear",
            "norm_out.linear", "proj_out",
        ]
        if identityKeys.contains(key) {
            return key
        }

        return nil
    }

    private static func normalizeBaseKey(_ key: String) -> String {
        var normalized = key

        let prefixesToRemove = [
            "base_model.model.",
            "diffusion_model.",
            "lora_unet_",
            "transformer.",
            "model.",
        ]

        for prefix in prefixesToRemove where normalized.hasPrefix(prefix) {
            normalized = String(normalized.dropFirst(prefix.count))
            break
        }

        if normalized.hasSuffix(".weight") {
            normalized = String(normalized.dropLast(".weight".count))
        }

        return normalized
    }

    private static func mapStandardKey(_ key: String) -> String? {
        var mapped = key

        // Diffusers feed-forward naming -> our naming.
        mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj", with: ".ff.linear_in")
        mapped = mapped.replacingOccurrences(of: ".ff.net.2", with: ".ff.linear_out")

        mapped = mapped.replacingOccurrences(of: ".ff_context.net.0.proj", with: ".ff_context.linear_in")
        mapped = mapped.replacingOccurrences(of: ".ff_context.net.2", with: ".ff_context.linear_out")

        // Alternative linear names seen in some exporters.
        mapped = mapped.replacingOccurrences(of: ".ff.linear1", with: ".ff.linear_in")
        mapped = mapped.replacingOccurrences(of: ".ff.linear2", with: ".ff.linear_out")

        mapped = mapped.replacingOccurrences(of: ".ff_context.linear1", with: ".ff_context.linear_in")
        mapped = mapped.replacingOccurrences(of: ".ff_context.linear2", with: ".ff_context.linear_out")

        // Joint attention output is an array `[Linear]` in our model: `to_out.0`.
        if mapped.hasPrefix("transformer_blocks."),
           mapped.contains(".attn.to_out"),
           !mapped.contains(".attn.to_out.0") {
            mapped = mapped.replacingOccurrences(of: ".attn.to_out", with: ".attn.to_out.0")
        }

        // Some FLUX LoRAs use `attn.proj` for the fused single-block projection.
        mapped = mapped.replacingOccurrences(of: ".attn.proj", with: ".attn.to_qkv_mlp_proj")
        mapped = mapped.replacingOccurrences(of: ".proj_out", with: ".attn.to_out")

        return mapped.isEmpty ? nil : mapped
    }

    private static func mapBFLKey(
        _ key: String,
        down: MLXArray,
        up: MLXArray
    ) -> [String: (down: MLXArray, up: MLXArray)]? {
        if let (block, remainder) = parseBFL(key: key, prefix: "double_blocks_") {
            return mapBFLDoubleBlock(block: block, remainder: remainder, down: down, up: up)
        }
        if let (block, remainder) = parseBFL(key: key, prefix: "single_blocks_") {
            return mapBFLSingleBlock(block: block, remainder: remainder, down: down, up: up)
        }
        return nil
    }

    private static func parseBFL(key: String, prefix: String) -> (block: Int, remainder: String)? {
        guard key.hasPrefix(prefix) else { return nil }
        let rest = key.dropFirst(prefix.count)
        guard let underscoreIndex = rest.firstIndex(of: "_") else { return nil }
        let blockString = rest[..<underscoreIndex]
        guard let block = Int(blockString) else { return nil }
        let remainderStart = rest.index(after: underscoreIndex)
        let remainder = String(rest[remainderStart...])
        return (block, remainder)
    }

    private static func mapBFLDoubleBlock(
        block: Int,
        remainder: String,
        down: MLXArray,
        up: MLXArray
    ) -> [String: (down: MLXArray, up: MLXArray)] {
        switch remainder {
        case "img_attn_qkv":
            return splitQKV(
                down: down,
                up: up,
                targets: [
                    "transformer_blocks.\(block).attn.to_q",
                    "transformer_blocks.\(block).attn.to_k",
                    "transformer_blocks.\(block).attn.to_v",
                ]
            )
        case "txt_attn_qkv":
            return splitQKV(
                down: down,
                up: up,
                targets: [
                    "transformer_blocks.\(block).attn.add_q_proj",
                    "transformer_blocks.\(block).attn.add_k_proj",
                    "transformer_blocks.\(block).attn.add_v_proj",
                ]
            )
        case "img_attn_proj":
            return ["transformer_blocks.\(block).attn.to_out.0": (down: down, up: up)]
        case "txt_attn_proj":
            return ["transformer_blocks.\(block).attn.to_add_out": (down: down, up: up)]
        case "img_mlp_0":
            return ["transformer_blocks.\(block).ff.linear_in": (down: down, up: up)]
        case "img_mlp_2":
            return ["transformer_blocks.\(block).ff.linear_out": (down: down, up: up)]
        case "txt_mlp_0":
            return ["transformer_blocks.\(block).ff_context.linear_in": (down: down, up: up)]
        case "txt_mlp_2":
            return ["transformer_blocks.\(block).ff_context.linear_out": (down: down, up: up)]
        default:
            return [:]
        }
    }

    private static func mapBFLSingleBlock(
        block: Int,
        remainder: String,
        down: MLXArray,
        up: MLXArray
    ) -> [String: (down: MLXArray, up: MLXArray)] {
        switch remainder {
        case "linear1":
            return [
                "single_transformer_blocks.\(block).attn.to_qkv_mlp_proj": expandBFLPackedSingleLinear1(
                    down: down,
                    up: up
                )
            ]
        case "linear2":
            return ["single_transformer_blocks.\(block).attn.to_out": (down: down, up: up)]
        default:
            return [:]
        }
    }

    /// BFL `single_blocks_{i}_linear1` LoRAs sometimes pack Q/K/V/MLP into a single key.
    ///
    /// When `down.shape[0] == 4 * up.shape[1]`, expand `up` into a block-diagonal matrix so
    /// `matmul(upExpanded, down)` matches the fused `to_qkv_mlp_proj` weight update.
    private static func expandBFLPackedSingleLinear1(
        down: MLXArray,
        up: MLXArray
    ) -> (down: MLXArray, up: MLXArray) {
        guard down.ndim == 2, up.ndim == 2 else {
            return (down: down, up: up)
        }

        let rank = up.shape[1]
        guard down.shape[0] == 4 * rank else {
            return (down: down, up: up)
        }

        let hidden = down.shape[1]
        let outputDim = up.shape[0]
        let qkvDim = 3 * hidden
        let mlpDim = outputDim - qkvDim
        guard mlpDim >= 0 else {
            return (down: down, up: up)
        }

        let qUp = up[0..<hidden, 0...]
        let kUp = up[hidden..<(2 * hidden), 0...]
        let vUp = up[(2 * hidden)..<qkvDim, 0...]
        let mlpUp = up[qkvDim..<outputDim, 0...]

        let zerosQRight = MLX.zeros([hidden, 3 * rank], dtype: up.dtype)
        let qExpanded = MLX.concatenated([qUp, zerosQRight], axis: 1)

        let zerosKLeft = MLX.zeros([hidden, rank], dtype: up.dtype)
        let zerosKRight = MLX.zeros([hidden, 2 * rank], dtype: up.dtype)
        let kExpanded = MLX.concatenated([zerosKLeft, kUp, zerosKRight], axis: 1)

        let zerosVLeft = MLX.zeros([hidden, 2 * rank], dtype: up.dtype)
        let zerosVRight = MLX.zeros([hidden, rank], dtype: up.dtype)
        let vExpanded = MLX.concatenated([zerosVLeft, vUp, zerosVRight], axis: 1)

        let zerosMLPLeft = MLX.zeros([mlpDim, 3 * rank], dtype: up.dtype)
        let mlpExpanded = MLX.concatenated([zerosMLPLeft, mlpUp], axis: 1)

        let upExpanded = MLX.concatenated([qExpanded, kExpanded, vExpanded, mlpExpanded], axis: 0)
        return (down: down, up: upExpanded)
    }

    private static func splitQKV(
        down: MLXArray,
        up: MLXArray,
        targets: [String]
    ) -> [String: (down: MLXArray, up: MLXArray)] {
        guard targets.count == 3, up.ndim == 2, down.ndim == 2 else { return [:] }
        guard up.shape[0] % 3 == 0 else { return [:] }

        let upChunk = up.shape[0] / 3
        let upSplits = [
            up[0..<upChunk, 0...],
            up[upChunk..<(2 * upChunk), 0...],
            up[(2 * upChunk)..<(3 * upChunk), 0...],
        ]

        let rank = up.shape[1]
        let downSplits: [MLXArray]
        if down.shape[0] == rank {
            downSplits = [down, down, down]
        } else if down.shape[0] == 3 * rank {
            downSplits = [
                down[0..<rank, 0...],
                down[rank..<(2 * rank), 0...],
                down[(2 * rank)..<(3 * rank), 0...],
            ]
        } else if down.shape[0] % 3 == 0 {
            let downChunk = down.shape[0] / 3
            downSplits = [
                down[0..<downChunk, 0...],
                down[downChunk..<(2 * downChunk), 0...],
                down[(2 * downChunk)..<(3 * downChunk), 0...],
            ]
        } else {
            downSplits = [down, down, down]
        }

        var mapped: [String: (down: MLXArray, up: MLXArray)] = [:]
        for (target, pair) in zip(targets, zip(downSplits, upSplits)) {
            mapped[target] = (down: pair.0, up: pair.1)
        }
        return mapped
    }
}
