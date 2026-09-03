import Foundation
import MLX
import MLXNN

public enum Cosmos3ModelLoaderError: LocalizedError, Sendable {
    case checkpointInventoryMismatch(missing: [String], unexpected: [String])

    public var errorDescription: String? {
        switch self {
        case .checkpointInventoryMismatch(let missing, let unexpected):
            let missingText = missing.prefix(8).joined(separator: ", ")
            let unexpectedText = unexpected.prefix(8).joined(separator: ", ")
            return "Cosmos3 checkpoint inventory mismatch"
                + " (missing: [\(missingText)], unexpected: [\(unexpectedText)])."
        }
    }
}

public enum Cosmos3CheckpointInventory {
    public static func transformerKeys(configuration: Cosmos3TransformerConfiguration) -> Set<String> {
        var keys: Set<String> = [
            "embed_tokens.weight",
            "norm.weight",
            "norm_moe_gen.weight",
            "proj_in.weight",
            "proj_in.bias",
            "proj_out.weight",
            "proj_out.bias",
            "time_embedder.linear_1.weight",
            "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight",
            "time_embedder.linear_2.bias",
        ]
        if configuration.includesLanguageModelHead {
            keys.insert("lm_head.weight")
        }
        if configuration.generatesActions {
            keys.formUnion([
                "action_modality_embed",
                "action_proj_in.fc.weight",
                "action_proj_in.bias.weight",
                "action_proj_out.fc.weight",
                "action_proj_out.bias.weight",
            ])
        }
        for index in 0..<configuration.layerCount {
            let prefix = "layers.\(index)."
            keys.formUnion([
                prefix + "input_layernorm.weight",
                prefix + "input_layernorm_moe_gen.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "post_attention_layernorm_moe_gen.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "mlp.down_proj.weight",
                prefix + "mlp_moe_gen.up_proj.weight",
                prefix + "mlp_moe_gen.down_proj.weight",
                prefix + "self_attn.to_q.weight",
                prefix + "self_attn.to_k.weight",
                prefix + "self_attn.to_v.weight",
                prefix + "self_attn.to_out.weight",
                prefix + "self_attn.add_q_proj.weight",
                prefix + "self_attn.add_k_proj.weight",
                prefix + "self_attn.add_v_proj.weight",
                prefix + "self_attn.to_add_out.weight",
                prefix + "self_attn.norm_added_q.weight",
                prefix + "self_attn.norm_added_k.weight",
            ])
            if configuration.feedForwardActivation == .siluGated {
                keys.formUnion([
                    prefix + "mlp.gate_proj.weight",
                    prefix + "mlp_moe_gen.gate_proj.weight",
                ])
            }
            if configuration.normalizesUnderstandingQueriesAndKeys {
                keys.formUnion([
                    prefix + "self_attn.norm_q.weight",
                    prefix + "self_attn.norm_k.weight",
                ])
            }
            if configuration.normalizesUnderstandingKeysForGeneration {
                keys.insert(prefix + "self_attn.k_norm_und_for_gen.weight")
            }
        }
        return keys
    }

    private static func logicalKey(for storedKey: String) -> String {
        for suffix in [".scales", ".biases"] where storedKey.hasSuffix(suffix) {
            return String(storedKey.dropLast(suffix.count)) + ".weight"
        }
        return storedKey
    }

    static func isUnusedSoundKey(_ key: String) -> Bool {
        key == "audio_modality_embed" || key.hasPrefix("audio_proj_in.")
            || key.hasPrefix("audio_proj_out.")
    }

    public static func validateTransformerIndex(
        _ index: HFSafetensorsIndex,
        configuration: Cosmos3TransformerConfiguration
    ) throws {
        let expected = transformerKeys(configuration: configuration)
        let actual = Set(index.weightMap.keys.filter { !isUnusedSoundKey($0) }.map(logicalKey))
        let missing = expected.subtracting(actual).sorted()
        let unexpected = actual.subtracting(expected).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw Cosmos3ModelLoaderError.checkpointInventoryMismatch(
                missing: missing,
                unexpected: unexpected
            )
        }
    }
}

public enum Cosmos3ModelLoader {
    public static func loadReasonerVision(
        resources: Cosmos3Resources,
        dtype: DType = .bfloat16,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Cosmos3ReasonerVisionModel {
        let configuration = try resources.loadReasonerConfiguration()
        progress?("Loading Cosmos3-Edge SigLIP2 reasoner vision tower")
        let model = Cosmos3ReasonerVisionModel(configuration: configuration)
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.visionEncoderURL,
            to: model,
            dtype: dtype,
            verify: .none,
            include: {
                $0.hasPrefix("model.visual.") || $0.hasPrefix("model.projector.")
            },
            mapper: { key, value in
                [(String(key.dropFirst("model.".count)), value)]
            },
            batchSize: 8
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadTransformer(
        resources: Cosmos3Resources,
        dtype: DType = .bfloat16,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Cosmos3OmniTransformerModel {
        let configuration = try resources.loadTransformerConfiguration()
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: resources.transformerIndexURL)
        )
        try Cosmos3CheckpointInventory.validateTransformerIndex(
            index,
            configuration: configuration
        )
        progress?("Loading Cosmos3 transformer")
        let model = Cosmos3OmniTransformerModel(configuration: configuration)
        if let quantization = configuration.quantization {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: resources.transformerIndexURL,
                to: model,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                progressHandler: { shard in
                    progress?(
                        "Loading Cosmos3 transformer shard "
                            + "\(shard.shardIndex + 1)/\(shard.shardCount)"
                    )
                }
            )
            eval(model.parameters().flattened().map(\.1))
            return model
        }
        let shardNames = index.shardFilenames
        for (index, shardName) in shardNames.enumerated() {
            progress?("Loading Cosmos3 transformer shard \(index + 1)/\(shardNames.count)")
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.transformerRootURL.appendingPathComponent(shardName),
                to: model,
                dtype: dtype,
                verify: .none,
                include: { !Cosmos3CheckpointInventory.isUnusedSoundKey($0) },
                batchSize: 8
            )
        }
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadVAE(
        resources: Cosmos3Resources,
        dtype: DType = .bfloat16,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2VAEModel {
        let configuration = try resources.loadVAEConfiguration()
        progress?("Loading Cosmos3-Edge Wan 4x16x16 VAE")
        let model = Wan2VAEModel(configuration: Wan2VAEConfiguration(
            latentChannels: configuration.latentDimension,
            encoderDimensions: configuration.baseDimension,
            decoderDimensions: configuration.decoderBaseDimension,
            imagePatchSize: configuration.patchSize,
            blockResampleShortcut: true,
            decoderResampleReducesChannels: false,
            latentMean: configuration.latentMeans,
            latentStandardDeviation: configuration.latentStandardDeviations
        ))
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.vaeURL,
            to: model,
            dtype: dtype,
            verify: .none,
            mapper: vaeWeightMapper,
            batchSize: 8
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    static func vaeWeightMapper(key: String, value: MLXArray) -> [(String, MLXArray)] {
        guard let mappedKey = mapVAEKey(key) else { return [] }
        let mappedValue: MLXArray
        if key.hasSuffix(".gamma") {
            // Diffusers stores Wan RMSNorm scales in channel-first broadcast
            // form [C, 1, 1, 1]. The native VAE is channel-last, so retain the
            // parameters as a vector and let MLX broadcast over the last axis.
            mappedValue = value.reshaped(-1)
        } else if value.ndim == 5 {
            let transposed = value.transposed(0, 2, 3, 4, 1)
            mappedValue = transposed.reshaped(-1).reshaped(transposed.shape)
        } else if value.ndim == 4 {
            let transposed = value.transposed(0, 2, 3, 1)
            mappedValue = transposed.reshaped(-1).reshaped(transposed.shape)
        } else {
            mappedValue = value
        }
        return [(mappedKey, mappedValue)]
    }

    static func mapVAEKey(_ key: String) -> String? {
        if key.hasPrefix("quant_conv.") {
            return "conv1." + key.dropFirst("quant_conv.".count)
        }
        if key.hasPrefix("post_quant_conv.") {
            return "conv2." + key.dropFirst("post_quant_conv.".count)
        }
        for side in ["encoder", "decoder"] {
            if key.hasPrefix("\(side).conv_in.") {
                return "\(side).conv1." + key.dropFirst("\(side).conv_in.".count)
            }
            if key.hasPrefix("\(side).conv_out.") {
                return "\(side).head.layer_2." + key.dropFirst("\(side).conv_out.".count)
            }
            if key == "\(side).norm_out.gamma" {
                return "\(side).head.layer_0.gamma"
            }
            if let mapped = mapVAEMiddleKey(key, side: side) {
                return mapped
            }
        }
        if let mapped = mapVAEBlockKey(
            key,
            sourcePrefix: "encoder.down_blocks.",
            targetPrefix: "encoder.downsamples.",
            resampleName: "downsampler"
        ) {
            return mapped
        }
        return mapVAEBlockKey(
            key,
            sourcePrefix: "decoder.up_blocks.",
            targetPrefix: "decoder.upsamples.",
            resampleName: "upsampler"
        )
    }

    private static func mapVAEMiddleKey(_ key: String, side: String) -> String? {
        let residualPrefix = "\(side).mid_block.resnets."
        if key.hasPrefix(residualPrefix) {
            let tail = String(key.dropFirst(residualPrefix.count))
            let parts = tail.split(separator: ".", maxSplits: 1).map(String.init)
            guard parts.count == 2, let block = Int(parts[0]), block < 2,
                  let suffix = mapVAEResidualSuffix(parts[1]) else {
                return nil
            }
            return "\(side).middle.\(block == 0 ? 0 : 2).\(suffix)"
        }
        let attentionPrefix = "\(side).mid_block.attentions.0."
        guard key.hasPrefix(attentionPrefix) else { return nil }
        let suffix = String(key.dropFirst(attentionPrefix.count))
        let mappedSuffix: String
        switch suffix {
        case "norm.gamma": mappedSuffix = "norm.gamma"
        case "to_qkv.weight": mappedSuffix = "to_qkv_weight"
        case "to_qkv.bias": mappedSuffix = "to_qkv_bias"
        case "proj.weight": mappedSuffix = "proj_weight"
        case "proj.bias": mappedSuffix = "proj_bias"
        default: return nil
        }
        return "\(side).middle.1.\(mappedSuffix)"
    }

    private static func mapVAEBlockKey(
        _ key: String,
        sourcePrefix: String,
        targetPrefix: String,
        resampleName: String
    ) -> String? {
        guard key.hasPrefix(sourcePrefix) else { return nil }
        let tail = String(key.dropFirst(sourcePrefix.count))
        let parts = tail.split(separator: ".").map(String.init)
        guard parts.count >= 4, let block = Int(parts[0]) else { return nil }
        if parts[1] == "resnets", let residual = Int(parts[2]) {
            let suffix = parts.dropFirst(3).joined(separator: ".")
            guard let mappedSuffix = mapVAEResidualSuffix(suffix) else { return nil }
            return "\(targetPrefix)\(block).\(sourcePrefix.hasPrefix("encoder") ? "downsamples" : "upsamples").\(residual).\(mappedSuffix)"
        }
        guard parts[1] == resampleName else { return nil }
        let layerIndex = sourcePrefix.hasPrefix("encoder") ? 2 : 3
        let target = "\(targetPrefix)\(block).\(sourcePrefix.hasPrefix("encoder") ? "downsamples" : "upsamples").\(layerIndex)"
        if parts.dropFirst(2).joined(separator: ".") == "resample.1.weight" {
            return target + ".resample_weight"
        }
        if parts.dropFirst(2).joined(separator: ".") == "resample.1.bias" {
            return target + ".resample_bias"
        }
        if parts.dropFirst(2).joined(separator: ".") == "time_conv.weight" {
            return target + ".time_conv.weight"
        }
        if parts.dropFirst(2).joined(separator: ".") == "time_conv.bias" {
            return target + ".time_conv.bias"
        }
        return nil
    }

    private static func mapVAEResidualSuffix(_ suffix: String) -> String? {
        switch suffix {
        case "norm1.gamma": "residual.layer_0.gamma"
        case "conv1.weight": "residual.layer_2.weight"
        case "conv1.bias": "residual.layer_2.bias"
        case "norm2.gamma": "residual.layer_3.gamma"
        case "conv2.weight": "residual.layer_6.weight"
        case "conv2.bias": "residual.layer_6.bias"
        case "conv_shortcut.weight": "shortcut.weight"
        case "conv_shortcut.bias": "shortcut.bias"
        default: nil
        }
    }
}
