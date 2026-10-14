#if !os(iOS)
import Foundation
import MLX
import MLXNN

extension Drama3Model {
    public func load(root: URL) throws {
        let source = try Drama3Weights(root: root)
        try validateLayout(source)
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened().filter {
            !$0.0.hasSuffix(".qkNormWeightBF16")
        })
        var loaded = Set<String>()
        func install(_ key: String, _ value: MLXArray) throws {
            guard let target = expected[key], target.shape == value.shape,
                  target.dtype == .uint32 ? value.dtype == .uint32 : value.dtype.isFloatingPoint,
                  loaded.insert(key).inserted else {
                throw Drama3Error.invalid("unexpected, duplicate, or incompatible tensor \(key): \(value.shape)")
            }
            try update(parameters: ModuleParameters.unflattened([(key, value)]), verify: .noUnusedKeys)
            MLX.eval(value)
        }
        for original in source.index.weightMap.keys.sorted() {
            if original.contains(".mlp.experts.") || original.hasSuffix("_scale_inv") { continue }
            guard let key = Self.mappedKey(original) else { throw Drama3Error.invalid("unknown tensor namespace: \(original)") }
            var value = try source.read(original)
            if key.hasSuffix(".conv1d.weight") { value = value.transposed(0, 2, 1) }
            try install(key, value)
        }
        for layer in 0..<config.text.numHiddenLayers {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                var packed: [MLXArray] = [], scales: [MLXArray] = [], biases: [MLXArray] = []
                for expert in 0..<config.text.numExperts {
                    try Task.checkCancellation()
                    let original = "model.language_model.layers.\(layer).mlp.experts.\(expert).\(projection)"
                    let q: (MLXArray, MLXArray, MLXArray)
                    if config.quantization != nil {
                        q = (try source.read(original + ".weight"), try source.read(original + ".scales"),
                             try source.read(original + ".biases"))
                    } else {
                        let result = MLX.quantized(try source.decodedWeight(original + ".weight"), groupSize: 64, bits: 4)
                        guard let bias = result.biases else { throw Drama3Error.invalid("affine quantizer omitted biases") }
                        q = (result.wq, result.scales, bias)
                    }
                    MLX.eval(q.0, q.1, q.2)
                    packed.append(q.0); scales.append(q.1); biases.append(q.2)
                }
                let base = "backbone.layers.\(layer).mlp.switch_mlp.\(projection)"
                try install(base + ".weight", MLX.stacked(packed))
                try install(base + ".scales", MLX.stacked(scales))
                try install(base + ".biases", MLX.stacked(biases))
                Memory.clearCache()
            }
        }
        guard loaded == Set(expected.keys) else {
            throw Drama3Error.invalid("checkpoint missing parameters: \(Set(expected.keys).subtracting(loaded).sorted().prefix(8))")
        }
    }

    /// Validate complete shapes and coverage before allocating or publishing weights.
    public func validateLayout(_ source: Drama3Weights) throws {
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened().filter {
            !$0.0.hasSuffix(".qkNormWeightBF16")
        }.map { ($0.0, $0.1.shape) })
        var seen = Set<String>(), expertCoverage: [String: Set<Int>] = [:]
        for name in source.index.weightMap.keys {
            let metadata = try source.metadata(name)
            if let marker = name.range(of: ".mlp.experts.") {
                let layer = String(name[..<marker.lowerBound]).replacingOccurrences(of: "model.language_model.", with: "backbone.")
                let parts = name[marker.upperBound...].split(separator: ".").map(String.init)
                guard parts.count == 3, let expert = Int(parts[0]), (0..<config.text.numExperts).contains(expert),
                      ["gate_proj", "up_proj", "down_proj"].contains(parts[1]) else {
                    throw Drama3Error.invalid("invalid expert namespace: \(name)")
                }
                let base = layer + ".mlp.switch_mlp." + parts[1]
                if config.quantization == nil {
                    guard parts[2] == "weight" || parts[2] == "weight_scale_inv",
                          let shape = expected[base + ".weight"] else {
                        throw Drama3Error.invalid("unexpected source expert: \(name)")
                    }
                    let rows = shape[1], columns = shape[2] * 8
                    if parts[2] == "weight_scale_inv" {
                        guard metadata.dtype == "F32", metadata.shape == [rows / 128, columns / 128] else {
                            throw Drama3Error.invalid("invalid expert block scales: \(name)")
                        }
                        continue
                    }
                    guard metadata.dtype == "F8_E4M3" || metadata.dtype == "F8_E4M3FN",
                          metadata.shape == [rows, columns], source.index.weightMap[name + "_scale_inv"] != nil else {
                        throw Drama3Error.invalid("invalid FP8 source expert: \(name)")
                    }
                    for suffix in ["weight", "scales", "biases"] { expertCoverage[base + "." + suffix, default: []].insert(expert) }
                } else {
                    let key = base + "." + parts[2]
                    guard ["weight", "scales", "biases"].contains(parts[2]),
                          let shape = expected[key], metadata.shape == Array(shape.dropFirst()),
                          parts[2] == "weight" ? metadata.dtype == "U32" : ["BF16", "F16", "F32"].contains(metadata.dtype) else {
                        throw Drama3Error.invalid("invalid packed expert: \(name)")
                    }
                    expertCoverage[key, default: []].insert(expert)
                }
            } else {
                guard let key = Self.mappedKey(name), let shape = expected[key] else {
                    throw Drama3Error.invalid("unknown checkpoint parameter: \(name)")
                }
                let actual = key.hasSuffix(".conv1d.weight") && metadata.shape.count == 3
                    ? [metadata.shape[0], metadata.shape[2], metadata.shape[1]] : metadata.shape
                guard actual == shape, ["BF16", "F16", "F32"].contains(metadata.dtype), seen.insert(key).inserted else {
                    throw Drama3Error.invalid("incompatible checkpoint tensor: \(name)")
                }
            }
        }
        for (key, experts) in expertCoverage {
            guard experts.count == config.text.numExperts else { throw Drama3Error.invalid("incomplete expert bank: \(key)") }
            seen.insert(key)
        }
        guard seen == Set(expected.keys) else {
            throw Drama3Error.invalid("incomplete checkpoint parameter coverage")
        }
    }

    public static func mappedKey(_ name: String) -> String? {
        if name.hasPrefix("model.language_model.") {
            return "backbone." + name.dropFirst("model.language_model.".count)
        }
        if name == "lm_head.weight" { return name }
        if name == "audio_decoder.codebook_embeddings.weight" { return "tower.codebook_embeddings.weight" }
        if name.hasPrefix("audio_decoder_mtp.heads.") {
            return "tower.heads." + name.dropFirst("audio_decoder_mtp.heads.".count)
        }
        var mapped: String
        if name.hasPrefix("audio_decoder_mtp_shared.layers.") {
            mapped = "tower.shared." + name.dropFirst("audio_decoder_mtp_shared.layers.".count)
        } else if name.hasPrefix("audio_decoder_mtp.layers.") {
            mapped = "tower.residual." + name.dropFirst("audio_decoder_mtp.layers.".count)
        } else { return nil }
        let replacements = [
            ("semantic_embedding.", "token_embedding."), ("codebook_embedding.", "token_embedding."),
            ("transformer_layer.attention.wqkv.", "qkv."), ("transformer_layer.attention.wo.", "out."),
            ("transformer_layer.attention.q_norm.", "q_norm."), ("transformer_layer.attention.k_norm.", "k_norm."),
            ("transformer_layer.attention_norm.", "input_norm."), ("transformer_layer.ffn_norm.", "post_norm."),
            ("transformer_layer.feed_forward.w1.", "gate."), ("transformer_layer.feed_forward.w3.", "up."),
            ("transformer_layer.feed_forward.w2.", "down.")
        ]
        for (old, new) in replacements { mapped = mapped.replacingOccurrences(of: old, with: new) }
        return mapped
    }
}

#endif
