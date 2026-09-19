import Foundation
import MLX

/// Converts original Tencent/Hugging Face checkpoint layouts in memory.
public enum AuKCheckpoint {
    public static func diffusion(_ url: URL) throws -> (AuKTensorStore, AuKTensorStore) {
        let raw = try loadArrays(url: url)
        var tensors = [String: MLXArray](), fusion = [String: MLXArray]()
        for (key, value) in raw where !key.hasPrefix("text_encoder.") {
            try requireFloating(value, key: key)
            if ["layer_weights", "layer_scale"].contains(key) { fusion[key] = value.asType(.float32); continue }
            var name = key.hasPrefix("transformer.") ? String(key.dropFirst(12)) : key
            if name == "rotary_embed.inv_freq" { fusion["inv_freq"] = value.asType(.float32); continue }
            name = name.replacingOccurrences(of: "time_mlp.2.", with: "time_mlp.1.")
                .replacingOccurrences(of: "conv_pos_embed.conv1d.2.", with: "conv_pos_embed.conv1d.1.")
            let tensor = value.asType(.float32)
            tensors[name] = name.contains(".conv1d.") && name.hasSuffix(".weight")
                ? tensor.transposed(0, 2, 1) : tensor
        }
        return (AuKTensorStore(tensors), AuKTensorStore(fusion))
    }

    public static func thinker(_ root: URL) throws -> AuKTensorStore {
        struct Index: Decodable { let weightMap: [String: String]
            enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
        }
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        let shards: [String]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL))
            shards = Set(index.weightMap.filter { wantedThinkerKey($0.key) }.values).sorted()
        } else { shards = ["model.safetensors"] }
        var tensors = [String: MLXArray]()
        for shard in shards {
            guard URL(fileURLWithPath: shard).lastPathComponent == shard else {
                throw AuKError.invalid("Invalid Thinker shard filename: \(shard)")
            }
            for (key, value) in try loadArrays(url: root.appendingPathComponent(shard)) where wantedThinkerKey(key) {
                try requireFloating(value, key: key)
                var name = String(key.dropFirst("thinker.".count))
                if name.hasPrefix("model.") { name = String(name.dropFirst(6)) }
                let tensor = value.asType(.float32)
                tensors[name] = name.hasPrefix("audio_tower.conv") && name.hasSuffix(".weight")
                    ? tensor.transposed(0, 2, 1) : tensor
            }
        }
        return AuKTensorStore(tensors)
    }

    private static func requireFloating(_ value: MLXArray, key: String) throws {
        guard value.dtype.isFloatingPoint else {
            throw AuKError.invalid("AuK requires original floating-point checkpoints; unsupported tensor: \(key)")
        }
    }

    static func wantedThinkerKey(_ key: String) -> Bool {
        key.hasPrefix("thinker.model.") || key.hasPrefix("thinker.audio_tower.")
    }

    public static func vae(_ url: URL) throws -> AuKTensorStore {
        let raw = try loadArrays(url: url)
        var folded = [String: MLXArray]()
        for (key, value) in raw {
            try requireFloating(value, key: key)
            if key.hasSuffix(".weight_v") { continue }
            if key.hasSuffix(".weight_g") {
                let base = String(key.dropLast(".weight_g".count))
                guard let direction = raw[base + ".weight_v"] else {
                    throw AuKError.invalid("Missing VAE weight normalization direction: \(base)")
                }
                let v = direction.asType(.float32)
                let norm = sqrt(sum(v * v, axes: Array(1..<v.ndim), keepDims: true))
                folded[base + ".weight"] = value.asType(.float32) * v / norm
            } else { folded[key] = value.asType(.float32) }
        }
        var result = [String: MLXArray]()
        func copy(_ source: String, _ destination: String, convolution: Bool = false, transpose: Bool = false) throws {
            guard let value = folded[source] else { throw AuKError.invalid("Missing VAE tensor: \(source)") }
            result[destination] = transpose ? value.transposed(1, 2, 0) : (convolution ? value.transposed(0, 2, 1) : value)
        }
        func conv(_ source: String, _ destination: String, transpose: Bool = false, bias: Bool = true) throws {
            try copy(source + ".weight", destination + ".weight", convolution: true, transpose: transpose)
            if bias { try copy(source + ".bias", destination + ".bias") }
        }
        try copy("global_mean", "global_mean")
        try copy("global_log_std", "global_log_std")
        try conv("audio_encoder.generator.0.layer", "audio_encoder.pre")
        for stage in 0..<6 {
            let source = "audio_encoder.generator."
            let target = "audio_encoder.stages.\(stage)"
            try conv(source + "\(2 + stage * 3).layer", target + ".down")
            for layer in 0..<6 {
                for (old, new) in [(1, 0), (3, 1)] {
                    try conv(source + "\(3 + stage * 3).layers.\(layer).\(old)", target + ".stack.layers.\(layer).\(new)")
                }
            }
        }
        try conv("audio_encoder.generator.20.layer", "audio_encoder.post")
        try conv("conv_pre", "decoder.conv_pre")
        try conv("conv_post", "decoder.conv_post", bias: false)
        for stage in 0..<6 { try conv("ups.\(stage).0", "decoder.ups.\(stage)", transpose: true) }
        for block in 0..<18 {
            let source = "resblocks.\(block)"
            for layer in 0..<3 {
                for group in ["convs1", "convs2"] {
                    try conv(source + ".\(group).\(layer)", "decoder." + source + ".\(group).\(layer)")
                }
            }
            for activation in 0..<6 {
                for parameter in ["alpha", "beta"] {
                    let key = source + ".activations.\(activation).act.\(parameter)"
                    try copy(key, "decoder." + key)
                }
            }
        }
        for parameter in ["alpha", "beta"] {
            try copy("activation_post.act.\(parameter)", "decoder.activation_post.act.\(parameter)")
        }
        return AuKTensorStore(result)
    }
}
