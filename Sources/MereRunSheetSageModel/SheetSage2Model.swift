import MLX
import MLXFast
import MLXNN

public final class SheetSage2DecoderCache {
    var keys: [MLXArray?]
    var values: [MLXArray?]
    var crossKeys: [MLXArray?]
    var crossValues: [MLXArray?]
    var offset = 0
    public init(layers: Int = 6) {
        keys = Array(repeating: nil, count: layers)
        values = keys
        crossKeys = keys
        crossValues = keys
    }
}

public struct SheetSage2Model {
    public let configuration: SheetSage2Configuration
    let encoder: MERT2Encoder
    let weights: SheetSage2Weights

    public init(configuration: SheetSage2Configuration, parent: [String: MLXArray], adapter: [String: MLXArray],
                dtype: DType = .float32) throws {
        self.configuration = configuration
        guard parent.values.allSatisfy({ $0.dtype == .float32 || $0.dtype == .bfloat16 }),
              adapter.values.allSatisfy({ $0.dtype == .float32 || $0.dtype == .bfloat16 }) else {
            throw SheetSage2Error.invalid("Expected FP32 or BF16 checkpoint tensors.")
        }
        var parent = parent.mapValues { $0.asType(.float32) }
        var remaining = adapter
        for layer in 0..<configuration.backbone.layers {
            for projection in ["query_proj", "key_proj", "value_proj", "out_proj"] {
                let name = "layers.\(layer).attn.\(projection)"
                guard let a = remaining.removeValue(forKey: "adapter." + name + ".lora_A.weight"),
                      let b = remaining.removeValue(forKey: "adapter." + name + ".lora_B.weight"),
                      let original = parent[name + ".weight"],
                      a.shape == [configuration.loraRank, configuration.backbone.hiddenSize],
                      b.shape == [configuration.backbone.hiddenSize, configuration.loraRank] else {
                    throw SheetSage2Error.invalid("Missing or incompatible attention adapter \(name).")
                }
                let merged = original + matmul(b.asType(.float32), a.asType(.float32)) * (configuration.loraAlpha / Float(configuration.loraRank))
                MLX.eval(merged)
                parent[name + ".weight"] = merged
            }
        }
        // Preserve frontend normalization buffers in FP32 when reducing the encoder's compute precision.
        encoder = try MERT2Encoder(configuration: configuration.backbone, arrays: Dictionary(uniqueKeysWithValues: parent.map {
            ($0.key, $0.key.hasPrefix("feature_extractor.") ? $0.value : $0.value.asType(dtype))
        }))
        var bank = SheetSage2Weights(values: remaining.mapValues { $0.asType(dtype) })
        var consumed = Set<String>()
        let width = configuration.hiddenSize
        try bank.require("layer_weight", [configuration.backbone.layers + 1], consumed: &consumed)
        try bank.requireLinear("encoder_projection", input: configuration.backbone.hiddenSize, output: width, consumed: &consumed)
        try bank.require("token_embedding.weight", [configuration.vocabSize, width], consumed: &consumed)
        try bank.require("decoder.embed_positions.weight", [configuration.maxOutputSequence + 2, width], consumed: &consumed)
        try bank.requireNorm("decoder.layernorm_embedding", size: width, consumed: &consumed)
        for layer in 0..<configuration.decoderLayers {
            let prefix = "decoder.layers.\(layer)"
            for attention in ["self_attn", "encoder_attn"] {
                for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    try bank.requireLinear(prefix + ".\(attention).\(projection)", input: width, output: width, consumed: &consumed)
                }
                try bank.requireNorm(prefix + ".\(attention)_layer_norm", size: width, consumed: &consumed)
            }
            try bank.requireNorm(prefix + ".final_layer_norm", size: width, consumed: &consumed)
            try bank.requireLinear(prefix + ".fc1", input: width, output: configuration.intermediateSize, consumed: &consumed)
            try bank.requireLinear(prefix + ".fc2", input: configuration.intermediateSize, output: width, consumed: &consumed)
        }
        guard Set(remaining.keys) == consumed else {
            throw SheetSage2Error.invalid("Unexpected SheetSage2 tensors: \(Set(remaining.keys).subtracting(consumed).sorted()).")
        }
        weights = bank
    }

    public func encode(_ audio: MLXArray, windowSamples: Int = 7_200_000,
                       progress: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        guard audio.ndim == 1, audio.size > 1024, audio.size <= windowSamples else {
            throw SheetSage2Error.invalid("Audio must fit one 300-second window at 24 kHz.")
        }
        let padded = concatenated([audio.asType(.float32), MLXArray.zeros([windowSamples - audio.size])])
        let mel = try encoder.mel(padded)
        let firstWeight = try encoder.weights.get("subsampling_module.0.convnext_layers.0.depthwise_block.1.weight")
        var hidden = try encoder.subsample(mel.asType(firstWeight.dtype))
        let mixing = try softmax(weights.get("layer_weight").asType(.float32))
        var mixed = hidden * mixing[0].asType(hidden.dtype)
        for index in 0..<configuration.backbone.layers {
            try Task.checkCancellation()
            hidden = try encoder.block(hidden, index: index)
            mixed = mixed + hidden * mixing[index + 1].asType(hidden.dtype)
            MLX.eval(hidden, mixed)
            progress?(index + 1, configuration.backbone.layers)
        }
        return try weights.linear(mixed, "encoder_projection")
    }

    public func logits(_ tokens: [Int], memory: MLXArray, cache: SheetSage2DecoderCache) throws -> MLXArray {
        guard !tokens.isEmpty, tokens.allSatisfy({ (0..<configuration.vocabSize).contains($0) }),
              cache.keys.count == configuration.decoderLayers,
              cache.offset + tokens.count <= configuration.maxOutputSequence,
              memory.ndim == 3, memory.dim(0) == 1, memory.dim(2) == configuration.hiddenSize else {
            throw SheetSage2Error.invalid("Invalid decoder tokens, memory, or context budget.")
        }
        let ids = MLXArray(tokens.map(Int32.init))
        let positions = MLXArray((cache.offset + 2)..<(cache.offset + 2 + tokens.count))
        var hidden = try (weights.get("token_embedding.weight")[ids] + weights.get("decoder.embed_positions.weight")[positions])
            .expandedDimensions(axis: 0)
        hidden = try weights.norm(hidden, "decoder.layernorm_embedding", epsilon: 1e-5)
        let width = configuration.hiddenSize
        let heads = configuration.numAttentionHeads
        let headDim = width / heads
        func project(_ value: MLXArray, _ name: String) throws -> MLXArray {
            try weights.linear(value, name).reshaped(1, -1, heads, headDim).transposed(0, 2, 1, 3)
        }
        for index in 0..<configuration.decoderLayers {
            try Task.checkCancellation()
            let prefix = "decoder.layers.\(index)"
            let query = try project(hidden, prefix + ".self_attn.q_proj")
            var keys = try project(hidden, prefix + ".self_attn.k_proj")
            var values = try project(hidden, prefix + ".self_attn.v_proj")
            if let previous = cache.keys[index], let oldValues = cache.values[index] {
                keys = concatenated([previous, keys], axis: 2)
                values = concatenated([oldValues, values], axis: 2)
            }
            cache.keys[index] = keys
            cache.values[index] = values
            let selfAttention = MLXFast.scaledDotProductAttention(queries: query, keys: keys, values: values,
                scale: 1 / Float(headDim).squareRoot(), mask: tokens.count > 1 ? .causal : .none)
            hidden = try weights.norm(hidden + weights.linear(selfAttention.transposed(0, 2, 1, 3).reshaped(1, -1, width),
                prefix + ".self_attn.out_proj"), prefix + ".self_attn_layer_norm", epsilon: 1e-5)
            if cache.crossKeys[index] == nil {
                cache.crossKeys[index] = try project(memory, prefix + ".encoder_attn.k_proj")
                cache.crossValues[index] = try project(memory, prefix + ".encoder_attn.v_proj")
            }
            guard let crossKeys = cache.crossKeys[index], let crossValues = cache.crossValues[index] else {
                throw SheetSage2Error.invalid("Missing cross-attention cache.")
            }
            let cross = try MLXFast.scaledDotProductAttention(queries: project(hidden, prefix + ".encoder_attn.q_proj"),
                keys: crossKeys, values: crossValues, scale: 1 / Float(headDim).squareRoot(), mask: .none)
            hidden = try weights.norm(hidden + weights.linear(cross.transposed(0, 2, 1, 3).reshaped(1, -1, width),
                prefix + ".encoder_attn.out_proj"), prefix + ".encoder_attn_layer_norm", epsilon: 1e-5)
            hidden = try weights.norm(hidden + weights.linear(gelu(weights.linear(hidden, prefix + ".fc1")), prefix + ".fc2"),
                prefix + ".final_layer_norm", epsilon: 1e-5)
            MLX.eval(hidden)
        }
        cache.offset += tokens.count
        return try matmul(hidden[0, tokens.count - 1], weights.get("token_embedding.weight").T)
    }
}
