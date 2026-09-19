import Foundation
import MLX
import MLXFast
import MLXNN

public struct AuKThinkerConfiguration: Decodable, Sendable {
    public struct Text: Decodable, Sendable {
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let intermediateSize: Int
        public let vocabSize: Int
        public let ropeTheta: Double
        public let rmsNormEps: Float
    }
    public struct Audio: Decodable, Sendable {
        public let dModel: Int
        public let encoderLayers: Int
        public let encoderAttentionHeads: Int
        public let encoderFfnDim: Int
        public let outputDim: Int
        public let nWindow: Int
        public let numMelBins: Int
    }
    public let textConfig: Text
    public let audioConfig: Audio
    public let audioTokenIndex: Int

    public static func load(from url: URL) throws -> Self {
        struct Root: Decodable { let thinkerConfig: AuKThinkerConfiguration }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Root.self, from: Data(contentsOf: url)).thinkerConfig
        guard config.textConfig.hiddenSize == 2048, config.textConfig.numHiddenLayers == 36,
              config.textConfig.numAttentionHeads == 16, config.textConfig.numKeyValueHeads == 2,
              config.audioConfig.nWindow == 100, config.audioConfig.numMelBins == 128,
              config.audioConfig.outputDim == 2048 else {
            throw AuKError.invalid("AuK requires the Qwen2.5-Omni-3B Thinker configuration")
        }
        return config
    }
}

public struct AuKThinker {
    let weights: AuKTensorStore
    let config: AuKThinkerConfiguration
    public init(weights: AuKTensorStore, configuration: AuKThinkerConfiguration) {
        self.weights = weights
        self.config = configuration
    }

    public func encode(tokens: [Int], audio: MLXArray?, layerWeights: MLXArray,
                       layerScale: MLXArray) throws -> MLXArray {
        guard !tokens.isEmpty, tokens.allSatisfy({ $0 >= 0 && $0 < config.textConfig.vocabSize }),
              layerWeights.size == config.textConfig.numHiddenLayers else {
            throw AuKError.invalid("Invalid AuK token IDs or layer fusion weights")
        }
        var h = try weights.tensor("embed_tokens.weight")[MLXArray(tokens)].expandedDimensions(axis: 0)
        if let audio {
            let features = try encodeAudio(audio)
            let positions = tokens.indices.filter { tokens[$0] == config.audioTokenIndex }
            guard positions.count == features.dim(0), let start = positions.first,
                  positions == Array(start..<(start + positions.count)) else {
                throw AuKError.invalid("AuK audio placeholder count does not match audio features")
            }
            h = concatenated([h[0..., ..<start], features.expandedDimensions(axis: 0),
                              h[0..., (start + positions.count)...]], axis: 1)
        }
        let cfg = config.textConfig
        let dimension = cfg.hiddenSize / cfg.numAttentionHeads
        let rotary = AuKRotary(frequencies: (0..<(dimension / 2)).map {
            Float(pow(cfg.ropeTheta, -Double(2 * $0) / Double(dimension)))
        })
        let n = tokens.count
        let positions = MLXArray(0..<n)
        let mask = which(positions.expandedDimensions(axis: 0) .> positions.expandedDimensions(axis: 1),
                         MLXArray(-Float.infinity), MLXArray(Float(0)))
        let fusion = softmax(layerWeights)
        var combined = zeros(like: h)
        for index in 0..<cfg.numHiddenLayers {
            let key = "layers.\(index)"
            let normalized = try weights.rms(h, key + ".input_layernorm", eps: cfg.rmsNormEps)
            let prefix = key + ".self_attn"
            let q = rotary.apply(aukHeads(try weights.linear(normalized, prefix + ".q_proj"), cfg.numAttentionHeads), interleaved: false)
            let k = rotary.apply(aukHeads(try weights.linear(normalized, prefix + ".k_proj"), cfg.numKeyValueHeads), interleaved: false)
            let v = aukHeads(try weights.linear(normalized, prefix + ".v_proj"), cfg.numKeyValueHeads)
            h = h + (try weights.linear(aukUnheads(aukAttention(q, k, v, mask: mask)), prefix + ".o_proj"))
            let x = try weights.rms(h, key + ".post_attention_layernorm", eps: cfg.rmsNormEps)
            let gate = try aukSiLU(weights.linear(x, key + ".mlp.gate_proj"))
            let up = try weights.linear(x, key + ".mlp.up_proj")
            h = h + (try weights.linear(gate * up, key + ".mlp.down_proj"))
            let state = index == cfg.numHiddenLayers - 1 ? try weights.rms(h, "norm", eps: cfg.rmsNormEps) : h
            combined = combined + MLXFast.layerNorm(state, weight: nil, bias: nil, eps: 1e-5) * fusion[index]
            eval(h, combined)
        }
        return combined * layerScale
    }

    func encodeAudio(_ mel: MLXArray) throws -> MLXArray {
        let cfg = config.audioConfig, window = config.audioConfig.nWindow * 2
        var chunks = [MLXArray](), lengths = [Int]()
        for offset in stride(from: 0, to: mel.dim(1), by: window) {
            let length = min(window, mel.dim(1) - offset)
            var chunk = mel[0..., offset..<(offset + length)]
            chunk = padded(chunk, widths: [.init(0), .init((0, window - length)), .init(0)])
            chunk = try gelu(weights.conv(chunk, "audio_tower.conv1", padding: 1))
            let valid = (MLXArray(0..<window) .< length).asType(.float32).reshaped(1, window, 1)
            chunk = try gelu(weights.conv(chunk * valid, "audio_tower.conv2", stride: 2, padding: 1))
            chunk = chunk + Self.positions(length: cfg.nWindow, dimension: cfg.dModel)
            lengths.append((length + 1) / 2)
            chunks.append(chunk[0..., ..<((length + 1) / 2)])
        }
        var h = concatenated(chunks, axis: 1)
        let ids = lengths.enumerated().flatMap { Array(repeating: Int32($0.offset), count: $0.element) }
        let blocks = MLXArray(ids)
        let mask = which(blocks.expandedDimensions(axis: 0) .== blocks.expandedDimensions(axis: 1),
                         MLXArray(Float(0)), MLXArray(-Float.infinity))
        for index in 0..<cfg.encoderLayers {
            let key = "audio_tower.layers.\(index)"
            let x = try weights.norm(h, key + ".self_attn_layer_norm")
            let prefix = key + ".self_attn"
            let q = aukHeads(try weights.linear(x, prefix + ".q_proj"), cfg.encoderAttentionHeads)
            let k = aukHeads(try weights.linear(x, prefix + ".k_proj"), cfg.encoderAttentionHeads)
            let v = aukHeads(try weights.linear(x, prefix + ".v_proj"), cfg.encoderAttentionHeads)
            h = h + (try weights.linear(aukUnheads(aukAttention(q, k, v, mask: mask)), prefix + ".out_proj"))
            let norm = try weights.norm(h, key + ".final_layer_norm")
            h = h + (try weights.linear(gelu(weights.linear(norm, key + ".fc1")), key + ".fc2"))
            eval(h)
        }
        let count = h.dim(1) / 2
        h = h[0, ..<(count * 2)].reshaped(count, 2, -1).mean(axis: 1)
        return try weights.linear(weights.norm(h, "audio_tower.ln_post"), "audio_tower.proj")
    }

    static func positions(length: Int, dimension: Int) -> MLXArray {
        let half = dimension / 2
        let values = (0..<length).flatMap { position in
            (0..<dimension).map { index -> Float in
                let angle = Double(position) * exp(-log(10000) * Double(index % half) / Double(half - 1))
                return Float(index < half ? sin(angle) : cos(angle))
            }
        }
        return MLXArray(values, [length, dimension])
    }
}
