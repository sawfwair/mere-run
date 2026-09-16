import MLX
import MLXFast
import MLXNN

struct YuE2Norm {
    let weight: MLXArray
    let epsilon: Float

    init(_ weights: inout YuE2Weights, _ prefix: String, size: Int, epsilon: Float) throws {
        weight = try weights.take(prefix + ".weight", [size])
        self.epsilon = epsilon
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = rsqrt(mean(square(x.asType(.float32)), axis: -1, keepDims: true) + epsilon)
        return x * scale.asType(x.dtype) * weight
    }
}

struct YuE2Attention {
    let query: YuE2Linear
    let key: YuE2Linear
    let value: YuE2Linear
    let output: YuE2Linear
    let queryNorm: YuE2Norm
    let keyNorm: YuE2Norm
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let theta: Float

    init(_ weights: inout YuE2Weights, _ prefix: String, configuration: YuE2Configuration) throws {
        let config = configuration
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        theta = config.ropeTheta
        query = try YuE2Linear(&weights, prefix + ".q_proj", input: config.hiddenSize, output: heads * headDim)
        key = try YuE2Linear(&weights, prefix + ".k_proj", input: config.hiddenSize, output: kvHeads * headDim)
        value = try YuE2Linear(&weights, prefix + ".v_proj", input: config.hiddenSize, output: kvHeads * headDim)
        output = try YuE2Linear(&weights, prefix + ".o_proj", input: heads * headDim, output: config.hiddenSize)
        queryNorm = try YuE2Norm(&weights, prefix + ".q_norm", size: headDim, epsilon: config.rmsNormEps)
        keyNorm = try YuE2Norm(&weights, prefix + ".k_norm", size: headDim, epsilon: config.rmsNormEps)
    }

    func project(_ input: MLXArray, offset: Int) -> (MLXArray, MLXArray, MLXArray) {
        let count = input.dim(1)
        let q = queryNorm(query(input).reshaped(1, count, heads, headDim)).transposed(0, 2, 1, 3)
        let k = keyNorm(key(input).reshaped(1, count, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = value(input).reshaped(1, count, kvHeads, headDim).transposed(0, 2, 1, 3)
        return (rotary(q, offset: offset), rotary(k, offset: offset), v)
    }

    private func rotary(_ x: MLXArray, offset: Int) -> MLXArray {
        let positions = MLXArray(offset..<(offset + x.dim(2))).asType(.float32).reshaped(-1, 1)
        let frequencies = 1 / pow(theta, MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32) / Float(headDim))
        let angles = positions * frequencies
        let cosine = cos(angles).asType(x.dtype)
        let sine = sin(angles).asType(x.dtype)
        let first = x[.ellipsis, ..<(headDim / 2)]
        let second = x[.ellipsis, (headDim / 2)...]
        return concatenated([first * cosine - second * sine, second * cosine + first * sine], axis: -1)
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, causal: Bool) -> MLXArray {
        let result = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(),
            mask: causal ? .causal : .none
        )
        return output(result.transposed(0, 2, 1, 3).reshaped(1, q.dim(2), heads * headDim))
    }
}

struct YuE2FeedForward {
    let gate: YuE2Linear
    let up: YuE2Linear
    let down: YuE2Linear

    init(_ weights: inout YuE2Weights, _ prefix: String, configuration: YuE2Configuration) throws {
        let config = configuration
        gate = try YuE2Linear(&weights, prefix + ".gate_proj", input: config.hiddenSize, output: config.intermediateSize)
        up = try YuE2Linear(&weights, prefix + ".up_proj", input: config.hiddenSize, output: config.intermediateSize)
        down = try YuE2Linear(&weights, prefix + ".down_proj", input: config.intermediateSize, output: config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

struct YuE2Branch {
    let norm: YuE2Norm
    let attention: YuE2Attention
    let postNorm: YuE2Norm
    let mlp: YuE2FeedForward

    init(_ weights: inout YuE2Weights, _ prefix: String, configuration: YuE2Configuration, acoustic: Bool) throws {
        norm = try YuE2Norm(&weights, prefix + (acoustic ? ".nar_input_layernorm" : ".input_layernorm"),
                           size: configuration.hiddenSize, epsilon: configuration.rmsNormEps)
        postNorm = try YuE2Norm(&weights, prefix + (acoustic ? ".nar_pre_mlp_layernorm" : ".post_attention_layernorm"),
                               size: configuration.hiddenSize, epsilon: configuration.rmsNormEps)
        attention = try YuE2Attention(&weights, prefix + (acoustic ? ".nar_self_attn" : ".self_attn"), configuration: configuration)
        mlp = try YuE2FeedForward(&weights, prefix + (acoustic ? ".nar_mlp" : ".mlp"), configuration: configuration)
    }
}

final class YuE2Model {
    let configuration: YuE2Configuration
    let embedding: MLXArray
    let head: YuE2Linear
    let norm: YuE2Norm
    let autoregressive: [YuE2Branch]
    let acoustic: [YuE2Branch]
    let latentInput: YuE2Linear
    let latentOutput: YuE2Linear
    let timeInput: YuE2Linear
    let timeOutput: YuE2Linear
    let positions: MLXArray

    init(configuration: YuE2Configuration, arrays: [String: MLXArray]) throws {
        self.configuration = configuration
        var weights = YuE2Weights(arrays)
        let config = configuration
        embedding = try weights.take("model.embed_tokens.weight", [config.vocabSize, config.hiddenSize])
        head = try YuE2Linear(&weights, "lm_head", input: config.hiddenSize, output: config.vocabSize)
        norm = try YuE2Norm(&weights, "model.norm", size: config.hiddenSize, epsilon: config.rmsNormEps)
        autoregressive = try (0..<config.numHiddenLayers).map {
            try YuE2Branch(&weights, "model.layers.\($0)", configuration: config, acoustic: false)
        }
        acoustic = try (0..<config.numHiddenLayers).map {
            try YuE2Branch(&weights, "model.layers.\($0)", configuration: config, acoustic: true)
        }
        latentInput = try YuE2Linear(&weights, "vae2llm", input: config.latentDim, output: config.hiddenSize, bias: true)
        latentOutput = try YuE2Linear(&weights, "llm2vae", input: config.hiddenSize, output: config.latentDim, bias: true)
        timeInput = try YuE2Linear(&weights, "time_embedder.mlp.0", input: 256, output: config.hiddenSize, bias: true)
        timeOutput = try YuE2Linear(&weights, "time_embedder.mlp.2", input: config.hiddenSize, output: config.hiddenSize, bias: true)
        positions = try weights.take("latent_pos_embed.pe", [config.maxLatentFrames, config.hiddenSize])
        try weights.finish()
    }

    func makeCache() -> [KVCacheSimple] { autoregressive.map { _ in KVCacheSimple() } }

    func logits(_ tokens: [Int], cache: [KVCacheSimple]) throws -> MLXArray {
        var hidden = embedding[MLXArray(tokens.map(Int32.init))].expandedDimensions(axis: 0)
        for (index, layer) in autoregressive.enumerated() {
            try Task.checkCancellation()
            let (query, key, value) = layer.attention.project(layer.norm(hidden), offset: cache[index].offset)
            let (keys, values) = cache[index].update(keys: key, values: value)
            hidden = hidden + layer.attention.attend(query, keys, values, causal: tokens.count > 1)
            hidden = hidden + layer.mlp(layer.postNorm(hidden))
        }
        return head(norm(hidden[0..., (tokens.count - 1)..<tokens.count, 0...])).reshaped(-1)
    }
}
