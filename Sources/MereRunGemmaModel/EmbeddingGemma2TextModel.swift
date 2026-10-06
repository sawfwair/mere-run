import Foundation
import MLX
import MLXFast
import MLXNN

/// Native bidirectional encoder. Checkpoint names match Google's language_model subtree.
public final class EmbeddingGemma2TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [EmbeddingGemma2EncoderLayer]
    @ModuleInfo var norm: EmbeddingGemma2RMSNorm
    @ModuleInfo var ple: EmbeddingGemma2PLE
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    public let config: EmbeddingGemma2Config.TextConfig

    public init(config: EmbeddingGemma2Config) throws {
        try config.validate()
        let text = config.textConfig
        self.config = text
        _embedTokens.wrappedValue = Embedding(embeddingCount: text.vocabSize, dimensions: text.hiddenSize)
        _layers.wrappedValue = (0..<text.numHiddenLayers).map { EmbeddingGemma2EncoderLayer(config: text, index: $0) }
        _norm.wrappedValue = EmbeddingGemma2RMSNorm(text.hiddenSize, eps: text.rmsNormEps)
        _ple.wrappedValue = EmbeddingGemma2PLE(config: text)
        _embeddingProjection.wrappedValue = Linear(text.hiddenSize, text.embeddingDim, bias: false)
        super.init()
    }

    public func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray) -> MLXArray {
        callAsFunction(embeddings: inputEmbeddings(inputIDs), attentionMask: attentionMask)
    }

    public func inputEmbeddings(_ inputIDs: MLXArray) -> MLXArray {
        embedTokens(inputIDs.asType(.int32))
            * MLXArray(Float(config.hiddenSize).squareRoot()).asType(embedTokens.weight.dtype)
    }

    public func callAsFunction(embeddings: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var hidden = embeddings
        let perLayer = ple(hidden)
        let fullMask = Self.attentionMask(validTokens: attentionMask, window: nil, dtype: hidden.dtype)
        let localMask = Self.attentionMask(validTokens: attentionMask, window: config.slidingWindow, dtype: hidden.dtype)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, perLayerInput: perLayer[0..., 0..., index, 0...],
                           mask: config.layerTypes[index] == "full_attention" ? fullMask : localMask)
        }
        return embeddingProjection(norm(hidden))
    }

    /// Padding excludes keys; local attention sees both directions through distance <= window.
    static func attentionMask(validTokens: MLXArray, window: Int?, dtype: DType) -> MLXArray {
        let length = validTokens.dim(1)
        var allowed = validTokens.asType(.int32).reshaped(-1, 1, 1, length) .> MLXArray(Int32(0))
        if let window {
            let positions = MLXArray(Int32(0)..<Int32(length))
            let distance = abs(positions.reshaped(length, 1) - positions.reshaped(1, length))
            allowed = allowed .&& (distance .<= MLXArray(Int32(window))).reshaped(1, 1, length, length)
        }
        return MLX.where(allowed, MLXArray(Float(0)).asType(dtype), MLXArray(Float(-1e9)).asType(dtype))
    }

    public static func pool(tokenEmbeddings: MLXArray, attentionMask: MLXArray, dimensions: Int) -> MLXArray {
        let valid = attentionMask.asType(.float32).expandedDimensions(axis: -1)
        let hidden = tokenEmbeddings.asType(.float32)
        let masked = MLX.where(valid .> MLXArray(Float(0)), hidden, MLXArray(Float(0)))
        let pooled = sum(masked, axis: 1) / maximum(sum(valid, axis: 1), MLXArray(Float(1e-9)))
        let truncated = pooled[0..., 0..<dimensions]
        let norms = maximum(sqrt(sum(truncated * truncated, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return truncated / norms
    }
}

final class EmbeddingGemma2RMSNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (Self.normalize(x, eps: eps) * weight.asType(.float32)).asType(x.dtype)
    }

    static func normalize(_ x: MLXArray, eps: Float) -> MLXArray {
        let full = x.asType(.float32)
        return full * pow(mean(full * full, axis: -1, keepDims: true) + MLXArray(eps), MLXArray(Float(-0.5)))
    }
}

final class EmbeddingGemma2PLE: Module {
    @ModuleInfo(key: "per_layer_model_projection") var projection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var norm: EmbeddingGemma2RMSNorm
    let config: EmbeddingGemma2Config.TextConfig

    init(config: EmbeddingGemma2Config.TextConfig) {
        self.config = config
        _projection.wrappedValue = Linear(config.hiddenSize, config.numHiddenLayers * config.hiddenSizePerLayerInput, bias: false)
        _norm.wrappedValue = EmbeddingGemma2RMSNorm(config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = projection(x) * MLXArray(pow(Float(config.hiddenSize), -0.5)).asType(x.dtype)
        return norm(projected.reshaped(x.dim(0), x.dim(1), config.numHiddenLayers, config.hiddenSizePerLayerInput))
    }
}

final class EmbeddingGemma2PLEBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var gate: Linear
    @ModuleInfo(key: "per_layer_projection") var projection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var norm: EmbeddingGemma2RMSNorm

    init(config: EmbeddingGemma2Config.TextConfig) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
        _projection.wrappedValue = Linear(config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
        _norm.wrappedValue = EmbeddingGemma2RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray) -> MLXArray {
        x + norm(projection(geluApproximate(gate(x)) * perLayerInput))
    }
}

final class EmbeddingGemma2Attention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: EmbeddingGemma2RMSNorm
    let headDim: Int
    let heads: Int
    let kvHeads: Int
    let eps: Float
    let rope: RoPE

    init(config: EmbeddingGemma2Config.TextConfig, index: Int) {
        let layer = config.layerConfig(at: index)
        headDim = layer.headDim
        heads = config.numAttentionHeads
        kvHeads = layer.numKeyValueHeads
        eps = config.rmsNormEps
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeParameters[config.layerTypes[index]]!.ropeTheta)
        _query.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _output.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _queryNorm.wrappedValue = EmbeddingGemma2RMSNorm(headDim, eps: eps)
        _keyNorm.wrappedValue = EmbeddingGemma2RMSNorm(headDim, eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let q = rope(queryNorm(query(x).reshaped(batch, length, heads, headDim)).transposed(0, 2, 1, 3))
        let k = rope(keyNorm(key(x).reshaped(batch, length, kvHeads, headDim)).transposed(0, 2, 1, 3))
        let v = EmbeddingGemma2RMSNorm.normalize(value(x).reshaped(batch, length, kvHeads, headDim), eps: eps)
            .asType(x.dtype).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1, mask: .array(mask))
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, heads * headDim))
    }
}

final class EmbeddingGemma2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(config: EmbeddingGemma2Config.TextConfig) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

final class EmbeddingGemma2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: EmbeddingGemma2Attention
    @ModuleInfo var mlp: EmbeddingGemma2MLP
    @ModuleInfo(key: "ple_block") var ple: EmbeddingGemma2PLEBlock
    @ModuleInfo(key: "input_layernorm") var inputNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var attentionNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postNorm: EmbeddingGemma2RMSNorm
    @ParameterInfo(key: "layer_scalar") var scalar: MLXArray

    init(config: EmbeddingGemma2Config.TextConfig, index: Int) {
        _attention.wrappedValue = EmbeddingGemma2Attention(config: config, index: index)
        _mlp.wrappedValue = EmbeddingGemma2MLP(config: config)
        _ple.wrappedValue = EmbeddingGemma2PLEBlock(config: config)
        _inputNorm.wrappedValue = EmbeddingGemma2RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        _attentionNorm.wrappedValue = EmbeddingGemma2RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        _preNorm.wrappedValue = EmbeddingGemma2RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = EmbeddingGemma2RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        _scalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray, mask: MLXArray) -> MLXArray {
        let attended = x + attentionNorm(attention(inputNorm(x), mask: mask))
        let hidden = attended + postNorm(mlp(preNorm(attended)))
        return ple(hidden, perLayerInput: perLayerInput) * scalar
    }
}
