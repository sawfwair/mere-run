#if !os(iOS)
// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MLX
import MLXFast
import MLXNN

final class D1Attention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var keyNorm: RMSNorm
    let heads: Int
    let kvHeads: Int
    let dimension: Int
    let rope: RoPE
    init(_ config: D1TextConfiguration) {
        heads = config.num_attention_heads; kvHeads = config.num_key_value_heads; dimension = config.headDim
        rope = RoPE(dimensions: dimension, traditional: false, base: config.theta)
        _query.wrappedValue = Linear(config.hidden_size, heads * dimension, bias: false)
        _key.wrappedValue = Linear(config.hidden_size, kvHeads * dimension, bias: false)
        _value.wrappedValue = Linear(config.hidden_size, kvHeads * dimension, bias: false)
        _output.wrappedValue = Linear(heads * dimension, config.hidden_size, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: dimension, eps: config.norm_eps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: dimension, eps: config.norm_eps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let q = rope(queryNorm(query(x).reshaped(batch, length, heads, dimension)).transposed(0, 2, 1, 3))
        let k = rope(keyNorm(key(x).reshaped(batch, length, kvHeads, dimension)).transposed(0, 2, 1, 3))
        let v = value(x).reshaped(batch, length, kvHeads, dimension).transposed(0, 2, 1, 3)
        return output(MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
            scale: 1 / sqrt(Float(dimension)), mask: mask).transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class D1ShortConv: Module {
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "conv") var conv: Conv1d
    let width: Int
    let omni: Bool
    init(_ config: D1TextConfiguration, omni: Bool) {
        width = config.hidden_size; self.omni = omni
        _input.wrappedValue = Linear(width, 3 * width, bias: false)
        _output.wrappedValue = Linear(width, width, bias: false)
        _conv.wrappedValue = Conv1d(inputChannels: width, outputChannels: width, kernelSize: 3, groups: width, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, prefix: Int) -> MLXArray {
        let projected = input(x)
        let b = projected[.ellipsis, 0..<width], c = projected[.ellipsis, width..<(2 * width)]
        let u = projected[.ellipsis, (2 * width)...]
        let bx = b * u
        let y: MLXArray
        if omni {
            // The last media position cannot read the first text position.
            let xp = padded(bx, widths: [[0, 0], [1, 1], [0, 0]])
            let length = x.dim(1), w = conv.weight
            let keep = MLXArray((0..<length).map { Float($0 == prefix - 1 ? 0 : 1) }).reshaped(1, length, 1)
            y = xp[0..., 0..<length, 0...] * w[0..., 0, 0].reshaped(1, 1, width)
                + xp[0..., 1..<(length + 1), 0...] * w[0..., 1, 0].reshaped(1, 1, width)
                + xp[0..., 2..<(length + 2), 0...] * keep * w[0..., 2, 0].reshaped(1, 1, width)
        } else { y = conv(padded(bx, widths: [[0, 0], [2, 0], [0, 0]])) }
        return output(c * y)
    }
}

final class D1MLP: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear
    init(_ config: D1TextConfiguration, omni: Bool) {
        let width = config.ffnWidth(omni: omni)
        _w1.wrappedValue = Linear(config.hidden_size, width, bias: false)
        _w3.wrappedValue = Linear(config.hidden_size, width, bias: false)
        _w2.wrappedValue = Linear(width, config.hidden_size, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class D1Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: D1Attention?
    @ModuleInfo var conv: D1ShortConv?
    @ModuleInfo(key: "feed_forward") var mlp: D1MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    init(_ config: D1TextConfiguration, kind: String, omni: Bool) {
        if kind == "full_attention" { _attention.wrappedValue = D1Attention(config) }
        else { _conv.wrappedValue = D1ShortConv(config, omni: omni) }
        _mlp.wrappedValue = D1MLP(config, omni: omni)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hidden_size, eps: config.norm_eps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hidden_size, eps: config.norm_eps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, prefix: Int, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let normalized = operatorNorm(x)
        let h = x + (attention?(normalized, mask: mask) ?? conv!(normalized, prefix: prefix))
        return h + mlp(ffnNorm(h))
    }
}

public final class D1Trunk: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    @ModuleInfo var layers: [D1Layer]
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    let omni: Bool
    public init(_ config: D1TextConfiguration, omni: Bool) {
        self.omni = omni
        _embedding.wrappedValue = Embedding(embeddingCount: config.vocab_size, dimensions: config.hidden_size)
        _layers.wrappedValue = config.layer_types.map { D1Layer(config, kind: $0, omni: omni) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hidden_size, eps: config.norm_eps)
        super.init()
    }
    public func embeddings(_ ids: MLXArray) -> MLXArray { embedding(ids) }
    public func logits(_ hidden: MLXArray) -> MLXArray { embedding.asLinear(hidden) }
    public func callAsFunction(_ embeddings: MLXArray, prefix: Int = 0) -> MLXArray {
        let length = embeddings.dim(1)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if !omni { mask = .causal }
        else if prefix == 0 { mask = .none }
        else {
            let positions = MLX.arange(length)
            let mediaQueries = (positions .< prefix).reshaped(1, 1, length, 1)
            let textKeys = (positions .>= prefix).reshaped(1, 1, 1, length)
            mask = .array(logicalAnd(mediaQueries, textKeys).asType(.float32) * -1e9)
        }
        return norm(layers.reduce(embeddings) { $1($0, prefix: prefix, mask: mask) })
    }
}
#endif
