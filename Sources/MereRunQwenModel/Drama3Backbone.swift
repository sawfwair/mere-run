#if !os(iOS)
import MLX
import MLXFast
import MLXNN
import MereRunKVCache

// DRAMA-3's pinned fish_omni_text graph has Qwen3.5 attention and MoE arithmetic.
// Register only the active attention family, so checkpoint coverage is exact.
private final class Drama3BackboneLayer: Module {
    @ModuleInfo(key: "linear_attn") var linear: Q35LinearAttention?
    @ModuleInfo(key: "self_attn") var full: Q35FullAttention?
    @ModuleInfo(key: "input_layernorm") var inputNorm: Q35RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: Q35RMSNorm
    @ModuleInfo var mlp: Q35FeedForward

    init(config: Q35Config, index: Int) {
        let text = config.textConfig
        _linear.wrappedValue = text.layerTypes[index] == "linear_attention" ? Q35LinearAttention(config: config) : nil
        _full.wrappedValue = text.layerTypes[index] == "full_attention" ? Q35FullAttention(config: config) : nil
        _inputNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _postNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _mlp.wrappedValue = Q35FeedForward(config: config)
    }

    func callAsFunction(_ x: MLXArray, cache: Q35LayerCache) -> MLXArray {
        let normalized = inputNorm(x)
        let attended: MLXArray
        switch cache {
        case .linear(let state): attended = linear!(normalized, cache: state)
        case .full(let state):
            let mask = createAttentionMask(h: normalized, cache: state)
            attended = full!(normalized, mask: mask, cache: state)
        }
        let residual = x + attended
        return residual + mlp(postNorm(residual))
    }
}

package final class Drama3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    @ModuleInfo private var layers: [Drama3BackboneLayer]
    @ModuleInfo private var norm: Q35RMSNorm
    private let config: Q35Config

    package init(config: Q35Config) {
        self.config = config
        let text = config.textConfig
        _embedding.wrappedValue = Embedding(embeddingCount: text.vocabSize, dimensions: text.hiddenSize)
        _layers.wrappedValue = (0..<text.numHiddenLayers).map { Drama3BackboneLayer(config: config, index: $0) }
        _norm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
    }

    package func embeddings(_ ids: MLXArray) -> MLXArray { embedding(ids) }

    package func makeCache() -> [Q35LayerCache] {
        config.textConfig.layerTypes.map {
            $0 == "linear_attention" ? .linear(Q35LinearCache()) : .full(KVCacheSimple())
        }
    }

    package func callAsFunction(_ embeddings: MLXArray, cache: [Q35LayerCache]) -> MLXArray {
        var hidden = embeddings
        for (layer, state) in zip(layers, cache) {
            hidden = layer(hidden, cache: state)
            MLX.eval(hidden)
        }
        return norm(hidden)
    }
}

#endif
