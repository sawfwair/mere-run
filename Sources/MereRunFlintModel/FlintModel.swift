import MLX
import MLXNN

package final class FlintDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: FlintAttention
    @ModuleInfo(key: "mlp") var mlp: FlintFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(config: FlintConfiguration, index: Int) {
        _attention.wrappedValue = FlintAttention(config: config, index: index)
        let path = "model.layers.\(index).mlp"
        if config.mlpOnlyLayers.contains(index) {
            _mlp.wrappedValue = FlintMLP(config: config, path: path, width: config.intermediateSize)
        } else {
            _mlp.wrappedValue = FlintMoE(config: config, path: path)
        }
        _inputNorm.wrappedValue = flintRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = flintRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: FlintCache?) -> MLXArray {
        let residual = x + attention(inputNorm(x), cache: cache)
        let normalized = postAttentionNorm(residual)
        let output = mlp(normalized)
        return residual + output
    }
}

package final class FlintBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: FlintEmbedding
    @ModuleInfo(key: "layers") var layers: [FlintDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    init(config: FlintConfiguration) {
        _embeddings.wrappedValue = FlintEmbedding(config: config)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { FlintDecoderLayer(config: config, index: $0) }
        _norm.wrappedValue = flintRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

package final class FlintCausalLM: Module {
    @ModuleInfo(key: "model") package var model: FlintBackbone
    @ModuleInfo(key: "lm_head") var head: FlintProjection
    package let config: FlintConfiguration
    package var asynchronousDecode = true

    package init(config: FlintConfiguration) throws {
        try config.validate()
        self.config = config
        _model.wrappedValue = FlintBackbone(config: config)
        _head.wrappedValue = FlintProjection(input: config.hiddenSize, output: config.vocabSize, policy: config.policy(for: "lm_head"))
        super.init()
    }

    package func makeCache(maxTokens: Int? = nil) -> [FlintCache] {
        config.layerTypes.map {
            FlintCache(window: $0 == .sliding ? config.slidingWindow : nil,
                       capacity: $0 == .full ? maxTokens : nil)
        }
    }

    package func callAsFunction(_ tokens: MLXArray, cache: [FlintCache]? = nil, lastPositionOnly: Bool = false) -> MLXArray {
        var hidden = model.embeddings(tokens)
        let enqueueBlocks = asynchronousDecode && tokens.size == 1
        var pendingCache: [MLXArray] = []
        for (index, layer) in model.layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index])
            if let row = cache?[index], let keys = row.keys, let values = row.values {
                pendingCache.append(contentsOf: [keys, values])
            }
            if enqueueBlocks {
                if (index + 1).isMultiple(of: 4) || index + 1 == model.layers.count {
                    // Submit bounded decode graphs without waiting after every layer.
                    // The decode engine evaluates the final logits before accepting a token.
                    asyncEval([hidden] + pendingCache)
                    pendingCache.removeAll(keepingCapacity: true)
                }
            } else {
                eval([hidden] + pendingCache)
                pendingCache.removeAll(keepingCapacity: true)
            }
        }
        if lastPositionOnly { hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...] }
        return head(model.norm(hidden).asType(.float32)).asType(.float32)
    }
}
