import MLX
import MLXNN

package final class KolibriDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: KolibriAttention
    @ModuleInfo(key: "mlp") var mlp: KolibriMoE
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attn_norm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var preFFNNorm: RMSNorm
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm

    init(config: KolibriConfiguration, index: Int) {
        _attention.wrappedValue = KolibriAttention(config: config, index: index)
        _mlp.wrappedValue = KolibriMoE(config: config, path: "model.layers.\(index).mlp")
        _inputNorm.wrappedValue = kolibriRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = kolibriRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFFNNorm.wrappedValue = kolibriRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFFNNorm.wrappedValue = kolibriRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KolibriCache?) -> MLXArray {
        let residual = x + postAttentionNorm(attention(inputNorm(x), cache: cache))
        return residual + postFFNNorm(mlp(preFFNNorm(residual)))
    }
}

package final class KolibriBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [KolibriDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    init(config: KolibriConfiguration) {
        _embeddings.wrappedValue = Embedding(weight: MLXArray.zeros([config.vocabSize, config.hiddenSize], dtype: .bfloat16))
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { KolibriDecoderLayer(config: config, index: $0) }
        _norm.wrappedValue = kolibriRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

package final class KolibriCausalLM: Module {
    @ModuleInfo(key: "model") package var model: KolibriBackbone
    @ModuleInfo(key: "lm_head") var head: KolibriProjection
    package let config: KolibriConfiguration

    package init(config: KolibriConfiguration) throws {
        try config.validate()
        self.config = config
        _model.wrappedValue = KolibriBackbone(config: config)
        _head.wrappedValue = KolibriProjection(input: config.hiddenSize, output: config.vocabSize,
                                              policy: config.quantization?["lm_head"])
        super.init()
    }

    package func observeExpertInputs(_ observer: ((String, MLXArray) -> Void)?) {
        for layer in model.layers { layer.mlp.experts.observeInput = observer }
    }

    package func makeCache() -> [KolibriCache] {
        config.layerTypes.map { KolibriCache(window: $0 == .sliding ? config.slidingWindow : nil) }
    }

    package func callAsFunction(_ tokens: MLXArray, cache: [KolibriCache]? = nil, lastPositionOnly: Bool = false) -> MLXArray {
        var hidden = model.embeddings(tokens)
        for (index, layer) in model.layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index])
            // Bound temporary expert selections and retained sliding-cache graphs per layer.
            eval(hidden)
            if let row = cache?[index], let keys = row.keys, let values = row.values { eval(keys, values) }
        }
        if lastPositionOnly { hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...] }
        // The released head computes logits in FP32, even with BF16 hidden states.
        return head(model.norm(hidden).asType(.float32)).asType(.float32)
    }
}
