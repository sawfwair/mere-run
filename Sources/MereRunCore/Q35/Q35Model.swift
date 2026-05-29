import Foundation
import MLX
import MLXFast
import MLXNN

enum Q35AttentionLayerType: String {
    case linear = "linear_attention"
    case full = "full_attention"

    init(_ raw: String) {
        self = Q35AttentionLayerType(rawValue: raw) ?? .linear
    }
}

public enum Q35LayerCache: @unchecked Sendable {
    case linear(Q35LinearCache)
    case full(KVCache)

    func fork() -> Q35LayerCache {
        switch self {
        case .linear(let cache):
            return .linear(cache.fork())
        case .full(let cache):
            return .full(cache.fork())
        }
    }

    func batched(with caches: [Q35LayerCache]) -> Q35LayerCache? {
        guard !caches.isEmpty else { return nil }
        switch self {
        case .linear(let cache):
            let typed = caches.compactMap { entry -> Q35LinearCache? in
                if case .linear(let linear) = entry {
                    return linear
                }
                return nil
            }
            guard typed.count == caches.count,
                  let batched = cache.batched(with: typed) else {
                return nil
            }
            return .linear(batched)
        case .full(let cache):
            let typed = caches.compactMap { entry -> KVCache? in
                if case .full(let full) = entry {
                    return full
                }
                return nil
            }
            guard typed.count == caches.count,
                  let batched = cache.batched(with: typed) else {
                return nil
            }
            return .full(batched)
        }
    }

    func unbatchedRows(count: Int) -> [Q35LayerCache]? {
        switch self {
        case .linear(let cache):
            return cache.unbatchedRows(count: count)?.map(Q35LayerCache.linear)
        case .full(let cache):
            return cache.unbatchedRows(count: count)?.map(Q35LayerCache.full)
        }
    }
}

final class Q35DecoderLayer: Module {
    @ModuleInfo(key: "linear_attn") var linearAttention: Q35LinearAttention
    @ModuleInfo(key: "self_attn") var selfAttention: Q35FullAttention
    @ModuleInfo(key: "mlp") var mlp: Q35MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let layerType: Q35AttentionLayerType
    let isMLPOnly: Bool

    init(config: Q35Config, layerIndex: Int) {
        let text = config.textConfig
        self.layerType = Q35AttentionLayerType(text.layerTypes[layerIndex])
        self.isMLPOnly = text.mlpOnlyLayers.contains(layerIndex)

        self._linearAttention.wrappedValue = Q35LinearAttention(config: config)
        self._selfAttention.wrappedValue = Q35FullAttention(config: config)
        self._mlp.wrappedValue = Q35MoE(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        fullMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: Q35LayerCache?
    ) -> MLXArray {
        var h = x
        if !isMLPOnly {
            let normed = inputLayerNorm(x)
            let attentionOut: MLXArray
            switch layerType {
            case .linear:
                let linearCache: Q35LinearCache?
                if case .linear(let cache)? = cache {
                    linearCache = cache
                } else {
                    linearCache = nil
                }
                attentionOut = linearAttention(normed, cache: linearCache)
            case .full:
                let fullCache: KVCache?
                if case .full(let cache)? = cache {
                    fullCache = cache
                } else {
                    fullCache = nil
                }
                attentionOut = selfAttention(normed, mask: fullMask, cache: fullCache)
            }
            h = x + attentionOut
        }

        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}

final class Q35Transformer: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Q35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: Q35Config) {
        let text = config.textConfig
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: text.vocabSize,
            dimensions: text.hiddenSize
        )
        self._layers.wrappedValue = (0..<text.numHiddenLayers).map { index in
            Q35DecoderLayer(config: config, layerIndex: index)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        super.init()
    }

    func embeddings(for inputIds: MLXArray) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        return embedTokens(tokenIds)
    }

    private func firstFullCache(from cache: [Q35LayerCache?]?) -> KVCache? {
        guard let cache else { return nil }
        for entry in cache {
            if case .full(let kv)? = entry {
                return kv
            }
        }
        return nil
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: [Q35LayerCache?]?,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hidden = inputEmbeddings ?? embeddings(for: inputIds)

        let fullMask = createAttentionMask(h: hidden, cache: firstFullCache(from: cache))

        for (index, layer) in layers.enumerated() {
            let layerCache = cache?[index] ?? nil
            hidden = layer(
                hidden,
                fullMask: fullMask,
                cache: layerCache
            )
        }

        return norm(hidden)
    }
}

struct Q35ForwardOutput {
    let hidden: MLXArray
    let logits: MLXArray
}

public final class Q35Model: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") var model: Q35Transformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Q35Config

    public init(config: Q35Config) {
        self.config = config
        self._model.wrappedValue = Q35Transformer(config: config)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.vocabSize,
            bias: false
        )
        super.init()
    }

    public func embeddings(for inputIds: MLXArray) -> MLXArray {
        model.embeddings(for: inputIds)
    }

    func logits(from hidden: MLXArray) -> MLXArray {
        lmHead(hidden)
    }

    func forward(
        _ inputIds: MLXArray,
        cache: [Q35LayerCache?]?,
        inputEmbeddings: MLXArray? = nil
    ) -> Q35ForwardOutput {
        let hidden = model(inputIds, cache: cache, inputEmbeddings: inputEmbeddings)
        return Q35ForwardOutput(hidden: hidden, logits: lmHead(hidden))
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [Q35LayerCache?]?,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        forward(inputIds, cache: cache, inputEmbeddings: inputEmbeddings).logits
    }
}
