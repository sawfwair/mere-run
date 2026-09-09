import Foundation
import MLX
import MLXFast
import MLXNN

package final class Gemma4LanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm

    let config: Gemma4TextConfig
    let firstKVSharedLayerIndex: Int
    let layerIndexToCacheIndex: [Int]
    private let embedScale: Float
    private let embedTokensPerLayerScale: Float
    private let perLayerInputScale: Float
    private let perLayerProjectionScale: Float

    package init(config: Gemma4TextConfig) {
        self.config = config
        self.embedScale = sqrt(Float(config.hiddenSize))
        self.embedTokensPerLayerScale = sqrt(Float(max(1, config.hiddenSizePerLayerInput)))
        self.perLayerInputScale = pow(2, -0.5)
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
        self.firstKVSharedLayerIndex = max(0, config.numHiddenLayers - config.numKVSharedLayers)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._embedTokensPerLayer.wrappedValue = Embedding(
            embeddingCount: config.vocabSizePerLayerInput,
            dimensions: max(1, config.numHiddenLayers * max(1, config.hiddenSizePerLayerInput))
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            Gemma4DecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize,
            max(1, config.numHiddenLayers * max(1, config.hiddenSizePerLayerInput)),
            bias: false
        )
        self._perLayerProjectionNorm.wrappedValue = RMSNorm(
            dimensions: max(1, config.hiddenSizePerLayerInput),
            eps: config.rmsNormEps
        )

        var cacheMap: [Int] = Array(0..<firstKVSharedLayerIndex)
        if firstKVSharedLayerIndex < config.numHiddenLayers {
            let concreteLayerTypes = Array(config.layerTypes.prefix(firstKVSharedLayerIndex))
            let sharedFullIndex = concreteLayerTypes.lastIndex(of: "full_attention") ?? 0
            let sharedSlidingIndex = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0
            for index in firstKVSharedLayerIndex..<config.numHiddenLayers {
                if config.layerTypes[index] == "full_attention" {
                    cacheMap.append(sharedFullIndex)
                } else {
                    cacheMap.append(sharedSlidingIndex)
                }
            }
        }
        self.layerIndexToCacheIndex = cacheMap
        super.init()
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        let hidden = embeddings(inputIds: tokenIds)
        return forward(
            embeddings: hidden,
            inputIds: tokenIds,
            cache: cache,
            mmTokenTypeIds: nil
        )
    }

    func embeddings(inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds) * MLXArray(embedScale).asType(embedTokens.weight.dtype)
    }

    func forward(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        forwardDetailed(
            embeddings: embeddings,
            inputIds: inputIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds,
            captureSharedKV: false
        ).hidden
    }

    func forwardDetailed(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil,
        captureSharedKV: Bool = true,
        attentionMask: MLXArray? = nil,
        captureHiddenStates: Bool = false
    ) -> Gemma4LanguageModelOutput {
        var hidden = embeddings
        var hiddenStates: [MLXArray]? = captureHiddenStates ? [hidden] : nil
        let perLayerInputs = inputIds.flatMap { tokenIds -> MLXArray? in
            guard config.hiddenSizePerLayerInput > 0 else { return nil }
            return projectPerLayerInputs(
                hiddenStates: hidden,
                inputIds: tokenIds
            )
        }
        let visionBlockIDs = Self.visionBlockIDs(from: mmTokenTypeIds, sequenceLength: hidden.dim(1))
        let caches = cache ?? makeCache()
        let forwardCaches = caches.map(Gemma4ForwardAttentionCache.init)
        for (index, layer) in layers.enumerated() {
            let cacheIndex = index < layerIndexToCacheIndex.count ? layerIndexToCacheIndex[index] : index
            let perLayerInput = perLayerInputs?[0..., 0..., index, 0...]
            hidden = layer(
                hidden,
                cache: cacheIndex < forwardCaches.count ? forwardCaches[cacheIndex] : nil,
                perLayerInput: perLayerInput,
                visionBlockIDs: visionBlockIDs,
                attentionMask: attentionMask
            )
            if captureHiddenStates {
                MLX.eval(hidden)
                hiddenStates?.append(hidden)
            }
        }
        let preNormHidden = hidden
        let normalizedHidden = norm(hidden)
        if captureHiddenStates, let lastIndex = hiddenStates?.indices.last {
            hiddenStates?[lastIndex] = normalizedHidden
        }
        let sharedKVStates = captureSharedKV ? collectSharedKVStates(from: caches) : [:]
        return Gemma4LanguageModelOutput(
            hidden: normalizedHidden,
            preNormHidden: preNormHidden,
            sharedKVStates: sharedKVStates,
            hiddenStates: hiddenStates
        )
    }

    package func forwardHiddenStates(
        inputIds: MLXArray,
        attentionMask: MLXArray
    ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]) {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        let output = forwardDetailed(
            embeddings: embeddings(inputIds: tokenIds),
            inputIds: tokenIds,
            captureSharedKV: false,
            attentionMask: attentionMask,
            captureHiddenStates: true
        )
        return (output.hidden, output.hiddenStates ?? [output.hidden])
    }

    func logits(_ inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        let hidden = self(inputIds, cache: cache)
        return embedTokens.asLinear(hidden)
    }

    /// Logits for the final position only. Prefill chunks never read the other
    /// positions' logits, and skipping them avoids a [seq, 262k] lm_head matmul
    /// plus its materialization per chunk.
    func lastPositionLogits(_ inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        var hidden = self(inputIds, cache: cache)
        let sequenceLength = hidden.dim(1)
        if sequenceLength > 1 {
            hidden = hidden[0..., (sequenceLength - 1)..., 0...]
        }
        return embedTokens.asLinear(hidden)
    }

    func logits(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        let hidden = forward(
            embeddings: embeddings,
            inputIds: inputIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        return embedTokens.asLinear(hidden)
    }

    func detailedLogits(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil
    ) -> Gemma4ForwardOutput {
        let output = forwardDetailed(
            embeddings: embeddings,
            inputIds: inputIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        return Gemma4ForwardOutput(
            logits: embedTokens.asLinear(output.hidden),
            hidden: output.preNormHidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    func makeCache(quantization: Gemma4KVCacheQuantization? = nil) -> [Gemma4AttentionCache] {
        config.layerTypes.prefix(firstKVSharedLayerIndex).map { layerType in
            let maxSize: Int? = layerType == "full_attention" ? nil : config.slidingWindow
            if let quantization, quantization.isEnabled {
                if quantization.scheme == .polar {
                    return Gemma4PolarKVCache(configuration: quantization, maxSize: maxSize)
                }
                return Gemma4QuantizedKVCache(configuration: quantization, maxSize: maxSize)
            }
            if let maxSize {
                return Gemma4SlidingKVCache(maxSize: maxSize)
            }
            return Gemma4FullKVCache()
        }
    }

    private func projectPerLayerInputs(
        hiddenStates: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let perLayerEmbedding = embedTokensPerLayer(inputIds)
            * MLXArray(embedTokensPerLayerScale).asType(hiddenStates.dtype)
        let reshapedEmbedding = perLayerEmbedding.reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            config.numHiddenLayers,
            max(1, config.hiddenSizePerLayerInput)
        )

        var projected = (perLayerModelProjection(hiddenStates) * MLXArray(perLayerProjectionScale).asType(hiddenStates.dtype)).reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            config.numHiddenLayers,
            max(1, config.hiddenSizePerLayerInput)
        )
        projected = perLayerProjectionNorm(projected)
        return (projected + reshapedEmbedding) * MLXArray(perLayerInputScale).asType(hiddenStates.dtype)
    }

    private func collectSharedKVStates(from caches: [Gemma4AttentionCache]) -> [String: Gemma4SharedKVState] {
        guard !caches.isEmpty else { return [:] }
        var states: [String: Gemma4SharedKVState] = [:]
        for index in 0..<min(layers.count, layerIndexToCacheIndex.count) {
            let cacheIndex = layerIndexToCacheIndex[index]
            guard cacheIndex < caches.count,
                  let state = caches[cacheIndex].currentState() else {
                continue
            }
            let maxSize = (caches[cacheIndex] as? Gemma4SlidingKVCache)?.configuredMaxSize
            states[layers[index].selfAttention.layerType] = Gemma4SharedKVState(
                keys: state.0,
                values: state.1,
                offset: caches[cacheIndex].offset,
                maxSize: maxSize
            )
        }
        return states
    }

    private static func visionBlockIDs(from mmTokenTypeIds: MLXArray?, sequenceLength: Int) -> MLXArray? {
        guard let mmTokenTypeIds, sequenceLength > 1 else { return nil }
        let typed = mmTokenTypeIds.asType(.int32)
        MLX.eval(typed)
        let values = typed.asArray(Int32.self)
        guard values.count >= sequenceLength else { return nil }

        var blockIDs = [Int32](repeating: -1, count: sequenceLength)
        var currentBlock: Int32 = -1
        var previousWasVision = false
        for index in 0..<sequenceLength {
            let value = values[index]
            let isVision = value == 1 || value == 2
            if isVision && !previousWasVision {
                currentBlock += 1
            }
            if isVision {
                blockIDs[index] = currentBlock
            }
            previousWasVision = isVision
        }
        return MLXArray(blockIDs)
    }
}
