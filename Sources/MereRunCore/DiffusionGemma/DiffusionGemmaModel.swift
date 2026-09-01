import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

final class DiffusionGemmaSwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray

    private let groupSize: Int
    private let bits: Int

    init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        parameters: DiffusionGemmaQuantizationParameters
    ) {
        groupSize = parameters.groupSize
        bits = parameters.bits
        let packedInput = max(1, (inputDims * parameters.bits + 31) / 32)
        let groups = max(1, (inputDims + parameters.groupSize - 1) / parameters.groupSize)
        _weight.wrappedValue = MLXArray.zeros(
            [numExperts, outputDims, packedInput],
            dtype: .uint32
        )
        _scales.wrappedValue = MLXArray.zeros(
            [numExperts, outputDims, groups],
            dtype: .bfloat16
        )
        _biases.wrappedValue = MLXArray.zeros(
            [numExperts, outputDims, groups],
            dtype: .bfloat16
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(-1)
        if x.ndim == 4 && x.dim(2) == topK {
            let flat = x.reshaped([batchTokens * topK, 1, inputDim])
            let flatIndices = indices.reshaped([batchTokens * topK])
            let output = applyFlat(flat, indices: flatIndices, sortedIndices: false)
            return output.reshaped([x.dim(0), x.dim(1), topK, output.dim(-1)])
        }
        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        return applyFlat(expanded, indices: indices, sortedIndices: false).squeezed(axis: -2)
    }

    func applyFlat(
        _ x: MLXArray,
        indices: MLXArray,
        sortedIndices: Bool
    ) -> MLXArray {
        portableGatherQuantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: sortedIndices
        )
    }
}

enum DiffusionGemmaExpertRouting {
    static let minimumSortedRouteCount = 64

    struct Plan {
        let order: MLXArray
        let inverseOrder: MLXArray
        let sortedIndices: MLXArray
        let tokenOrder: MLXArray
    }

    static func shouldSort(routeCount: Int) -> Bool {
        routeCount >= minimumSortedRouteCount
    }

    static func sortedPlan(indices: MLXArray, topK: Int) -> Plan {
        let flatIndices = indices.reshaped([indices.size])
        let order = argSort(flatIndices, axis: 0)
        return Plan(
            order: order,
            inverseOrder: argSort(order, axis: 0),
            sortedIndices: take(flatIndices, order, axis: 0),
            tokenOrder: order.floorDivide(topK)
        )
    }
}

final class DiffusionGemmaExperts: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: DiffusionGemmaSwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: DiffusionGemmaSwitchLinear

    private let intermediateSize: Int

    init(
        config: DiffusionGemmaTextConfig,
        quantization: DiffusionGemmaQuantizationConfig,
        layerIndex: Int
    ) {
        intermediateSize = config.moeIntermediateSize
        let prefix = "model.decoder.layers.\(layerIndex).experts"
        _gateUpProj.wrappedValue = DiffusionGemmaSwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: 2 * config.moeIntermediateSize,
            numExperts: config.numExperts,
            parameters: quantization.parameters(for: "\(prefix).gate_up_proj")
        )
        _downProj.wrappedValue = DiffusionGemmaSwitchLinear(
            inputDims: config.moeIntermediateSize,
            outputDims: config.hiddenSize,
            numExperts: config.numExperts,
            parameters: quantization.parameters(for: "\(prefix).down_proj")
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let routeCount = batchTokens * topK
        guard DiffusionGemmaExpertRouting.shouldSort(routeCount: routeCount) else {
            let gateUp = gateUpProj(x, indices: indices)
            let gate = gateUp[.ellipsis, ..<intermediateSize]
            let up = gateUp[.ellipsis, intermediateSize...]
            let routed = downProj(geluApproximate(gate) * up, indices: indices)
            return (routed * MLX.expandedDimensions(weights, axis: weights.ndim)).sum(axis: -2)
        }

        let inputDim = x.dim(-1)
        let plan = DiffusionGemmaExpertRouting.sortedPlan(indices: indices, topK: topK)
        let sortedInput = take(
            x.reshaped([batchTokens, inputDim]),
            plan.tokenOrder,
            axis: 0
        ).reshaped([routeCount, 1, inputDim])
        let gateUp = gateUpProj.applyFlat(
            sortedInput,
            indices: plan.sortedIndices,
            sortedIndices: true
        )
        let gate = gateUp[.ellipsis, ..<intermediateSize]
        let up = gateUp[.ellipsis, intermediateSize...]
        let sortedRouted = downProj.applyFlat(
            geluApproximate(gate) * up,
            indices: plan.sortedIndices,
            sortedIndices: true
        )
        let routed = take(sortedRouted, plan.inverseOrder, axis: 0).reshaped([
            x.dim(0),
            x.dim(1),
            topK,
            sortedRouted.dim(-1),
        ])
        return (routed * MLX.expandedDimensions(weights, axis: weights.ndim)).sum(axis: -2)
    }
}

final class DiffusionGemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: DiffusionGemmaTextConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class DiffusionGemmaRouter: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    private let topK: Int
    private let eps: Float
    private let rootSize: Float

    init(config: DiffusionGemmaTextConfig) {
        topK = config.topKExperts
        eps = config.rmsNormEps
        rootSize = pow(Float(config.hiddenSize), -0.5)
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        _perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let normWeight = scale * MLXArray(rootSize).asType(scale.dtype)
        let scores = proj(MLXFast.rmsNorm(x, weight: normWeight, eps: eps))
        let k = min(topK, scores.dim(-1))
        let indices = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        var weights = softmax(takeAlong(scores, indices, axis: -1), axis: -1)
        weights = weights * take(perExpertScale, indices, axis: 0)
        return (indices, weights)
    }
}

final class DiffusionGemmaAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let layerType: String
    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let slidingWindow: Int
    private let rope: any OffsetLayer
    private let eps: Float

    init(config: DiffusionGemmaTextConfig, layerIndex: Int) {
        layerType = config.layerTypes[layerIndex]
        let full = layerType == "full_attention"
        numHeads = config.numAttentionHeads
        numKVHeads = full ? config.numGlobalKeyValueHeads : config.numKeyValueHeads
        headDim = full ? config.globalHeadDim : config.headDim
        slidingWindow = config.slidingWindow
        eps = config.rmsNormEps

        let ropeConfig = config.ropeParameters[layerType]
        if ropeConfig?.ropeType == "proportional" {
            rope = Gemma4ProportionalRoPE(
                dims: headDim,
                base: ropeConfig?.ropeTheta ?? 1_000_000,
                partialRotaryFactor: ropeConfig?.partialRotaryFactor ?? 0.25
            )
        } else {
            rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: ropeConfig?.ropeTheta ?? 10_000
            )
        }

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = full ? nil : Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func causal(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let queryOffset = cache.offset
        let projected = project(x, offset: queryOffset)
        guard let state = cache.attentionState(appending: projected.keys, values: projected.values) else {
            preconditionFailure("DiffusionGemma cache failed to retain appended state.")
        }
        let mask = causalMask(
            queryLength: x.dim(1),
            queryOffset: queryOffset,
            keyLength: state.0.dim(2),
            dtype: x.dtype
        )
        return attend(projected.queries, keys: state.0, values: state.1, mask: mask, input: x)
    }

    func canvas(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let projected = project(x, offset: cache.offset)
        var prefix = cache.currentState()
        if layerType == "sliding_attention", let state = prefix {
            let keep = max(0, slidingWindow - 1)
            if state.0.dim(2) > keep {
                let start = state.0.dim(2) - keep
                prefix = (
                    state.0[0..., 0..., start..., 0...],
                    state.1[0..., 0..., start..., 0...]
                )
            }
        }
        let keys = prefix.map { concatenated([$0.0, projected.keys], axis: 2) } ?? projected.keys
        let values = prefix.map { concatenated([$0.1, projected.values], axis: 2) } ?? projected.values
        return attend(projected.queries, keys: keys, values: values, mask: .none, input: x)
    }

    private func project(_ x: MLXArray, offset: Int) -> (
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray
    ) {
        let batch = x.dim(0)
        let length = x.dim(1)
        var queries = qNorm(qProj(x).reshaped(batch, length, numHeads, headDim))
            .transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)
        let rawKeys = kProj(x)
        var keys = kNorm(rawKeys.reshaped(batch, length, numKVHeads, headDim))
            .transposed(0, 2, 1, 3)
        keys = rope(keys, offset: offset)
        let rawValues = vProj?(x) ?? rawKeys
        let values = gemma4RMSNormNoScale(
            rawValues.reshaped(batch, length, numKVHeads, headDim),
            eps: eps
        ).transposed(0, 2, 1, 3)
        return (queries, keys, values)
    }

    private func attend(
        _ queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        input: MLXArray
    ) -> MLXArray {
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1,
            mask: mask
        )
        return oProj(attended.transposed(0, 2, 1, 3).reshaped(
            input.dim(0),
            input.dim(1),
            numHeads * headDim
        ))
    }

    private func causalMask(
        queryLength: Int,
        queryOffset: Int,
        keyLength: Int,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard queryLength > 1 else { return .none }
        let keyStart = max(0, queryOffset + queryLength - keyLength)
        let queries = MLXArray(Int32(queryOffset)..<Int32(queryOffset + queryLength)).reshaped(queryLength, 1)
        let keys = MLXArray(Int32(keyStart)..<Int32(keyStart + keyLength)).reshaped(1, keyLength)
        var allowed = keys .<= queries
        if layerType == "sliding_attention" {
            allowed = allowed .&& (keys .> (queries - Int32(slidingWindow)))
        }
        let typed = allowed.asType(dtype).reshaped(1, 1, queryLength, keyLength)
        let zeros = MLXArray.zeros(typed.shape, dtype: dtype)
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(typed .> MLXArray(0).asType(dtype), zeros, negative))
    }
}

final class DiffusionGemmaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: DiffusionGemmaAttention
    @ModuleInfo(key: "mlp") var mlp: DiffusionGemmaMLP
    @ModuleInfo(key: "router") var router: DiffusionGemmaRouter
    @ModuleInfo(key: "experts") var experts: DiffusionGemmaExperts
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(
        config: DiffusionGemmaTextConfig,
        quantization: DiffusionGemmaQuantizationConfig,
        layerIndex: Int
    ) {
        _selfAttention.wrappedValue = DiffusionGemmaAttention(config: config, layerIndex: layerIndex)
        _mlp.wrappedValue = DiffusionGemmaMLP(config: config)
        _router.wrappedValue = DiffusionGemmaRouter(config: config)
        _experts.wrappedValue = DiffusionGemmaExperts(
            config: config,
            quantization: quantization,
            layerIndex: layerIndex
        )
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache,
        decoder: Bool,
        scalar: MLXArray? = nil
    ) -> MLXArray {
        let attentionResidual = x
        var hidden = inputLayerNorm(x)
        hidden = decoder
            ? selfAttention.canvas(hidden, cache: cache)
            : selfAttention.causal(hidden, cache: cache)
        hidden = attentionResidual + postAttentionLayerNorm(hidden)

        let feedForwardResidual = hidden
        var dense = mlp(preFeedforwardLayerNorm(hidden))
        dense = postFeedforwardLayerNorm1(dense)

        let route = router(hidden)
        var sparse = experts(
            preFeedforwardLayerNorm2(hidden),
            indices: route.indices,
            weights: route.weights
        )
        sparse = postFeedforwardLayerNorm2(sparse)

        hidden = feedForwardResidual + postFeedforwardLayerNorm(dense + sparse)
        return hidden * (scalar ?? layerScalar)
    }
}

final class DiffusionGemmaSelfConditioning: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "post_norm") var postNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    private let eps: Float

    init(config: DiffusionGemmaTextConfig) {
        eps = config.rmsNormEps
        _preNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ embeddings: MLXArray, signal: MLXArray) -> MLXArray {
        let normed = preNorm(signal)
        let projected = downProj(geluApproximate(gateProj(normed)) * upProj(normed))
        return gemma4RMSNormNoScale(embeddings + projected, eps: eps)
    }
}

final class DiffusionGemmaDecoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [DiffusionGemmaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "self_conditioning") var selfConditioning: DiffusionGemmaSelfConditioning

    let config: DiffusionGemmaTextConfig
    private let embedScale: Float

    init(config: DiffusionGemmaConfig) {
        self.config = config.textConfig
        embedScale = sqrt(Float(config.textConfig.hiddenSize))
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.textConfig.vocabSize,
            dimensions: config.textConfig.hiddenSize
        )
        _layers.wrappedValue = (0..<config.textConfig.numHiddenLayers).map {
            DiffusionGemmaDecoderLayer(
                config: config.textConfig,
                quantization: config.quantization,
                layerIndex: $0
            )
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.textConfig.hiddenSize,
            eps: config.textConfig.rmsNormEps
        )
        _selfConditioning.wrappedValue = DiffusionGemmaSelfConditioning(config: config.textConfig)
        super.init()
    }

    func embeddings(_ tokenIds: MLXArray) -> MLXArray {
        embedTokens(tokenIds.asType(.int32)) * MLXArray(embedScale).asType(embedTokens.weight.dtype)
    }

    func causal(
        _ tokenIds: MLXArray,
        caches: [Gemma4AttentionCache],
        encoderScalars: [MLXArray]
    ) {
        var hidden = embeddings(tokenIds)
        for index in layers.indices {
            hidden = layers[index](
                hidden,
                cache: caches[index],
                decoder: false,
                scalar: encoderScalars[index]
            )
        }
        _ = norm(hidden)
    }

    func canvasLogits(
        _ tokenIds: MLXArray,
        caches: [Gemma4AttentionCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        var hidden = embeddings(tokenIds)
        if let selfConditioningLogits,
           let quantizedEmbedding = embedTokens as? PreQuantizedEmbedding {
            let probabilities = softmax(selfConditioningLogits, axis: -1)
            let signal = quantizedMM(
                probabilities.asType(hidden.dtype),
                quantizedEmbedding.weight,
                scales: quantizedEmbedding.scales,
                biases: quantizedEmbedding.biases,
                transpose: false,
                groupSize: quantizedEmbedding.groupSize,
                bits: quantizedEmbedding.bits,
                mode: quantizedEmbedding.mode
            ) * MLXArray(embedScale).asType(hidden.dtype)
            hidden = selfConditioning(hidden, signal: signal)
        } else {
            hidden = selfConditioning(hidden, signal: MLXArray.zeros(hidden.shape, dtype: hidden.dtype))
        }
        for index in layers.indices {
            hidden = layers[index](hidden, cache: caches[index], decoder: true)
        }
        var logits = embedTokens.asLinear(norm(hidden))
        let softcap = MLXArray(config.finalLogitSoftcapping).asType(.float32)
        logits = tanh(logits.asType(.float32) / softcap) * softcap
        return logits
    }

    func makeCaches() -> [Gemma4AttentionCache] {
        config.layerTypes.map { layerType -> Gemma4AttentionCache in
            if layerType == "full_attention" {
                return Gemma4FullKVCache()
            }
            return Gemma4SlidingKVCache(maxSize: config.slidingWindow)
        }
    }
}

final class DiffusionGemmaEncoderScalar: Module {
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    override init() {
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }
}

final class DiffusionGemmaEncoderLanguageModel: Module {
    @ModuleInfo(key: "layers") var layers: [DiffusionGemmaEncoderScalar]

    init(layerCount: Int) {
        _layers.wrappedValue = (0..<layerCount).map { _ in DiffusionGemmaEncoderScalar() }
        super.init()
    }
}

final class DiffusionGemmaEncoder: Module {
    @ModuleInfo(key: "language_model") var languageModel: DiffusionGemmaEncoderLanguageModel

    init(layerCount: Int) {
        _languageModel.wrappedValue = DiffusionGemmaEncoderLanguageModel(layerCount: layerCount)
        super.init()
    }
}

final class DiffusionGemmaBackbone: Module {
    @ModuleInfo(key: "decoder") var decoder: DiffusionGemmaDecoder
    @ModuleInfo(key: "encoder") var encoder: DiffusionGemmaEncoder

    init(config: DiffusionGemmaConfig) {
        _decoder.wrappedValue = DiffusionGemmaDecoder(config: config)
        _encoder.wrappedValue = DiffusionGemmaEncoder(layerCount: config.textConfig.numHiddenLayers)
        super.init()
    }
}

public final class DiffusionGemmaLanguageModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") var model: DiffusionGemmaBackbone

    public let config: DiffusionGemmaConfig

    public init(config: DiffusionGemmaConfig) {
        self.config = config
        _model.wrappedValue = DiffusionGemmaBackbone(config: config)
        super.init()
    }

    func makeCaches() -> [Gemma4AttentionCache] {
        model.decoder.makeCaches()
    }

    func updateCache(_ tokenIds: MLXArray, caches: [Gemma4AttentionCache]) {
        model.decoder.causal(
            tokenIds,
            caches: caches,
            encoderScalars: model.encoder.languageModel.layers.map(\.layerScalar)
        )
    }

    func canvasLogits(
        _ tokenIds: MLXArray,
        caches: [Gemma4AttentionCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        model.decoder.canvasLogits(
            tokenIds,
            caches: caches,
            selfConditioningLogits: selfConditioningLogits
        )
    }
}
