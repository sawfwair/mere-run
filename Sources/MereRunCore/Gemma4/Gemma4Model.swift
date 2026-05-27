import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

@inline(__always)
private func gemma4RepeatAlongHeads(_ x: MLXArray, heads: Int) -> MLXArray {
    MLX.repeated(x, count: heads, axis: 1)
}

@inline(__always)
private func gemma4RMSNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    let normalized = x32 * rsqrt(variance + MLXArray(eps))
    return normalized.asType(dtype)
}

/// Proportional RoPE for Gemma 4 full-attention layers.
///
/// Frequencies are computed relative to the **full** head dimension (not just the
/// rotated portion), and rotation is applied to the first `rotatedDims/2`
/// elements of each half of the head — matching HF's rotate_half convention.
final class Gemma4ProportionalRoPE: Module, OffsetLayer {
    private let dims: Int
    private let rotatedDims: Int
    private let traditional: Bool
    private let freqs: MLXArray?

    init(dims: Int, traditional: Bool = false, base: Float = 10_000, partialRotaryFactor: Float = 1.0, factor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional
        let ropeAngles = Int(partialRotaryFactor * Float(dims / 2))
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2))
                / MLXArray(Float(dims))
            self.freqs = MLXArray(factor) * pow(MLXArray(base), exponents)
        } else {
            self.freqs = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0 else { return x }

        let head = x[0..., 0..., 0..., ..<dims]
        let half = dims / 2
        let rotHalf = rotatedDims / 2

        let left = head[0..., 0..., 0..., ..<half]
        let right = head[0..., 0..., 0..., half...]

        let toRotate = concatenated(
            [left[0..., 0..., 0..., ..<rotHalf], right[0..., 0..., 0..., ..<rotHalf]],
            axis: -1
        )
        let rotated = MLXFast.RoPE(
            toRotate,
            dimensions: rotatedDims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )

        let newLeft = concatenated(
            [rotated[0..., 0..., 0..., ..<rotHalf], left[0..., 0..., 0..., rotHalf...]],
            axis: -1
        )
        let newRight = concatenated(
            [rotated[0..., 0..., 0..., rotHalf...], right[0..., 0..., 0..., rotHalf...]],
            axis: -1
        )
        let newHead = concatenated([newLeft, newRight], axis: -1)

        if x.dim(-1) > dims {
            return concatenated([newHead, x[0..., 0..., 0..., dims...]], axis: -1)
        }
        return newHead
    }
}

protocol Gemma4AttentionCache: AnyObject {
    var offset: Int { get }
    func currentState() -> (MLXArray, MLXArray)?
    func append(keys: MLXArray, values: MLXArray)
    func fork() -> Gemma4AttentionCache
    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache?
    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]?
    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray?
}

extension Gemma4AttentionCache {
    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        nil
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        nil
    }

    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        nil
    }
}

final class Gemma4FullKVCache: Gemma4AttentionCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset: Int = 0

    func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        return (keys, values)
    }

    func append(keys: MLXArray, values: MLXArray) {
        if let existingKeys = self.keys, let existingValues = self.values {
            self.keys = concatenated([existingKeys, keys], axis: 2)
            self.values = concatenated([existingValues, values], axis: 2)
        } else {
            self.keys = keys
            self.values = values
        }
        self.offset += keys.dim(2)
    }

    func fork() -> Gemma4AttentionCache {
        let copy = Gemma4FullKVCache()
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        return copy
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4FullKVCache],
              !typed.isEmpty,
              typed.allSatisfy({ $0.offset == offset }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4FullKVCache()
        copy.keys = concatenated(states.map(\.0), axis: 0)
        copy.values = concatenated(states.map(\.1), axis: 0)
        copy.offset = offset
        return copy
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0, let keys, let values, keys.dim(0) == count, values.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4FullKVCache()
            copy.keys = keys[index..<(index + 1), 0..., 0..., 0...]
            copy.values = values[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            return copy
        }
    }
}

final class Gemma4SlidingKVCache: Gemma4AttentionCache {
    private let maxSize: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset: Int = 0

    init(maxSize: Int) {
        self.maxSize = max(1, maxSize)
    }

    func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        return (keys, values)
    }

    func append(keys: MLXArray, values: MLXArray) {
        let combinedKeys: MLXArray
        let combinedValues: MLXArray
        if let existingKeys = self.keys, let existingValues = self.values {
            combinedKeys = concatenated([existingKeys, keys], axis: 2)
            combinedValues = concatenated([existingValues, values], axis: 2)
        } else {
            combinedKeys = keys
            combinedValues = values
        }

        let totalLength = combinedKeys.dim(2)
        if totalLength > maxSize {
            let start = totalLength - maxSize
            self.keys = combinedKeys[0..., 0..., start..., 0...]
            self.values = combinedValues[0..., 0..., start..., 0...]
        } else {
            self.keys = combinedKeys
            self.values = combinedValues
        }
        self.offset += keys.dim(2)
    }

    func fork() -> Gemma4AttentionCache {
        let copy = Gemma4SlidingKVCache(maxSize: maxSize)
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        return copy
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4SlidingKVCache],
              !typed.isEmpty,
              typed.allSatisfy({ $0.offset == offset && $0.maxSize == maxSize }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4SlidingKVCache(maxSize: maxSize)
        copy.keys = concatenated(states.map(\.0), axis: 0)
        copy.values = concatenated(states.map(\.1), axis: 0)
        copy.offset = offset
        return copy
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0, let keys, let values, keys.dim(0) == count, values.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4SlidingKVCache(maxSize: maxSize)
            copy.keys = keys[index..<(index + 1), 0..., 0..., 0...]
            copy.values = values[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            return copy
        }
    }
}

final class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4TextConfig, layerIndex: Int) {
        let firstSharedIndex = config.numHiddenLayers - config.numKVSharedLayers
        let isSharedKVLayer = config.numKVSharedLayers > 0 && layerIndex >= firstSharedIndex
        let widthMultiplier = (config.useDoubleWideMLP && isSharedKVLayer) ? 2 : 1
        let intermediate = config.intermediateSize * widthMultiplier

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode

    init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        groupSize: Int = 16,
        bits: Int = 4,
        mode: QuantizationMode = .nvfp4,
        bias: Bool = false
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let scale = sqrt(1.0 / Float(max(1, inputDims)))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )
        let groups = max(1, (inputDims + groupSize - 1) / groupSize)
        self._scales.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        self._biases.wrappedValue = nil
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)
        let batchTokens = batch * sequenceLength

        let flatX: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatX = x.reshaped([batchTokens * topK, 1, inputDim])
        } else {
            var expanded = x.reshaped([batchTokens, 1, inputDim])
            expanded = MLX.expandedDimensions(expanded, axis: 1)
            expanded = MLX.repeated(expanded, count: topK, axis: 1)
            flatX = expanded.reshaped([batchTokens * topK, 1, inputDim])
        }

        let flatIndices = indices.reshaped([batchTokens * topK])
        let output: MLXArray
        if let scales {
            output = gatherQuantizedMM(
                flatX,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: flatIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                sortedIndices: false
            )
        } else {
            output = gatherMM(
                flatX,
                weight.swappedAxes(-1, -2),
                rhsIndices: flatIndices,
                sortedIndices: false
            )
        }

        let outputDim = output.dim(2)
        var reshaped = output.reshaped([batchTokens, topK, outputDim])
        reshaped = reshaped.reshaped([batch, sequenceLength, topK, outputDim])

        if let bias {
            let selectedBias = take(bias, flatIndices, axis: 0)
                .reshaped([batch, sequenceLength, topK, outputDim])
            return reshaped + selectedBias
        }
        return reshaped
    }
}

final class Gemma4SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4SwitchLinear

    init(config: Gemma4TextConfig) {
        self._gateProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: max(1, config.moeIntermediateSize),
            numExperts: max(1, config.numExperts)
        )
        self._upProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: max(1, config.moeIntermediateSize),
            numExperts: max(1, config.numExperts)
        )
        self._downProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: max(1, config.moeIntermediateSize),
            outputDims: config.hiddenSize,
            numExperts: max(1, config.numExperts)
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let up = upProj(x, indices: indices)
        let gate = gateProj(x, indices: indices)
        return downProj(geluApproximate(gate) * up, indices: indices)
    }
}

final class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    private let topK: Int
    private let eps: Float
    private let rootSize: Float

    init(config: Gemma4TextConfig) {
        self.topK = max(1, config.topKExperts)
        self.eps = config.rmsNormEps
        self.rootSize = pow(Float(max(1, config.hiddenSize)), -0.5)
        self._proj.wrappedValue = Linear(config.hiddenSize, max(1, config.numExperts), bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([max(1, config.numExperts)])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let normWeight = scale * MLXArray(rootSize).asType(scale.dtype)
        let routedInput = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
        let expertScores = proj(routedInput)
        let k = min(topK, expertScores.dim(-1))
        let indices = argPartition(-expertScores, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        var weights = takeAlong(expertScores, indices, axis: -1)
        weights = softmax(weights, axis: -1)
        weights = weights * take(perExpertScale, indices, axis: 0)
        return (indices, weights)
    }
}

final class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: Gemma4SwitchGLU

    init(config: Gemma4TextConfig) {
        self._switchGLU.wrappedValue = Gemma4SwitchGLU(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let routed = switchGLU(x, indices: indices)
        return (routed * MLX.expandedDimensions(weights, axis: weights.ndim)).sum(axis: -2)
    }
}

final class Gemma4Attention: Module {
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
    private let windowSize: Int
    private let rope: any OffsetLayer
    private let rmsNormEps: Float
    private let isKVSharedLayer: Bool
    private let useKeyEqualsValue: Bool

    init(config: Gemma4TextConfig, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self.numHeads = config.numAttentionHeads
        self.windowSize = config.slidingWindow
        self.rmsNormEps = config.rmsNormEps

        let isFullAttention = layerType == "full_attention"
        self.headDim = isFullAttention ? (config.globalHeadDim ?? config.headDim) : config.headDim
        self.numKVHeads = isFullAttention ? (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads) : config.numKeyValueHeads
        self.isKVSharedLayer = config.numKVSharedLayers > 0 && layerIndex >= (config.numHiddenLayers - config.numKVSharedLayers)
        self.useKeyEqualsValue = config.attentionKEqV && isFullAttention

        let ropeConfig = config.ropeParameters[layerType] ?? config.ropeParameters["sliding_attention"]
        let ropeBase = ropeConfig?.ropeTheta ?? 10_000
        let partial = max(0, min(1, ropeConfig?.partialRotaryFactor ?? 1))
        let ropeType = ropeConfig?.ropeType ?? "default"
        if ropeType == "proportional" {
            self.rope = Gemma4ProportionalRoPE(
                dims: headDim,
                traditional: false,
                base: ropeBase,
                partialRotaryFactor: partial
            )
        } else {
            let ropeDims = max(1, Int(Float(headDim) * partial))
            self.rope = RoPE(
                dimensions: ropeDims,
                traditional: false,
                base: ropeBase
            )
        }

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = useKeyEqualsValue
            ? nil
            : Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let sequenceLength = x.dim(1)

        let rawQueries = qProj(x).reshaped(batchSize, sequenceLength, numHeads, headDim)
        var queries = qNorm(rawQueries).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)

        let scale: Float = 1.0
        let repeats = max(1, numHeads / max(1, numKVHeads))

        let keys: MLXArray
        let values: MLXArray
        if isKVSharedLayer, let cache {
            if sequenceLength == 1,
               let attended = cache.specializedAttention(queries: queries, repeats: repeats, scale: scale) {
                let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
                return oProj(reshaped)
            }
            if let shared = cache.currentState() {
                keys = shared.0
                values = shared.1
            } else {
                var rawKeys = kProj(x).reshaped(batchSize, sequenceLength, numKVHeads, headDim)
                var rawValues = useKeyEqualsValue
                    ? rawKeys
                    : vProj!(x).reshaped(batchSize, sequenceLength, numKVHeads, headDim)

                rawKeys = kNorm(rawKeys)
                rawValues = gemma4RMSNormNoScale(rawValues, eps: rmsNormEps)

                var computedKeys = rawKeys.transposed(0, 2, 1, 3)
                let computedValues = rawValues.transposed(0, 2, 1, 3)
                computedKeys = rope(computedKeys, offset: offset)

                cache.append(keys: computedKeys, values: computedValues)
                let updated = cache.currentState()!
                keys = updated.0
                values = updated.1
            }
        } else {
            var rawKeys = kProj(x).reshaped(batchSize, sequenceLength, numKVHeads, headDim)
            var rawValues = useKeyEqualsValue
                ? rawKeys
                : vProj!(x).reshaped(batchSize, sequenceLength, numKVHeads, headDim)

            rawKeys = kNorm(rawKeys)
            rawValues = gemma4RMSNormNoScale(rawValues, eps: rmsNormEps)

            var computedKeys = rawKeys.transposed(0, 2, 1, 3)
            let computedValues = rawValues.transposed(0, 2, 1, 3)
            computedKeys = rope(computedKeys, offset: offset)

            if let cache {
                cache.append(keys: computedKeys, values: computedValues)
                if sequenceLength == 1,
                   let attended = cache.specializedAttention(queries: queries, repeats: repeats, scale: scale) {
                    let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
                    return oProj(reshaped)
                }
                let updated = cache.currentState()!
                keys = updated.0
                values = updated.1
            } else {
                keys = computedKeys
                values = computedValues
            }
        }

        var broadcastKeys = keys
        var broadcastValues = values
        if repeats > 1 {
            broadcastKeys = gemma4RepeatAlongHeads(broadcastKeys, heads: repeats)
            broadcastValues = gemma4RepeatAlongHeads(broadcastValues, heads: repeats)
        }

        let mask = makeAttentionMask(
            queryLength: sequenceLength,
            queryOffset: offset,
            keyLength: broadcastKeys.dim(2),
            windowSize: layerType == "sliding_attention" ? windowSize : nil,
            dtype: x.dtype
        )

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: broadcastKeys,
            values: broadcastValues,
            scale: scale,
            mask: mask
        )
        let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
        return oProj(reshaped)
    }

    private func makeAttentionMask(
        queryLength: Int,
        queryOffset: Int,
        keyLength: Int,
        windowSize: Int?,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard queryLength > 1 else {
            return .none
        }

        let keyStart = max(0, queryOffset + queryLength - keyLength)
        let queryPositions = MLXArray(Int32(queryOffset)..<Int32(queryOffset + queryLength)).reshaped(queryLength, 1)
        let keyPositions = MLXArray(Int32(keyStart)..<Int32(keyStart + keyLength)).reshaped(1, keyLength)

        var allowed = keyPositions .<= queryPositions
        if let windowSize {
            allowed = allowed .&& (keyPositions .> (queryPositions - Int32(windowSize)))
        }

        let allowedTyped = allowed.asType(dtype).reshaped(1, 1, queryLength, keyLength)
        let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(allowedTyped .> MLXArray(0).asType(dtype), zeros, negative))
    }
}

final class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "router") var router: Gemma4Router?
    @ModuleInfo(key: "experts") var experts: Gemma4Experts?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    private let hasPerLayerInput: Bool

    init(config: Gemma4TextConfig, layerIndex: Int) {
        self._selfAttention.wrappedValue = Gemma4Attention(config: config, layerIndex: layerIndex)
        self._mlp.wrappedValue = Gemma4MLP(config: config, layerIndex: layerIndex)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4Router(config: config)
            self._experts.wrappedValue = Gemma4Experts(config: config)
            self._preFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self._router.wrappedValue = nil
            self._experts.wrappedValue = nil
            self._preFeedforwardLayerNorm2.wrappedValue = nil
            self._postFeedforwardLayerNorm1.wrappedValue = nil
            self._postFeedforwardLayerNorm2.wrappedValue = nil
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0
        self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, max(1, config.hiddenSizePerLayerInput), bias: false)
        self._perLayerProjection.wrappedValue = Linear(max(1, config.hiddenSizePerLayerInput), config.hiddenSize, bias: false)
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        perLayerInput: MLXArray?
    ) -> MLXArray {
        let attentionResidual = x
        var hidden = inputLayerNorm(x)
        hidden = selfAttention(hidden, cache: cache)
        hidden = postAttentionLayerNorm(hidden)
        hidden = attentionResidual + hidden

        let mlpResidual = hidden
        if let router, let experts, let preFeedforwardLayerNorm2, let postFeedforwardLayerNorm1, let postFeedforwardLayerNorm2 {
            var dense = preFeedforwardLayerNorm(hidden)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let route = router(hidden)
            var sparse = preFeedforwardLayerNorm2(hidden)
            sparse = experts(sparse, indices: route.indices, weights: route.weights)
            sparse = postFeedforwardLayerNorm2(sparse)

            hidden = dense + sparse
        } else {
            hidden = preFeedforwardLayerNorm(hidden)
            hidden = mlp(hidden)
        }
        hidden = postFeedforwardLayerNorm(hidden)
        hidden = mlpResidual + hidden

        if hasPerLayerInput, let perLayerInput {
            let gateResidual = hidden
            var gate = perLayerInputGate(hidden)
            gate = geluApproximate(gate)
            gate = gate * perLayerInput
            gate = perLayerProjection(gate)
            gate = postPerLayerInputNorm(gate)
            hidden = gateResidual + gate
        }

        return hidden * layerScalar
    }
}

final class Gemma4LanguageModel: Module {
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

    init(config: Gemma4TextConfig) {
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

        var hidden = embedTokens(tokenIds) * MLXArray(embedScale).asType(embedTokens.weight.dtype)
        let perLayerInputs = projectPerLayerInputs(
            hiddenStates: hidden,
            inputIds: tokenIds
        )

        let caches = cache ?? makeCache()
        for (index, layer) in layers.enumerated() {
            let cacheIndex = index < layerIndexToCacheIndex.count ? layerIndexToCacheIndex[index] : index
            let perLayerInput = perLayerInputs[0..., 0..., index, 0...]
            hidden = layer(
                hidden,
                cache: cacheIndex < caches.count ? caches[cacheIndex] : nil,
                perLayerInput: perLayerInput
            )
        }
        return norm(hidden)
    }

    func logits(_ inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        let hidden = self(inputIds, cache: cache)
        return embedTokens.asLinear(hidden)
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
}

public final class Gemma4TextCausalLM: Module {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel

    public let config: Gemma4TextConfig
    private let finalLogitSoftcapping: Float?

    public init(config: Gemma4TextConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._languageModel.wrappedValue = Gemma4LanguageModel(config: config)
        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [AnyObject]? = nil
    ) -> MLXArray {
        let typedCache = cache as? [Gemma4AttentionCache]
        var logits = languageModel.logits(inputIds, cache: typedCache)
        if let finalLogitSoftcapping {
            let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    public func makeCache(quantization: Gemma4KVCacheQuantization? = nil) -> [AnyObject] {
        languageModel.makeCache(quantization: quantization)
    }
}
