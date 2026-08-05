import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

@inline(__always)
private func lfm2Swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

public final class LFM2ConvCache: @unchecked Sendable {
    var state: MLXArray?

    public init() {}

    public func fork() -> LFM2ConvCache {
        let copy = LFM2ConvCache()
        copy.state = state
        return copy
    }

    func batched(with caches: [LFM2ConvCache]) -> LFM2ConvCache? {
        guard !caches.isEmpty else { return nil }
        let states = caches.compactMap(\.state)
        guard states.count == caches.count,
              let first = states.first,
              states.allSatisfy({ Array($0.shape.dropFirst()) == Array(first.shape.dropFirst()) }) else {
            return nil
        }
        let result = LFM2ConvCache()
        result.state = concatenated(states, axis: 0)
        return result
    }

    func unbatchedRows(count: Int) -> [LFM2ConvCache]? {
        guard count > 0, let state, state.dim(0) == count else { return nil }
        return (0..<count).map { index in
            let result = LFM2ConvCache()
            result.state = state[index..<(index + 1), 0..., 0...]
            return result
        }
    }
}

public enum LFM2LayerCache: @unchecked Sendable {
    case attention(KVCache)
    case conv(LFM2ConvCache)

    func fork() -> LFM2LayerCache {
        switch self {
        case .attention(let cache):
            return .attention(cache.fork())
        case .conv(let cache):
            return .conv(cache.fork())
        }
    }

    var offset: Int? {
        if case .attention(let cache) = self {
            return cache.offset
        }
        return nil
    }

    var batchSignature: String {
        switch self {
        case .attention(let cache):
            if cache.supportsVariablePositionBatching {
                return "attention:variable"
            }
            return "attention:\(cache.offset)"
        case .conv(let cache):
            let shape = cache.state?.shape.map(String.init).joined(separator: "x") ?? "nil"
            return "conv:\(shape)"
        }
    }

    func batched(with caches: [LFM2LayerCache]) -> LFM2LayerCache? {
        guard !caches.isEmpty else { return nil }
        switch self {
        case .attention(let first):
            let typed = caches.compactMap { cache -> KVCache? in
                guard case .attention(let value) = cache else { return nil }
                return value
            }
            guard typed.count == caches.count, let batched = first.batched(with: typed) else {
                return nil
            }
            return .attention(batched)
        case .conv(let first):
            let typed = caches.compactMap { cache -> LFM2ConvCache? in
                guard case .conv(let value) = cache else { return nil }
                return value
            }
            guard typed.count == caches.count, let batched = first.batched(with: typed) else {
                return nil
            }
            return .conv(batched)
        }
    }

    func unbatchedRows(count: Int) -> [LFM2LayerCache]? {
        switch self {
        case .attention(let cache):
            return cache.unbatchedRows(count: count)?.map(LFM2LayerCache.attention)
        case .conv(let cache):
            return cache.unbatchedRows(count: count)?.map(LFM2LayerCache.conv)
        }
    }
}

final class LFM2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let scale: Float
    private let rope: RoPE

    init(config: LFM2Config) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(max(1, config.headDim)))
        self.rope = RoPE(
            dimensions: max(1, config.headDim),
            traditional: false,
            base: config.ropeParameters?.ropeTheta ?? config.ropeTheta
        )

        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: false
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._outProj.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim,
            config.hiddenSize,
            bias: false
        )
        self._qLayerNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.normEps)
        self._kLayerNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.normEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)

        let queriesProjected = qProj(x).reshaped(batch, sequence, numHeads, headDim)
        let keysProjected = kProj(x).reshaped(batch, sequence, numKVHeads, headDim)
        let valuesProjected = vProj(x).reshaped(batch, sequence, numKVHeads, headDim)

        var queries = qLayerNorm(queriesProjected).transposed(0, 2, 1, 3)
        var keys = kLayerNorm(keysProjected).transposed(0, 2, 1, 3)
        var values = valuesProjected.transposed(0, 2, 1, 3)

        if let rowOffsets = cache?.rowOffsets, rowOffsets.count == batch {
            queries = applyRoPEByRow(queries, rowOffsets: rowOffsets)
            keys = applyRoPEByRow(keys, rowOffsets: rowOffsets)
        } else {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        if let cache {
            let cached = cache.update(keys: keys, values: values)
            keys = cached.0
            values = cached.1
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        let output = attended.transposed(0, 2, 1, 3)
            .reshaped(batch, sequence, numHeads * headDim)
        return outProj(output)
    }

    private func applyRoPEByRow(_ value: MLXArray, rowOffsets: [Int]) -> MLXArray {
        guard !rowOffsets.isEmpty else {
            return rope(value, offset: 0)
        }
        let rows = rowOffsets.enumerated().map { index, offset in
            rope(value[index..<(index + 1), 0..., 0..., 0...], offset: offset)
        }
        return concatenated(rows, axis: 0)
    }
}

final class LFM2ShortConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    private let hiddenSize: Int
    private let cacheLength: Int

    init(config: LFM2Config) {
        self.hiddenSize = config.hiddenSize
        self.cacheLength = max(1, config.convLCache)
        self._conv.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: max(1, config.convLCache),
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: config.hiddenSize,
            bias: config.convBias
        )
        self._inProj.wrappedValue = Linear(
            config.hiddenSize,
            3 * config.hiddenSize,
            bias: config.convBias
        )
        self._outProj.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: config.convBias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: LFM2ConvCache?) -> MLXArray {
        let projected = inProj(x)
        let bGate = projected[.ellipsis, 0..<hiddenSize]
        let cGate = projected[.ellipsis, hiddenSize..<(2 * hiddenSize)]
        let values = projected[.ellipsis, (2 * hiddenSize)...]
        var convInput = bGate * values
        let keep = max(0, cacheLength - 1)

        if let cache {
            let state = cache.state ?? MLXArray.zeros(
                [x.dim(0), keep, hiddenSize],
                dtype: x.dtype
            )
            convInput = concatenated([state, convInput], axis: 1)
            if keep > 0 {
                cache.state = convInput[0..., (convInput.dim(1) - keep)..., 0...]
            } else {
                cache.state = MLXArray.zeros([x.dim(0), 0, hiddenSize], dtype: x.dtype)
            }
        } else {
            convInput = padded(
                convInput,
                widths: [[0, 0], [keep, 0], [0, 0]],
                value: MLXArray(0.0).asType(x.dtype)
            )
        }

        return outProj(cGate * conv(convInput))
    }
}

final class LFM2SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?

    let groupSize: Int
    let bits: Int

    init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        groupSize: Int,
        bits: Int,
        quantized: Bool
    ) {
        self.groupSize = groupSize
        self.bits = bits

        let scale = sqrt(1.0 / Float(max(1, inputDims)))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )
        let groups = max(1, (inputDims + groupSize - 1) / groupSize)
        self._scales.wrappedValue = quantized
            ? MLXArray.zeros([numExperts, outputDims, groups])
            : nil
        self._biases.wrappedValue = quantized
            ? MLXArray.zeros([numExperts, outputDims, groups])
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)

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
            let resolved = resolvedQuantization(inputDim: inputDim, scales: scales)
            output = portableGatherQuantizedMM(
                flatX,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: flatIndices,
                transpose: true,
                groupSize: resolved.groupSize,
                bits: resolved.bits,
                mode: .affine,
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

        let outDim = output.dim(2)
        return output
            .reshaped([batchTokens, topK, outDim])
            .reshaped([x.dim(0), x.dim(1), topK, outDim])
    }

    private func resolvedQuantization(inputDim: Int, scales: MLXArray) -> (groupSize: Int, bits: Int) {
        var resolvedBits = bits
        let packedInputDim = weight.dim(weight.ndim - 1)
        let numerator = packedInputDim * 32
        if inputDim > 0, numerator % inputDim == 0 {
            let inferredBits = numerator / inputDim
            if (2...8).contains(inferredBits) {
                resolvedBits = inferredBits
            }
        }

        var resolvedGroupSize = groupSize
        let scaleGroups = scales.dim(scales.ndim - 1)
        if inputDim > 0, scaleGroups > 0, inputDim % scaleGroups == 0 {
            resolvedGroupSize = inputDim / scaleGroups
        }
        return (resolvedGroupSize, resolvedBits)
    }
}

enum LFM2MoEAccelerationPolicy {
    static let fusedAffine8MoEEnabled = RoutedMoERouting.parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LFM2_FUSED_AFFINE8_MOE"],
        default: true
    )
}

final class LFM2SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: LFM2SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: LFM2SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: LFM2SwitchLinear

    init(config: LFM2Config) {
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8
        let quantized = config.quantization != nil

        self._gateProj.wrappedValue = LFM2SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized
        )
        self._upProj.wrappedValue = LFM2SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized
        )
        self._downProj.wrappedValue = LFM2SwitchLinear(
            inputDims: config.moeIntermediateSize,
            outputDims: config.hiddenSize,
            numExperts: config.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        return unsorted(x, indices: indices)
    }

    private func unsorted(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        if LFM2MoEAccelerationPolicy.fusedAffine8MoEEnabled,
           gateProj.groupSize == upProj.groupSize,
           gateProj.bits == upProj.bits,
           let gateScales = gateProj.scales,
           let gateBiases = gateProj.biases,
           let upScales = upProj.scales,
           let upBiases = upProj.biases,
           let fused = RoutedMoERouting.fusedGatherAffine8SwiGLU(
               x,
               gateWeight: gateProj.weight,
               gateScales: gateScales,
               gateBiases: gateBiases,
               upWeight: upProj.weight,
               upScales: upScales,
               upBiases: upBiases,
               expertIndices: indices,
               topK: topK,
               groupSize: gateProj.groupSize,
               bits: gateProj.bits
           ) {
            return downProj(
                fused.reshaped([
                    batch,
                    sequenceLength,
                    topK,
                    fused.dim(-1),
                ]),
                indices: indices
            )
        }
        let up = upProj(x, indices: indices)
        let gate = gateProj(x, indices: indices)
        return downProj(lfm2Swiglu(gate, up), indices: indices)
    }
}

final class LFM2FeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "up_proj") var upProj: Linear?
    @ModuleInfo(key: "down_proj") var downProj: Linear?
    @ModuleInfo(key: "w1") var w1: Linear?
    @ModuleInfo(key: "w2") var w2: Linear?
    @ModuleInfo(key: "w3") var w3: Linear?
    @ModuleInfo(key: "gate") var gate: Linear?
    @ModuleInfo(key: "switch_mlp") var switchMLP: LFM2SwitchGLU?
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray?

    private let usesDense: Bool
    private let usesDenseWeightNames: Bool
    private let topK: Int
    private let normTopKProb: Bool

    init(config: LFM2Config, layerIndex: Int) {
        self.usesDense = config.modelType == "lfm2" || layerIndex < config.numDenseLayers
        self.usesDenseWeightNames = config.modelType == "lfm2"
        self.topK = max(1, config.numExpertsPerTok)
        self.normTopKProb = config.normTopKProb
        if usesDenseWeightNames {
            self._w1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._w2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
            self._w3.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        } else if usesDense {
            self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        } else {
            self._gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            self._switchMLP.wrappedValue = LFM2SwitchGLU(config: config)
            self._expertBias.wrappedValue = config.useExpertBias
                ? MLXArray.zeros([config.numExperts])
                : nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesDenseWeightNames, let w1, let w2, let w3 {
            return w2(lfm2Swiglu(w1(x), w3(x)))
        }
        if usesDense, let gateProj, let upProj, let downProj {
            return downProj(lfm2Swiglu(gateProj(x), upProj(x)))
        }

        guard let gate, let switchMLP else {
            preconditionFailure("LFM2 sparse feed-forward modules were not initialized")
        }
        var scores = softmax(gate(x).asType(.float32), axis: -1)
        if let expertBias {
            scores = scores + expertBias
        }

        let indices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
        scores = takeAlong(scores, indices, axis: -1)
        if normTopKProb, topK > 1 {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + MLXArray(1e-20))
        }
        scores = scores.asType(x.dtype)

        let switched = switchMLP(x, indices: indices)
        var routed = switched * MLX.expandedDimensions(scores, axis: scores.ndim)
        routed = routed.sum(axis: -2)
        return routed
    }
}

final class LFM2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: LFM2Attention?
    @ModuleInfo(key: "conv") var conv: LFM2ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM2FeedForward
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    let isAttentionLayer: Bool

    init(config: LFM2Config, layerIndex: Int) {
        self.isAttentionLayer = config.fullAttentionLayerIndexes.contains(layerIndex)
        if isAttentionLayer {
            self._selfAttention.wrappedValue = LFM2Attention(config: config)
        } else {
            self._conv.wrappedValue = LFM2ShortConv(config: config)
        }
        self._feedForward.wrappedValue = LFM2FeedForward(config: config, layerIndex: layerIndex)
        self._operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: LFM2LayerCache?
    ) -> MLXArray {
        let normed = operatorNorm(x)
        let operatorOut: MLXArray
        if isAttentionLayer {
            let attentionCache: KVCache?
            if case .attention(let kv)? = cache {
                attentionCache = kv
            } else {
                attentionCache = nil
            }
            guard let selfAttention else {
                preconditionFailure("LFM2 attention module was not initialized")
            }
            operatorOut = selfAttention(normed, mask: attentionMask, cache: attentionCache)
        } else {
            let convCache: LFM2ConvCache?
            if case .conv(let c)? = cache {
                convCache = c
            } else {
                convCache = nil
            }
            guard let conv else {
                preconditionFailure("LFM2 convolution module was not initialized")
            }
            operatorOut = conv(normed, cache: convCache)
        }

        let hidden = x + operatorOut
        return hidden + feedForward(ffnNorm(hidden))
    }
}

final class LFM2Transformer: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LFM2DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(config: LFM2Config) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { index in
            LFM2DecoderLayer(config: config, layerIndex: index)
        }
        self._embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        super.init()
    }

    func embeddings(for inputIds: MLXArray) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        return embedTokens(tokenIds)
    }

    private func firstAttentionCache(from cache: [LFM2LayerCache?]?) -> KVCache? {
        guard let cache else { return nil }
        for entry in cache {
            if case .attention(let kv)? = entry {
                return kv
            }
        }
        return nil
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: [LFM2LayerCache?]?,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hidden = inputEmbeddings ?? embeddings(for: inputIds)
        let attentionMask = createAttentionMask(h: hidden, cache: firstAttentionCache(from: cache))

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                attentionMask: attentionMask,
                cache: cache?[index] ?? nil
            )
        }
        return embeddingNorm(hidden)
    }
}

struct LFM2ForwardOutput {
    let hidden: MLXArray
    let logits: MLXArray
}

public final class LFM2Model: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") var model: LFM2Transformer

    public let config: LFM2Config

    public init(config: LFM2Config) {
        self.config = config
        self._model.wrappedValue = LFM2Transformer(config: config)
        super.init()
    }

    public func embeddings(for inputIds: MLXArray) -> MLXArray {
        model.embeddings(for: inputIds)
    }

    func forward(
        _ inputIds: MLXArray,
        cache: [LFM2LayerCache?]?,
        inputEmbeddings: MLXArray? = nil
    ) -> LFM2ForwardOutput {
        let hidden = model(inputIds, cache: cache, inputEmbeddings: inputEmbeddings)
        return LFM2ForwardOutput(hidden: hidden, logits: model.embedTokens.asLinear(hidden))
    }

    /// Forward for prefill chunks: hidden states flow through every position
    /// (the caches need them all), but the tied-embedding lm_head projects
    /// only the final position — prefill consumers read exactly the last
    /// position, and the full-chunk projection is a [chunk, vocab] matmul
    /// plus its materialization per chunk.
    func forwardPrefill(
        _ inputIds: MLXArray,
        cache: [LFM2LayerCache?]?,
        inputEmbeddings: MLXArray? = nil
    ) -> LFM2ForwardOutput {
        var hidden = model(inputIds, cache: cache, inputEmbeddings: inputEmbeddings)
        let sequenceLength = hidden.dim(1)
        if sequenceLength > 1 {
            hidden = hidden[0..., (sequenceLength - 1)..., 0...]
        }
        return LFM2ForwardOutput(hidden: hidden, logits: model.embedTokens.asLinear(hidden))
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [LFM2LayerCache?]?,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        forward(inputIds, cache: cache, inputEmbeddings: inputEmbeddings).logits
    }
}
