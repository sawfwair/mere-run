import Foundation
import MLX
import MLXFast
import MLXNN

private func swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

private func repeatAlongHeads(_ x: MLXArray, heads: Int) -> MLXArray {
    MLX.repeated(x, count: heads, axis: 1)
}

private func createMask(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    createAttentionMask(h: h, cache: cache)
}

private final class GLM47SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let groupSize: Int
    let bits: Int

    init(inputDims: Int, outputDims: Int, numExperts: Int, groupSize: Int, bits: Int, bias: Bool) {
        self.groupSize = groupSize
        self.bits = bits
        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )
        let groups = max(1, (inputDims + groupSize - 1) / groupSize)
        self._scales.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        self._biases.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)

        // SwitchGLU calls gate/up/down in sequence; downProj sees [B,L,topK,H].
        // Detect already-expanded inputs to avoid double expansion.
        let flatX: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatX = x.reshaped([batchTokens * topK, 1, inputDim])
        } else {
            var expanded = x.reshaped([batchTokens, 1, inputDim])
            expanded = MLX.expandedDimensions(expanded, axis: 1)
            expanded = MLX.repeated(expanded, count: topK, axis: 1)
            flatX = expanded.reshaped([batchTokens * topK, 1, inputDim])
        }

        let flatIdx = indices.reshaped([batchTokens * topK])

        let output: MLXArray
        if let scales {
            output = gatherQuantizedMM(
                flatX,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: flatIdx,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                sortedIndices: false
            )
        } else {
            output = gatherMM(
                flatX,
                weight,
                rhsIndices: flatIdx,
                sortedIndices: false
            )
        }

        let outDim = output.dim(2)
        var reshaped = output.reshaped([batchTokens, topK, outDim])
        reshaped = reshaped.reshaped([x.dim(0), x.dim(1), topK, outDim])

        if let bias {
            let biasExpanded = bias.reshaped([1, 1, 1, outDim])
            return reshaped + biasExpanded
        }

        return reshaped
    }
}

private final class GLM47SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: GLM47SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: GLM47SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: GLM47SwitchLinear

    init(inputDims: Int, hiddenDims: Int, numExperts: Int, groupSize: Int, bits: Int) {
        self._gateProj.wrappedValue = GLM47SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        self._upProj.wrappedValue = GLM47SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        self._downProj.wrappedValue = GLM47SwitchLinear(
            inputDims: hiddenDims,
            outputDims: inputDims,
            numExperts: numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let up = upProj(x, indices: indices)
        let gate = gateProj(x, indices: indices)
        let activated = swiglu(gate, up)
        return downProj(activated, indices: indices)
    }
}

private final class GLM47MoEGate: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    init(config: GLM47FlashConfig) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        let nExperts = config.nRoutedExperts ?? 0
        self._weight.wrappedValue = MLXArray.zeros([nExperts, config.hiddenSize])
        self._correctionBias.wrappedValue = MLXArray.zeros([nExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let gates = MLX.matmul(x, weight.transposed())
        var scores = MLX.sigmoid(gates.asType(.float32))
        let origScores = scores

        scores = scores + correctionBias

        if nGroup > 1 {
            // n_group > 1 is unused for GLM-4.7 Flash; skip grouping.
        }

        let kth = topK - 1
        let inds = argPartition(-scores, kth: kth, axis: -1)[.ellipsis, 0..<topK]
        var topScores = takeAlong(origScores, inds, axis: -1)
        if topK > 1 && normTopkProb {
            let denom = topScores.sum(axis: -1, keepDims: true)
            topScores = topScores / denom
        }
        topScores = topScores * routedScalingFactor
        return (inds, topScores)
    }
}

private final class GLM47FlashAttention: Module {
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    @ModuleInfo(key: "o_proj") var oProj: Linear

    private let numHeads: Int
    private let qkRopeHeadDim: Int
    private let qkNopeHeadDim: Int
    private let vHeadDim: Int
    private let qHeadDim: Int
    private let scale: Float
    private let rope: RoPE
    private let kvLoraRank: Int

    init(config: GLM47FlashConfig) {
        self.numHeads = config.numAttentionHeads
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.vHeadDim = config.vHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = 1.0 / sqrt(Float(self.qHeadDim))
        self.kvLoraRank = config.kvLoraRank

        self._qAProj.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._qBProj.wrappedValue = Linear(config.qLoraRank, config.numAttentionHeads * self.qHeadDim, bias: false)

        self._kvAProj.wrappedValue = Linear(
            config.hiddenSize,
            config.kvLoraRank + config.qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        let kvOutDim = config.numAttentionHeads * (self.qHeadDim - config.qkRopeHeadDim + config.vHeadDim)
        self._kvBProj.wrappedValue = Linear(config.kvLoraRank, kvOutDim, bias: false)

        self._oProj.wrappedValue = Linear(config.numAttentionHeads * config.vHeadDim, config.hiddenSize, bias: config.attentionBias)
        self.rope = RoPE(dimensions: config.qkRopeHeadDim, traditional: config.ropeTraditional ?? true, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let q = qBProj(qALayerNorm(qAProj(x)))
        var qStates = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        var qNope = qStates[.ellipsis, 0..<qkNopeHeadDim]
        var qPe = qStates[.ellipsis, qkNopeHeadDim...]

        let compressedKV = kvAProj(x)
        var kvNope = compressedKV[.ellipsis, 0..<kvLoraRank]
        let kPeRaw = compressedKV[.ellipsis, kvLoraRank...]
        kvNope = kvALayerNorm(kvNope)
        var kv = kvBProj(kvNope).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        var kNope = kv[.ellipsis, 0..<qkNopeHeadDim]
        var values = kv[.ellipsis, qkNopeHeadDim...]

        var kPe = kPeRaw.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)
        kPe = repeatAlongHeads(kPe, heads: numHeads)

        var keys = MLX.concatenated([kNope, kPe], axis: -1)

        if let cache {
            let cached = cache.update(keys: keys, values: values)
            keys = cached.0
            values = cached.1
        }

        let queries = MLX.concatenated([qNope, qPe], axis: -1)

        let qF32 = queries.asType(.float32)
        let kF32 = keys.asType(.float32)
        let vF32 = values.asType(.float32)
        var out = MLXFast.scaledDotProductAttention(
            queries: qF32,
            keys: kF32,
            values: vF32,
            scale: scale,
            mask: mask
        )
        out = out.asType(queries.dtype)
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(out)
    }
}

private protocol GLM47MLP: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

private final class GLM47FlashDenseMLP: Module, GLM47MLP {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: GLM47FlashConfig, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let inDim = hiddenSize ?? config.hiddenSize
        let hiddenDim = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(inDim, hiddenDim, bias: false)
        self._upProj.wrappedValue = Linear(inDim, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, inDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(swiglu(gateProj(x), upProj(x)))
    }
}

private final class GLM47FlashMoE: Module, GLM47MLP {
    @ModuleInfo(key: "switch_mlp") var switchMLP: GLM47SwitchGLU
    @ModuleInfo(key: "gate") var gate: GLM47MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM47FlashDenseMLP?

    init(config: GLM47FlashConfig) {
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8
        let experts = config.nRoutedExperts ?? 0
        self._switchMLP.wrappedValue = GLM47SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: experts,
            groupSize: groupSize,
            bits: bits
        )
        self._gate.wrappedValue = GLM47MoEGate(config: config)
        if let shared = config.nSharedExperts, shared > 0 {
            self._sharedExperts.wrappedValue = GLM47FlashDenseMLP(
                config: config,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * shared
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, indices: inds)
        y = (y * scores.expandedDimensions(axis: scores.ndim)).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

fileprivate final class GLM47FlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: GLM47FlashAttention
    let mlp: GLM47MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: GLM47FlashConfig, layerIndex: Int) {
        self._selfAttention.wrappedValue = GLM47FlashAttention(config: config)
        if let routed = config.nRoutedExperts,
           layerIndex >= config.firstKDenseReplace,
           layerIndex % 1 == 0,
           routed > 0 {
            self.mlp = GLM47FlashMoE(config: config)
        } else {
            self.mlp = GLM47FlashDenseMLP(config: config)
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}

public final class GLM47FlashModel: Module {
    @ModuleInfo(key: "model") var model: GLM47FlashTransformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(config: GLM47FlashConfig) {
        self._model.wrappedValue = GLM47FlashTransformer(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputIds, cache: cache)
        return lmHead(hidden)
    }
}

public final class GLM47FlashTransformer: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [GLM47FlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: GLM47FlashConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            GLM47FlashDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        var h = embedTokens(tokenIds)
        let mask = createMask(h: h, cache: cache?.first)

        for (idx, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[idx])
        }
        return norm(h)
    }
}
