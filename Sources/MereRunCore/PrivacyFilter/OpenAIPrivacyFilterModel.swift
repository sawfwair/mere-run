import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN
import MLXRandom

public struct OpenAIPrivacyFilterModelOutput {
    public let lastHiddenState: MLXArray
    public let logits: MLXArray
}

final class OpenAIPrivacyFilterYarnRoPE: Module {
    private let dims: Int
    private let traditional: Bool
    private let mscale: Float
    private let freqs: MLXArray

    init(config: OpenAIPrivacyFilterConfig) {
        let params = config.ropeParameters
        let dims = config.headDim
        self.dims = dims
        self.traditional = true

        func correctionDim(numRotations: Float) -> Float {
            let numerator = Float(dims) * log(Float(params.originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
            let denominator = 2 * log(params.ropeTheta)
            return numerator / denominator
        }

        let low = max(Int(floor(correctionDim(numRotations: params.betaFast))), 0)
        let high = min(Int(ceil(correctionDim(numRotations: params.betaSlow))), dims - 1)
        let rampDenominator: Float = low == high ? 0.001 : Float(high - low)

        let half = max(1, dims / 2)
        let halfIndices = MLXArray((0..<half).map { Float($0) })
        let evenIndices = MLXArray(Array(stride(from: 0, to: dims, by: 2)).map { Float($0) })
        let freqExtra = MLX.pow(MLXArray(params.ropeTheta), evenIndices / Float(dims))
        let freqInter = MLXArray(params.factor) * freqExtra
        let ramp = MLX.clip((halfIndices - Float(low)) / rampDenominator, min: 0.0, max: 1.0)
        let freqMask = MLXArray(1.0) - ramp
        self.freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (MLXArray(1.0) - freqMask))

        if params.factor <= 1.0 {
            self.mscale = 1.0
        } else {
            self.mscale = 0.1 * log(params.factor) + 1.0
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = mscale == 1.0 ? x : x * MLXArray(mscale).asType(x.dtype)
        return MLXFast.RoPE(
            input,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: 0,
            freqs: freqs
        )
    }
}

final class OpenAIPrivacyFilterAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "sinks") var sinks: MLXArray

    private let headDim: Int
    private let numAttentionHeads: Int
    private let numKeyValueHeads: Int
    private let scale: Float
    private let rope: OpenAIPrivacyFilterYarnRoPE

    init(config: OpenAIPrivacyFilterConfig) {
        self.headDim = config.headDim
        self.numAttentionHeads = config.numAttentionHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.scale = 1.0 / sqrt(Float(max(1, config.headDim)))
        self.rope = OpenAIPrivacyFilterYarnRoPE(config: config)

        self._qProj.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: config.attentionBias)
        self._sinks.wrappedValue = MLX.zeros([config.numAttentionHeads], dtype: .float32)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)

        var q = qProj(x).reshaped(batch, sequenceLength, numAttentionHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, sequenceLength, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, sequenceLength, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        q = rope(q)
        k = rope(k)

        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask.map { .array($0.asType(q.dtype)) } ?? .none,
            sinks: sinks.asType(q.dtype)
        )
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, sequenceLength, numAttentionHeads * headDim))
    }
}

@inline(__always)
private func openAIPrivacyFilterSwiGLU(up: MLXArray, gate: MLXArray) -> MLXArray {
    let clippedGate = MLX.clip(gate, max: 7.0)
    let clippedUp = MLX.clip(up, min: -7.0, max: 7.0)
    let glu = clippedGate * MLX.sigmoid(clippedGate * MLXArray(1.702).asType(gate.dtype))
    return (clippedUp + MLXArray(1.0).asType(up.dtype)) * glu
}

final class OpenAIPrivacyFilterSwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    private let groupSize: Int
    private let bits: Int

    init(inputDims: Int, outputDims: Int, numExperts: Int, groupSize: Int = 64, bits: Int = 4, bias: Bool = true) {
        self.groupSize = groupSize
        self.bits = bits
        let scale = sqrt(1.0 / Float(max(1, inputDims)))
        self._weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [numExperts, outputDims, inputDims])
        self._scales.wrappedValue = nil
        self._biases.wrappedValue = nil
        self._bias.wrappedValue = bias ? MLX.zeros([numExperts, outputDims]) : nil
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

final class OpenAIPrivacyFilterSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: OpenAIPrivacyFilterSwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: OpenAIPrivacyFilterSwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: OpenAIPrivacyFilterSwitchLinear

    init(config: OpenAIPrivacyFilterConfig) {
        self._gateProj.wrappedValue = OpenAIPrivacyFilterSwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.intermediateSize,
            numExperts: config.numLocalExperts,
            bias: true
        )
        self._upProj.wrappedValue = OpenAIPrivacyFilterSwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.intermediateSize,
            numExperts: config.numLocalExperts,
            bias: true
        )
        self._downProj.wrappedValue = OpenAIPrivacyFilterSwitchLinear(
            inputDims: config.intermediateSize,
            outputDims: config.hiddenSize,
            numExperts: config.numLocalExperts,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let up = upProj(x, indices: indices)
        let gate = gateProj(x, indices: indices)
        let activated = openAIPrivacyFilterSwiGLU(up: up, gate: gate)
        return downProj(activated, indices: indices)
    }
}

final class OpenAIPrivacyFilterMLP: Module {
    @ModuleInfo(key: "experts") var experts: OpenAIPrivacyFilterSwitchGLU
    @ModuleInfo(key: "router") var router: Linear

    private let numExpertsPerToken: Int

    init(config: OpenAIPrivacyFilterConfig) {
        self.numExpertsPerToken = config.numExpertsPerTok
        self._experts.wrappedValue = OpenAIPrivacyFilterSwitchGLU(config: config)
        self._router.wrappedValue = Linear(config.hiddenSize, config.numLocalExperts, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routerLogits = router(x).asType(.float32)
        let k = max(1, numExpertsPerToken)
        let indices = argPartition(-routerLogits, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        let topValues = takeAlong(routerLogits, indices, axis: -1)
        let weights = softmax(topValues, axis: -1).asType(x.dtype)
        let expertOutput = experts(x, indices: indices)
        let weighted = expertOutput * MLX.expandedDimensions(weights, axis: weights.ndim)
        return weighted.sum(axis: -2)
    }
}

final class OpenAIPrivacyFilterEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: OpenAIPrivacyFilterAttention
    @ModuleInfo(key: "mlp") var mlp: OpenAIPrivacyFilterMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: OpenAIPrivacyFilterConfig) {
        self._selfAttention.wrappedValue = OpenAIPrivacyFilterAttention(config: config)
        self._mlp.wrappedValue = OpenAIPrivacyFilterMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = inputLayerNorm(x)
        hidden = selfAttention(hidden, mask: mask)
        let attended = x + hidden
        hidden = postAttentionLayerNorm(attended)
        return attended + mlp(hidden)
    }
}

final class OpenAIPrivacyFilterBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [OpenAIPrivacyFilterEncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private let slidingWindow: Int

    init(config: OpenAIPrivacyFilterConfig) {
        self.slidingWindow = config.slidingWindow
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            OpenAIPrivacyFilterEncoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var hidden = embedTokens(inputIds)
        let mask = Self.bidirectionalSlidingWindowMask(
            attentionMask: attentionMask,
            seqLen: hidden.dim(1),
            window: slidingWindow,
            dtype: hidden.dtype
        )

        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return norm(hidden)
    }

    static func bidirectionalSlidingWindowMask(
        attentionMask: MLXArray?,
        seqLen: Int,
        window: Int,
        dtype: DType
    ) -> MLXArray {
        let idx = MLXArray((0..<seqLen).map { Int32($0) }).asType(.int32)
        let i = idx.reshaped(seqLen, 1)
        let j = idx.reshaped(1, seqLen)
        let dist = MLX.abs(i - j)
        let keepGeom = (dist .<= MLXArray(Int32(window))).asType(.bool)
        var keep = broadcast(keepGeom.reshaped(1, 1, seqLen, seqLen), to: [1, 1, seqLen, seqLen])

        if let attentionMask {
            let batch = attentionMask.dim(0)
            let keyKeep = (attentionMask .> MLXArray(Float(0))).asType(.bool).reshaped(batch, 1, 1, seqLen)
            keep = logicalAnd(
                broadcast(keep, to: [batch, 1, seqLen, seqLen]),
                broadcast(keyKeep, to: [batch, 1, seqLen, seqLen])
            )
        }

        let batch = keep.dim(0)
        let zeros = MLX.zeros([batch, 1, seqLen, seqLen], dtype: dtype)
        let negative = MLXArray(-1.0e9).asType(dtype)
        return MLX.where(keep, zeros, zeros + negative)
    }
}

public final class OpenAIPrivacyFilterModel: Module {
    @ModuleInfo(key: "model") var model: OpenAIPrivacyFilterBackbone
    @ModuleInfo(key: "score") var score: Linear

    public let config: OpenAIPrivacyFilterConfig

    public init(config: OpenAIPrivacyFilterConfig) {
        self.config = config
        self._model.wrappedValue = OpenAIPrivacyFilterBackbone(config: config)
        self._score.wrappedValue = Linear(config.hiddenSize, config.numLabels, bias: true)
        super.init()
    }

    public func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray?) -> OpenAIPrivacyFilterModelOutput {
        precondition(inputIds.ndim == 2, "inputIds must be [batch, sequence]")
        let lastHiddenState = model(inputIds: inputIds, attentionMask: attentionMask)
        let logits = score(lastHiddenState)
        return OpenAIPrivacyFilterModelOutput(lastHiddenState: lastHiddenState, logits: logits)
    }

    public static func sanitizeWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasPrefix("original.") {
            return []
        }
        if key.contains("mlp.experts.gate_up_proj_bias") {
            let (gateBias, upBias) = value.split(axis: -1)
            return [
                (key.replacingOccurrences(of: "gate_up_proj_bias", with: "gate_proj.bias"), gateBias.contiguous()),
                (key.replacingOccurrences(of: "gate_up_proj_bias", with: "up_proj.bias"), upBias.contiguous()),
            ]
        }
        if key.contains("mlp.experts.gate_up_proj") {
            let (gate, up) = value.split(axis: -1)
            return [
                (key.replacingOccurrences(of: "gate_up_proj", with: "gate_proj.weight"), gate.swappedAxes(-1, -2).contiguous()),
                (key.replacingOccurrences(of: "gate_up_proj", with: "up_proj.weight"), up.swappedAxes(-1, -2).contiguous()),
            ]
        }
        if key.hasSuffix("mlp.experts.down_proj") {
            return [(key + ".weight", value.swappedAxes(-1, -2).contiguous())]
        }
        if key.hasSuffix("mlp.experts.down_proj_bias") {
            return [(key.replacingOccurrences(of: "down_proj_bias", with: "down_proj.bias"), value)]
        }
        return [(key, value)]
    }
}
