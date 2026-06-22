import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

struct WooshRoPECache {
    let cos: MLXArray
    let sin: MLXArray
}

enum WooshRoPE {
    static func precompute(config: WooshDiTConfig, maxSeqLen: Int? = nil, audioFPSMultiplier: Float? = nil) -> WooshRoPECache {
        let dim = max(config.qkRopeHeadDim, config.qkvHeadDim)
        var seqLen = Int(ceil(Float(maxSeqLen ?? config.maxSeqLen) / Float(config.patchSize)))
        if let multiplier = config.ropeLenMultiplier {
            seqLen *= multiplier
        }
        let originalSeqLen = Int(ceil(Float(config.originalSeqLen) / Float(config.patchSize)))

        var freqs: [Float] = []
        freqs.reserveCapacity(dim / 2)
        for index in stride(from: 0, to: dim, by: 2) {
            freqs.append(1.0 / pow(config.ropeTheta, Float(index) / Float(dim)))
        }

        if seqLen > originalSeqLen {
            let low = max(0, min(dim - 1, Int(floor(correctionDim(rotations: Float(config.betaFast), dim: dim, base: config.ropeTheta, maxSeqLen: originalSeqLen)))))
            let high = max(0, min(dim - 1, Int(ceil(correctionDim(rotations: Float(config.betaSlow), dim: dim, base: config.ropeTheta, maxSeqLen: originalSeqLen)))))
            for index in 0..<freqs.count {
                var ramp = (Float(index) - Float(low)) / Float(max(1, high - low))
                ramp = min(1, max(0, ramp))
                let smooth = 1 - ramp
                freqs[index] = (freqs[index] / config.ropeFactor) * (1 - smooth) + freqs[index] * smooth
            }
        }

        var cosValues = [Float]()
        var sinValues = [Float]()
        cosValues.reserveCapacity(seqLen * freqs.count)
        sinValues.reserveCapacity(seqLen * freqs.count)
        for pos in 0..<seqLen {
            let position = Float(pos) * (audioFPSMultiplier ?? 1)
            for freq in freqs {
                let angle = position * freq
                cosValues.append(cos(angle))
                sinValues.append(sin(angle))
            }
        }
        return WooshRoPECache(
            cos: MLXArray(cosValues).reshaped(seqLen, freqs.count),
            sin: MLXArray(sinValues).reshaped(seqLen, freqs.count)
        )
    }

    static func constantDescription(config: WooshDiTConfig) -> WooshRoPECache {
        let half = max(config.qkRopeHeadDim, config.qkvHeadDim) / 2
        return WooshRoPECache(
            cos: MLXArray.ones([config.maxDescriptionLength + config.nMemoryTokensDescription, half]),
            sin: MLXArray.zeros([config.maxDescriptionLength + config.nMemoryTokensDescription, half])
        )
    }

    static func apply(_ x: MLXArray, cache: WooshRoPECache) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let heads = x.dim(2)
        let dim = x.dim(3)
        let half = dim / 2
        let paired = x.reshaped(batch, sequence, heads, half, 2)
        let real = paired[0..., 0..., 0..., 0..., 0]
        let imaginary = paired[0..., 0..., 0..., 0..., 1]
        let cosSlice = cache.cos[0..<sequence, 0..<half].expandedDimensions(axes: [0, 2])
        let sinSlice = cache.sin[0..<sequence, 0..<half].expandedDimensions(axes: [0, 2])
        let outReal = real * cosSlice - imaginary * sinSlice
        let outImaginary = real * sinSlice + imaginary * cosSlice
        return MLX.stacked([outReal, outImaginary], axis: -1).reshaped(x.shape)
    }

    private static func correctionDim(rotations: Float, dim: Int, base: Float, maxSeqLen: Int) -> Float {
        Float(dim) * log(Float(maxSeqLen) / (rotations * 2 * Float.pi)) / (2 * log(base))
    }
}

final class WooshFourierFeaturesTime: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inFeatures: Int = 1, outFeatures: Int) {
        self._weight.wrappedValue = MLXRandom.normal([outFeatures / 2, inFeatures])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let f = MLXArray(2 * Float.pi).asType(input.dtype) * matmul(input, weight.transposed())
        return MLX.concatenated([MLX.cos(f), MLX.sin(f)], axis: -1)
    }
}

final class WooshFixedFourierFeaturesTime: Module {
    @ParameterInfo(key: "freqs") var freqs: MLXArray
    private let timeFactor: Float

    init(outFeatures: Int, maxPeriod: Float = 10000, timeFactor: Float = 1000) {
        self.timeFactor = timeFactor
        let half = outFeatures / 2
        let values = (0..<half).map { index in
            exp(-log(maxPeriod) * Float(index) / Float(half))
        }
        self._freqs.wrappedValue = MLXArray(values)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let args = input.asType(.float32) * MLXArray(timeFactor) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1).asType(input.dtype)
    }
}

final class WooshLinearSiLU2: Module {
    @ModuleInfo(key: "first") var first: Linear
    @ModuleInfo(key: "second") var second: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        self._first.wrappedValue = Linear(inputDim, hiddenDim, bias: true)
        self._second.wrappedValue = Linear(hiddenDim, outputDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(second(silu(first(x))))
    }
}

final class WooshLinearSiLUToLinear: Module {
    @ModuleInfo(key: "first") var first: Linear
    @ModuleInfo(key: "second") var second: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        self._first.wrappedValue = Linear(inputDim, hiddenDim, bias: true)
        self._second.wrappedValue = Linear(hiddenDim, outputDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        second(silu(first(x)))
    }
}

struct WooshDiTState {
    var x: MLXArray
    var t: MLXArray
    var mPlus: MLXArray
    var description: MLXArray
    var descriptionMask: MLXArray
    var videoFeatures: MLXArray
    var rope: WooshRoPECache
    var descriptionRoPE: WooshRoPECache
    var videoRoPE: WooshRoPECache
    var logvar: MLXArray?
}

public struct WooshCondition {
    public let crossAttention: MLXArray
    public let crossAttentionMask: MLXArray
    public let cfg: MLXArray?
    public let videoFeatures: MLXArray?

    public init(
        crossAttention: MLXArray,
        crossAttentionMask: MLXArray,
        cfg: MLXArray? = nil,
        videoFeatures: MLXArray? = nil
    ) {
        self.crossAttention = crossAttention
        self.crossAttentionMask = crossAttentionMask
        self.cfg = cfg
        self.videoFeatures = videoFeatures
    }
}

final class WooshInputProcessing: Module {
    @ModuleInfo(key: "timestep_features") var timestepFeatures: WooshFourierFeaturesTime
    @ModuleInfo(key: "to_timestep_embed") var toTimestepEmbed: WooshLinearSiLUToLinear
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "to_cond_embed") var toCondEmbed: WooshLinearSiLUToLinear
    @ParameterInfo(key: "memory_tokens_rope") var memoryTokensRope: MLXArray
    @ParameterInfo(key: "memory_tokens_description") var memoryTokensDescription: MLXArray
    @ModuleInfo(key: "timestep_logvar") var timestepLogvar: WooshFourierFeaturesTime
    @ModuleInfo(key: "to_logvar") var toLogvar: WooshLinearSiLUToLinear
    @ParameterInfo(key: "description_pad") var descriptionPad: MLXArray

    private let config: WooshDiTConfig
    private let rope: WooshRoPECache
    private let descriptionRoPE: WooshRoPECache

    init(config: WooshDiTConfig) {
        self.config = config
        self.rope = WooshRoPE.precompute(config: config)
        self.descriptionRoPE = WooshRoPE.constantDescription(config: config)
        self._timestepFeatures.wrappedValue = WooshFourierFeaturesTime(outFeatures: config.timestepFeaturesDim)
        self._toTimestepEmbed.wrappedValue = WooshLinearSiLUToLinear(
            inputDim: config.timestepFeaturesDim,
            hiddenDim: config.interDim,
            outputDim: config.dim
        )
        self._projectIn.wrappedValue = Linear(config.ioChannels * config.patchSize, config.dim, bias: true)
        self._toCondEmbed.wrappedValue = WooshLinearSiLUToLinear(
            inputDim: config.condTokenDim,
            hiddenDim: config.interDim,
            outputDim: config.dim
        )
        self._memoryTokensRope.wrappedValue = MLXRandom.normal([1, config.nMemoryTokensRope, config.dim])
        self._memoryTokensDescription.wrappedValue = MLXRandom.normal([1, config.nMemoryTokensDescription, config.dim])
        self._timestepLogvar.wrappedValue = WooshFourierFeaturesTime(outFeatures: config.timestepFeaturesDim)
        self._toLogvar.wrappedValue = WooshLinearSiLUToLinear(
            inputDim: config.timestepFeaturesDim,
            hiddenDim: 128,
            outputDim: 1
        )
        self._descriptionPad.wrappedValue = MLXRandom.normal([config.maxDescriptionLength, config.condTokenDim])
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray, cond: WooshCondition, mask: MLXArray?) -> WooshDiTState {
        let batch = x.dim(0)
        let mPlus = toTimestepEmbed(timestepFeatures(t.expandedDimensions(axis: 1)))
        let tEmbed = silu(mPlus)
        let xEmbedded = embedX(x)
        let memory = MLX.broadcast(memoryTokensRope, to: [batch, config.nMemoryTokensRope, config.dim])
        let paddedDescription = padDescription(cond.crossAttention, mask: cond.crossAttentionMask)
        let description = toCondEmbed(paddedDescription)
        let descriptionMemory = config.nMemoryTokensDescription > 0
            ? MLX.broadcast(memoryTokensDescription, to: [batch, config.nMemoryTokensDescription, config.dim])
            : MLXArray.zeros([batch, 0, config.dim], dtype: x.dtype)
        let descriptionFull = config.nMemoryTokensDescription > 0
            ? MLX.concatenated([descriptionMemory, description], axis: 1)
            : description
        let descriptionMask = MLXArray.ones([batch, descriptionFull.dim(1)], dtype: cond.crossAttentionMask.dtype)

        return WooshDiTState(
            x: MLX.concatenated([memory, xEmbedded], axis: 1),
            t: tEmbed,
            mPlus: mPlus,
            description: descriptionFull,
            descriptionMask: descriptionMask,
            videoFeatures: MLXArray.zeros([batch, 0, config.dim], dtype: x.dtype),
            rope: rope,
            descriptionRoPE: descriptionRoPE,
            videoRoPE: rope,
            logvar: config.estimateLogvar
                ? toLogvar(timestepLogvar(t.expandedDimensions(axis: 1)))[0..., 0]
                : nil
        )
    }

    private func embedX(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let channels = x.dim(1)
        let time = x.dim(2)
        let transposed = x.transposed(0, 2, 1)
        if config.patchSize == 1 {
            return projectIn(transposed)
        }
        let patched = transposed.reshaped(batch, time / config.patchSize, channels * config.patchSize)
        return projectIn(patched)
    }

    private func padDescription(_ description: MLXArray, mask: MLXArray) -> MLXArray {
        guard config.noDescriptionMask else {
            return description
        }
        let batch = description.dim(0)
        let pad = MLX.broadcast(descriptionPad.expandedDimensions(axis: 0), to: [batch, config.maxDescriptionLength, config.condTokenDim])
        return MLX.where(mask.asType(.bool).expandedDimensions(axis: -1), description, pad)
    }
}

final class WooshFlowMapPreprocessing: Module {
    @ModuleInfo(key: "old_preprocessing") var oldPreprocessing: WooshInputProcessing
    @ModuleInfo(key: "timestep_features_t") var timestepFeaturesT: WooshFixedFourierFeaturesTime
    @ModuleInfo(key: "timestep_features_r") var timestepFeaturesR: WooshFixedFourierFeaturesTime
    @ModuleInfo(key: "cfg_features") var cfgFeatures: WooshFixedFourierFeaturesTime
    @ModuleInfo(key: "to_timestep_embed") var toTimestepEmbed: WooshLinearSiLU2
    @ModuleInfo(key: "timestep_logvar") var timestepLogvar: WooshFourierFeaturesTime
    @ModuleInfo(key: "to_logvar") var toLogvar: WooshLinearSiLUToLinear

    private let config: WooshDiTConfig

    init(config: WooshDiTConfig) {
        self.config = config
        self._oldPreprocessing.wrappedValue = WooshInputProcessing(config: config)
        self._timestepFeaturesT.wrappedValue = WooshFixedFourierFeaturesTime(outFeatures: config.timestepFeaturesDim, timeFactor: 1)
        self._timestepFeaturesR.wrappedValue = WooshFixedFourierFeaturesTime(outFeatures: config.timestepFeaturesDim, timeFactor: 1)
        self._cfgFeatures.wrappedValue = WooshFixedFourierFeaturesTime(outFeatures: config.timestepFeaturesDim, timeFactor: 1)
        self._toTimestepEmbed.wrappedValue = WooshLinearSiLU2(
            inputDim: config.timestepFeaturesDim * 3,
            hiddenDim: config.interDim,
            outputDim: config.dim
        )
        self._timestepLogvar.wrappedValue = WooshFourierFeaturesTime(outFeatures: config.timestepFeaturesDim)
        self._toLogvar.wrappedValue = WooshLinearSiLUToLinear(
            inputDim: config.timestepFeaturesDim * 2,
            hiddenDim: 128,
            outputDim: 1
        )
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray, r: MLXArray, cond: WooshCondition, mask: MLXArray?) -> WooshDiTState {
        var state = oldPreprocessing(x, t: t, cond: cond, mask: mask)
        let cfg = cond.cfg ?? MLXArray.ones([x.dim(0)], dtype: x.dtype)
        let combined = MLX.concatenated([
            timestepFeaturesT(t.expandedDimensions(axis: 1)),
            timestepFeaturesR(r.expandedDimensions(axis: 1)),
            cfgFeatures(cfg.expandedDimensions(axis: 1)),
        ], axis: -1)
        state.t = toTimestepEmbed(combined)
        let logvarIn = MLX.concatenated([
            timestepLogvar(t.expandedDimensions(axis: 1)),
            timestepLogvar(r.expandedDimensions(axis: 1)),
        ], axis: -1)
        state.logvar = toLogvar(logvarIn)[0..., 0]
        return state
    }
}

final class WooshModalityAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "norm_q") var normQ: WooshRMSNorm
    @ModuleInfo(key: "norm_k") var normK: WooshRMSNorm
    @ModuleInfo(key: "mod_proj") var modProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    private let config: WooshDiTConfig
    private let scale: Float
    private let keyPath: KeyPath<WooshDiTState, MLXArray>
    private let ropePath: KeyPath<WooshDiTState, WooshRoPECache>

    init(config: WooshDiTConfig, keyPath: KeyPath<WooshDiTState, MLXArray>, ropePath: KeyPath<WooshDiTState, WooshRoPECache>) {
        self.config = config
        self.scale = 1.0 / sqrt(Float(config.headDim))
        self.keyPath = keyPath
        self.ropePath = ropePath
        self._qkv.wrappedValue = Linear(config.dim, config.dim * 3, bias: true)
        self._normQ.wrappedValue = WooshRMSNorm(dimensions: config.headDim)
        self._normK.wrappedValue = WooshRMSNorm(dimensions: config.headDim)
        self._modProj.wrappedValue = Linear(config.dim, config.dim * 3, bias: true)
        self._outProj.wrappedValue = Linear(config.dim, config.dim, bias: true)
    }

    struct Precomputed {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let gate: MLXArray
        let sequenceLength: Int
    }

    func precompute(_ state: WooshDiTState) -> Precomputed {
        let x = state[keyPath: keyPath]
        let batch = x.dim(0)
        let sequence = x.dim(1)
        var normed = WooshTensorOps.layerNormNoAffine(x)
        let modulation = MLX.split(modProj(state.t).expandedDimensions(axis: 1), parts: 3, axis: -1)
        normed = (MLXArray(1).asType(x.dtype) + modulation[1]) * normed + modulation[0]
        let qkvValue = qkv(normed)
        let parts = MLX.split(qkvValue, parts: 3, axis: -1)
        var q = parts[0].reshaped(batch, sequence, config.nHeads, config.headDim)
        var k = parts[1].reshaped(batch, sequence, config.nHeads, config.headDim)
        let v = parts[2].reshaped(batch, sequence, config.nHeads, config.headDim).transposed(0, 2, 1, 3)
        q = normQ(q)
        k = normK(k)
        let qRope = WooshRoPE.apply(q[0..., 0..., 0..., 0..<config.qkRopeHeadDim], cache: state[keyPath: ropePath])
        let kRope = WooshRoPE.apply(k[0..., 0..., 0..., 0..<config.qkRopeHeadDim], cache: state[keyPath: ropePath])
        let qFull = MLX.concatenated([qRope, q[0..., 0..., 0..., config.qkRopeHeadDim..<config.headDim]], axis: -1)
        let kFull = MLX.concatenated([kRope, k[0..., 0..., 0..., config.qkRopeHeadDim..<config.headDim]], axis: -1)
        return Precomputed(
            q: qFull.transposed(0, 2, 1, 3),
            k: kFull.transposed(0, 2, 1, 3),
            v: v,
            gate: modulation[2],
            sequenceLength: sequence
        )
    }

    func finish(attended: MLXArray, residual: MLXArray, gate: MLXArray) -> MLXArray {
        let batch = residual.dim(0)
        let sequence = residual.dim(1)
        let projected = outProj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, config.dim))
        return residual + projected * gate
    }

    func callAsFunction(_ state: inout WooshDiTState, target: WritableKeyPath<WooshDiTState, MLXArray>) {
        let residual = state[keyPath: keyPath]
        let pre = precompute(state)
        let attended = MLXFast.scaledDotProductAttention(
            queries: pre.q,
            keys: pre.k,
            values: pre.v,
            scale: scale,
            mask: .none
        )
        state[keyPath: target] = finish(attended: attended, residual: residual, gate: pre.gate)
    }
}

final class WooshRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        WooshTensorOps.rmsNorm(x, weight: weight, eps: eps)
    }
}

final class WooshModalityMLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "mod_proj") var modProj: Linear

    private let config: WooshDiTConfig

    init(config: WooshDiTConfig) {
        self.config = config
        self._w1.wrappedValue = Linear(config.dim, config.interDim, bias: true)
        self._w2.wrappedValue = Linear(config.interDim, config.dim, bias: true)
        self._modProj.wrappedValue = Linear(config.dim, config.dim * 3, bias: true)
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray) -> MLXArray {
        var hidden = WooshTensorOps.layerNormNoAffine(x)
        let modulation = MLX.split(modProj(t).expandedDimensions(axis: 1), parts: 3, axis: -1)
        hidden = (MLXArray(1).asType(x.dtype) + modulation[1]) * hidden + modulation[0]
        hidden = w2(WooshTensorOps.geluTanh(w1(hidden)))
        return x + hidden * modulation[2]
    }
}

final class WooshMMMModalities: Module {
    @ModuleInfo(key: "x") var x: WooshModalityAttention
    @ModuleInfo(key: "description") var description: WooshModalityAttention

    init(config: WooshDiTConfig) {
        self._x.wrappedValue = WooshModalityAttention(config: config, keyPath: \.x, ropePath: \.rope)
        self._description.wrappedValue = WooshModalityAttention(
            config: config,
            keyPath: \.description,
            ropePath: \.descriptionRoPE
        )
    }
}

final class WooshMMMFFNs: Module {
    @ModuleInfo(key: "x") var x: WooshModalityMLP
    @ModuleInfo(key: "description") var description: WooshModalityMLP

    init(config: WooshDiTConfig) {
        self._x.wrappedValue = WooshModalityMLP(config: config)
        self._description.wrappedValue = WooshModalityMLP(config: config)
    }
}

final class WooshMMMAttention: Module {
    @ModuleInfo(key: "modalities") var modalities: WooshMMMModalities
    private let config: WooshDiTConfig
    private let scale: Float

    init(config: WooshDiTConfig) {
        self.config = config
        self.scale = 1.0 / sqrt(Float(config.headDim))
        self._modalities.wrappedValue = WooshMMMModalities(config: config)
    }

    func callAsFunction(_ state: inout WooshDiTState) {
        let xResidual = state.x
        let descriptionResidual = state.description
        let xPre = modalities.x.precompute(state)
        let descriptionPre = modalities.description.precompute(state)
        let q = MLX.concatenated([xPre.q, descriptionPre.q], axis: 2)
        let k = MLX.concatenated([xPre.k, descriptionPre.k], axis: 2)
        let v = MLX.concatenated([xPre.v, descriptionPre.v], axis: 2)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        let xAttended = attended[0..., 0..., 0..<xPre.sequenceLength, 0...]
        let descAttended = attended[0..., 0..., xPre.sequenceLength..<(xPre.sequenceLength + descriptionPre.sequenceLength), 0...]
        state.x = modalities.x.finish(attended: xAttended, residual: xResidual, gate: xPre.gate)
        state.description = modalities.description.finish(
            attended: descAttended,
            residual: descriptionResidual,
            gate: descriptionPre.gate
        )
    }
}

final class WooshMMMBlock: Module {
    @ModuleInfo(key: "attn") var attention: WooshMMMAttention
    @ModuleInfo(key: "ffns") var ffns: WooshMMMFFNs

    init(config: WooshDiTConfig) {
        self._attention.wrappedValue = WooshMMMAttention(config: config)
        self._ffns.wrappedValue = WooshMMMFFNs(config: config)
    }

    func callAsFunction(_ state: inout WooshDiTState) {
        attention(&state)
        state.x = ffns.x(state.x, t: state.t)
        state.description = ffns.description(state.description, t: state.t)
    }
}

final class WooshSingleStreamBlock: Module {
    @ModuleInfo(key: "qkv_mlp") var qkvMLP: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm_q") var normQ: WooshRMSNorm
    @ModuleInfo(key: "norm_k") var normK: WooshRMSNorm
    @ModuleInfo(key: "mod_proj") var modProj: Linear

    private let config: WooshDiTConfig
    private let scale: Float

    init(config: WooshDiTConfig) {
        self.config = config
        self.scale = 1.0 / sqrt(Float(config.headDim))
        self._qkvMLP.wrappedValue = Linear(config.dim, config.dim * 3 + config.interDim, bias: true)
        self._outProj.wrappedValue = Linear(config.dim + config.interDim, config.dim, bias: true)
        self._normQ.wrappedValue = WooshRMSNorm(dimensions: config.headDim)
        self._normK.wrappedValue = WooshRMSNorm(dimensions: config.headDim)
        self._modProj.wrappedValue = Linear(config.dim, config.dim * 3, bias: true)
    }

    func callAsFunction(_ state: inout WooshDiTState) {
        let xLen = state.x.dim(1)
        let descLen = state.description.dim(1)
        let residualX = state.x
        let residualDescription = state.description
        var hidden = MLX.concatenated([state.x, state.description], axis: 1)
        let batch = hidden.dim(0)
        let sequence = hidden.dim(1)
        hidden = WooshTensorOps.layerNormNoAffine(hidden)
        let modulation = MLX.split(modProj(state.t).expandedDimensions(axis: 1), parts: 3, axis: -1)
        hidden = (MLXArray(1).asType(hidden.dtype) + modulation[1]) * hidden + modulation[0]
        let projected = qkvMLP(hidden)
        let qkv = projected[0..., 0..., 0..<(config.dim * 3)]
        let mlp = projected[0..., 0..., (config.dim * 3)..<(config.dim * 3 + config.interDim)]
        let qkvParts = MLX.split(qkv, parts: 3, axis: -1)
        var q = qkvParts[0].reshaped(batch, sequence, config.nHeads, config.headDim)
        var k = qkvParts[1].reshaped(batch, sequence, config.nHeads, config.headDim)
        let v = qkvParts[2].reshaped(batch, sequence, config.nHeads, config.headDim).transposed(0, 2, 1, 3)
        q = normQ(q)
        k = normK(k)

        let cos = MLX.concatenated([state.rope.cos[0..<xLen, 0...], state.descriptionRoPE.cos[0..<descLen, 0...]], axis: 0)
        let sin = MLX.concatenated([state.rope.sin[0..<xLen, 0...], state.descriptionRoPE.sin[0..<descLen, 0...]], axis: 0)
        let cache = WooshRoPECache(cos: cos, sin: sin)
        let qRope = WooshRoPE.apply(q[0..., 0..., 0..., 0..<config.qkRopeHeadDim], cache: cache)
        let kRope = WooshRoPE.apply(k[0..., 0..., 0..., 0..<config.qkRopeHeadDim], cache: cache)
        q = MLX.concatenated([qRope, q[0..., 0..., 0..., config.qkRopeHeadDim..<config.headDim]], axis: -1)
        k = MLX.concatenated([kRope, k[0..., 0..., 0..., config.qkRopeHeadDim..<config.headDim]], axis: -1)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        let z = attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, config.dim)
        let out = outProj(MLX.concatenated([z, WooshTensorOps.geluTanh(mlp)], axis: -1)) * modulation[2]
        state.x = residualX + out[0..., 0..<xLen, 0...]
        state.description = residualDescription + out[0..., xLen..<(xLen + descLen), 0...]
    }
}

final class WooshDiTLayer: Module {
    @ModuleInfo(key: "attn") var attention: WooshMMMAttention?
    @ModuleInfo(key: "ffns") var ffns: WooshMMMFFNs?
    @ModuleInfo(key: "qkv_mlp") var qkvMLP: Linear?
    @ModuleInfo(key: "out_proj") var outProj: Linear?
    @ModuleInfo(key: "norm_q") var normQ: WooshRMSNorm?
    @ModuleInfo(key: "norm_k") var normK: WooshRMSNorm?
    @ModuleInfo(key: "mod_proj") var modProj: Linear?

    private let singleStream: WooshSingleStreamBlock?
    private let isMultimodal: Bool

    init(config: WooshDiTConfig, index: Int) {
        if index < config.nMultimodalLayers {
            let block = WooshMMMBlock(config: config)
            self._attention.wrappedValue = block.attention
            self._ffns.wrappedValue = block.ffns
            self._qkvMLP.wrappedValue = nil
            self._outProj.wrappedValue = nil
            self._normQ.wrappedValue = nil
            self._normK.wrappedValue = nil
            self._modProj.wrappedValue = nil
            self.singleStream = nil
            self.isMultimodal = true
        } else {
            let block = WooshSingleStreamBlock(config: config)
            self._attention.wrappedValue = nil
            self._ffns.wrappedValue = nil
            self._qkvMLP.wrappedValue = block.qkvMLP
            self._outProj.wrappedValue = block.outProj
            self._normQ.wrappedValue = block.normQ
            self._normK.wrappedValue = block.normK
            self._modProj.wrappedValue = block.modProj
            self.singleStream = block
            self.isMultimodal = false
        }
    }

    func callAsFunction(_ state: inout WooshDiTState) {
        if isMultimodal {
            if let attention {
                attention(&state)
            }
            if let ffns {
                state.x = ffns.x(state.x, t: state.t)
                state.description = ffns.description(state.description, t: state.t)
            }
        } else if let singleStream {
            singleStream(&state)
        }
    }
}

final class WooshPostProcessing: Module {
    @ModuleInfo(key: "mod_proj") var modProj: Linear
    @ModuleInfo(key: "linear") var linear: Linear
    private let config: WooshDiTConfig

    init(config: WooshDiTConfig) {
        self.config = config
        self._modProj.wrappedValue = Linear(config.dim, config.dim * 2, bias: true)
        self._linear.wrappedValue = Linear(config.dim, config.ioChannels * config.patchSize, bias: true)
    }

    func callAsFunction(_ state: WooshDiTState) -> MLXArray {
        var x = state.x[0..., config.nMemoryTokensRope..<state.x.dim(1), 0...]
        if config.adalnLastLayer {
            if !config.adalnLastLayerNomod {
                let parts = MLX.split(modProj(state.t).expandedDimensions(axis: 1), parts: 2, axis: -1)
                x = (MLXArray(1).asType(x.dtype) + parts[1]) * WooshTensorOps.layerNormNoAffine(x) + parts[0]
            } else {
                x = WooshTensorOps.layerNormNoAffine(x)
            }
        }
        x = linear(x)
        let batch = x.dim(0)
        let time = x.dim(1)
        if config.patchSize == 1 {
            return x.transposed(0, 2, 1)
        }
        return x.reshaped(batch, time * config.patchSize, config.ioChannels).transposed(0, 2, 1)
    }
}

final class WooshFlowMapPostProcessing: Module {
    @ModuleInfo(key: "old_postprocessing") var oldPostprocessing: WooshPostProcessing

    init(config: WooshDiTConfig) {
        self._oldPostprocessing.wrappedValue = WooshPostProcessing(config: config)
    }

    func callAsFunction(_ state: WooshDiTState) -> MLXArray {
        -oldPostprocessing(state)
    }
}

public final class WooshFlowMapDiT: Module {
    @ModuleInfo(key: "preprocessing") var preprocessing: WooshFlowMapPreprocessing
    @ModuleInfo(key: "layers") var layers: [WooshDiTLayer]
    @ModuleInfo(key: "postprocessing") var postprocessing: WooshFlowMapPostProcessing

    private let config: WooshDiTConfig

    public init(config: WooshDiTConfig = WooshDiTConfig()) {
        self.config = config
        self._preprocessing.wrappedValue = WooshFlowMapPreprocessing(config: config)
        self._layers.wrappedValue = (0..<config.nLayers).map { WooshDiTLayer(config: config, index: $0) }
        self._postprocessing.wrappedValue = WooshFlowMapPostProcessing(config: config)
    }

    public func callAsFunction(x: MLXArray, t: MLXArray, r: MLXArray, cond: WooshCondition) -> MLXArray {
        var state = preprocessing(x, t: t, r: r, cond: cond, mask: nil)
        for index in layers.indices {
            layers[index](&state)
        }
        return postprocessing(state)
    }
}

public final class WooshLatentDiT: Module {
    @ModuleInfo(key: "preprocessing") var preprocessing: WooshInputProcessing
    @ModuleInfo(key: "layers") var layers: [WooshDiTLayer]
    @ModuleInfo(key: "postprocessing") var postprocessing: WooshPostProcessing

    private let config: WooshDiTConfig

    public init(config: WooshDiTConfig = WooshDiTConfig()) {
        self.config = config
        self._preprocessing.wrappedValue = WooshInputProcessing(config: config)
        self._layers.wrappedValue = (0..<config.nLayers).map { WooshDiTLayer(config: config, index: $0) }
        self._postprocessing.wrappedValue = WooshPostProcessing(config: config)
    }

    public func callAsFunction(x: MLXArray, t: MLXArray, cond: WooshCondition) -> MLXArray {
        var state = preprocessing(x, t: t, cond: cond, mask: nil)
        for index in layers.indices {
            layers[index](&state)
        }
        return postprocessing(state)
    }
}
