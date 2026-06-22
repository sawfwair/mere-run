import Foundation
import MLX
import MLXFast
import MLXNN

public final class WooshVideoLatentModel: Module {
    @ModuleInfo(key: "conditioners") var conditioners: WooshVideoConditioners
    @ModuleInfo(key: "dit") var dit: WooshVideoLatentDiT

    public init(config: WooshDiTConfig = WooshDiTConfig()) {
        self._conditioners.wrappedValue = WooshVideoConditioners(config: config)
        self._dit.wrappedValue = WooshVideoLatentDiT(config: config)
    }

    public func callAsFunction(x: MLXArray, t: MLXArray, cond: WooshCondition) throws -> MLXArray {
        let encodedVideo = try conditioners.videoFeatures.encode(cond.videoFeatures)
        return dit(x: x, t: t, cond: cond, encodedVideoFeatures: encodedVideo)
    }
}

public final class WooshVideoFlowMapModel: Module {
    @ModuleInfo(key: "conditioners") var conditioners: WooshVideoConditioners
    @ModuleInfo(key: "dit") var dit: WooshVideoFlowMapDiT

    public init(config: WooshDiTConfig = WooshDiTConfig()) {
        self._conditioners.wrappedValue = WooshVideoConditioners(config: config)
        self._dit.wrappedValue = WooshVideoFlowMapDiT(config: config)
    }

    public func callAsFunction(x: MLXArray, t: MLXArray, r: MLXArray, cond: WooshCondition) throws -> MLXArray {
        let encodedVideo = try conditioners.videoFeatures.encode(cond.videoFeatures)
        return dit(x: x, t: t, r: r, cond: cond, encodedVideoFeatures: encodedVideo)
    }
}

public final class WooshVideoConditioners: Module {
    @ModuleInfo(key: "video_features") var videoFeatures: WooshVideoFeatureConditioner

    init(config: WooshDiTConfig) {
        self._videoFeatures.wrappedValue = WooshVideoFeatureConditioner(config: config)
    }
}

public final class WooshVideoFeatureConditioner: Module {
    @ModuleInfo(key: "linear_encoder") var linearEncoder: Linear

    init(config: WooshDiTConfig) {
        self._linearEncoder.wrappedValue = Linear(768, config.dim, bias: false)
    }

    func encode(_ rawFeatures: MLXArray?) throws -> MLXArray {
        guard let rawFeatures else {
            throw WooshError.missingVideoFeatures
        }
        guard rawFeatures.ndim == 3, rawFeatures.dim(2) == 768 else {
            throw WooshError.invalidVideoFeatureShape(rawFeatures.shape)
        }
        return linearEncoder(rawFeatures.asType(.float32))
    }
}

final class WooshVideoInputProcessing: Module {
    @ModuleInfo(key: "old_preprocessing") var oldPreprocessing: WooshInputProcessing

    private let videoRoPE: WooshRoPECache

    init(config: WooshDiTConfig) {
        self.videoRoPE = WooshRoPE.precompute(config: config, audioFPSMultiplier: 100.0 / 24.0)
        self._oldPreprocessing.wrappedValue = WooshInputProcessing(config: config)
    }

    func callAsFunction(
        _ x: MLXArray,
        t: MLXArray,
        cond: WooshCondition,
        encodedVideoFeatures: MLXArray,
        mask: MLXArray?
    ) -> WooshDiTState {
        var state = oldPreprocessing(x, t: t, cond: cond, mask: mask)
        state.videoFeatures = encodedVideoFeatures
        state.videoRoPE = videoRoPE
        return state
    }
}

final class WooshVideoFlowMapPreprocessing: Module {
    @ModuleInfo(key: "old_preprocessing") var oldPreprocessing: WooshVideoInputProcessing
    @ModuleInfo(key: "timestep_features_t") var timestepFeaturesT: WooshFixedFourierFeaturesTime
    @ModuleInfo(key: "timestep_features_r") var timestepFeaturesR: WooshFixedFourierFeaturesTime
    @ModuleInfo(key: "cfg_features") var cfgFeatures: WooshFixedFourierFeaturesTime
    @ModuleInfo(key: "to_timestep_embed") var toTimestepEmbed: WooshLinearSiLU2
    @ModuleInfo(key: "timestep_logvar") var timestepLogvar: WooshFourierFeaturesTime
    @ModuleInfo(key: "to_logvar") var toLogvar: WooshLinearSiLUToLinear

    private let config: WooshDiTConfig

    init(config: WooshDiTConfig) {
        self.config = config
        self._oldPreprocessing.wrappedValue = WooshVideoInputProcessing(config: config)
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

    func callAsFunction(
        _ x: MLXArray,
        t: MLXArray,
        r: MLXArray,
        cond: WooshCondition,
        encodedVideoFeatures: MLXArray,
        mask: MLXArray?
    ) -> WooshDiTState {
        var state = oldPreprocessing(
            x,
            t: t,
            cond: cond,
            encodedVideoFeatures: encodedVideoFeatures,
            mask: mask
        )
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

final class WooshUnmodulatedModalityAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "norm_q") var normQ: WooshRMSNorm
    @ModuleInfo(key: "norm_k") var normK: WooshRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    private let config: WooshDiTConfig
    private let keyPath: KeyPath<WooshDiTState, MLXArray>
    private let ropePath: KeyPath<WooshDiTState, WooshRoPECache>

    init(config: WooshDiTConfig, keyPath: KeyPath<WooshDiTState, MLXArray>, ropePath: KeyPath<WooshDiTState, WooshRoPECache>) {
        self.config = config
        self.keyPath = keyPath
        self.ropePath = ropePath
        self._qkv.wrappedValue = Linear(config.dim, config.dim * 3, bias: true)
        self._normQ.wrappedValue = WooshRMSNorm(dimensions: config.headDim)
        self._normK.wrappedValue = WooshRMSNorm(dimensions: config.headDim)
        self._outProj.wrappedValue = Linear(config.dim, config.dim, bias: true)
    }

    struct Precomputed {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let sequenceLength: Int
    }

    func precompute(_ state: WooshDiTState) -> Precomputed {
        let x = state[keyPath: keyPath]
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let normed = WooshTensorOps.layerNormNoAffine(x)
        let parts = MLX.split(qkv(normed), parts: 3, axis: -1)
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
            sequenceLength: sequence
        )
    }

    func finish(attended: MLXArray, residual: MLXArray) -> MLXArray {
        let batch = residual.dim(0)
        let sequence = residual.dim(1)
        return residual + outProj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, config.dim))
    }
}

final class WooshUnmodulatedModalityMLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear

    init(config: WooshDiTConfig) {
        self._w1.wrappedValue = Linear(config.dim, config.interDim, bias: true)
        self._w2.wrappedValue = Linear(config.interDim, config.dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + w2(WooshTensorOps.geluTanh(w1(WooshTensorOps.layerNormNoAffine(x))))
    }
}

final class WooshVideoMMMModalities: Module {
    @ModuleInfo(key: "x") var x: WooshModalityAttention
    @ModuleInfo(key: "description") var description: WooshModalityAttention
    @ModuleInfo(key: "video_features") var videoFeatures: WooshUnmodulatedModalityAttention

    init(config: WooshDiTConfig) {
        self._x.wrappedValue = WooshModalityAttention(config: config, keyPath: \.x, ropePath: \.rope)
        self._description.wrappedValue = WooshModalityAttention(
            config: config,
            keyPath: \.description,
            ropePath: \.descriptionRoPE
        )
        self._videoFeatures.wrappedValue = WooshUnmodulatedModalityAttention(
            config: config,
            keyPath: \.videoFeatures,
            ropePath: \.videoRoPE
        )
    }
}

final class WooshVideoMMMFFNs: Module {
    @ModuleInfo(key: "x") var x: WooshModalityMLP
    @ModuleInfo(key: "description") var description: WooshModalityMLP
    @ModuleInfo(key: "video_features") var videoFeatures: WooshUnmodulatedModalityMLP

    init(config: WooshDiTConfig) {
        self._x.wrappedValue = WooshModalityMLP(config: config)
        self._description.wrappedValue = WooshModalityMLP(config: config)
        self._videoFeatures.wrappedValue = WooshUnmodulatedModalityMLP(config: config)
    }
}

final class WooshVideoMMMAttention: Module {
    @ModuleInfo(key: "modalities") var modalities: WooshVideoMMMModalities
    private let scale: Float

    init(config: WooshDiTConfig) {
        self.scale = 1.0 / sqrt(Float(config.headDim))
        self._modalities.wrappedValue = WooshVideoMMMModalities(config: config)
    }

    func callAsFunction(_ state: inout WooshDiTState) {
        let xResidual = state.x
        let descriptionResidual = state.description
        let videoResidual = state.videoFeatures
        let xPre = modalities.x.precompute(state)
        let descriptionPre = modalities.description.precompute(state)
        let videoPre = modalities.videoFeatures.precompute(state)
        let q = MLX.concatenated([xPre.q, descriptionPre.q, videoPre.q], axis: 2)
        let k = MLX.concatenated([xPre.k, descriptionPre.k, videoPre.k], axis: 2)
        let v = MLX.concatenated([xPre.v, descriptionPre.v, videoPre.v], axis: 2)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        let descStart = xPre.sequenceLength
        let videoStart = descStart + descriptionPre.sequenceLength
        let xAttended = attended[0..., 0..., 0..<xPre.sequenceLength, 0...]
        let descAttended = attended[0..., 0..., descStart..<videoStart, 0...]
        let videoAttended = attended[0..., 0..., videoStart..<(videoStart + videoPre.sequenceLength), 0...]
        state.x = modalities.x.finish(attended: xAttended, residual: xResidual, gate: xPre.gate)
        state.description = modalities.description.finish(
            attended: descAttended,
            residual: descriptionResidual,
            gate: descriptionPre.gate
        )
        state.videoFeatures = modalities.videoFeatures.finish(attended: videoAttended, residual: videoResidual)
    }
}

final class WooshVideoMMMBlock: Module {
    @ModuleInfo(key: "attn") var attention: WooshVideoMMMAttention
    @ModuleInfo(key: "ffns") var ffns: WooshVideoMMMFFNs

    init(config: WooshDiTConfig) {
        self._attention.wrappedValue = WooshVideoMMMAttention(config: config)
        self._ffns.wrappedValue = WooshVideoMMMFFNs(config: config)
    }

    func callAsFunction(_ state: inout WooshDiTState) {
        attention(&state)
        state.x = ffns.x(state.x, t: state.t)
        state.description = ffns.description(state.description, t: state.t)
        state.videoFeatures = ffns.videoFeatures(state.videoFeatures)
    }
}

final class WooshVideoSingleStreamBlock: Module {
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
        let videoLen = state.videoFeatures.dim(1)
        let residualX = state.x
        let residualDescription = state.description
        let residualVideo = state.videoFeatures
        var hidden = MLX.concatenated([state.x, state.description, state.videoFeatures], axis: 1)
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

        let cos = MLX.concatenated([
            state.rope.cos[0..<xLen, 0...],
            state.descriptionRoPE.cos[0..<descLen, 0...],
            state.videoRoPE.cos[0..<videoLen, 0...],
        ], axis: 0)
        let sin = MLX.concatenated([
            state.rope.sin[0..<xLen, 0...],
            state.descriptionRoPE.sin[0..<descLen, 0...],
            state.videoRoPE.sin[0..<videoLen, 0...],
        ], axis: 0)
        let cache = WooshRoPECache(cos: cos, sin: sin)
        let qRope = WooshRoPE.apply(q[0..., 0..., 0..., 0..<config.qkRopeHeadDim], cache: cache)
        let kRope = WooshRoPE.apply(k[0..., 0..., 0..., 0..<config.qkRopeHeadDim], cache: cache)
        q = MLX.concatenated([qRope, q[0..., 0..., 0..., config.qkRopeHeadDim..<config.headDim]], axis: -1)
        k = MLX.concatenated([kRope, k[0..., 0..., 0..., config.qkRopeHeadDim..<config.headDim]], axis: -1)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v,
            scale: scale,
            mask: .none
        )
        let z = attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, config.dim)
        let out = outProj(MLX.concatenated([z, WooshTensorOps.geluTanh(mlp)], axis: -1)) * modulation[2]
        let descEnd = xLen + descLen
        state.x = residualX + out[0..., 0..<xLen, 0...]
        state.description = residualDescription + out[0..., xLen..<descEnd, 0...]
        state.videoFeatures = residualVideo + out[0..., descEnd..<(descEnd + videoLen), 0...]
    }
}

final class WooshVideoDiTLayer: Module {
    @ModuleInfo(key: "attn") var attention: WooshVideoMMMAttention?
    @ModuleInfo(key: "ffns") var ffns: WooshVideoMMMFFNs?
    @ModuleInfo(key: "qkv_mlp") var qkvMLP: Linear?
    @ModuleInfo(key: "out_proj") var outProj: Linear?
    @ModuleInfo(key: "norm_q") var normQ: WooshRMSNorm?
    @ModuleInfo(key: "norm_k") var normK: WooshRMSNorm?
    @ModuleInfo(key: "mod_proj") var modProj: Linear?

    private let singleStream: WooshVideoSingleStreamBlock?
    private let isMultimodal: Bool

    init(config: WooshDiTConfig, index: Int) {
        if index < config.nMultimodalLayers {
            let block = WooshVideoMMMBlock(config: config)
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
            let block = WooshVideoSingleStreamBlock(config: config)
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
                state.videoFeatures = ffns.videoFeatures(state.videoFeatures)
            }
        } else if let singleStream {
            singleStream(&state)
        }
    }
}

public final class WooshVideoLatentDiT: Module {
    @ModuleInfo(key: "preprocessing") var preprocessing: WooshVideoInputProcessing
    @ModuleInfo(key: "layers") var layers: [WooshVideoDiTLayer]
    @ModuleInfo(key: "postprocessing") var postprocessing: WooshPostProcessing

    public init(config: WooshDiTConfig = WooshDiTConfig()) {
        self._preprocessing.wrappedValue = WooshVideoInputProcessing(config: config)
        self._layers.wrappedValue = (0..<config.nLayers).map { WooshVideoDiTLayer(config: config, index: $0) }
        self._postprocessing.wrappedValue = WooshPostProcessing(config: config)
    }

    public func callAsFunction(
        x: MLXArray,
        t: MLXArray,
        cond: WooshCondition,
        encodedVideoFeatures: MLXArray
    ) -> MLXArray {
        var state = preprocessing(x, t: t, cond: cond, encodedVideoFeatures: encodedVideoFeatures, mask: nil)
        for index in layers.indices {
            layers[index](&state)
        }
        return postprocessing(state)
    }
}

public final class WooshVideoFlowMapDiT: Module {
    @ModuleInfo(key: "preprocessing") var preprocessing: WooshVideoFlowMapPreprocessing
    @ModuleInfo(key: "layers") var layers: [WooshVideoDiTLayer]
    @ModuleInfo(key: "postprocessing") var postprocessing: WooshFlowMapPostProcessing

    public init(config: WooshDiTConfig = WooshDiTConfig()) {
        self._preprocessing.wrappedValue = WooshVideoFlowMapPreprocessing(config: config)
        self._layers.wrappedValue = (0..<config.nLayers).map { WooshVideoDiTLayer(config: config, index: $0) }
        self._postprocessing.wrappedValue = WooshFlowMapPostProcessing(config: config)
    }

    public func callAsFunction(
        x: MLXArray,
        t: MLXArray,
        r: MLXArray,
        cond: WooshCondition,
        encodedVideoFeatures: MLXArray
    ) -> MLXArray {
        var state = preprocessing(x, t: t, r: r, cond: cond, encodedVideoFeatures: encodedVideoFeatures, mask: nil)
        for index in layers.indices {
            layers[index](&state)
        }
        return postprocessing(state)
    }
}
