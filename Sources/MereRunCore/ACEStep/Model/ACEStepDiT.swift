import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepDiT: Module {
    let config: ACEStepConfig
    let rope: RoPE

    @ModuleInfo(key: "layers") var layers: [ACEStepDiTLayer]

    @ModuleInfo(key: "proj_in") var projIn: Conv1d
    @ModuleInfo(key: "proj_out") var projOut: ConvTransposed1d

    @ModuleInfo(key: "time_embed") var timeEmbed: ACEStepTimestepEmbedding
    @ModuleInfo(key: "time_embed_r") var timeEmbedR: ACEStepTimestepEmbedding

    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: Linear

    @ModuleInfo(key: "norm_out") var normOut: RMSNorm
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(config: ACEStepConfig) {
        self.config = config
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta,
            scale: 1.0
        )

        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { ACEStepDiTLayer(config: config, layerIdx: $0) }

        self._projIn.wrappedValue = Conv1d(
            inputChannels: config.inChannels,
            outputChannels: config.hiddenSize,
            kernelSize: config.patchSize,
            stride: config.patchSize,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )

        self._projOut.wrappedValue = ConvTransposed1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.audioAcousticHiddenDim,
            kernelSize: config.patchSize,
            stride: config.patchSize,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )

        self._timeEmbed.wrappedValue = ACEStepTimestepEmbedding(inChannels: 256, timeEmbedDim: config.hiddenSize)
        self._timeEmbedR.wrappedValue = ACEStepTimestepEmbedding(inChannels: 256, timeEmbedDim: config.hiddenSize)

        self._conditionEmbedder.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)

        self._normOut.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, config.hiddenSize])
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        timestepR: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray?,
        contextLatents: MLXArray
    ) -> MLXArray {
        let (tembT, timestepProjT) = timeEmbed(timestep)
        let (tembR, timestepProjR) = timeEmbedR(timestep - timestepR)
        let temb = tembT + tembR
        let timestepEmbedding = timestepProjT + timestepProjR

        let originalSeqLen = hiddenStates.dim(1)
        var x = MLX.concatenated([contextLatents, hiddenStates], axis: -1)

        let padLength = (config.patchSize - (originalSeqLen % config.patchSize)) % config.patchSize
        if padLength > 0 {
            x = padded(x, widths: [[0, 0], [0, padLength], [0, 0]])
        }

        x = projIn(x)
        let encoderProjected = conditionEmbedder(encoderHiddenStates)

        let seqLen = x.dim(1)
        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard config.useSlidingWindow, let window = config.slidingWindow, window > 0 else {
                return fullMask
            }
            let mask = makeSlidingWindowMask(seqLen: seqLen, window: window)
            return .array(mask)
        }()

        let encoderMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let encoderAttentionMask else { return .none }
            let mask = makeCrossAttentionMask(attentionMask: encoderAttentionMask, queryLen: seqLen)
            return .array(mask)
        }()

        for layer in layers {
            let selfMask: MLXFast.ScaledDotProductAttentionMaskMode = (layer.attentionType == "sliding_attention") ? slidingMask : fullMask
            x = layer(
                x,
                timestepEmbedding: timestepEmbedding,
                selfAttentionMask: selfMask,
                encoderHiddenStates: encoderProjected,
                encoderAttentionMask: encoderMask,
                rope: rope
            )
        }

        let outMod = scaleShiftTable + temb.expandedDimensions(axis: 1)
        let shift = outMod[0..., 0..<1, 0...]
        let scale = outMod[0..., 1..<2, 0...]
        x = (normOut(x) * (1 + scale) + shift).asType(x.dtype)

        x = projOut(x)
        x = x[0..., 0..<originalSeqLen, 0...]
        return x
    }

    private func makeSlidingWindowMask(seqLen: Int, window: Int) -> MLXArray {
        // Additive mask: 0 for allowed, -1e9 for disallowed.
        let idx = MLXArray((0..<seqLen).map { Int32($0) }).asType(.int32)
        let i = idx.reshaped(seqLen, 1)
        let j = idx.reshaped(1, seqLen)
        let dist = MLX.abs(i - j)
        let keep = dist .<= MLXArray(Int32(window))
        let zeros = MLX.zeros([seqLen, seqLen], dtype: .float32)
        let negInf = MLXArray(-1.0e9)
        let additive = MLX.where(keep, zeros, zeros + negInf)
        return additive.reshaped(1, 1, seqLen, seqLen)
    }

    private func makeCrossAttentionMask(attentionMask: MLXArray, queryLen: Int) -> MLXArray {
        // attentionMask: [B, S] (1 = keep, 0 = mask)
        let B = attentionMask.dim(0)
        let S = attentionMask.dim(1)

        let keep = (attentionMask .> MLXArray(Float(0))).asType(.bool)
        let keepExpanded = keep.reshaped(B, 1, 1, S)
        let keepBroadcast = MLX.broadcast(keepExpanded, to: [B, 1, queryLen, S])

        let zeros = MLX.zeros([B, 1, queryLen, S], dtype: .float32)
        let negInf = MLXArray(-1.0e9)
        return MLX.where(keepBroadcast, zeros, zeros + negInf)
    }
}
