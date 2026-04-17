import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepAudioTokenDetokenizer: Module {
    let config: ACEStepConfig
    let rope: RoPE

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_tokens") var specialTokens: MLXArray
    @ModuleInfo(key: "layers") var layers: [ACEStepEncoderLayer]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(config: ACEStepConfig) {
        self.config = config
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta,
            scale: 1.0
        )

        self._embedTokens.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._specialTokens.wrappedValue = MLXArray.zeros([1, config.poolWindowSize, config.hiddenSize])
        self._layers.wrappedValue = (0..<config.numAttentionPoolerHiddenLayers).map { ACEStepEncoderLayer(config: config, layerIdx: $0) }
        self._projOut.wrappedValue = Linear(config.hiddenSize, config.audioAcousticHiddenDim, bias: true)
    }

    /// Detokenize quantized 5Hz tokens `[B, T5, 2048]` back to 25Hz latents `[B, T25, 64]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T5 = x.dim(1)
        let P = config.poolWindowSize

        var h = embedTokens(x) // [B, T5, D]

        h = h.expandedDimensions(axis: 2) // [B, T5, 1, D]
        h = broadcast(h, to: [B, T5, P, config.hiddenSize])

        let st = broadcast(specialTokens, to: [B, T5, P, config.hiddenSize])
        h = h + st

        h = h.reshaped(B * T5, P, config.hiddenSize)

        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard config.useSlidingWindow, let window = config.slidingWindow, window > 0 else { return fullMask }
            return ACEStepAttentionMasks.bidirectionalMask(attentionMask: nil, seqLen: P, slidingWindow: window)
        }()

        for layer in layers {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = (layer.attentionType == "sliding_attention") ? slidingMask : fullMask
            h = layer(h, mask: mask, rope: rope)
        }

        h = norm(h)
        h = projOut(h)

        h = h.reshaped(B, T5, P, config.audioAcousticHiddenDim)
        return h.reshaped(B, T5 * P, config.audioAcousticHiddenDim)
    }
}

