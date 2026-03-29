import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepAttentionPooler: Module {
    let config: ACEStepConfig
    let rope: RoPE

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_token") var specialToken: MLXArray
    @ModuleInfo(key: "layers") var layers: [ACEStepEncoderLayer]

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
        self._specialToken.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        self._layers.wrappedValue = (0..<config.numAttentionPoolerHiddenLayers).map { ACEStepEncoderLayer(config: config, layerIdx: $0) }
    }

    /// Pool patches `[B, T, P, D]` into `[B, T, D]` using a special token per time step.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let P = x.dim(2)

        var h = embedTokens(x)
        let cls = broadcast(specialToken, to: [B, T, 1, config.hiddenSize])
        h = MLX.concatenated([cls, h], axis: 2) // [B, T, 1+P, D]

        h = h.reshaped(B * T, 1 + P, config.hiddenSize)

        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard config.useSlidingWindow, let window = config.slidingWindow, window > 0 else { return fullMask }
            return ACEStepAttentionMasks.bidirectionalMask(attentionMask: nil, seqLen: 1 + P, slidingWindow: window)
        }()

        for layer in layers {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = (layer.attentionType == "sliding_attention") ? slidingMask : fullMask
            h = layer(h, mask: mask, rope: rope)
        }

        h = norm(h)

        let pooled = h[0..., 0, 0...] // [B*T, D]
        return pooled.reshaped(B, T, config.hiddenSize)
    }
}

