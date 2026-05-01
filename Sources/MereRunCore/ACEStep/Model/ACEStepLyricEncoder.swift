import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepLyricEncoder: Module {
    let config: ACEStepConfig
    let rope: RoPE

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "layers") var layers: [ACEStepEncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: ACEStepConfig) {
        self.config = config
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta,
            scale: 1.0
        )

        self._embedTokens.wrappedValue = Linear(config.textHiddenDim, config.hiddenSize, bias: true)
        self._layers.wrappedValue = (0..<config.numLyricEncoderHiddenLayers).map { ACEStepEncoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(inputsEmbeds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var x = embedTokens(inputsEmbeds)

        let seqLen = x.dim(1)
        let fullMask = ACEStepAttentionMasks.bidirectionalMask(attentionMask: attentionMask, seqLen: seqLen, slidingWindow: nil)
        let slidingMask = ACEStepAttentionMasks.bidirectionalMask(
            attentionMask: attentionMask,
            seqLen: seqLen,
            slidingWindow: (config.useSlidingWindow ? config.slidingWindow : nil)
        )

        for layer in layers {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = (layer.attentionType == "sliding_attention") ? slidingMask : fullMask
            x = layer(x, mask: mask, rope: rope)
        }

        return norm(x)
    }
}

