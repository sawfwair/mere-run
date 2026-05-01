import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepEncoderLayer: Module {
    let attentionType: String

    @ModuleInfo(key: "self_attn") var selfAttn: ACEStepAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: ACEStepMLP

    init(config: ACEStepConfig, layerIdx: Int) {
        self.attentionType = config.layerTypes?[safe: layerIdx] ?? "full_attention"

        self._selfAttn.wrappedValue = ACEStepAttention(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._mlp.wrappedValue = ACEStepMLP(config: config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        rope: RoPE
    ) -> MLXArray {
        var x = hiddenStates

        // Self-attention
        let residual1 = x
        let normed = inputLayerNorm(x)
        let attnOut = selfAttn(normed, mask: mask, rope: rope)
        x = residual1 + attnOut

        // MLP
        let residual2 = x
        let postNormed = postAttentionLayerNorm(x)
        let mlpOut = mlp(postNormed)
        x = residual2 + mlpOut

        return x
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard index >= 0, index < count else { return nil }
        return self[index]
    }
}

