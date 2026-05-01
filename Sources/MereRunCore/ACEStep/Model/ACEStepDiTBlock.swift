import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: ACEStepConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class ACEStepDiTLayer: Module {
    let attentionType: String

    @ModuleInfo(key: "self_attn_norm") var selfAttnNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: ACEStepAttention

    @ModuleInfo(key: "cross_attn_norm") var crossAttnNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: ACEStepAttention

    @ModuleInfo(key: "mlp_norm") var mlpNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: ACEStepMLP

    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(config: ACEStepConfig, layerIdx: Int) {
        self.attentionType = config.layerTypes?[safe: layerIdx] ?? "full_attention"

        self._selfAttnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._selfAttn.wrappedValue = ACEStepAttention(config: config)

        self._crossAttnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._crossAttn.wrappedValue = ACEStepAttention(config: config)

        self._mlpNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._mlp.wrappedValue = ACEStepMLP(config: config)

        self._scaleShiftTable.wrappedValue = MLXArray.zeros([1, 6, config.hiddenSize])
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        timestepEmbedding: MLXArray,
        selfAttentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        rope: RoPE
    ) -> MLXArray {
        // (scale_shift_table + timestep_embedding).chunk(6, dim=1)
        let mod = (scaleShiftTable + timestepEmbedding).asType(hiddenStates.dtype)

        let shiftMsa = mod[0..., 0..<1, 0...]
        let scaleMsa = mod[0..., 1..<2, 0...]
        let gateMsa = mod[0..., 2..<3, 0...]
        let cShiftMsa = mod[0..., 3..<4, 0...]
        let cScaleMsa = mod[0..., 4..<5, 0...]
        let cGateMsa = mod[0..., 5..<6, 0...]

        var x = hiddenStates

        // Self-attn (AdaLN + gated residual)
        var normed = selfAttnNorm(x)
        normed = (normed * (1 + scaleMsa) + shiftMsa).asType(x.dtype)
        let attnOut = selfAttn(normed, mask: selfAttentionMask, rope: rope)
        x = (x + attnOut * gateMsa).asType(x.dtype)

        // Cross-attn (standard residual)
        let crossNormed = crossAttnNorm(x)
        let crossOut = crossAttn(crossNormed, mask: encoderAttentionMask, encoderHiddenStates: encoderHiddenStates)
        x = x + crossOut

        // MLP (AdaLN + gated residual)
        var mlpIn = mlpNorm(x)
        mlpIn = (mlpIn * (1 + cScaleMsa) + cShiftMsa).asType(x.dtype)
        let mlpOut = mlp(mlpIn)
        x = (x + mlpOut * cGateMsa).asType(x.dtype)

        return x
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard index >= 0, index < count else { return nil }
        return self[index]
    }
}
