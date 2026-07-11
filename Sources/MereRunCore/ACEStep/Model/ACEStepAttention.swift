import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: ACEStepConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.ropeTheta = config.ropeTheta

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: config.attentionBias)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        encoderHiddenStates: MLXArray? = nil,
        rope: RoPE? = nil,
        ropeOffset: Int = 0
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let qLen = hiddenStates.dim(1)

        let kInput = encoderHiddenStates ?? hiddenStates
        let kLen = kInput.dim(1)

        var q = qProj(hiddenStates)
        var k = kProj(kInput)
        var v = vProj(kInput)

        q = qNorm(q.reshaped(B, qLen, numHeads, headDim)).transposed(0, 2, 1, 3)
        k = kNorm(k.reshaped(B, kLen, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        v = v.reshaped(B, kLen, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if encoderHiddenStates == nil, rope != nil {
            q = applySplitRoPE(q, offset: ropeOffset)
            k = applySplitRoPE(k, offset: ropeOffset)
        }

        let qF32 = q.asType(.float32)
        let kF32 = k.asType(.float32)
        let vF32 = v.asType(.float32)
        var out = MLXFast.scaledDotProductAttention(
            queries: qF32,
            keys: kF32,
            values: vF32,
            scale: scale,
            mask: mask
        )
        out = out.asType(q.dtype)

        out = out.transposed(0, 2, 1, 3).reshaped(B, qLen, -1)
        return oProj(out)
    }

    private func applySplitRoPE(_ x: MLXArray, offset: Int) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        let seqLen = x32.dim(2)
        let half = headDim / 2

        let positions = MLXArray((offset..<(offset + seqLen)).map { Float($0) }).asType(.float32)
        let indices = MLXArray((0..<half).map { Float($0) }).asType(.float32)
        let freqs = MLX.exp(-MLXArray(Float(log(ropeTheta))) * indices / Float(half))
        let args = positions[0..., .newAxis] * freqs[.newAxis]
        let cos = MLX.cos(args).reshaped(1, 1, seqLen, half)
        let sin = MLX.sin(args).reshaped(1, 1, seqLen, half)

        let x1 = x32[0..., 0..., 0..., 0..<half]
        let x2 = x32[0..., 0..., 0..., half...]
        let out1 = x1 * cos - x2 * sin
        let out2 = x2 * cos + x1 * sin
        return MLX.concatenated([out1, out2], axis: 3).asType(dtype)
    }
}
