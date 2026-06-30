import Foundation
import MLX
import MLXFast
import MLXNN

final class Q35FullAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Q35RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Q35RMSNorm

    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let hasOutputGate: Bool
    private let scale: Float
    private let rotaryDimensions: Int
    private let rope: RoPE
    private let mropeRotary: Qwen3VLRotaryEmbedding?

    init(config: Q35Config) {
        let text = config.textConfig
        self.numHeads = text.numAttentionHeads
        self.numKVHeads = text.numKeyValueHeads
        self.headDim = text.headDim
        self.hasOutputGate = text.attnOutputGate
        self.scale = 1.0 / sqrt(Float(max(1, text.headDim)))

        let qOutput = text.numAttentionHeads * text.headDim * (text.attnOutputGate ? 2 : 1)
        let kvOutput = text.numKeyValueHeads * text.headDim

        self._qProj.wrappedValue = Linear(text.hiddenSize, qOutput, bias: text.attentionBias)
        self._kProj.wrappedValue = Linear(text.hiddenSize, kvOutput, bias: text.attentionBias)
        self._vProj.wrappedValue = Linear(text.hiddenSize, kvOutput, bias: text.attentionBias)
        self._oProj.wrappedValue = Linear(text.numAttentionHeads * text.headDim, text.hiddenSize, bias: text.attentionBias)
        self._qNorm.wrappedValue = Q35RMSNorm(dimensions: text.headDim, eps: text.rmsNormEps)
        self._kNorm.wrappedValue = Q35RMSNorm(dimensions: text.headDim, eps: text.rmsNormEps)

        let ropeDims = max(1, Int(Float(text.headDim) * text.ropeParameters.partialRotaryFactor))
        self.rotaryDimensions = min(text.headDim, ropeDims)
        self.rope = RoPE(
            dimensions: ropeDims,
            traditional: false,
            base: text.ropeParameters.ropeTheta
        )
        if text.ropeParameters.mropeInterleaved == true,
           let mropeSection = text.ropeParameters.mropeSection,
           !mropeSection.isEmpty {
            self.mropeRotary = Qwen3VLRotaryEmbedding(
                dim: ropeDims,
                base: text.ropeParameters.ropeTheta,
                mropeSection: mropeSection
            )
        } else {
            self.mropeRotary = nil
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)

        let qProjection = qProj(x).reshaped(b, s, numHeads, hasOutputGate ? headDim * 2 : headDim)

        let queries: MLXArray
        let gate: MLXArray?
        if hasOutputGate {
            queries = qProjection[.ellipsis, 0..<headDim]
            gate = qProjection[.ellipsis, headDim...].reshaped(b, s, numHeads * headDim)
        } else {
            queries = qProjection
            gate = nil
        }

        let keys = kProj(x).reshaped(b, s, numKVHeads, headDim)
        let values = vProj(x).reshaped(b, s, numKVHeads, headDim)

        var q = qNorm(queries).transposed(0, 2, 1, 3)
        var k = kNorm(keys).transposed(0, 2, 1, 3)
        var v = values.transposed(0, 2, 1, 3)

        if let positionIds, mropeRotary != nil {
            (q, k) = applyMRoPE(q, k, positionIds: positionIds)
        } else if let rowOffsets = cache?.rowOffsets, rowOffsets.count == b {
            q = applyRoPEByRow(q, rowOffsets: rowOffsets)
            k = applyRoPEByRow(k, rowOffsets: rowOffsets)
        } else {
            let offset = cache?.offset ?? 0
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }

        if let cache {
            let cached = cache.update(keys: k, values: v)
            k = cached.0
            v = cached.1
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        var out = attn.transposed(0, 2, 1, 3).reshaped(b, s, numHeads * headDim)
        if let gate {
            out = out * MLX.sigmoid(gate)
        }
        return oProj(out)
    }

    private func applyMRoPE(
        _ queries: MLXArray,
        _ keys: MLXArray,
        positionIds: MLXArray
    ) -> (MLXArray, MLXArray) {
        guard let mropeRotary else { return (queries, keys) }

        let rotaryWidth = min(rotaryDimensions, queries.dim(-1), keys.dim(-1))
        let qRotary = queries[0..., 0..., 0..., 0..<rotaryWidth]
        let kRotary = keys[0..., 0..., 0..., 0..<rotaryWidth]
        let (cos, sin) = mropeRotary(positionIds: positionIds, dtype: queries.dtype)
        let rotated = applyRotaryPosEmb(qRotary, kRotary, cos: cos, sin: sin)

        guard rotaryWidth < queries.dim(-1) else {
            return rotated
        }

        let qPass = queries[0..., 0..., 0..., rotaryWidth...]
        let kPass = keys[0..., 0..., 0..., rotaryWidth...]
        return (
            MLX.concatenated([rotated.0, qPass], axis: -1),
            MLX.concatenated([rotated.1, kPass], axis: -1)
        )
    }

    private func applyRoPEByRow(_ value: MLXArray, rowOffsets: [Int]) -> MLXArray {
        guard !rowOffsets.isEmpty else {
            return rope(value, offset: 0)
        }
        let rows = rowOffsets.enumerated().map { index, offset in
            rope(value[index..<(index + 1), 0..., 0..., 0...], offset: offset)
        }
        return concatenated(rows, axis: 0)
    }
}
