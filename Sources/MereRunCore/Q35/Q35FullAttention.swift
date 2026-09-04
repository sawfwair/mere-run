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
    @ModuleInfo(key: "indexer") var indexer: Q38QSAIndexer?

    /// Row-concatenated q/k/v quantized weights so each attention call issues
    /// one quantized matmul instead of three (bit-identical: quantized
    /// packing is per-output-row). Lives outside the module tree; invalidated
    /// whenever the source modules are replaced. MERERUN_Q35_FUSED_QKV=0
    /// falls back to separate projections. Trades a second resident copy of
    /// the attention projection weights for the fused matmul.
    private var fusedQKV: Gemma4FusedQuantizedProjection?
    private static let fusedQKVEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_Q35_FUSED_QKV"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()
    private static let logFusionOnce: Void = {
        FileHandle.standardError.write(Data("[q35] fused_qkv=true\n".utf8))
    }()

    private func resolvedFusedQKV() -> Gemma4FusedQuantizedProjection? {
        guard Self.fusedQKVEnabled else { return nil }
        if let fusedQKV, fusedQKV.matches([qProj, kProj, vProj]) {
            return fusedQKV
        }
        fusedQKV = Gemma4FusedQuantizedProjection.fuse([qProj, kProj, vProj])
        if fusedQKV != nil {
            _ = Self.logFusionOnce
        }
        return fusedQKV
    }

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
        self._qNorm.wrappedValue = Q35RMSNorm(
            dimensions: text.headDim,
            eps: text.rmsNormEps,
            zeroCenteredWeight: !text.isQwen4Exp
        )
        self._kNorm.wrappedValue = Q35RMSNorm(
            dimensions: text.headDim,
            eps: text.rmsNormEps,
            zeroCenteredWeight: !text.isQwen4Exp
        )
        self._indexer.wrappedValue = text.isQwen4Exp ? Q38QSAIndexer(config: config) : nil

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
        positionIds: MLXArray? = nil,
        targetVerify: Bool = false
    ) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)
        let offsets = cache?.rowOffsets ?? Array(repeating: cache?.offset ?? 0, count: b)

        let qFlat: MLXArray
        let kFlat: MLXArray
        let vFlat: MLXArray
        if let fused = resolvedFusedQKV() {
            let parts = fused.callSplit(x)
            qFlat = parts[0]
            kFlat = parts[1]
            vFlat = parts[2]
        } else {
            qFlat = qProj(x)
            kFlat = kProj(x)
            vFlat = vProj(x)
        }

        let qProjection = qFlat.reshaped(b, s, numHeads, hasOutputGate ? headDim * 2 : headDim)

        let queries: MLXArray
        let gate: MLXArray?
        if hasOutputGate {
            queries = qProjection[.ellipsis, 0..<headDim]
            gate = qProjection[.ellipsis, headDim...].reshaped(b, s, numHeads * headDim)
        } else {
            queries = qProjection
            gate = nil
        }

        let keys = kFlat.reshaped(b, s, numKVHeads, headDim)
        let values = vFlat.reshaped(b, s, numKVHeads, headDim)

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

        let queryCount = q.dim(2)
        let keyCount = k.dim(2)
        let attn: MLXArray
        if let indexer, targetVerify, b == 1, (2...32).contains(queryCount),
           (queryCount <= 9 || Q38WideVerificationPolicy.exactSparseAttention),
           keyCount >= queryCount, case .causal = mask {
            // Flash-Next's BF16 dense attention and sparse reductions can
            // round differently with multiple query rows. Also, a block can
            // cross the dense/QSA boundary. Keep each row's attention on its
            // exact serial history; Q/K/V and output projections stay batched.
            attn = q38VerificationAttention(
                hidden: x, queries: q, keys: k, values: v, offsets: offsets,
                positionIds: positionIds, cache: cache as? Q38QSACache, indexer: indexer
            )
        } else if let indexer, let sparse = indexer.attention(
            hidden: x, queries: q, keys: k, values: v, offsets: offsets,
            positionIds: positionIds, cache: cache as? Q38QSACache, scale: scale
        ) {
            attn = sparse
        } else if targetVerify,
           b == 1,
           (6...9).contains(queryCount),
           keyCount >= queryCount,
           case .causal = mask {
            // The fused SDPA vector path changes arithmetic above five query
            // rows. Preserve the serial greedy trajectory by presenting the
            // same bottom-right-aligned causal windows as two <=5-row calls.
            let split = 5
            let keySplit = keyCount - (queryCount - split)
            let first = MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., 0..<split, 0...],
                keys: k[0..., 0..., 0..<keySplit, 0...],
                values: v[0..., 0..., 0..<keySplit, 0...],
                scale: scale,
                mask: .causal
            )
            let second = MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., split..., 0...],
                keys: k,
                values: v,
                scale: scale,
                mask: .causal
            )
            attn = MLX.concatenated([first, second], axis: 2)
        } else {
            attn = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: mask
            )
        }

        var out = attn.transposed(0, 2, 1, 3).reshaped(b, s, numHeads * headDim)
        if let gate {
            out = out * MLX.sigmoid(gate)
        }
        return oProj(out)
    }

    private func q38VerificationAttention(
        hidden: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray,
        offsets: [Int], positionIds: MLXArray?, cache: Q38QSACache?, indexer: Q38QSAIndexer
    ) -> MLXArray {
        let count = queries.dim(2)
        let prefix = keys.dim(2) - count
        let rows = (0..<count).map { row in
            let query = queries[0..., 0..., row..<(row + 1), 0...]
            let rowKeys = keys[0..., 0..., 0..<(prefix + row + 1), 0...]
            let rowValues = values[0..., 0..., 0..<(prefix + row + 1), 0...]
            let positions = positionIds.map {
                $0.ndim == 3 ? $0[0..., 0..., row..<(row + 1)] : $0[0..., row..<(row + 1)]
            }
            return indexer.attention(
                hidden: hidden[0..., row..<(row + 1), 0...],
                queries: query, keys: rowKeys, values: rowValues, offsets: [offsets[0] + row],
                positionIds: positions, cache: cache, scale: scale
            ) ?? MLXFast.scaledDotProductAttention(
                queries: query, keys: rowKeys, values: rowValues, scale: scale, mask: .none
            )
        }
        return MLX.concatenated(rows, axis: 2)
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
