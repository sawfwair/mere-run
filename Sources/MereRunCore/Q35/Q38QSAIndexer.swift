import Foundation
import MLX
import MLXNN

final class Q38QSAIndexer: Module {
    @ModuleInfo(key: "index_qk_proj") var indexQKProjection: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: Q35RMSNorm
    @ModuleInfo(key: "k_layernorm") var keyNorm: Q35RMSNorm

    let budget: Int
    private let headCount: Int
    private let kvHeadCount: Int
    private let headDimension: Int
    private let ratio: Int
    private let rotaryDimensions: Int
    private let rotary: Qwen3VLRotaryEmbedding

    init(config: Q35Config) {
        let text = config.textConfig
        self.budget = text.indexerBudget
        self.headCount = text.indexerHeadCount
        self.kvHeadCount = text.indexerKVHeadCount
        self.headDimension = text.indexerHeadDimension
        self.ratio = text.indexerCompressionRatio
        self.rotaryDimensions = Int(Float(text.headDim) * text.ropeParameters.partialRotaryFactor)
        self.rotary = Qwen3VLRotaryEmbedding(
            dim: rotaryDimensions,
            base: text.ropeParameters.ropeTheta,
            mropeSection: text.ropeParameters.mropeSection ?? []
        )
        self._indexQKProjection.wrappedValue = Linear(
            text.hiddenSize, (headCount + kvHeadCount) * headDimension, bias: false
        )
        self._queryNorm.wrappedValue = Q35RMSNorm(
            dimensions: headDimension, eps: text.rmsNormEps, zeroCenteredWeight: false
        )
        self._keyNorm.wrappedValue = Q35RMSNorm(
            dimensions: headDimension, eps: text.rmsNormEps, zeroCenteredWeight: false
        )
        super.init()
    }

    func attention(
        hidden: MLXArray,
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        offsets: [Int],
        positionIds: MLXArray?,
        cache: Q38QSACache?,
        scale: Float
    ) -> MLXArray? {
        let batch = hidden.dim(0)
        let count = hidden.dim(1)
        let projected = indexQKProjection(hidden)
        let indexQueries = queryNorm(projected[.ellipsis, 0..<(headCount * headDimension)]
            .reshaped(batch, count, headCount, headDimension)).transposed(0, 2, 1, 3)
        var rawKeys = projected[.ellipsis, (headCount * headDimension)...]
            .reshaped(batch, count, kvHeadCount, headDimension).transposed(0, 2, 1, 3)
        var positions = Self.positionRows(batch: batch, count: count, offsets: offsets, positionIds: positionIds)
        let rotatedQueries = rotate(indexQueries, positions: positions)
        if let cache {
            (rawKeys, positions) = cache.updateIndexer(
                keys: rawKeys,
                positions: MLX.broadcast(positions, to: [batch, kvHeadCount, count, 3])
            )
        }
        // Still record raw index keys below the budget: later decode can cross
        // the boundary, including on a fork restored from a short prompt.
        guard keys.dim(2) > budget else { return nil }
        precondition(rawKeys.dim(2) == keys.dim(2), "QSA needs indexer history aligned with target KV")
        let compressed = compressedKeys(rawKeys, positions: positions)
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: count, by: Q38SparseAttention.queryChunkSize) {
            let end = min(count, start + Q38SparseAttention.queryChunkSize)
            let indexQ = rotatedQueries[0..., 0..., start..<end, 0...]
            let scoreHeads = MLX.matmul(
                indexQ.asType(.float32), compressed.asType(.float32).swappedAxes(-1, -2)
            )
            let scores = MLX.maximum(scoreHeads, 0).sum(axis: 1) / sqrt(Float(headDimension))
            let selected = Q38SparseAttention.select(
                scores: scores, offsets: offsets.map { $0 + start }, budget: budget, ratio: ratio
            )
            let output = Q38SparseAttention.attend(
                queries: queries[0..., 0..., start..<end, 0...],
                keys: keys, values: values, indices: selected.indices, valid: selected.valid, scale: scale
            )
            if count > Q38SparseAttention.queryChunkSize { MLX.eval(output) }
            outputs.append(output)
        }
        return outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 2)
    }

    func compressedKeys(_ rawKeys: MLXArray, positions: MLXArray) -> MLXArray {
        let groups = rawKeys.dim(2) / ratio
        let complete = groups * ratio
        let pooled = rawKeys[0..., 0..., 0..<complete, 0...]
            .reshaped(rawKeys.dim(0), kvHeadCount, groups, ratio, headDimension)
            .asType(.float32).mean(axis: 3).asType(rawKeys.dtype)
        let firstPositions = positions[0..., 0..<1, 0..<complete, 0...]
            .reshaped(positions.dim(0), 1, groups, ratio, 3)[0..., 0..., 0..., 0, 0...]
        return rotate(keyNorm(pooled), positions: firstPositions)
    }

    private func rotate(_ value: MLXArray, positions: MLXArray) -> MLXArray {
        let ids = positions.squeezed(axis: 1).transposed(2, 0, 1)
        let (cos, sin) = rotary(positionIds: ids, dtype: value.dtype)
        let rotating = value[.ellipsis, 0..<rotaryDimensions]
        let rotated = applyRotaryPosEmb(rotating, rotating, cos: cos, sin: sin).0
        return rotaryDimensions == headDimension ? rotated
            : MLX.concatenated([rotated, value[.ellipsis, rotaryDimensions...]], axis: -1)
    }

    static func positionRows(
        batch: Int, count: Int, offsets: [Int], positionIds: MLXArray?
    ) -> MLXArray {
        if let positionIds, positionIds.ndim == 3 {
            return positionIds.transposed(1, 2, 0).expandedDimensions(axis: 1)
        }
        let positions = positionIds ?? (
            MLXArray(offsets.map(Int32.init)).reshaped(batch, 1)
                + MLXArray(0..<Int32(count)).reshaped(1, count)
        )
        return MLX.broadcast(positions.reshaped(batch, 1, count, 1), to: [batch, 1, count, 3])
    }
}
