import Foundation
import MLX
import MLXNN

/// Qwen4Exp's compressed micro-block selector. The mathematical reference is
/// Ollama's MIT-licensed qwen4_exp/qsa.go; see THIRD_PARTY_NOTICES.md.
enum Q38SparseAttention {
    static let queryChunkSize = 16

    static func select(
        scores: MLXArray,
        offsets: [Int],
        budget: Int,
        ratio: Int
    ) -> (indices: MLXArray, valid: MLXArray) {
        let batch = scores.dim(0)
        let count = scores.dim(1)
        let blocks = scores.dim(2)
        let selectedCount = min(blocks, budget / ratio)
        let visible = MLXArray(offsets.map(Int32.init)).reshaped(batch, 1, 1)
            + MLXArray(1...Int32(count)).reshaped(1, count, 1)
        let visibleBlocks = MLX.floorDivide(visible, MLXArray(Int32(ratio)))
        let blockIDs = MLXArray(0..<Int32(blocks)).reshaped(1, 1, blocks)
        let selected: MLXArray
        if blocks > selectedCount {
            let masked = MLX.which(blockIDs .< visibleBlocks, scores, -Float.infinity)
            selected = MLX.argPartition(-masked, kth: selectedCount - 1, axis: -1)[
                .ellipsis, 0..<selectedCount
            ].asType(.int32)
        } else {
            selected = MLX.broadcast(blockIDs, to: [batch, count, blocks])
        }
        let withinBlock = MLXArray(0..<Int32(ratio)).reshaped(1, 1, 1, ratio)
        let tokens = selected.expandedDimensions(axis: -1) * ratio + withinBlock
        let selectedValid = MLX.broadcast(
            (selected .< visibleBlocks).expandedDimensions(axis: -1),
            to: tokens.shape
        )
        let tailStart = visibleBlocks * ratio
        let tail = tailStart + MLXArray(0..<Int32(ratio - 1)).reshaped(1, 1, ratio - 1)
        let valid = MLX.concatenated([
            selectedValid.reshaped(batch, count, selectedCount * ratio),
            tail .< visible,
        ], axis: -1)
        let indices = MLX.concatenated([
            tokens.reshaped(batch, count, selectedCount * ratio), tail,
        ], axis: -1)
        return (MLX.which(valid, indices, MLXArray(Int32(0))), valid)
    }

    /// Each query gathers at most budget + ratio - 1 tokens. Query tiling and
    /// eager tile evaluation bound the temporary KV working set during prefill.
    static func attend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        indices: MLXArray,
        valid: MLXArray,
        scale: Float
    ) -> MLXArray {
        let batch = queries.dim(0)
        let heads = queries.dim(1)
        let kvHeads = keys.dim(1)
        let count = queries.dim(2)
        let dimension = queries.dim(3)
        let selectedKeys = gather(keys, indices: indices).asType(.float32)
        let selectedValues = gather(values, indices: indices).asType(.float32)
        let groupedQueries = queries.asType(.float32)
            .reshaped(batch, kvHeads, heads / kvHeads, count, 1, dimension)
        let groupedKeys = selectedKeys.expandedDimensions(axis: 2).swappedAxes(-1, -2)
        let scores = MLX.matmul(groupedQueries, groupedKeys).squeezed(axis: -2) * scale
        let masked = MLX.which(
            valid.expandedDimensions(axes: [1, 2]), scores, -Float.infinity
        )
        let probabilities = MLX.softmax(masked, axis: -1, precise: true)
        return MLX.matmul(
            probabilities.expandedDimensions(axis: -2),
            selectedValues.expandedDimensions(axis: 2)
        ).reshaped(batch, heads, count, dimension).asType(queries.dtype)
    }

    private static func gather(_ history: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = history.dim(0)
        let heads = history.dim(1)
        let length = history.dim(2)
        let dimension = history.dim(3)
        let batchOffsets = (MLXArray(0..<Int32(batch)) * length).reshaped(batch, 1, 1)
        let flat = history.transposed(1, 0, 2, 3).reshaped(heads, batch * length, dimension)
        return MLX.take(flat, indices + batchOffsets, axis: 1).transposed(1, 0, 2, 3, 4)
    }
}
