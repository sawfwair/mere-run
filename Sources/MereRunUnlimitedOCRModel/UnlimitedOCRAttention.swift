import Foundation
import MLX
import MLXFast

package final class UnlimitedOCRCache {
    package init() {}
    package var keys: MLXArray?
    package var values: MLXArray?
    package var indexKeys: MLXArray?
}

package enum UnlimitedOCRAttention {
    /// Each query attends at most 12 remote blocks and 128 local tokens. Gathering
    /// selected rows avoids a dense [heads, sequence, sequence] attention matrix.
    package static func attend(
        q: MLXArray, k: MLXArray, v: MLXArray, indexQ: MLXArray, indexK: MLXArray,
        head: MLXArray, state: UnlimitedOCRPageState, offset: Int,
        scale: Float, floor: Float, sigma: Float, blockSize: Int = 32,
        localBlocks: Int = 4, budget: Int = 12, topM: Int = 16
    ) -> MLXArray {
        let total = k.dim(1)
        let count = q.dim(1)
        let dim = q.dim(2)
        let blocks = (total + blockSize - 1) / blockSize
        let padding = blocks * blockSize - total
        let indexDim = indexK.dim(-1)
        let padded = padding == 0 ? indexK : concatenated([
            indexK, MLXArray.zeros([padding, indexDim], dtype: .float32)
        ], axis: 0)
        let grouped = padded.reshaped(blocks, blockSize, indexDim)
        let positions = MLXArray(0..<(blocks * blockSize)).reshaped(blocks, blockSize)
        let validKeys = positions .< total
        let summaryLogits = clip(UnlimitedOCRMath.scaleRouterScores((grouped * head).sum(axis: -1), dimension: indexDim), min: -10000, max: 10000)
        let summaryWeights = softmax(which(validKeys, summaryLogits, MLXArray(-Float.infinity)), axis: -1)
        let summaries = (grouped * summaryWeights.expandedDimensions(axis: -1)).sum(axis: 1)
        let imagePages = state.imagePages
        let pageRanges: [(Int32, Int32)] = (0..<blocks).map { block in
            let pages = imagePages[(block * blockSize)..<min(total, (block + 1) * blockSize)].filter { $0 >= 0 }
            return (pages.min() ?? -1, pages.max() ?? -1)
        }
        var outputs: [MLXArray] = []
        // Bound the gathered KV working set, including the 1024px SAM prefill.
        for start in stride(from: 0, to: count, by: 16) {
            let end = min(start + 16, count)
            let rows = end - start
            let queryPositions = MLXArray((offset + start)..<(offset + end))
            var remote = [Bool](repeating: false, count: rows * blocks)
            for row in 0..<rows {
                let position = offset + start + row
                let page = state.queryPages[position]
                let allowed = pageRanges.map { $0.0 >= 0 && page >= $0.0 && page <= $0.1 }
                // The upstream fallback applies after intersecting with remote eligibility.
                let maxBlock = position / blockSize - localBlocks
                let hasCandidate = (0..<blocks).contains { $0 <= maxBlock && allowed[$0] }
                for block in 0..<blocks {
                    remote[row * blocks + block] = block <= maxBlock && (!hasCandidate || allowed[block])
                }
            }
            let candidate = MLXArray(remote).reshaped(rows, blocks)
            let scores = clip(UnlimitedOCRMath.scaleRouterScores(matmul(indexQ[start..<end], summaries.T), dimension: indexDim), min: -10000, max: 10000)
            let masked = which(candidate, UnlimitedOCRMath.scaleRouterScores(scores, dimension: indexDim, factor: scale), MLXArray(Float(-10000)))
            let mCount = min(topM, blocks)
            let order = argSort(-masked, axis: -1)[.ellipsis, ..<mCount]
            let sorted = takeAlong(masked, order, axis: -1)
            let prior = UnlimitedOCRMath.entmax(sorted)
            let selectedCount = min(budget, mCount)
            let selected = order[.ellipsis, ..<selectedCount]
            let valid = takeAlong(candidate, selected, axis: -1)
            let logPrior = log(maximum(prior[.ellipsis, ..<selectedCount], floor))
            let denominator = maximum(valid.sum(axis: -1, keepDims: true).asType(.float32), 1)
            let center = which(valid, logPrior, MLXArray(Float(0))).sum(axis: -1, keepDims: true) / denominator
            let bias = (logPrior - center) / sigma
            let remotePositions = (selected.expandedDimensions(axis: -1) * blockSize
                + MLXArray(0..<blockSize)).reshaped(rows, -1)
            let repeatedBias = broadcast(bias.expandedDimensions(axis: -1), to: [rows, selectedCount, blockSize]).reshaped(rows, -1)
            let repeatedValid = broadcast(valid.expandedDimensions(axis: -1), to: [rows, selectedCount, blockSize]).reshaped(rows, -1)
            let queryPos = queryPositions.expandedDimensions(axis: -1)
            let window = blockSize * localBlocks
            let remoteValid = repeatedValid .&& (remotePositions .< total)
                .&& (remotePositions .<= queryPos) .&& (remotePositions .<= queryPos - window)
            let localPositions = queryPos - window + 1 + MLXArray(0..<window)
            let indices = concatenated([remotePositions, localPositions], axis: -1)
            var mask = concatenated([
                which(remoteValid, repeatedBias, MLXArray(-Float.infinity)),
                which(localPositions .>= 0, MLXArray(Float(0)), MLXArray(-Float.infinity))
            ], axis: -1)
            // Transformers prefill uses the query dtype for Q/K/V and bias.
            // Its incremental decode explicitly promotes all four to FP32.
            let attentionType: DType = count > 1 ? q.dtype : .float32
            mask = mask.asType(attentionType)
            let safeIndices = clip(indices, min: 0, max: total - 1)
            let selectedK = k.take(safeIndices, axis: 1).transposed(1, 0, 2, 3)
            let selectedV = v.take(safeIndices, axis: 1).transposed(1, 0, 2, 3)
            let queries = q[0..., start..<end, 0...].transposed(1, 0, 2).expandedDimensions(axis: 2)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries.asType(attentionType), keys: selectedK.asType(attentionType), values: selectedV.asType(attentionType),
                scale: 1 / sqrt(Float(dim)), mask: mask.reshaped(rows, 1, 1, -1)
            ).asType(q.dtype).squeezed(axis: 2)
            eval(output)
            outputs.append(output)
        }
        return concatenated(outputs, axis: 0).transposed(1, 0, 2)
    }
}
