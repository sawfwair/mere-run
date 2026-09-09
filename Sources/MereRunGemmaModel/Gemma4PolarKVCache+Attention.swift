import Foundation
import MLX
import MLXFast

extension Gemma4PolarKVCache {
    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        guard queries.dim(2) == 1 else {
            return nil
        }

        if let fused = fusedSpecializedAttention(queries: queries, repeats: repeats, scale: scale) {
            return fused
        }
        return chunkedSpecializedAttention(queries: queries, repeats: repeats, scale: scale)
    }

    func fusedSpecializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        guard Self.supportsFastKernels,
              queries.dim(2) == 1,
              leadingKeys == nil,
              leadingValues == nil,
              let polarKeys,
              let polarValues else {
            return nil
        }

        if let scoreValueOutput = scoreValueSpecializedAttention(
            queries: queries,
            keyState: polarKeys,
            valueState: polarValues,
            repeats: repeats,
            scale: scale
        ) {
            return scoreValueOutput
        }

        let queries32 = queries.asType(.float32)
        let rotatedQueries = MLX.matmul(queries32, polarKeys.rotationTransposed)
        var runningWeighted: MLXArray?
        var runningNormalizer: MLXArray?
        var runningMax: MLXArray?

        let totalTokens = polarKeys.tokenCount
        var start = 0
        while start < totalTokens {
            let end = min(start + Self.decodeChunkSize, totalTokens)
            applyPolarChunkFused(
                queries: rotatedQueries,
                keyState: polarKeys,
                valueState: polarValues,
                tokenRange: start..<end,
                repeats: repeats,
                scale: scale,
                runningWeighted: &runningWeighted,
                runningNormalizer: &runningNormalizer,
                runningMax: &runningMax
            )
            start = end
        }

        guard let runningWeighted, let runningNormalizer else {
            return nil
        }
        return MLX.matmul(runningWeighted / runningNormalizer, polarValues.rotation).asType(queries.dtype)
    }

    func scoreValueSpecializedAttention(
        queries: MLXArray,
        keyState: Gemma4PolarTensorState,
        valueState: Gemma4PolarTensorState,
        repeats: Int,
        scale: Float
    ) -> MLXArray? {
        let totalTokens = keyState.tokenCount
        guard totalTokens > 0 else {
            return nil
        }

        let queries32 = queries.asType(.float32)
        let rotatedQueries = MLX.matmul(queries32, keyState.rotationTransposed)
        let tokenCounts = MLXArray([UInt32(totalTokens)])
        let scoreKernel = Gemma4PolarFastKernels.scoreKernel(
            bits: keyState.bits,
            dim: queries.dim(3),
            packedWidth: keyState.packedWidth,
            repeats: repeats
        )
        let scores = scoreKernel(
            [rotatedQueries, keyState.packed, keyState.norms, keyState.centroids, scale, tokenCounts],
            template: [
                ("Bits", keyState.bits),
                ("Dim", queries.dim(3)),
                ("PackedWidth", keyState.packedWidth),
                ("RepeatCount", repeats),
            ],
            grid: (32, queries.dim(1), queries.dim(0) * totalTokens),
            threadGroup: (32, 1, 1),
            outputShapes: [[queries.dim(0), queries.dim(1), 1, totalTokens]],
            outputDTypes: [.float32]
        )[0]

        let scoreMax = scores.max(axis: -1, keepDims: true)
        let weights = exp(scores - scoreMax)
        let normalizer = weights.sum(axis: -1, keepDims: true)
        let weightedValueKernel = Gemma4PolarFastKernels.weightedValueKernel(
            bits: valueState.bits,
            dim: queries.dim(3),
            packedWidth: valueState.packedWidth,
            repeats: repeats
        )
        let weighted = weightedValueKernel(
            [weights, valueState.packed, valueState.norms, valueState.centroids, tokenCounts],
            template: [
                ("Bits", valueState.bits),
                ("Dim", queries.dim(3)),
                ("PackedWidth", valueState.packedWidth),
                ("RepeatCount", repeats),
            ],
            grid: (32, queries.dim(1), queries.dim(0) * queries.dim(3)),
            threadGroup: (32, 1, 1),
            outputShapes: [[queries.dim(0), queries.dim(1), 1, queries.dim(3)]],
            outputDTypes: [.float32]
        )[0]

        return MLX.matmul(weighted / normalizer, valueState.rotation).asType(queries.dtype)
    }

    func chunkedSpecializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        let queries32 = queries.asType(.float32)

        var runningWeighted: MLXArray?
        var runningNormalizer: MLXArray?
        var runningMax: MLXArray?

        if let leadingKeys, let leadingValues {
            applyChunk(
                queries: queries32,
                keys: leadingKeys,
                values: leadingValues,
                repeats: repeats,
                scale: scale,
                runningWeighted: &runningWeighted,
                runningNormalizer: &runningNormalizer,
                runningMax: &runningMax
            )
        }

        if let polarKeys, let polarValues {
            let totalTokens = polarKeys.tokenCount
            if totalTokens > 0 {
                var start = 0
                while start < totalTokens {
                    let end = min(start + Self.decodeChunkSize, totalTokens)
                    applyChunk(
                        queries: queries32,
                        keys: polarKeys.dequantized(tokenRange: start..<end),
                        values: polarValues.dequantized(tokenRange: start..<end),
                        repeats: repeats,
                        scale: scale,
                        runningWeighted: &runningWeighted,
                        runningNormalizer: &runningNormalizer,
                        runningMax: &runningMax
                    )
                    start = end
                }
            }
        }

        guard let runningWeighted, let runningNormalizer else {
            return nil
        }
        return (runningWeighted / runningNormalizer).asType(queries.dtype)
    }

    func applyPolarChunkFused(
        queries: MLXArray,
        keyState: Gemma4PolarTensorState,
        valueState: Gemma4PolarTensorState,
        tokenRange: Range<Int>,
        repeats: Int,
        scale: Float,
        runningWeighted: inout MLXArray?,
        runningNormalizer: inout MLXArray?,
        runningMax: inout MLXArray?
    ) {
        let tokenCounts = MLXArray([UInt32(tokenRange.count), UInt32(tokenRange.lowerBound)])
        let fusedChunkDecodeKernel = Gemma4PolarFastKernels.fusedChunkDecodeKernel(
            bits: keyState.bits,
            dim: queries.dim(3),
            packedWidth: keyState.packedWidth,
            repeats: repeats
        )

        let weightedAndStats = fusedChunkDecodeKernel(
            [queries, keyState.packed, keyState.norms, valueState.packed, valueState.norms, keyState.centroids, scale, tokenCounts],
            template: [
                ("Bits", keyState.bits),
                ("Dim", queries.dim(3)),
                ("PackedWidth", keyState.packedWidth),
                ("RepeatCount", repeats),
            ],
            grid: (32, queries.dim(1), queries.dim(0) * queries.dim(3)),
            threadGroup: (32, 1, 1),
            outputShapes: [
                [queries.dim(0), queries.dim(1), 1, queries.dim(3)],
                [queries.dim(0), queries.dim(1), 1, 1],
                [queries.dim(0), queries.dim(1), 1, 1],
            ],
            outputDTypes: [.float32, .float32, .float32]
        )
        let chunkWeighted = weightedAndStats[0]
        let chunkNormalizer = weightedAndStats[1]
        let chunkMax = weightedAndStats[2]

        guard let currentWeighted = runningWeighted,
              let currentNormalizer = runningNormalizer,
              let currentMax = runningMax else {
            runningWeighted = chunkWeighted
            runningNormalizer = chunkNormalizer
            runningMax = chunkMax
            return
        }

        let mergedMax = MLX.maximum(currentMax, chunkMax)
        let currentScale = exp(currentMax - mergedMax)
        let chunkScale = exp(chunkMax - mergedMax)
        runningWeighted = currentWeighted * currentScale + chunkWeighted * chunkScale
        runningNormalizer = currentNormalizer * currentScale + chunkNormalizer * chunkScale
        runningMax = mergedMax
    }

    func applyChunk(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        repeats: Int,
        scale: Float,
        runningWeighted: inout MLXArray?,
        runningNormalizer: inout MLXArray?,
        runningMax: inout MLXArray?
    ) {
        guard keys.dim(2) > 0 else {
            return
        }

        var broadcastKeys = keys.asType(.float32)
        var broadcastValues = values.asType(.float32)
        if repeats > 1 {
            broadcastKeys = MLX.repeated(broadcastKeys, count: repeats, axis: 1)
            broadcastValues = MLX.repeated(broadcastValues, count: repeats, axis: 1)
        }

        let scores = MLX.matmul(queries, broadcastKeys.transposed(0, 1, 3, 2)) * MLXArray(scale)
        let chunkMax = scores.max(axis: -1, keepDims: true)
        let shifted = exp(scores - chunkMax)
        let chunkWeighted = MLX.matmul(shifted, broadcastValues)
        let chunkNormalizer = shifted.sum(axis: -1, keepDims: true)

        guard let currentWeighted = runningWeighted,
              let currentNormalizer = runningNormalizer,
              let currentMax = runningMax else {
            runningWeighted = chunkWeighted
            runningNormalizer = chunkNormalizer
            runningMax = chunkMax
            return
        }

        let mergedMax = MLX.maximum(currentMax, chunkMax)
        let currentScale = exp(currentMax - mergedMax)
        let chunkScale = exp(chunkMax - mergedMax)
        runningWeighted = currentWeighted * currentScale + chunkWeighted * chunkScale
        runningNormalizer = currentNormalizer * currentScale + chunkNormalizer * chunkScale
        runningMax = mergedMax
    }

}
