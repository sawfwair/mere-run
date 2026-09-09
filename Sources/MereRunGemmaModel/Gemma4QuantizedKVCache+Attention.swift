import Foundation
import MLX
import MLXFast

extension Gemma4QuantizedKVCache {
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
        guard Self.supportsFastKernels else {
            return nil
        }
        guard queries.dim(2) == 1 else {
            return nil
        }

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

        if let quantizedKeys, let quantizedValues {
            let totalTokens = quantizedKeys.tokenCount
            if totalTokens > 0 {
                var start = 0
                while start < totalTokens {
                    let end = min(start + Self.decodeChunkSize, totalTokens)
                    guard applyQuantizedChunkFused(
                        queries: queries32,
                        keyState: quantizedKeys,
                        valueState: quantizedValues,
                        tokenRange: start..<end,
                        repeats: repeats,
                        scale: scale,
                        runningWeighted: &runningWeighted,
                        runningNormalizer: &runningNormalizer,
                        runningMax: &runningMax
                    ) else {
                        return nil
                    }
                    start = end
                }
            }
        }

        guard let runningWeighted, let runningNormalizer else {
            return nil
        }
        return (runningWeighted / runningNormalizer).asType(queries.dtype)
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

        if let quantizedKeys, let quantizedValues {
            let totalTokens = quantizedKeys.tokenCount
            if totalTokens > 0 {
                var start = 0
                while start < totalTokens {
                    let end = min(start + Self.decodeChunkSize, totalTokens)
                    let chunkKeys = quantizedKeys.dequantized(tokenRange: start..<end)
                    let chunkValues = quantizedValues.dequantized(tokenRange: start..<end)
                    applyChunk(
                        queries: queries32,
                        keys: chunkKeys,
                        values: chunkValues,
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

    func applyQuantizedChunkFused(
        queries: MLXArray,
        keyState: Gemma4QuantizedTensorState,
        valueState: Gemma4QuantizedTensorState,
        tokenRange: Range<Int>,
        repeats: Int,
        scale: Float,
        runningWeighted: inout MLXArray?,
        runningNormalizer: inout MLXArray?,
        runningMax: inout MLXArray?
    ) -> Bool {
        guard let keyBiases = keyState.biases,
              let valueBiases = valueState.biases else {
            return false
        }
        guard let fusedChunkDecodeKernel = makeFusedChunkDecodeKernel(
            keyState: keyState,
            valueState: valueState,
            repeats: repeats,
            dim: queries.dim(3)
        ) else {
            return false
        }

        let tokenCounts = MLXArray([UInt32(tokenRange.count), UInt32(tokenRange.lowerBound)])
        let weightedAndStats = fusedChunkDecodeKernel(
            [queries, keyState.weight, keyState.scales, keyBiases, valueState.weight, valueState.scales, valueBiases, scale, tokenCounts],
            template: fusedChunkDecodeTemplateArguments(
                keyState: keyState,
                valueState: valueState,
                repeats: repeats,
                dim: queries.dim(3)
            ),
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
            return true
        }

        let mergedMax = MLX.maximum(currentMax, chunkMax)
        let currentScale = exp(currentMax - mergedMax)
        let chunkScale = exp(chunkMax - mergedMax)
        runningWeighted = currentWeighted * currentScale + chunkWeighted * chunkScale
        runningNormalizer = currentNormalizer * currentScale + chunkNormalizer * chunkScale
        runningMax = mergedMax
        return true
    }

    func makeScoreKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        return Gemma4AffineFastKernels.scoreKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    func makeWeightedValueKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.weightedValueKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    func makeWeightedValueFromScoresKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.weightedValueFromScoresKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    func makeWeightedValueAndNormalizerFromScoresKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.weightedValueAndNormalizerFromScoresKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    func makeFusedChunkDecodeKernel(
        keyState: Gemma4QuantizedTensorState,
        valueState: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.fusedChunkDecodeKernel(
            keyBits: keyState.bits,
            valueBits: valueState.bits,
            groupSize: keyState.groupSize,
            dim: dim,
            keyPackedWidth: keyState.packedWidth,
            valuePackedWidth: valueState.packedWidth,
            keyGroupCount: keyState.groupCount,
            valueGroupCount: valueState.groupCount,
            repeats: repeats
        )
    }

    func scoreTemplateArguments(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("Bits", state.bits),
            ("GroupSize", state.groupSize),
            ("Dim", dim),
            ("PackedWidth", state.packedWidth),
            ("GroupCount", state.groupCount),
            ("RepeatCount", repeats),
        ]
    }

    func valueTemplateArguments(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("Bits", state.bits),
            ("GroupSize", state.groupSize),
            ("Dim", dim),
            ("PackedWidth", state.packedWidth),
            ("GroupCount", state.groupCount),
            ("RepeatCount", repeats),
        ]
    }

    func weightedValueFromScoresTemplateArguments(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("Bits", state.bits),
            ("GroupSize", state.groupSize),
            ("Dim", dim),
            ("PackedWidth", state.packedWidth),
            ("GroupCount", state.groupCount),
            ("RepeatCount", repeats),
        ]
    }

    func fusedChunkDecodeTemplateArguments(
        keyState: Gemma4QuantizedTensorState,
        valueState: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("KeyBits", keyState.bits),
            ("ValueBits", valueState.bits),
            ("GroupSize", keyState.groupSize),
            ("Dim", dim),
            ("KeyPackedWidth", keyState.packedWidth),
            ("ValuePackedWidth", valueState.packedWidth),
            ("KeyGroupCount", keyState.groupCount),
            ("ValueGroupCount", valueState.groupCount),
            ("RepeatCount", repeats),
        ]
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
