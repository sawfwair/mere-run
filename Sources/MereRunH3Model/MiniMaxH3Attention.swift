import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

final class MiniMaxH3Attention: Module {
    package let heads: Int
    package let headDimension: Int
    package let innerDimension: Int
    package let scale: Float

    @ModuleInfo(key: "qkv_proj") package var queryKeyValue: Linear
    @ModuleInfo(key: "q_norm") package var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") package var keyNorm: RMSNorm
    @ModuleInfo(key: "out_proj") package var output: Linear
    package var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled
    package var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases)
    package var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)?
    package var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)?

    package init(configuration: MiniMaxH3TransformerConfiguration) {
        self.heads = configuration.attentionHeadCount
        self.headDimension = configuration.attentionHeadDimension
        self.innerDimension = heads * headDimension
        self.scale = 1 / sqrt(Float(headDimension))
        self._queryKeyValue.wrappedValue = Linear(
            configuration.hiddenSize,
            3 * innerDimension,
            bias: false
        )
        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: headDimension,
            eps: configuration.queryKeyNormEpsilon
        )
        self._keyNorm.wrappedValue = RMSNorm(
            dimensions: headDimension,
            eps: configuration.queryKeyNormEpsilon
        )
        self._output.wrappedValue = Linear(innerDimension, configuration.hiddenSize, bias: false)
    }

    package func callAsFunction(_ input: MLXArray, rope: MiniMaxH3RotaryEmbedding?) -> MLXArray {
        let projected = project(input, rope: rope)
        let attended = scaledDotProductAttention(
            queries: projected[0],
            keys: projected[1],
            values: projected[2],
            maximumQueryTokens: nil,
            maximumHeadsPerKernel: nil
        )
        return projectOutput(attended)
    }

    package func project(_ input: MLXArray, rope: MiniMaxH3RotaryEmbedding?) -> [MLXArray] {
        if exactKernelMode == .affineQ8,
           enabledExactKernelStages.contains(.qkvProjection) {
            if let rope, let weights = miniMaxH3AffineQ8Weights(queryKeyValue) {
                if let projected = MiniMaxH3FusedKernels.projectHeadMajorQKVAffineInt8(
                    input: input,
                    weightCodes: weights.codes,
                    weightScales: weights.scales,
                    weightBiases: weights.biases,
                    queryNormWeight: queryNorm.weight,
                    keyNormWeight: keyNorm.weight,
                    ropeCosine: rope.cosine,
                    ropeSine: rope.sine,
                    eps: queryNorm.eps
                ) {
                    exactKernelDispatchHandler?(.qkvProjection)
                    return [projected.query, projected.key, projected.value]
                }
                exactKernelFallbackHandler?(
                    .qkvProjection,
                    "input=\(input.dtype):\(input.shape) q_norm=\(queryNorm.weight.dtype) "
                        + "k_norm=\(keyNorm.weight.dtype) rope=\(rope.cosine.dtype):"
                        + "\(rope.cosine.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.qkvProjection, "weight-or-rope-contract")
            }
        }
        let globalProjection = queryKeyValue(input)
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.qkvLayout) {
            if let rope {
                if let projected = MiniMaxH3FusedKernels.prepareHeadMajorQKV(
                    projected: globalProjection,
                    queryNormWeight: queryNorm.weight,
                    keyNormWeight: keyNorm.weight,
                    ropeCosine: rope.cosine,
                    ropeSine: rope.sine,
                    eps: queryNorm.eps
                ) {
                    exactKernelDispatchHandler?(.qkvLayout)
                    return [projected.query, projected.key, projected.value]
                }
                exactKernelFallbackHandler?(
                    .qkvLayout,
                    "projection=\(globalProjection.dtype):\(globalProjection.shape) "
                        + "q_norm=\(queryNorm.weight.dtype) k_norm=\(keyNorm.weight.dtype) "
                        + "rope=\(rope.cosine.dtype):\(rope.cosine.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.qkvLayout, "rope-unavailable")
            }
        }
        // The MLX-Serve artifact deinterleaves the released checkpoint's
        // per-head rows into three global Q/K/V slabs before quantization.
        // Do not apply the raw-checkpoint interleave a second time here.
        let projected = miniMaxH3SplitProjectedQKV(
            globalProjection, heads: heads, headDimension: headDimension
        )
        var query = queryNorm(projected[0])
        var key = keyNorm(projected[1])
        if let rope {
            query = rope.apply(query)
            key = rope.apply(key)
        }
        // Chunked SDPA reuses the complete K/V tensors for every query slice.
        // Materialize the transposed head-major views once so MLX does not
        // repack the same strided K/V storage inside every attention kernel.
        query = query.transposed(0, 2, 1, 3).contiguous()
        key = key.transposed(0, 2, 1, 3).contiguous()
        let value = projected[2].transposed(0, 2, 1, 3).contiguous()
        return [query, key, value]
    }

    package func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        maximumQueryTokens: Int?,
        maximumHeadsPerKernel: Int?,
        maximumKernelsPerEvaluation: Int = 1
    ) -> MLXArray {
        if let maximumQueryTokens { precondition(maximumQueryTokens > 0) }
        if let maximumHeadsPerKernel { precondition(maximumHeadsPerKernel > 0) }
        let queryChunkSize = min(maximumQueryTokens ?? queries.dim(2), queries.dim(2))
        let headChunkSize = min(maximumHeadsPerKernel ?? queries.dim(1), queries.dim(1))
        guard queryChunkSize < queries.dim(2) || headChunkSize < queries.dim(1) else {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
        }

        precondition(maximumKernelsPerEvaluation > 0)
        var headOutputs: [MLXArray] = []
        headOutputs.reserveCapacity((queries.dim(1) + headChunkSize - 1) / headChunkSize)
        var pending: [MLXArray] = []
        pending.reserveCapacity(maximumKernelsPerEvaluation)
        for headStart in stride(from: 0, to: queries.dim(1), by: headChunkSize) {
            let headEnd = min(headStart + headChunkSize, queries.dim(1))
            var queryOutputs: [MLXArray] = []
            queryOutputs.reserveCapacity((queries.dim(2) + queryChunkSize - 1) / queryChunkSize)
            for queryStart in stride(from: 0, to: queries.dim(2), by: queryChunkSize) {
                let queryEnd = min(queryStart + queryChunkSize, queries.dim(2))
                let chunk = MLXFast.scaledDotProductAttention(
                    queries: queries[
                        0..., headStart..<headEnd, queryStart..<queryEnd, 0...
                    ],
                    keys: keys[0..., headStart..<headEnd, 0..., 0...],
                    values: values[0..., headStart..<headEnd, 0..., 0...],
                    scale: scale,
                    mask: .none
                )
                queryOutputs.append(chunk)
                pending.append(chunk)
                if pending.count == maximumKernelsPerEvaluation {
                    MLX.eval(pending)
                    pending.removeAll(keepingCapacity: true)
                }
            }
            headOutputs.append(MLX.concatenated(queryOutputs, axis: 2))
        }
        if !pending.isEmpty {
            MLX.eval(pending)
        }
        return MLX.concatenated(headOutputs, axis: 1)
    }

    package func projectOutput(_ attended: MLXArray) -> MLXArray {
        if exactKernelMode == .affineQ8,
           enabledExactKernelStages.contains(.attentionOutput) {
            if let weights = miniMaxH3AffineQ8Weights(output) {
                if let projected = MiniMaxH3FusedKernels.projectHeadMajorAttentionAffineInt8(
                    attention: attended,
                    weightCodes: weights.codes,
                    weightScales: weights.scales,
                    weightBiases: weights.biases
                ) {
                    exactKernelDispatchHandler?(.attentionOutput)
                    return projected
                }
                exactKernelFallbackHandler?(
                    .attentionOutput,
                    "attention=\(attended.dtype):\(attended.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.attentionOutput, "weight-contract")
            }
        }
        let batch = attended.dim(0)
        let sequence = attended.dim(2)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, innerDimension))
    }
}
