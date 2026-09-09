import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

final class MiniMaxH3TransformerBlock: Module {
    @ModuleInfo(key: "norm1") package var attentionNorm: RMSNorm
    @ModuleInfo(key: "attn") package var attention: MiniMaxH3Attention
    @ModuleInfo(key: "norm2") package var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "mlp") package var feedForward: MiniMaxH3FeedForward
    @ModuleInfo(key: "adaln_proj") package var adaLN: MiniMaxH3AdaLNProjection?
    var adaLNWeightsAvailable: Bool
    package var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled {
        didSet {
            attention.exactKernelMode = exactKernelMode
            feedForward.exactKernelMode = exactKernelMode
        }
    }
    package var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases) {
        didSet {
            attention.enabledExactKernelStages = enabledExactKernelStages
            feedForward.enabledExactKernelStages = enabledExactKernelStages
        }
    }
    package var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)? {
        didSet {
            attention.exactKernelDispatchHandler = exactKernelDispatchHandler
            feedForward.exactKernelDispatchHandler = exactKernelDispatchHandler
        }
    }
    package var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)? {
        didSet {
            attention.exactKernelFallbackHandler = exactKernelFallbackHandler
            feedForward.exactKernelFallbackHandler = exactKernelFallbackHandler
        }
    }

    package init(configuration: MiniMaxH3TransformerConfiguration, includeAdaLN: Bool) {
        self.adaLNWeightsAvailable = includeAdaLN
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._attention.wrappedValue = MiniMaxH3Attention(configuration: configuration)
        self._feedForwardNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._feedForward.wrappedValue = MiniMaxH3FeedForward(configuration: configuration)
        self._adaLN.wrappedValue = includeAdaLN
            ? MiniMaxH3AdaLNProjection(
                inputDimension: configuration.timeEmbeddingDimension,
                hiddenSize: configuration.hiddenSize,
                partCount: 6,
                modalityCount: 3
            )
            : nil
    }

    package var includesAdaLN: Bool {
        adaLNWeightsAvailable
    }

    package var supportsAffineQ8ExactKernels: Bool {
        miniMaxH3AffineQ8Weights(attention.queryKeyValue) != nil
            && miniMaxH3AffineQ8Weights(attention.output) != nil
            && miniMaxH3AffineQ8Weights(feedForward.input) != nil
            && miniMaxH3AffineQ8Weights(feedForward.output) != nil
            && attention.queryNorm.weight.dtype == .bfloat16
            && attention.keyNorm.weight.dtype == .bfloat16
            && feedForwardNorm.weight.dtype == .bfloat16
    }

    #if DEBUG
    package func feedForwardForBenchmark(_ value: MLXArray) -> MLXArray {
        feedForward(value)
    }

    package func feedForwardInputForBenchmark(_ value: MLXArray) -> MLXArray {
        feedForward.project(value)
    }

    package func feedForwardOutputForBenchmark(_ value: MLXArray) -> MLXArray {
        feedForward.projectOutput(value)
    }
    #endif

    package func discardAdaLNWeights() {
        guard adaLNWeightsAvailable else { return }
        update(modules: ModuleChildren.unflattened([
            ("adaln_proj", MiniMaxH3AdaLNProjection(discarded: ())),
        ]))
        adaLNWeightsAvailable = false
    }

    func prepareAttentionInput(
        _ value: MLXArray,
        modulation: MLXArray,
        adaLNIndices: MLXArray
    ) -> MLXArray {
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.attentionAdaLN) {
            if let prepared = MiniMaxH3FusedKernels.prepareAttentionInput(
                input: value,
                normWeight: attentionNorm.weight,
                modulation: modulation,
                rowIndices: adaLNIndices,
                eps: attentionNorm.eps
            ) {
                exactKernelDispatchHandler?(.attentionAdaLN)
                return prepared
            }
            exactKernelFallbackHandler?(
                .attentionAdaLN,
                "input=\(value.dtype):\(value.shape) "
                    + "norm=\(attentionNorm.weight.dtype) "
                    + "modulation=\(modulation.dtype):\(modulation.shape) "
                    + "indices=\(adaLNIndices.dtype):\(adaLNIndices.shape)"
            )
        }
        let parts = MLX.split(modulation, parts: 6, axis: -1)
        let shift = MLX.take(parts[0], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scale = MLX.take(parts[1], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        return attentionNorm(value) * (1 + scale) + shift
    }

    package func callAsFunction(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.gateAdaLN),
           let exact = exactBoundaryCall(
               value,
               timeEmbedding: timeEmbedding,
               adaLNIndices: adaLNIndices,
               rope: rope,
               cachedModulation: cachedModulation
           ) {
            return exact
        }
        let attended = attentionResidual(
            value,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            rope: rope,
            cachedModulation: cachedModulation
        )
        return feedForwardResidual(
            attended,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            cachedModulation: cachedModulation
        )
    }

    func exactBoundaryCall(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray? {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else {
                exactKernelFallbackHandler?(.gateAdaLN, "adaln-unavailable")
                return nil
            }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        let attentionOutput = attention(attentionInput, rope: rope)
        guard let boundary = MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
            residual: value,
            attentionOutput: attentionOutput,
            normWeight: feedForwardNorm.weight,
            modulation: completeModulation,
            rowIndices: adaLNIndices,
            eps: feedForwardNorm.eps
        ) else {
            exactKernelFallbackHandler?(
                .gateAdaLN,
                "residual=\(value.dtype):\(value.shape) "
                    + "attention=\(attentionOutput.dtype):\(attentionOutput.shape) "
                    + "norm=\(feedForwardNorm.weight.dtype) "
                    + "modulation=\(completeModulation.dtype):\(completeModulation.shape) "
                    + "indices=\(adaLNIndices.dtype):\(adaLNIndices.shape)"
            )
            return nil
        }
        exactKernelDispatchHandler?(.gateAdaLN)
        return boundary.residual
            + boundary.feedForwardGate * feedForward(boundary.feedForwardInput)
    }

    package func attentionResidual(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let modulation = MLX.split(completeModulation, parts: 6, axis: -1)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        return value + gateAttention * attention(attentionInput, rope: rope)
    }

    package func feedForwardResidual(
        _ attended: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 6, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let shiftFeedForward = MLX.take(modulation[3], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scaleFeedForward = MLX.take(modulation[4], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let gateFeedForward = MLX.take(modulation[5], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let feedForwardInput = feedForwardNorm(attended) * (1 + scaleFeedForward) + shiftFeedForward
        return attended + gateFeedForward * feedForward(feedForwardInput)
    }

    package func attentionProjection(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> [MLXArray] {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let modulation = MLX.split(completeModulation, parts: 6, axis: -1)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        return attention.project(attentionInput, rope: rope) + [gateAttention]
    }

    package func fastH3AttentionProjection(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?,
        compressionGate: MiniMaxH3FastH3CompressionGate
    ) -> [MLXArray] {
        fastH3AttentionProjection(
            value,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            rope: rope,
            cachedModulation: cachedModulation,
            compressionGateStorage: compressionGate.storage,
            compressionGateParameters: compressionGate.parameters
        )
    }

    package func fastH3AttentionProjection(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?,
        compressionGateStorage: MiniMaxH3FastH3CompressionGate.Storage,
        compressionGateParameters: [MLXArray]
    ) -> [MLXArray] {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let modulation = MLX.split(completeModulation, parts: 6, axis: -1)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        let projected = attention.project(attentionInput, rope: rope)
        let compressed = MiniMaxH3FastH3CompressionGate.project(
            attentionInput,
            storage: compressionGateStorage,
            parameters: compressionGateParameters
        ).reshaped(
            attentionInput.dim(0),
            attentionInput.dim(1),
            attention.heads,
            attention.headDimension
        ).transposed(0, 2, 1, 3).contiguous()
        return projected + [gateAttention, compressed]
    }

    package func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        maximumQueryTokens: Int,
        maximumHeadsPerKernel: Int?,
        maximumKernelsPerEvaluation: Int,
        dynamicSparseRequest: DynamicSparseAttentionRequest? = nil
    ) -> MLXArray {
        if let dynamicSparseRequest,
           let sparse = DynamicSparseAttention.call(
               queries: queries,
               keys: keys,
               values: values,
               request: dynamicSparseRequest,
               scale: attention.scale,
               maximumQueryTokens: maximumQueryTokens,
               maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
           ) {
            return sparse
        }
        return attention.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            maximumQueryTokens: maximumQueryTokens,
            maximumHeadsPerKernel: maximumHeadsPerKernel,
            maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
        )
    }

    package func attentionProjectionResidual(
        _ value: MLXArray,
        attended: MLXArray,
        gate: MLXArray
    ) -> MLXArray {
        value + gate * attention.projectOutput(attended)
    }

    package func postAttention(
        _ value: MLXArray,
        attended: MLXArray,
        gate: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        let projected = postAttentionProjection(
            value,
            attended: attended,
            gate: gate,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            cachedModulation: cachedModulation
        )
        return feedForwardProjectionResidual(
            projected[0],
            projected: projected[1],
            gate: projected[2]
        )
    }

    package func postAttentionProjection(
        _ value: MLXArray,
        attended: MLXArray,
        gate: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> [MLXArray] {
        let attentionOutput = attention.projectOutput(attended)
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.gateAdaLN) {
            let completeModulation: MLXArray
            if let cachedModulation {
                completeModulation = cachedModulation
            } else {
                guard let adaLN else {
                    preconditionFailure("MiniMax-H3 AdaLN cache is required")
                }
                completeModulation = adaLN.concatenated(timeEmbedding)
            }
            if let boundary = MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
                residual: value,
                attentionOutput: attentionOutput,
                normWeight: feedForwardNorm.weight,
                modulation: completeModulation,
                rowIndices: adaLNIndices,
                eps: feedForwardNorm.eps
            ) {
                exactKernelDispatchHandler?(.gateAdaLN)
                return [
                    boundary.residual,
                    feedForward.project(boundary.feedForwardInput),
                    boundary.feedForwardGate,
                ]
            }
            exactKernelFallbackHandler?(
                .gateAdaLN,
                "residual=\(value.dtype):\(value.shape) "
                    + "attention=\(attentionOutput.dtype):\(attentionOutput.shape) "
                    + "norm=\(feedForwardNorm.weight.dtype) "
                    + "modulation=\(completeModulation.dtype):\(completeModulation.shape) "
                    + "indices=\(adaLNIndices.dtype):\(adaLNIndices.shape)"
            )
        }
        let attentionResidual = value + gate * attentionOutput
        let feedForwardParts = feedForwardProjection(
            attentionResidual,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            cachedModulation: cachedModulation
        )
        return [
            attentionResidual,
            feedForwardParts[0],
            feedForwardParts[1],
        ]
    }

    package func feedForwardProjection(
        _ attended: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> [MLXArray] {
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 6, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let shiftFeedForward = MLX.take(modulation[3], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scaleFeedForward = MLX.take(modulation[4], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let gateFeedForward = MLX.take(modulation[5], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let feedForwardInput = feedForwardNorm(attended) * (1 + scaleFeedForward) + shiftFeedForward
        return [feedForward.project(feedForwardInput), gateFeedForward]
    }

    package func feedForwardProjectionResidual(
        _ attended: MLXArray,
        projected: MLXArray,
        gate: MLXArray
    ) -> MLXArray {
        attended + gate * feedForward.projectOutput(projected)
    }

    package func precomputeModulation(timeEmbedding: MLXArray) -> MLXArray {
        guard adaLNWeightsAvailable, let adaLN else {
            preconditionFailure("MiniMax-H3 AdaLN weights are not loaded")
        }
        return adaLN.concatenated(timeEmbedding)
    }
}
