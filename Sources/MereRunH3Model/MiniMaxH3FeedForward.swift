import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

final class MiniMaxH3FeedForward: Module {
    @ModuleInfo(key: "fc1") package var input: Linear
    @ModuleInfo(key: "fc2") package var output: Linear
    package var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled
    package var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases)
    package var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)?
    package var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)?

    package init(configuration: MiniMaxH3TransformerConfiguration) {
        self._input.wrappedValue = Linear(
            configuration.hiddenSize,
            2 * configuration.feedForwardSize,
            bias: false
        )
        self._output.wrappedValue = Linear(
            configuration.feedForwardSize,
            configuration.hiddenSize,
            bias: false
        )
    }

    package func callAsFunction(_ value: MLXArray) -> MLXArray {
        projectOutput(project(value))
    }

    package func project(_ value: MLXArray) -> MLXArray {
        if exactKernelMode.usesAffineQ8FeedForward,
           enabledExactKernelStages.contains(.feedForwardInput) {
            if let weights = miniMaxH3AffineQ8Weights(input) {
                let projected = exactKernelMode.usesTiledAffineQ8FeedForward
                    ? MiniMaxH3FusedKernels.projectFeedForwardInputAffineInt8SwiGLUTiled(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                    : MiniMaxH3FusedKernels.projectFeedForwardInputAffineInt8SwiGLU(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                if let projected {
                    exactKernelDispatchHandler?(.feedForwardInput)
                    return projected
                }
                exactKernelFallbackHandler?(
                    .feedForwardInput,
                    "input=\(value.dtype):\(value.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.feedForwardInput, "weight-contract")
            }
        }
        let parts = MLX.split(input(value), parts: 2, axis: -1)
        return MLXNN.silu(parts[0]) * parts[1]
    }

    package func projectOutput(_ value: MLXArray) -> MLXArray {
        if exactKernelMode.usesAffineQ8FeedForward,
           enabledExactKernelStages.contains(.feedForwardOutput) {
            if let weights = miniMaxH3AffineQ8Weights(output) {
                let projected = exactKernelMode.usesTiledAffineQ8FeedForward
                    ? MiniMaxH3FusedKernels.projectFeedForwardOutputAffineInt8Tiled(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                    : MiniMaxH3FusedKernels.projectFeedForwardOutputAffineInt8(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                if let projected {
                    exactKernelDispatchHandler?(.feedForwardOutput)
                    return projected
                }
                exactKernelFallbackHandler?(
                    .feedForwardOutput,
                    "input=\(value.dtype):\(value.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.feedForwardOutput, "weight-contract")
            }
        }
        return output(value)
    }
}
