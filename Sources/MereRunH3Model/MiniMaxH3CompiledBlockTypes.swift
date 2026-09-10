import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

typealias MiniMaxH3CompiledBlockForward = @Sendable ([MLXArray]) -> [MLXArray]

struct MiniMaxH3CompiledBlockForwards {
    package let attentionProjection: MiniMaxH3CompiledBlockForward
    package let fastH3AttentionProjection: MiniMaxH3CompiledBlockForward?
    package let attentionOutput: MiniMaxH3CompiledBlockForward
    package let feedForwardProjection: MiniMaxH3CompiledBlockForward
    package let feedForwardOutput: MiniMaxH3CompiledBlockForward
    package let postAttentionProjection: MiniMaxH3CompiledBlockForward
    package let postAttention: MiniMaxH3CompiledBlockForward
}

#if DEBUG
/// Test-only harness for timing the exact production H3 block schedules at
/// realistic packed-row counts without loading the complete checkpoint.
package final class MiniMaxH3BlockScheduleBenchmark {
    package enum Schedule {
        case splitPostAttention
        case fusedFeedForward
        case fusedPostAttention
    }

    package let rowCount: Int
    let maximumQueryTokens: Int
    let maximumHeadsPerKernel: Int?
    let maximumKernelsPerEvaluation: Int
    let block: MiniMaxH3TransformerBlock
    let originalParameters: [(String, MLXArray)]
    let forwards: MiniMaxH3CompiledBlockForwards
    let fusedFeedForward: MiniMaxH3CompiledBlockForward
    let hidden: MLXArray
    let timeEmbedding: MLXArray
    let adaLNIndices: MLXArray
    let rope: MiniMaxH3RotaryEmbedding
    let cachedModulation: MLXArray

    package init(
        rowCount: Int,
        maximumQueryTokens: Int,
        maximumKernelsPerEvaluation: Int,
        maximumHeadsPerKernel: Int? = nil,
        dtype: DType = .bfloat16,
        weightSeed: UInt64? = nil,
        inputSeed: UInt64? = nil
    ) {
        precondition(rowCount > 0)
        precondition(maximumQueryTokens > 0)
        precondition(maximumKernelsPerEvaluation > 0)
        if let maximumHeadsPerKernel { precondition(maximumHeadsPerKernel > 0) }
        self.rowCount = rowCount
        self.maximumQueryTokens = maximumQueryTokens
        self.maximumHeadsPerKernel = maximumHeadsPerKernel
        self.maximumKernelsPerEvaluation = maximumKernelsPerEvaluation

        let configuration = MiniMaxH3TransformerConfiguration()
        if let weightSeed { MLXRandom.seed(weightSeed) }
        let block = MiniMaxH3TransformerBlock(configuration: configuration, includeAdaLN: false)
        block.update(parameters: block.parameters().mapValues { $0.asType(dtype) })
        MLX.eval(block.parameters())
        self.block = block
        self.originalParameters = block.parameters().flattened()

        let attentionProjection = MLX.compile(inputs: [block]) { inputs in
            block.attentionProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                rope: MiniMaxH3RotaryEmbedding(cosine: inputs[3], sine: inputs[4]),
                cachedModulation: inputs[5]
            )
        }
        let attentionOutput = MLX.compile(inputs: [block]) { inputs in
            [block.attentionProjectionResidual(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2]
            )]
        }
        let feedForwardProjection = MLX.compile(inputs: [block]) { inputs in
            block.feedForwardProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs[3]
            )
        }
        let feedForwardOutput = MLX.compile(inputs: [block]) { inputs in
            [block.feedForwardProjectionResidual(
                inputs[0],
                projected: inputs[1],
                gate: inputs[2]
            )]
        }
        let feedForward = MLX.compile(inputs: [block]) { inputs in
            [block.feedForwardResidual(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs[3]
            )]
        }
        let postAttention = MLX.compile(inputs: [block]) { inputs in
            [block.postAttention(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs[5]
            )]
        }
        let postAttentionProjection = MLX.compile(inputs: [block]) { inputs in
            block.postAttentionProjection(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs[5]
            )
        }
        self.forwards = MiniMaxH3CompiledBlockForwards(
            attentionProjection: attentionProjection,
            fastH3AttentionProjection: nil,
            attentionOutput: attentionOutput,
            feedForwardProjection: feedForwardProjection,
            feedForwardOutput: feedForwardOutput,
            postAttentionProjection: postAttentionProjection,
            postAttention: postAttention
        )
        self.fusedFeedForward = feedForward

        if let inputSeed { MLXRandom.seed(inputSeed) }
        self.hidden = MLXRandom.normal([1, rowCount, configuration.hiddenSize])
            .asType(dtype)
        self.timeEmbedding = MLXArray.zeros(
            [3, configuration.timeEmbeddingDimension],
            dtype: dtype
        )
        self.adaLNIndices = MLXArray((0..<rowCount).map { Int32($0 % 9) })
        self.rope = MiniMaxH3RotaryEmbedding(
            cosine: MLXArray.ones(
                [1, rowCount, 1, 6 * configuration.ropeFrequencyCount],
                dtype: dtype
            ),
            sine: MLXArray.zeros(
                [1, rowCount, 1, 6 * configuration.ropeFrequencyCount],
                dtype: dtype
            )
        )
        self.cachedModulation = (
            MLXRandom.normal([9, 6 * configuration.hiddenSize]) * Float(0.1)
        ).asType(dtype)
        MLX.eval(
            hidden,
            timeEmbedding,
            adaLNIndices,
            rope.cosine,
            rope.sine,
            cachedModulation
        )
    }

    package func useOriginalWeights(from source: MiniMaxH3BlockScheduleBenchmark) {
        block.update(parameters: ModuleParameters.unflattened(source.originalParameters))
    }

    package var defaultInput: MLXArray { hidden }

    package func callAsFunction(
        schedule: Schedule,
        maximumQueryTokens: Int? = nil,
        maximumHeadsPerKernel: Int? = nil,
        maximumKernelsPerEvaluation: Int? = nil,
        input: MLXArray? = nil
    ) -> MLXArray {
        let hidden = input ?? self.hidden
        let projectedAttention = projectAttention(input: hidden)
        MLX.eval(projectedAttention)
        let attended = attend(
            projectedAttention,
            maximumQueryTokens: maximumQueryTokens,
            maximumHeadsPerKernel: maximumHeadsPerKernel,
            maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
        )
        return postAttention(
            schedule: schedule,
            attended: attended,
            gate: projectedAttention[3],
            input: hidden
        )
    }

    package func projectAttention(input: MLXArray? = nil) -> [MLXArray] {
        forwards.attentionProjection([
            input ?? hidden,
            timeEmbedding,
            adaLNIndices,
            rope.cosine,
            rope.sine,
            cachedModulation,
        ])
    }

    package func attend(
        _ projectedAttention: [MLXArray],
        maximumQueryTokens: Int? = nil,
        maximumHeadsPerKernel: Int? = nil,
        maximumKernelsPerEvaluation: Int? = nil
    ) -> MLXArray {
        block.scaledDotProductAttention(
            queries: projectedAttention[0],
            keys: projectedAttention[1],
            values: projectedAttention[2],
            maximumQueryTokens: maximumQueryTokens ?? self.maximumQueryTokens,
            maximumHeadsPerKernel: maximumHeadsPerKernel ?? self.maximumHeadsPerKernel,
            maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
                ?? self.maximumKernelsPerEvaluation
        )
    }

    package func attentionOutput(
        input: MLXArray,
        attended: MLXArray,
        gate: MLXArray
    ) -> MLXArray {
        let output = forwards.attentionOutput([input, attended, gate])[0]
        MLX.eval(output)
        return output
    }

    package func splitFeedForward(input: MLXArray) -> MLXArray {
        let projected = forwards.feedForwardProjection([
            input,
            timeEmbedding,
            adaLNIndices,
            cachedModulation,
        ])
        MLX.eval(projected)
        let output = forwards.feedForwardOutput([input, projected[0], projected[1]])[0]
        MLX.eval(output)
        return output
    }

    package func makeFusedBoundary(
        to successor: MiniMaxH3BlockScheduleBenchmark
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        MLX.compile(inputs: [block, successor.block]) { inputs in
            let nextHidden = self.block.feedForwardResidual(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs[3]
            )
            return [nextHidden] + successor.block.attentionProjection(
                nextHidden,
                timeEmbedding: inputs[4],
                adaLNIndices: inputs[5],
                rope: MiniMaxH3RotaryEmbedding(cosine: inputs[6], sine: inputs[7]),
                cachedModulation: inputs[8]
            )
        }
    }

    package func fusedBoundaryInputs(
        to successor: MiniMaxH3BlockScheduleBenchmark,
        input: MLXArray
    ) -> [MLXArray] {
        [
            input,
            timeEmbedding,
            adaLNIndices,
            cachedModulation,
            successor.timeEmbedding,
            successor.adaLNIndices,
            successor.rope.cosine,
            successor.rope.sine,
            successor.cachedModulation,
        ]
    }

    package func postAttention(
        schedule: Schedule,
        attended: MLXArray,
        gate: MLXArray,
        input: MLXArray? = nil
    ) -> MLXArray {
        let hidden = input ?? self.hidden
        switch schedule {
        case .splitPostAttention:
            let attendedHidden = attentionOutput(input: hidden, attended: attended, gate: gate)
            return splitFeedForward(input: attendedHidden)
        case .fusedFeedForward:
            let attentionOutput = forwards.attentionOutput([
                hidden,
                attended,
                gate,
            ])[0]
            MLX.eval(attentionOutput)
            let output = fusedFeedForward([
                attentionOutput,
                timeEmbedding,
                adaLNIndices,
                cachedModulation,
            ])[0]
            MLX.eval(output)
            return output
        case .fusedPostAttention:
            let output = forwards.postAttention([
                hidden,
                attended,
                gate,
                timeEmbedding,
                adaLNIndices,
                cachedModulation,
            ])[0]
            MLX.eval(output)
            return output
        }
    }
}
#endif
