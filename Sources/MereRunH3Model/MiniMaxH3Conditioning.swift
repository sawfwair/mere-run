import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

final class MiniMaxH3TokenRefinerBlock: Module {
    @ModuleInfo(key: "norm1") package var attentionNorm: RMSNorm
    @ModuleInfo(key: "attn") package var attention: MiniMaxH3Attention
    @ModuleInfo(key: "norm2") package var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "mlp") package var feedForward: MiniMaxH3FeedForward

    package init(configuration: MiniMaxH3TransformerConfiguration) {
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
    }

    package func callAsFunction(_ value: MLXArray) -> MLXArray {
        let attended = value + attention(attentionNorm(value), rope: nil)
        return attended + feedForward(feedForwardNorm(attended))
    }
}

final class MiniMaxH3TokenRefiner: Module {
    @ModuleInfo(key: "blocks") package var blocks: [MiniMaxH3TokenRefinerBlock]
    @ModuleInfo(key: "final_norm") package var finalNorm: RMSNorm

    package init(configuration: MiniMaxH3TransformerConfiguration) {
        self._blocks.wrappedValue = (0..<configuration.refinerLayerCount).map { _ in
            MiniMaxH3TokenRefinerBlock(configuration: configuration)
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
    }

    package func callAsFunction(_ value: MLXArray) -> MLXArray {
        finalNorm(blocks.reduce(value) { hidden, block in block(hidden) })
    }
}

final class MiniMaxH3AdaLNProjection: Module {
    package let hiddenSize: Int
    package let partCount: Int
    package let modalityCount: Int
    @ModuleInfo(key: "linear") package var linear: Linear

    package init(inputDimension: Int, hiddenSize: Int, partCount: Int, modalityCount: Int) {
        self.hiddenSize = hiddenSize
        self.partCount = partCount
        self.modalityCount = modalityCount
        self._linear.wrappedValue = Linear(
            inputDimension,
            partCount * modalityCount * hiddenSize,
            bias: true
        )
    }

    convenience init(discarded: Void) {
        self.init(inputDimension: 1, hiddenSize: 1, partCount: 1, modalityCount: 1)
    }

    package func callAsFunction(_ timeEmbedding: MLXArray) -> [MLXArray] {
        let projected = miniMaxH3Linear(linear, MLXNN.silu(timeEmbedding))
            .reshaped(timeEmbedding.dim(0) * modalityCount, partCount * hiddenSize)
        return MLX.split(projected, parts: partCount, axis: -1)
    }

    package func concatenated(_ timeEmbedding: MLXArray) -> MLXArray {
        MLX.concatenated(self(timeEmbedding), axis: -1)
    }
}

final class MiniMaxH3TimeEmbedding: Module {
    @ModuleInfo(key: "proj_in") package var input: Linear
    @ModuleInfo(key: "proj_out") package var output: Linear

    package init(configuration: MiniMaxH3TransformerConfiguration) {
        self._input.wrappedValue = Linear(
            configuration.timeFrequencyDimension,
            configuration.timeEmbeddingHiddenSize,
            bias: true
        )
        self._output.wrappedValue = Linear(
            configuration.timeEmbeddingHiddenSize,
            configuration.timeEmbeddingDimension,
            bias: true
        )
    }

    package init(discarded: Void) {
        self._input.wrappedValue = Linear(1, 1, bias: false)
        self._output.wrappedValue = Linear(1, 1, bias: false)
    }

    package func callAsFunction(_ value: MLXArray) -> MLXArray {
        miniMaxH3Linear(output, MLXNN.silu(miniMaxH3Linear(input, value)))
    }
}
