import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

package struct MiniMaxH3TransformerPreparedContext {
    package let text: MLXArray
    package let adaLNIndices: MLXArray
    package let rope: MiniMaxH3RotaryEmbedding
    package let layout: MiniMaxH3PackedLayout
    package let fastVSA: MiniMaxH3FastVSAPreparedContext?
}

package struct MiniMaxH3TokenReductionState {
    package let reducedHidden: MLXArray
    package let originalVideo: MLXArray
    package let pooledBaseline: MLXArray
}

package struct MiniMaxH3TokenReductionMap {
    package let targetVideoStart: Int
    package let reducedVideoRowCount: Int
    package let firstVideoIndices: MLXArray
    package let secondVideoIndices: MLXArray
    package let parentVideoIndices: MLXArray
    package let absoluteSourceIndices: MLXArray

    package init(layout: MiniMaxH3PackedLayout) {
        precondition(layout.targetVideoRows.upperBound == layout.sequenceLength)
        let spatialHeight = layout.latentHeight / 2
        let spatialWidth = layout.latentWidth / 2
        let reducedWidth = (spatialWidth + 1) / 2
        precondition(spatialHeight > 0 && spatialWidth > 0)
        precondition(
            layout.targetVideoRows.count
                == layout.videoLatentFrames * spatialHeight * spatialWidth
        )

        var first: [Int32] = []
        var second: [Int32] = []
        first.reserveCapacity(layout.videoLatentFrames * spatialHeight * reducedWidth)
        second.reserveCapacity(first.capacity)
        for temporal in 0..<layout.videoLatentFrames {
            for height in 0..<spatialHeight {
                let rowStart = (temporal * spatialHeight + height) * spatialWidth
                for reducedColumn in 0..<reducedWidth {
                    let firstIndex = rowStart + reducedColumn * 2
                    first.append(Int32(firstIndex))
                    second.append(Int32(min(firstIndex + 1, rowStart + spatialWidth - 1)))
                }
            }
        }
        var parents: [Int32] = []
        parents.reserveCapacity(layout.targetVideoRows.count)
        for temporal in 0..<layout.videoLatentFrames {
            for height in 0..<spatialHeight {
                let reducedRowStart = (temporal * spatialHeight + height) * reducedWidth
                for column in 0..<spatialWidth {
                    parents.append(Int32(reducedRowStart + column / 2))
                }
            }
        }
        let prefix = (0..<layout.targetVideoRows.lowerBound).map(Int32.init)
        let video = first.map { Int32(layout.targetVideoRows.lowerBound) + $0 }
        self.targetVideoStart = layout.targetVideoRows.lowerBound
        self.reducedVideoRowCount = first.count
        self.firstVideoIndices = MLXArray(first)
        self.secondVideoIndices = MLXArray(second)
        self.parentVideoIndices = MLXArray(parents)
        self.absoluteSourceIndices = MLXArray(prefix + video)
    }

    package func pool(_ hidden: MLXArray) -> MiniMaxH3TokenReductionState {
        precondition(hidden.ndim == 3)
        let originalVideo = hidden[0..., targetVideoStart..., 0...]
        let first = MLX.take(originalVideo, firstVideoIndices, axis: 1).asType(.float32)
        let second = MLX.take(originalVideo, secondVideoIndices, axis: 1).asType(.float32)
        let pooled = ((first + second) * 0.5).asType(hidden.dtype)
        let reduced = targetVideoStart == 0
            ? pooled
            : MLX.concatenated([hidden[0..., 0..<targetVideoStart, 0...], pooled], axis: 1)
        return MiniMaxH3TokenReductionState(
            reducedHidden: reduced,
            originalVideo: originalVideo,
            pooledBaseline: pooled
        )
    }

    package func restore(
        _ reducedHidden: MLXArray,
        state: MiniMaxH3TokenReductionState,
        updateScale: Float
    ) -> MLXArray {
        precondition(reducedHidden.dim(1) == targetVideoStart + reducedVideoRowCount)
        let reducedVideo = reducedHidden[0..., targetVideoStart..., 0...]
        let current = MLX.take(reducedVideo, parentVideoIndices, axis: 1).asType(.float32)
        let baseline = MLX.take(
            state.pooledBaseline,
            parentVideoIndices,
            axis: 1
        ).asType(.float32)
        let restoredVideo = (
            state.originalVideo.asType(.float32) + updateScale * (current - baseline)
        ).asType(reducedHidden.dtype)
        return targetVideoStart == 0
            ? restoredVideo
            : MLX.concatenated([
                reducedHidden[0..., 0..<targetVideoStart, 0...],
                restoredVideo,
            ], axis: 1)
    }
}

package struct MiniMaxH3TokenReductionPreparedContext {
    package let map: MiniMaxH3TokenReductionMap
    package let reducedContext: MiniMaxH3TransformerPreparedContext
}

package struct MiniMaxH3AdaLNStep {
    package init(timeEmbedding: MLXArray, blockModulations: [MLXArray], finalModulation: MLXArray) {
        self.timeEmbedding = timeEmbedding
        self.blockModulations = blockModulations
        self.finalModulation = finalModulation
    }

    package let timeEmbedding: MLXArray
    package let blockModulations: [MLXArray]
    package let finalModulation: MLXArray
}
