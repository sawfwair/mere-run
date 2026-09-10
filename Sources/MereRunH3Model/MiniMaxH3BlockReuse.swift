import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

public struct MiniMaxH3TransformerOutput {
    public let videoVelocityRows: MLXArray
    public let audioVelocityRows: MLXArray
}

package struct MiniMaxH3BlockReuseResult {
    package let output: MiniMaxH3TransformerOutput
    package let refreshedTailResidual: MLXArray?
}

package struct MiniMaxH3FirstBlockChange: Sendable, Equatable {
    package let videoGlobal: Float
    package let audioGlobal: Float
    package let videoTemporalMaximum: Float
    package let audioTemporalMaximum: Float

    package var isFinite: Bool {
        videoGlobal.isFinite
            && audioGlobal.isFinite
            && videoTemporalMaximum.isFinite
            && audioTemporalMaximum.isFinite
    }

    package static func measure(
        current: MLXArray,
        previous: MLXArray,
        layout: MiniMaxH3PackedLayout
    ) -> Self {
        let audioRowCount = layout.targetAudioRows.count
        let videoRowCount = layout.targetVideoRows.count
        let totalRowCount = audioRowCount + videoRowCount
        precondition(current.shape == previous.shape)
        precondition(current.ndim == 3 && current.dim(0) == 1)
        precondition(current.dim(1) == totalRowCount)
        precondition(audioRowCount == layout.audioLatentFrames * 2)
        precondition(videoRowCount.isMultiple(of: layout.videoLatentFrames))

        let currentFloat = current.asType(.float32)
        let previousFloat = previous.asType(.float32)
        let currentAudio = currentFloat[0..., 0..<audioRowCount, 0...]
        let previousAudio = previousFloat[0..., 0..<audioRowCount, 0...]
        let currentVideo = currentFloat[0..., audioRowCount..<totalRowCount, 0...]
        let previousVideo = previousFloat[0..., audioRowCount..<totalRowCount, 0...]

        func relativeGlobal(_ value: MLXArray, _ reference: MLXArray) -> MLXArray {
            let numerator = MLX.mean(MLX.abs(value - reference))
            let denominator = MLX.maximum(
                MLX.mean(MLX.abs(reference)),
                MLXArray(Float(1e-8))
            )
            return numerator / denominator
        }

        let videoRowsPerFrame = videoRowCount / layout.videoLatentFrames
        let videoDifference = MLX.abs(currentVideo - previousVideo).reshaped(
            1,
            layout.videoLatentFrames,
            videoRowsPerFrame,
            current.dim(2)
        )
        let videoReference = MLX.abs(previousVideo).reshaped(
            1,
            layout.videoLatentFrames,
            videoRowsPerFrame,
            current.dim(2)
        )
        let videoTemporal = MLX.mean(videoDifference, axes: [0, 2, 3])
            / MLX.maximum(
                MLX.mean(videoReference, axes: [0, 2, 3]),
                MLXArray(Float(1e-8))
            )

        let audioDifference = MLX.abs(currentAudio - previousAudio).reshaped(
            1,
            2,
            layout.audioLatentFrames,
            current.dim(2)
        )
        let audioReference = MLX.abs(previousAudio).reshaped(
            1,
            2,
            layout.audioLatentFrames,
            current.dim(2)
        )
        let audioTemporal = MLX.mean(audioDifference, axes: [0, 1, 3])
            / MLX.maximum(
                MLX.mean(audioReference, axes: [0, 1, 3]),
                MLXArray(Float(1e-8))
            )

        let metrics = MLX.stacked([
            relativeGlobal(currentVideo, previousVideo),
            relativeGlobal(currentAudio, previousAudio),
            MLX.max(videoTemporal),
            MLX.max(audioTemporal),
        ]).asType(.float32)
        MLX.eval(metrics)
        let values = metrics.asArray(Float.self)
        return .init(
            videoGlobal: values[0],
            audioGlobal: values[1],
            videoTemporalMaximum: values[2],
            audioTemporalMaximum: values[3]
        )
    }
}

package struct MiniMaxH3AdaptiveBlockReuseResult {
    package let output: MiniMaxH3TransformerOutput
    package let refreshedFirstResidual: MLXArray?
    package let refreshedTargetTailResidual: MLXArray?
    package let reusedTail: Bool
    package let change: MiniMaxH3FirstBlockChange?
}
