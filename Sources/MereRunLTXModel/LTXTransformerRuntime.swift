import Foundation
import MLX
import MLXFast
import MLXNN

package func computeAudioLatentFrameCount(videoFrames: Int, fps: Double) -> Int {
    let duration = Double(videoFrames) / max(1, fps)
    let latentsPerSecond = Double(LTXAudioLatentSampleRate) / Double(LTXAudioHopLength) / Double(LTXAudioLatentDownsampleFactor)
    return max(1, Int((duration * latentsPerSecond).rounded(.toNearestOrEven)))
}

package func createAudioPositionGrid(
    batchSize: Int,
    audioFrames: Int,
    sampleRate: Float = Float(LTXAudioLatentSampleRate),
    hopLength: Float = Float(LTXAudioHopLength),
    downsampleFactor: Int = LTXAudioLatentDownsampleFactor,
    causalFix: Bool = true
) -> MLXArray {
    var values = [Float](repeating: 0, count: batchSize * audioFrames * 2)
    for b in 0..<batchSize {
        for t in 0..<audioFrames {
            var startFrame = Float(t * downsampleFactor)
            var endFrame = Float((t + 1) * downsampleFactor)
            if causalFix {
                let shift = Float(1 - downsampleFactor)
                startFrame = max(0, startFrame + shift)
                endFrame = max(0, endFrame + shift)
            }

            let startSeconds = (startFrame * hopLength) / sampleRate
            let endSeconds = (endFrame * hopLength) / sampleRate
            let idx = (b * audioFrames + t) * 2
            values[idx] = startSeconds
            values[idx + 1] = endSeconds
        }
    }
    return MLXArray(values).reshaped(batchSize, 1, audioFrames, 2).asType(.float32)
}

package protocol LTXUnifiedAVTransformerRuntime: Module {
    func forward(
        videoLatent: MLXArray,
        videoKeyframesMask: MLXArray?,
        videoAttentionMask: MLXArray?,
        audioLatent: MLXArray,
        timestep: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        audioSigma: MLXArray,
        perturbation: LTXAudioToVideoPerturbation
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray)
}

package func prepareLTXSelfAttentionMask(
    _ mask: MLXArray?,
    dtype: DType
) -> MLXArray? {
    guard let mask else { return nil }
    let typed = mask.asType(dtype)
    let epsilon = MLXArray(Float(1e-7)).asType(dtype)
    let negative = MLX.full(
        typed.shape,
        values: MLXArray(Float(-1e9)).asType(dtype)
    )
    let bias = MLX.where(
        typed .> MLXArray(0).asType(dtype),
        MLX.log(MLX.maximum(typed, epsilon)),
        negative
    )
    return bias.expandedDimensions(axis: 1)
}
