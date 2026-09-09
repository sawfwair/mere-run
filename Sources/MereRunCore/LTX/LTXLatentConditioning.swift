import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func upsampleLatents(
    _ latents: MLXArray,
    upsampler: LTXLatentUpsampler,
    latentMean: MLXArray,
    latentStd: MLXArray
) -> MLXArray {
    let dtype = latents.dtype
    let mean = latentMean.asType(dtype).reshaped(1, -1, 1, 1, 1)
    let std = latentStd.asType(dtype).reshaped(1, -1, 1, 1, 1)

    var x = latents
    x = x * std + mean
    x = upsampler(x)
    x = (x - mean) / std
    return x
}

func upsampleLatentsTemporally(
    _ latents: MLXArray,
    upsampler: LTXTemporalLatentUpsampler,
    latentMean: MLXArray,
    latentStd: MLXArray
) -> MLXArray {
    let dtype = latents.dtype
    let mean = latentMean.asType(dtype).reshaped(1, -1, 1, 1, 1)
    let std = latentStd.asType(dtype).reshaped(1, -1, 1, 1, 1)

    var x = latents
    x = x * std + mean
    x = upsampler(x)
    x = (x - mean) / std
    return x
}

func applyLatentConditioning(
    baseLatent: MLXArray,
    conditionedLatent: MLXArray,
    frameIndex: Int,
    strength: Float,
    endConditionedLatent: MLXArray? = nil,
    endFrameIndex: Int = -1,
    endStrength: Float = 1.0
) -> LTXLatentConditioningState {
    let b = baseLatent.dim(0)
    let c = baseLatent.dim(1)
    let f = baseLatent.dim(2)
    let h = baseLatent.dim(3)
    let w = baseLatent.dim(4)
    let condFrames = conditionedLatent.dim(2)
    let dtype = baseLatent.dtype

    let condEnd = min(frameIndex + condFrames, f)
    let oneMinusStrength = MLXArray(1.0 - strength).asType(dtype)

    // Optional end keyframe: condition a second image at the tail of the clip so
    // LTX interpolates a directed start->end motion. Defaults to the last latent
    // frame(s). Start conditioning takes precedence on any overlap.
    let endCondFrames = endConditionedLatent?.dim(2) ?? 0
    let endStart = endConditionedLatent != nil
        ? (endFrameIndex >= 0 ? endFrameIndex : max(0, f - endCondFrames))
        : f
    let endStop = min(endStart + endCondFrames, f)
    let oneMinusEndStrength = MLXArray(1.0 - endStrength).asType(dtype)

    var latentFrames: [MLXArray] = []
    var cleanFrames: [MLXArray] = []
    var maskFrames: [MLXArray] = []
    latentFrames.reserveCapacity(f)
    cleanFrames.reserveCapacity(f)
    maskFrames.reserveCapacity(f)

    for frame in 0..<f {
        if frame >= frameIndex, frame < condEnd {
            let condIdx = frame - frameIndex
            let condSlice = conditionedLatent[0..., 0..., condIdx..<condIdx + 1, 0..., 0...]
            latentFrames.append(condSlice)
            cleanFrames.append(condSlice)
            maskFrames.append(MLX.full([b, 1, 1, 1, 1], values: oneMinusStrength))
        } else if let endLatent = endConditionedLatent, frame >= endStart, frame < endStop {
            let condIdx = frame - endStart
            let condSlice = endLatent[0..., 0..., condIdx..<condIdx + 1, 0..., 0...]
            latentFrames.append(condSlice)
            cleanFrames.append(condSlice)
            maskFrames.append(MLX.full([b, 1, 1, 1, 1], values: oneMinusEndStrength))
        } else {
            latentFrames.append(baseLatent[0..., 0..., frame..<frame + 1, 0..., 0...])
            cleanFrames.append(MLX.zeros([b, c, 1, h, w], dtype: dtype))
            maskFrames.append(MLX.ones([b, 1, 1, 1, 1], dtype: dtype))
        }
    }

    return LTXLatentConditioningState(
        latent: MLX.concatenated(latentFrames, axis: 2),
        cleanLatent: MLX.concatenated(cleanFrames, axis: 2),
        denoiseMask: MLX.concatenated(maskFrames, axis: 2)
    )
}
