import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func denoiseAVLoop(
    videoLatents: MLXArray,
    audioLatents: MLXArray,
    videoRope: (cos: MLXArray, sin: MLXArray),
    audioRope: (cos: MLXArray, sin: MLXArray),
    videoCrossRope: (cos: MLXArray, sin: MLXArray),
    audioCrossRope: (cos: MLXArray, sin: MLXArray),
    videoContext: MLXArray,
    audioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    videoConditioning: LTXLatentConditioningState?,
    ancestralNoiseSeed: Int? = nil
) -> (MLXArray, MLXArray) {
    var currentVideo = videoLatents
    var currentAudio = audioLatents
    let dtype = videoLatents.dtype
    if let ancestralNoiseSeed {
        MLXRandom.seed(UInt64(bitPattern: Int64(ancestralNoiseSeed)))
    }

    for i in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[i]
        let nextSigma = sigmas[i + 1]

        let b = currentVideo.dim(0)
        let c = currentVideo.dim(1)
        let f = currentVideo.dim(2)
        let h = currentVideo.dim(3)
        let w = currentVideo.dim(4)
        let videoTokenCount = f * h * w
        let flatVideo = currentVideo.transposed(0, 2, 3, 4, 1).reshaped(b, videoTokenCount, c)

        let ab = currentAudio.dim(0)
        let ac = currentAudio.dim(1)
        let at = currentAudio.dim(2)
        let af = currentAudio.dim(3)
        let flatAudio = currentAudio.transposed(0, 2, 1, 3).reshaped(ab, at, ac * af)

        let videoTimesteps: MLXArray
        if let videoConditioning {
            let mask = videoConditioning.denoiseMask.reshaped(b, 1, f, 1, 1)
            let broadcastMask = broadcast(mask, to: [b, 1, f, h, w]).reshaped(b, videoTokenCount)
            videoTimesteps = MLXArray(sigma).asType(dtype) * broadcastMask
        } else {
            videoTimesteps = MLX.full([b, videoTokenCount], values: MLXArray(sigma).asType(dtype))
        }
        let audioTimesteps = MLX.full([ab, at], values: MLXArray(sigma).asType(dtype))
        let globalTimestep = MLX.full([b], values: MLXArray(sigma).asType(dtype))

        let velocity = transformer.forward(
            videoLatent: flatVideo,
            videoKeyframesMask: makeLTXVideoKeyframesMask(
                batchSize: b,
                tokenCount: videoTokenCount,
                tokensPerFirstFrame: h * w,
                dtype: flatVideo.dtype
            ),
            videoAttentionMask: nil,
            audioLatent: flatAudio,
            timestep: globalTimestep,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            videoContext: videoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            audioSigma: globalTimestep,
            perturbation: .none
        )

        let videoVelocity = velocity.videoVelocity
            .reshaped(b, f, h, w, c)
            .transposed(0, 4, 1, 2, 3)
        let audioVelocity = velocity.audioVelocity
            .reshaped(ab, at, ac, af)
            .transposed(0, 2, 1, 3)
        MLX.eval(videoVelocity, audioVelocity)

        var denoisedVideo = toDenoised(noisy: currentVideo, velocity: videoVelocity, sigma: sigma)
        if let videoConditioning {
            let one = MLXArray(1.0).asType(denoisedVideo.dtype)
            denoisedVideo = denoisedVideo * videoConditioning.denoiseMask + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
        }
        let denoisedAudio = toDenoised(noisy: currentAudio, velocity: audioVelocity, sigma: sigma)
        MLX.eval(denoisedVideo, denoisedAudio)

        if nextSigma > 0, ancestralNoiseSeed != nil {
            let videoNoise = MLXRandom.normal(currentVideo.shape).asType(dtype)
            let audioNoise = MLXRandom.normal(currentAudio.shape).asType(dtype)
            currentVideo = ltxAncestralEulerStep(
                sample: currentVideo,
                denoised: denoisedVideo,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: videoNoise
            )
            currentAudio = ltxAncestralEulerStep(
                sample: currentAudio,
                denoised: denoisedAudio,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: audioNoise
            )
            if let videoConditioning {
                let one = MLXArray(1.0).asType(currentVideo.dtype)
                currentVideo = currentVideo * videoConditioning.denoiseMask
                    + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
            }
        } else if nextSigma > 0 {
            let sigmaArr = MLXArray(sigma).asType(dtype)
            let nextArr = MLXArray(nextSigma).asType(dtype)
            currentVideo = denoisedVideo + nextArr * (currentVideo - denoisedVideo) / sigmaArr
            currentAudio = denoisedAudio + nextArr * (currentAudio - denoisedAudio) / sigmaArr
        } else {
            currentVideo = denoisedVideo
            currentAudio = denoisedAudio
        }
        MLX.eval(currentVideo, currentAudio)
    }

    return (currentVideo, currentAudio)
}
