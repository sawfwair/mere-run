import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func denoiseLTX25AVTokenLoop(
    videoState: LTX25VideoTokenState,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    videoContext: MLXArray,
    audioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    ancestralNoiseSeed: Int? = nil,
    audioConditioning: LTXLatentConditioningState? = nil
) -> (video: LTX25VideoTokenState, audio: MLXArray) {
    var currentVideo = videoState
    var currentAudio = audioLatents
    let dtype = videoState.latent.dtype
    if let ancestralNoiseSeed {
        MLXRandom.seed(UInt64(bitPattern: Int64(ancestralNoiseSeed)))
    }

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let audioShape = (
            batch: currentAudio.dim(0),
            channels: currentAudio.dim(1),
            frames: currentAudio.dim(2),
            melBins: currentAudio.dim(3)
        )
        let flatAudio = currentAudio
            .transposed(0, 2, 1, 3)
            .reshaped(audioShape.batch, audioShape.frames, audioShape.channels * audioShape.melBins)
        let videoTimesteps = currentVideo.denoiseMask.squeezed(axis: -1)
            * MLXArray(sigma).asType(dtype)
        let audioTimesteps = audioConditioning.map {
            $0.denoiseMask[0..., 0, 0..., 0] * MLXArray(sigma).asType(dtype)
        } ?? MLX.full(
            [audioShape.batch, audioShape.frames],
            values: MLXArray(sigma).asType(dtype)
        )
        let globalTimestep = MLX.full(
            [currentVideo.targetShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )
        let velocity = transformer.forward(
            videoLatent: currentVideo.latent,
            videoKeyframesMask: currentVideo.keyframesMask,
            videoAttentionMask: currentVideo.attentionMask,
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
        var denoisedVideo = toDenoised(
            noisy: currentVideo.latent,
            velocity: velocity.videoVelocity,
            sigma: sigma
        )
        let one = MLXArray(1).asType(dtype)
        denoisedVideo = denoisedVideo * currentVideo.denoiseMask
            + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
        let audioVelocity = velocity.audioVelocity
            .reshaped(audioShape.batch, audioShape.frames, audioShape.channels, audioShape.melBins)
            .transposed(0, 2, 1, 3)
        var denoisedAudio = toDenoised(
            noisy: currentAudio,
            velocity: audioVelocity,
            sigma: sigma
        )
        if let audioConditioning {
            denoisedAudio = denoisedAudio * audioConditioning.denoiseMask
                + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
        }

        if nextSigma > 0, ancestralNoiseSeed != nil {
            currentVideo.latent = ltxAncestralEulerStep(
                sample: currentVideo.latent,
                denoised: denoisedVideo,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentVideo.latent.shape).asType(dtype)
            )
            currentAudio = ltxAncestralEulerStep(
                sample: currentAudio,
                denoised: denoisedAudio,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentAudio.shape).asType(dtype)
            )
            currentVideo.latent = currentVideo.latent * currentVideo.denoiseMask
                + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
            if let audioConditioning {
                currentAudio = currentAudio * audioConditioning.denoiseMask
                    + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
            }
        } else if nextSigma > 0 {
            let sigmaArray = MLXArray(sigma).asType(dtype)
            let nextArray = MLXArray(nextSigma).asType(dtype)
            currentVideo.latent = denoisedVideo
                + nextArray * (currentVideo.latent - denoisedVideo) / sigmaArray
            currentAudio = denoisedAudio
                + nextArray * (currentAudio - denoisedAudio) / sigmaArray
        } else {
            currentVideo.latent = denoisedVideo
            currentAudio = denoisedAudio
        }
        if let audioConditioning {
            currentAudio = currentAudio * audioConditioning.denoiseMask
                + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
        }
        MLX.eval(currentVideo.latent, currentAudio)
    }
    return (currentVideo, currentAudio)
}

func predictLTX25JointAVTokenDenoised(
    videoState: LTX25VideoTokenState,
    flatAudio: MLXArray,
    videoTimesteps: MLXArray,
    audioTimesteps: MLXArray,
    globalTimestep: MLXArray,
    videoContext: MLXArray,
    audioContext: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    transformer: any LTXUnifiedAVTransformerRuntime,
    perturbation: LTXAudioToVideoPerturbation,
    audioShape: (batch: Int, channels: Int, frames: Int, melBins: Int)
) -> (video: MLXArray, audio: MLXArray) {
    let output = transformer.forward(
        videoLatent: videoState.latent,
        videoKeyframesMask: videoState.keyframesMask,
        videoAttentionMask: videoState.attentionMask,
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
        perturbation: perturbation
    )
    let video = (
        videoState.latent.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(videoState.latent.dtype)
    let audio = (
        flatAudio.asType(.float32)
            - audioTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.audioVelocity.asType(.float32)
    ).asType(flatAudio.dtype)
        .reshaped(audioShape.batch, audioShape.frames, audioShape.channels, audioShape.melBins)
        .transposed(0, 2, 1, 3)
    MLX.eval(video, audio)
    return (video, audio)
}

struct LTXGuidedAVPrediction {
    let video: MLXArray
    let audio: MLXArray
    let unconditionalVideo: MLXArray
    let unconditionalAudio: MLXArray
}

final class LTXGuidanceProjectionCacheMetrics {
    var buildSeconds = 0.0
    var buildCount = 0
    var reuseCount = 0
    var fallbackCount = 0
}
