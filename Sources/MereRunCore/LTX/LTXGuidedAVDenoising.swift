import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func predictJointAVDenoised(
    flatVideo: MLXArray,
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
    videoShape: (batch: Int, channels: Int, frames: Int, height: Int, width: Int),
    audioShape: (batch: Int, channels: Int, frames: Int, melBins: Int)
) -> (video: MLXArray, audio: MLXArray) {
    let output = transformer.forward(
        videoLatent: flatVideo,
        videoKeyframesMask: makeLTXVideoKeyframesMask(
            batchSize: videoShape.batch,
            tokenCount: flatVideo.dim(1),
            tokensPerFirstFrame: videoShape.height * videoShape.width,
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
        perturbation: perturbation
    )
    let videoFlat = (
        flatVideo.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(flatVideo.dtype)
    let audioFlat = (
        flatAudio.asType(.float32)
            - audioTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.audioVelocity.asType(.float32)
    ).asType(flatAudio.dtype)
    let video = videoFlat
        .reshaped(
            videoShape.batch,
            videoShape.frames,
            videoShape.height,
            videoShape.width,
            videoShape.channels
        )
        .transposed(0, 4, 1, 2, 3)
    let audio = audioFlat
        .reshaped(
            audioShape.batch,
            audioShape.frames,
            audioShape.channels,
            audioShape.melBins
        )
        .transposed(0, 2, 1, 3)
    MLX.eval(video, audio)
    return (video, audio)
}

func denoiseGuidedAVLoop(
    videoLatents: MLXArray,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray,
    positiveAudioContext: MLXArray,
    negativeAudioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    videoConditioning: LTXLatentConditioningState?,
    videoGuidance: LTXMultiModalGuidance,
    audioGuidance: LTXMultiModalGuidance
) -> (video: MLXArray, audio: MLXArray) {
    var currentVideo = videoLatents
    var currentAudio = audioLatents
    let dtype = videoLatents.dtype

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let videoShape = (
            batch: currentVideo.dim(0),
            channels: currentVideo.dim(1),
            frames: currentVideo.dim(2),
            height: currentVideo.dim(3),
            width: currentVideo.dim(4)
        )
        let audioShape = (
            batch: currentAudio.dim(0),
            channels: currentAudio.dim(1),
            frames: currentAudio.dim(2),
            melBins: currentAudio.dim(3)
        )
        let videoTokenCount = videoShape.frames * videoShape.height * videoShape.width
        let flatVideo = currentVideo
            .transposed(0, 2, 3, 4, 1)
            .reshaped(videoShape.batch, videoTokenCount, videoShape.channels)
        let flatAudio = currentAudio
            .transposed(0, 2, 1, 3)
            .reshaped(audioShape.batch, audioShape.frames, audioShape.channels * audioShape.melBins)

        let videoTimesteps: MLXArray
        if let videoConditioning {
            let mask = videoConditioning.denoiseMask
                .reshaped(videoShape.batch, 1, videoShape.frames, 1, 1)
            let flattenedMask = broadcast(
                mask,
                to: [
                    videoShape.batch,
                    1,
                    videoShape.frames,
                    videoShape.height,
                    videoShape.width,
                ]
            ).reshaped(videoShape.batch, videoTokenCount)
            videoTimesteps = MLXArray(sigma).asType(dtype) * flattenedMask
        } else {
            videoTimesteps = MLX.full(
                [videoShape.batch, videoTokenCount],
                values: MLXArray(sigma).asType(dtype)
            )
        }
        let audioTimesteps = MLX.full(
            [audioShape.batch, audioShape.frames],
            values: MLXArray(sigma).asType(dtype)
        )
        let globalTimestep = MLX.full(
            [videoShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )

        func predict(
            videoContext: MLXArray,
            audioContext: MLXArray,
            perturbation: LTXAudioToVideoPerturbation
        ) -> (video: MLXArray, audio: MLXArray) {
            predictJointAVDenoised(
                flatVideo: flatVideo,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                globalTimestep: globalTimestep,
                videoContext: videoContext,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: perturbation,
                videoShape: videoShape,
                audioShape: audioShape
            )
        }

        let conditioned = predict(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            perturbation: .none
        )
        let negative = predict(
            videoContext: negativeVideoContext,
            audioContext: negativeAudioContext,
            perturbation: .none
        )
        let perturbed = predict(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            perturbation: .spatioTemporal(
                videoBlocks: videoGuidance.spatioTemporalBlocks,
                audioBlocks: audioGuidance.spatioTemporalBlocks
            )
        )
        let isolated = predict(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            perturbation: .isolatedModalities
        )

        var denoisedVideo = videoGuidance.combine(
            conditioned: conditioned.video,
            negativeText: negative.video,
            perturbed: perturbed.video,
            isolatedModality: isolated.video
        )
        let denoisedAudio = audioGuidance.combine(
            conditioned: conditioned.audio,
            negativeText: negative.audio,
            perturbed: perturbed.audio,
            isolatedModality: isolated.audio
        )
        if let videoConditioning {
            let one = MLXArray(1).asType(denoisedVideo.dtype)
            denoisedVideo = denoisedVideo * videoConditioning.denoiseMask
                + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
        }

        let sigma32 = MLXArray(sigma)
        let delta32 = MLXArray(nextSigma - sigma)
        let videoVelocity = (currentVideo.asType(.float32) - denoisedVideo.asType(.float32)) / sigma32
        let audioVelocity = (currentAudio.asType(.float32) - denoisedAudio.asType(.float32)) / sigma32
        currentVideo = (currentVideo.asType(.float32) + videoVelocity * delta32).asType(dtype)
        currentAudio = (currentAudio.asType(.float32) + audioVelocity * delta32).asType(dtype)
        MLX.eval(currentVideo, currentAudio)
    }

    return (currentVideo, currentAudio)
}
