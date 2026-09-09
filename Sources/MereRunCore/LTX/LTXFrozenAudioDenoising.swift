import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func makeLTXAudioToVideoVideoRopes(
    latentFrames: Int,
    height: Int,
    width: Int,
    fps: Double
) -> (selfAttention: LTXRope, crossAttention: LTXRope) {
    let positions = createPositionGrid(
        batchSize: 1,
        numFrames: latentFrames,
        height: height,
        width: width,
        temporalScale: 8,
        spatialScale: 32,
        fps: Float(fps),
        causalFix: true
    )
    return makeLTXAudioToVideoVideoRopes(positions: positions)
}

func makeLTXAudioToVideoVideoRopes(
    positions: MLXArray
) -> (selfAttention: LTXRope, crossAttention: LTXRope) {
    let selfAttention = precomputeSplitRope(
        positions: positions,
        dim: 4_096,
        theta: 10_000,
        maxPos: [20, 2_048, 2_048],
        numHeads: 32
    )
    let crossPositions = positions[0..., 0..<1, 0..., 0...]
    let crossAttention = precomputeSplitRope(
        positions: crossPositions,
        dim: 2_048,
        theta: 10_000,
        maxPos: [20],
        numHeads: 32
    )
    return (selfAttention, crossAttention)
}

func predictFrozenAudioVideoDenoised(
    flatVideo: MLXArray,
    flatAudio: MLXArray,
    videoTimesteps: MLXArray,
    audioTimesteps: MLXArray,
    videoSigma: MLXArray,
    audioSigma: MLXArray,
    videoContext: MLXArray,
    audioContext: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    transformer: any LTXUnifiedAVTransformerRuntime,
    perturbation: LTXAudioToVideoPerturbation,
    outputShape: (batch: Int, channels: Int, frames: Int, height: Int, width: Int)
) -> MLXArray {
    let output = transformer.forward(
        videoLatent: flatVideo,
        videoKeyframesMask: makeLTXVideoKeyframesMask(
            batchSize: outputShape.batch,
            tokenCount: flatVideo.dim(1),
            tokensPerFirstFrame: outputShape.height * outputShape.width,
            dtype: flatVideo.dtype
        ),
        videoAttentionMask: nil,
        audioLatent: flatAudio,
        timestep: videoSigma,
        videoTimesteps: videoTimesteps,
        audioTimesteps: audioTimesteps,
        videoContext: videoContext,
        audioContext: audioContext,
        videoRope: videoRope,
        audioRope: audioRope,
        videoCrossRope: videoCrossRope,
        audioCrossRope: audioCrossRope,
        audioSigma: audioSigma,
        perturbation: perturbation
    )
    let denoisedFlat = (
        flatVideo.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(flatVideo.dtype)
    let denoised = denoisedFlat
        .reshaped(
            outputShape.batch,
            outputShape.frames,
            outputShape.height,
            outputShape.width,
            outputShape.channels
        )
        .transposed(0, 4, 1, 2, 3)
    MLX.eval(denoised)
    return denoised
}

func denoiseFrozenAudioVideoLoop(
    videoLatents: MLXArray,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray?,
    audioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    videoConditioning: LTXLatentConditioningState?,
    guidance: LTXAudioToVideoGuidance?,
    debugLabel: String? = nil
) throws -> (video: MLXArray, audio: MLXArray) {
    var currentVideo = videoLatents
    let dtype = videoLatents.dtype
    let audioBatch = audioLatents.dim(0)
    let audioChannels = audioLatents.dim(1)
    let audioFrames = audioLatents.dim(2)
    let audioMelBins = audioLatents.dim(3)
    let flatAudio = audioLatents
        .transposed(0, 2, 1, 3)
        .reshaped(audioBatch, audioFrames, audioChannels * audioMelBins)
    let audioTimesteps = MLX.zeros([audioBatch, audioFrames], dtype: dtype)
    let audioSigma = MLX.zeros([audioBatch], dtype: dtype)

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let batch = currentVideo.dim(0)
        let channels = currentVideo.dim(1)
        let frames = currentVideo.dim(2)
        let height = currentVideo.dim(3)
        let width = currentVideo.dim(4)
        let tokenCount = frames * height * width
        let flatVideo = currentVideo
            .transposed(0, 2, 3, 4, 1)
            .reshaped(batch, tokenCount, channels)

        let videoTimesteps: MLXArray
        if let videoConditioning {
            let mask = videoConditioning.denoiseMask.reshaped(batch, 1, frames, 1, 1)
            let flattenedMask = broadcast(
                mask,
                to: [batch, 1, frames, height, width]
            ).reshaped(batch, tokenCount)
            videoTimesteps = MLXArray(sigma).asType(dtype) * flattenedMask
        } else {
            videoTimesteps = MLX.full(
                [batch, tokenCount],
                values: MLXArray(sigma).asType(dtype)
            )
        }
        let videoSigma = MLX.full(
            [batch],
            values: MLXArray(sigma).asType(dtype)
        )
        let outputShape = (batch, channels, frames, height, width)
        if index == 0, let debugLabel {
            let parityIO = LTXAudioToVideoParityIO()
            try parityIO.save(flatVideo, suffix: "\(debugLabel)_step0_video_input")
            try parityIO.save(flatAudio, suffix: "\(debugLabel)_step0_audio_input")
            try parityIO.save(videoTimesteps, suffix: "\(debugLabel)_step0_video_timesteps")
            try parityIO.save(videoRope.cos, suffix: "\(debugLabel)_video_rope_cos")
            try parityIO.save(videoRope.sin, suffix: "\(debugLabel)_video_rope_sin")
            try parityIO.save(audioRope.cos, suffix: "\(debugLabel)_audio_rope_cos")
            try parityIO.save(audioRope.sin, suffix: "\(debugLabel)_audio_rope_sin")
        }

        let conditioned = predictFrozenAudioVideoDenoised(
            flatVideo: flatVideo,
            flatAudio: flatAudio,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            videoSigma: videoSigma,
            audioSigma: audioSigma,
            videoContext: positiveVideoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            transformer: transformer,
            perturbation: .none,
            outputShape: outputShape
        )
        if index == 0, let debugLabel {
            let conditionedTokens = conditioned
                .transposed(0, 2, 3, 4, 1)
                .reshaped(batch, tokenCount, channels)
            try LTXAudioToVideoParityIO().save(
                conditionedTokens,
                suffix: "\(debugLabel)_step0_conditioned_x0"
            )
        }

        var denoised = conditioned
        if let guidance {
            let negativeText: MLXArray
            if guidance.classifierFreeScale == 1 {
                negativeText = conditioned
            } else if let negativeVideoContext {
                negativeText = predictFrozenAudioVideoDenoised(
                    flatVideo: flatVideo,
                    flatAudio: flatAudio,
                    videoTimesteps: videoTimesteps,
                    audioTimesteps: audioTimesteps,
                    videoSigma: videoSigma,
                    audioSigma: audioSigma,
                    videoContext: negativeVideoContext,
                    audioContext: audioContext,
                    videoRope: videoRope,
                    audioRope: audioRope,
                    videoCrossRope: videoCrossRope,
                    audioCrossRope: audioCrossRope,
                    transformer: transformer,
                    perturbation: .none,
                    outputShape: outputShape
                )
            } else {
                preconditionFailure("A negative video context is required for classifier-free guidance.")
            }

            let perturbed = guidance.spatioTemporalScale == 0 ? conditioned : predictFrozenAudioVideoDenoised(
                flatVideo: flatVideo,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                videoSigma: videoSigma,
                audioSigma: audioSigma,
                videoContext: positiveVideoContext,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: .spatioTemporal(blocks: guidance.spatioTemporalBlocks),
                outputShape: outputShape
            )
            let isolated = guidance.audioToVideoScale == 1 ? conditioned : predictFrozenAudioVideoDenoised(
                flatVideo: flatVideo,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                videoSigma: videoSigma,
                audioSigma: audioSigma,
                videoContext: positiveVideoContext,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: .isolatedModalities,
                outputShape: outputShape
            )
            denoised = guidance.combine(
                conditioned: conditioned,
                negativeText: negativeText,
                perturbed: perturbed,
                isolatedAudio: isolated
            )
            MLX.eval(denoised)
            if index == 0, let debugLabel {
                let parityIO = LTXAudioToVideoParityIO()
                let variants = [
                    ("negative_x0", negativeText),
                    ("perturbed_x0", perturbed),
                    ("isolated_x0", isolated),
                    ("guided_x0", denoised),
                ]
                for (suffix, array) in variants {
                    let tokens = array
                        .transposed(0, 2, 3, 4, 1)
                        .reshaped(batch, tokenCount, channels)
                    try parityIO.save(tokens, suffix: "\(debugLabel)_step0_\(suffix)")
                }
            }
        }

        if let videoConditioning {
            let one = MLXArray(1).asType(denoised.dtype)
            denoised = denoised * videoConditioning.denoiseMask
                + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
        }
        let sigma32 = MLXArray(sigma)
        let delta32 = MLXArray(nextSigma - sigma)
        let velocity32 = (currentVideo.asType(.float32) - denoised.asType(.float32)) / sigma32
        currentVideo = (currentVideo.asType(.float32) + velocity32 * delta32).asType(dtype)
        MLX.eval(currentVideo)
    }
    return (currentVideo, audioLatents)
}
