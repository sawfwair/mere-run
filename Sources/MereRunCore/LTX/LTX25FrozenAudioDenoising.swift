import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func predictFrozenLTX25AudioVideoDenoised(
    videoState: LTX25VideoTokenState,
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
    perturbation: LTXAudioToVideoPerturbation
) -> MLXArray {
    let output = transformer.forward(
        videoLatent: videoState.latent,
        videoKeyframesMask: videoState.keyframesMask,
        videoAttentionMask: videoState.attentionMask,
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
    let denoised = (
        videoState.latent.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(videoState.latent.dtype)
    MLX.eval(denoised)
    return denoised
}

func denoiseFrozenLTX25AudioVideoTokenLoop(
    videoState: LTX25VideoTokenState,
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
    guidance: LTXAudioToVideoGuidance?
) -> LTX25VideoTokenState {
    var current = videoState
    let dtype = current.latent.dtype
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
        let videoTimesteps = current.denoiseMask.squeezed(axis: -1)
            * MLXArray(sigma).asType(dtype)
        let videoSigma = MLX.full(
            [current.targetShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )

        func predict(
            context: MLXArray,
            perturbation: LTXAudioToVideoPerturbation
        ) -> MLXArray {
            predictFrozenLTX25AudioVideoDenoised(
                videoState: current,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                videoSigma: videoSigma,
                audioSigma: audioSigma,
                videoContext: context,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: perturbation
            )
        }

        let conditioned = predict(context: positiveVideoContext, perturbation: .none)
        var denoised = conditioned
        if let guidance {
            let negativeText: MLXArray
            if guidance.classifierFreeScale == 1 {
                negativeText = conditioned
            } else if let negativeVideoContext {
                negativeText = predict(context: negativeVideoContext, perturbation: .none)
            } else {
                preconditionFailure("A negative video context is required for classifier-free guidance.")
            }
            let perturbed = guidance.spatioTemporalScale == 0
                ? conditioned
                : predict(
                    context: positiveVideoContext,
                    perturbation: .spatioTemporal(blocks: guidance.spatioTemporalBlocks)
                )
            let isolated = guidance.audioToVideoScale == 1
                ? conditioned
                : predict(context: positiveVideoContext, perturbation: .isolatedModalities)
            denoised = guidance.combine(
                conditioned: conditioned,
                negativeText: negativeText,
                perturbed: perturbed,
                isolatedAudio: isolated
            )
        }
        let one = MLXArray(1).asType(dtype)
        denoised = denoised * current.denoiseMask
            + current.cleanLatent * (one - current.denoiseMask)
        let velocity = (
            current.latent.asType(.float32) - denoised.asType(.float32)
        ) / MLXArray(sigma)
        current.latent = (
            current.latent.asType(.float32) + velocity * MLXArray(nextSigma - sigma)
        ).asType(dtype)
        MLX.eval(current.latent)
    }
    return current
}
