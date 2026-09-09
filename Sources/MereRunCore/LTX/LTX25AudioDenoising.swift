import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func denoiseLTX25AudioOnlyLoop(
    audioLatents: MLXArray,
    audioRope: LTXRope,
    positiveContext: MLXArray,
    negativeContext: MLXArray,
    transformer: LTXAudioOnlyTransformerV2,
    sigmas: [Float],
    guidance: LTXTextToAudioGuidance
) -> MLXArray {
    var current = audioLatents
    let batch = current.dim(0)
    let channels = current.dim(1)
    let frames = current.dim(2)
    let melBins = current.dim(3)
    let dtype = current.dtype
    var lastDenoised: MLXArray?

    func denoised(
        context: MLXArray,
        sigma: Float,
        skippedBlocks: Set<Int>
    ) -> MLXArray {
        let flat = current
            .transposed(0, 2, 1, 3)
            .reshaped(batch, frames, channels * melBins)
        let timesteps = MLX.full(
            [batch, frames],
            values: MLXArray(sigma).asType(dtype)
        )
        let global = MLX.full([batch], values: MLXArray(sigma).asType(dtype))
        let velocity = transformer.forward(
            audioLatent: flat,
            timestep: global,
            audioTimesteps: timesteps,
            audioContext: context,
            audioRope: audioRope,
            skippedSelfAttentionBlocks: skippedBlocks
        )
        return toDenoised(
            noisy: current,
            velocity: velocity
                .reshaped(batch, frames, channels, melBins)
                .transposed(0, 2, 1, 3),
            sigma: sigma
        )
    }

    for index in 0..<(sigmas.count - 1) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let guided: MLXArray
        if guidance.shouldSkip(step: index), let lastDenoised {
            guided = lastDenoised
        } else {
            let conditioned = denoised(context: positiveContext, sigma: sigma, skippedBlocks: [])
            let negative = guidance.classifierFreeScale == 1
                ? conditioned
                : denoised(context: negativeContext, sigma: sigma, skippedBlocks: [])
            let perturbed = guidance.spatioTemporalScale == 0
                ? conditioned
                : denoised(
                    context: positiveContext,
                    sigma: sigma,
                    skippedBlocks: guidance.spatioTemporalBlocks
                )
            guided = guidance.combine(
                conditioned: conditioned,
                negativeText: negative,
                perturbed: perturbed
            )
            lastDenoised = guided
        }
        if nextSigma == 0 {
            current = guided
        } else {
            let sigmaArray = MLXArray(sigma).asType(dtype)
            let nextArray = MLXArray(nextSigma).asType(dtype)
            current = guided + nextArray * (current - guided) / sigmaArray
        }
        MLX.eval(current)
    }
    return current
}
