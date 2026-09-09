import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func denoiseGuidedLTX25AVTokenLoop(
    videoState: LTX25VideoTokenState,
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
    videoGuidance: LTXMultiModalGuidance,
    audioGuidance: LTXMultiModalGuidance,
    sampler: LTXSamplerConfiguration,
    seed: Int,
    guidanceProjectionCache: LTXGuidanceProjectionCacheMode,
    guidanceProjectionCacheMetrics: LTXGuidanceProjectionCacheMetrics,
    teaCacheController: LTXTeaCacheController?,
    teaCachePipelineStage: LTXTeaCachePipelineStage,
    audioConditioning: LTXLatentConditioningState? = nil
) -> (video: LTX25VideoTokenState, audio: MLXArray) {
    var currentVideo = videoState
    var currentAudio = audioLatents
    let dtype = videoState.latent.dtype

    MLXRandom.seed(UInt64(bitPattern: Int64(seed &+ sampler.noiseSeedOffset)))
    var lastVideo: MLXArray?
    var lastAudio: MLXArray?
    var previousVideoVelocity: MLXArray?
    var previousAudioVelocity: MLXArray?

    func prediction(
        video: LTX25VideoTokenState,
        audio: MLXArray,
        sigma: Float,
        stepIndex: Int,
        teaCacheStage: LTXTeaCacheStage = .primary,
        teaCacheStepCount: Int,
        forceUnconditional: Bool = false,
        cachedVideo: MLXArray? = nil,
        cachedAudio: MLXArray? = nil
    ) -> LTXGuidedAVPrediction {
        predictGuidedLTX25AV(
            videoState: video,
            audioLatents: audio,
            sigma: sigma,
            stepIndex: stepIndex,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            positiveVideoContext: positiveVideoContext,
            negativeVideoContext: negativeVideoContext,
            positiveAudioContext: positiveAudioContext,
            negativeAudioContext: negativeAudioContext,
            transformer: transformer,
            videoGuidance: videoGuidance,
            audioGuidance: audioGuidance,
            guidanceProjectionCache: guidanceProjectionCache,
            guidanceProjectionCacheMetrics: guidanceProjectionCacheMetrics,
            teaCacheController: teaCacheController,
            teaCacheStage: teaCacheStage,
            teaCachePipelineStage: teaCachePipelineStage,
            teaCacheStepCount: teaCacheStepCount,
            audioConditioning: audioConditioning,
            forceUnconditional: forceUnconditional,
            lastVideo: cachedVideo,
            lastAudio: cachedAudio
        )
    }

    if sampler.mode == .res2s {
        let fullStepCount = sigmas.count - 1
        var workingSigmas = sigmas
        if workingSigmas.last == 0 {
            workingSigmas.removeLast()
            workingSigmas.append(contentsOf: [0.0011, 0])
        }
        var stepNoiseStream = LTXRandomKeyStream(seed: sampler.noiseSeedOffset)
        var substepNoiseStream = LTXRandomKeyStream(seed: sampler.substepNoiseSeedOffset)
        for index in 0..<fullStepCount {
            let sigma = workingSigmas[index]
            let nextSigma = workingSigmas[index + 1]
            var anchorVideo = currentVideo.latent.asType(.float32)
            var anchorAudio = currentAudio.asType(.float32)
            let first = prediction(
                video: currentVideo,
                audio: currentAudio,
                sigma: sigma,
                stepIndex: index,
                teaCacheStepCount: fullStepCount + 1,
                cachedVideo: lastVideo,
                cachedAudio: lastAudio
            )
            lastVideo = first.video
            lastAudio = first.audio
            let step = -log(Double(nextSigma) / Double(sigma))
            let coefficients = LTXRes2s.coefficients(step: step)
            var epsilonVideo = first.video.asType(.float32) - anchorVideo
            var epsilonAudio = first.audio.asType(.float32) - anchorAudio
            let midpointSigma = sqrt(sigma * nextSigma)
            var midpointVideo = anchorVideo
                + MLXArray(Float(step * coefficients.a21)) * epsilonVideo
            var midpointAudio = anchorAudio
                + MLXArray(Float(step * coefficients.a21)) * epsilonAudio
            midpointVideo = ltxRes2sSDEStep(
                sample: anchorVideo,
                denoised: midpointVideo,
                sigma: sigma,
                nextSigma: midpointSigma,
                eta: 0.5,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorVideo.shape,
                    dtype: dtype,
                    stream: &substepNoiseStream
                )
            )
            midpointAudio = ltxRes2sSDEStep(
                sample: anchorAudio,
                denoised: midpointAudio,
                sigma: sigma,
                nextSigma: midpointSigma,
                eta: 0.5,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorAudio.shape,
                    dtype: dtype,
                    stream: &substepNoiseStream
                )
            )
            if sampler.res2sBongMath, step < 0.5, sigma > 0.03 {
                for _ in 0..<sampler.res2sBongMathMaxIterations {
                    anchorVideo = midpointVideo.asType(.float32)
                        - MLXArray(Float(step * coefficients.a21)) * epsilonVideo
                    anchorAudio = midpointAudio.asType(.float32)
                        - MLXArray(Float(step * coefficients.a21)) * epsilonAudio
                    epsilonVideo = first.video.asType(.float32) - anchorVideo
                    epsilonAudio = first.audio.asType(.float32) - anchorAudio
                }
            }
            var midpointState = currentVideo
            midpointState.latent = midpointVideo.asType(dtype)
            let second = prediction(
                video: midpointState,
                audio: midpointAudio.asType(dtype),
                sigma: midpointSigma,
                stepIndex: index,
                teaCacheStage: .midpoint,
                teaCacheStepCount: fullStepCount
            )
            let nextVideoEstimate = anchorVideo + MLXArray(Float(step))
                * (MLXArray(Float(coefficients.b1)) * epsilonVideo
                    + MLXArray(Float(coefficients.b2))
                        * (second.video.asType(.float32) - anchorVideo))
            let nextAudioEstimate = anchorAudio + MLXArray(Float(step))
                * (MLXArray(Float(coefficients.b1)) * epsilonAudio
                    + MLXArray(Float(coefficients.b2))
                        * (second.audio.asType(.float32) - anchorAudio))
            currentVideo.latent = ltxRes2sSDEStep(
                sample: anchorVideo,
                denoised: nextVideoEstimate,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorVideo.shape,
                    dtype: dtype,
                    stream: &stepNoiseStream
                )
            )
            currentAudio = ltxRes2sSDEStep(
                sample: anchorAudio,
                denoised: nextAudioEstimate,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorAudio.shape,
                    dtype: dtype,
                    stream: &stepNoiseStream
                )
            )
            let one = MLXArray(1).asType(dtype)
            currentVideo.latent = currentVideo.latent * currentVideo.denoiseMask
                + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
            if let audioConditioning {
                currentAudio = currentAudio * audioConditioning.denoiseMask
                    + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
            }
            MLX.eval(currentVideo.latent, currentAudio)
        }
        if workingSigmas.last == 0 {
            let finalSigma = workingSigmas[fullStepCount]
            let final = prediction(
                video: currentVideo,
                audio: currentAudio,
                sigma: finalSigma,
                stepIndex: fullStepCount,
                teaCacheStepCount: fullStepCount + 1,
                cachedVideo: lastVideo,
                cachedAudio: lastAudio
            )
            currentVideo.latent = final.video
            currentAudio = final.audio
            MLX.eval(currentVideo.latent, currentAudio)
        }
        return (currentVideo, currentAudio)
    }

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        var result = prediction(
            video: currentVideo,
            audio: currentAudio,
            sigma: sigma,
            stepIndex: index,
            teaCacheStepCount: max(1, sigmas.count - 1),
            forceUnconditional: sampler.mode == .cfgPlusPlus,
            cachedVideo: lastVideo,
            cachedAudio: lastAudio
        )
        lastVideo = result.video
        lastAudio = result.audio

        if sampler.mode == .gradientEstimatingEuler {
            let videoVelocity = (
                currentVideo.latent.asType(.float32) - result.video.asType(.float32)
            ) / MLXArray(sigma)
            let audioVelocity = (
                currentAudio.asType(.float32) - result.audio.asType(.float32)
            ) / MLXArray(sigma)
            if let previousVideoVelocity, let previousAudioVelocity {
                let correctedVideo = previousVideoVelocity
                    + MLXArray(sampler.gradientEstimationGamma)
                        * (videoVelocity - previousVideoVelocity)
                let correctedAudio = previousAudioVelocity
                    + MLXArray(sampler.gradientEstimationGamma)
                        * (audioVelocity - previousAudioVelocity)
                result = LTXGuidedAVPrediction(
                    video: currentVideo.latent.asType(.float32)
                        - MLXArray(sigma) * correctedVideo,
                    audio: currentAudio.asType(.float32)
                        - MLXArray(sigma) * correctedAudio,
                    unconditionalVideo: result.unconditionalVideo,
                    unconditionalAudio: result.unconditionalAudio
                )
            }
            previousVideoVelocity = videoVelocity
            previousAudioVelocity = audioVelocity
        }

        switch sampler.mode {
        case .euler, .gradientEstimatingEuler:
            currentVideo.latent = ltxEulerStep(
                sample: currentVideo.latent,
                denoised: result.video,
                sigma: sigma,
                nextSigma: nextSigma
            )
            currentAudio = ltxEulerStep(
                sample: currentAudio,
                denoised: result.audio,
                sigma: sigma,
                nextSigma: nextSigma
            )
        case .eulerAncestral:
            currentVideo.latent = ltxAncestralEulerStep(
                sample: currentVideo.latent,
                denoised: result.video,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentVideo.latent.shape).asType(dtype),
                eta: sampler.eta
            )
            currentAudio = ltxAncestralEulerStep(
                sample: currentAudio,
                denoised: result.audio,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentAudio.shape).asType(dtype),
                eta: sampler.eta
            )
        case .cfgPlusPlus:
            currentVideo.latent = ltxCfgPlusPlusStep(
                sample: currentVideo.latent,
                denoised: result.video,
                unconditionalDenoised: result.unconditionalVideo,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: MLXRandom.normal(currentVideo.latent.shape).asType(dtype)
            )
            currentAudio = ltxCfgPlusPlusStep(
                sample: currentAudio,
                denoised: result.audio,
                unconditionalDenoised: result.unconditionalAudio,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: MLXRandom.normal(currentAudio.shape).asType(dtype)
            )
        case .res2s:
            preconditionFailure("Res2s is handled by the dedicated second-order loop.")
        }
        let one = MLXArray(1).asType(dtype)
        currentVideo.latent = currentVideo.latent * currentVideo.denoiseMask
            + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
        if let audioConditioning {
            currentAudio = currentAudio * audioConditioning.denoiseMask
                + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
        }
        MLX.eval(currentVideo.latent, currentAudio)
    }
    return (currentVideo, currentAudio)
}
