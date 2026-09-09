import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func predictGuidedLTX25AV(
    videoState: LTX25VideoTokenState,
    audioLatents: MLXArray,
    sigma: Float,
    stepIndex: Int,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray,
    positiveAudioContext: MLXArray,
    negativeAudioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    videoGuidance: LTXMultiModalGuidance,
    audioGuidance: LTXMultiModalGuidance,
    guidanceProjectionCache: LTXGuidanceProjectionCacheMode,
    guidanceProjectionCacheMetrics: LTXGuidanceProjectionCacheMetrics,
    teaCacheController: LTXTeaCacheController?,
    teaCacheStage: LTXTeaCacheStage,
    teaCachePipelineStage: LTXTeaCachePipelineStage,
    teaCacheStepCount: Int,
    audioConditioning: LTXLatentConditioningState?,
    forceUnconditional: Bool,
    lastVideo: MLXArray?,
    lastAudio: MLXArray?
) -> LTXGuidedAVPrediction {
    let dtype = videoState.latent.dtype
    let audioShape = (
        batch: audioLatents.dim(0),
        channels: audioLatents.dim(1),
        frames: audioLatents.dim(2),
        melBins: audioLatents.dim(3)
    )
    let flatAudio = audioLatents
        .transposed(0, 2, 1, 3)
        .reshaped(audioShape.batch, audioShape.frames, audioShape.channels * audioShape.melBins)
    let videoTimesteps = videoState.denoiseMask.squeezed(axis: -1)
        * MLXArray(sigma).asType(dtype)
    let audioTimesteps = audioConditioning.map {
        $0.denoiseMask[0..., 0, 0..., 0] * MLXArray(sigma).asType(dtype)
    } ?? MLX.full(
        [audioShape.batch, audioShape.frames],
        values: MLXArray(sigma).asType(dtype)
    )
    let globalTimestep = MLX.full(
        [videoState.targetShape.batch],
        values: MLXArray(sigma).asType(dtype)
    )

    let needsUnconditional = forceUnconditional
        || videoGuidance.classifierFreeScale != 1
        || audioGuidance.classifierFreeScale != 1
    let needsPerturbed = videoGuidance.spatioTemporalScale != 0
        || audioGuidance.spatioTemporalScale != 0
    let needsIsolated = videoGuidance.modalityScale != 1 || audioGuidance.modalityScale != 1
    let positivePredictionCount = 1 + (needsPerturbed ? 1 : 0) + (needsIsolated ? 1 : 0)
    let transformerV2 = transformer as? LTXUnifiedAVTransformerV2
    let cacheDecision = ltxGuidanceProjectionCacheDecision(
        mode: guidanceProjectionCache,
        positivePredictionCount: positivePredictionCount,
        batchSize: videoState.targetShape.batch,
        videoTextTokens: positiveVideoContext.dim(1),
        audioTextTokens: positiveAudioContext.dim(1),
        blockCount: 48,
        bytesPerElement: 2,
        activeMemoryBytes: UInt64(Memory.activeMemory),
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )
    let textProjectionCache: LTXV2TextProjectionCache?
    if cacheDecision.shouldCache, let transformerV2 {
        let buildStart = ltxMonotonicSeconds()
        textProjectionCache = transformerV2.prepareTextProjectionCache(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            timestep: globalTimestep,
            audioSigma: globalTimestep
        )
        guidanceProjectionCacheMetrics.buildSeconds += ltxMonotonicSeconds() - buildStart
        guidanceProjectionCacheMetrics.buildCount += 1
        guidanceProjectionCacheMetrics.reuseCount += positivePredictionCount - 1
    } else {
        textProjectionCache = nil
        if guidanceProjectionCache != .disabled, positivePredictionCount > 1 {
            guidanceProjectionCacheMetrics.fallbackCount += 1
        }
    }
    defer {
        transformerV2?.useTextProjectionCache(nil)
        transformerV2?.useTeaCache(controller: nil, request: nil)
    }

    func predict(
        videoContext: MLXArray,
        audioContext: MLXArray,
        perturbation: LTXAudioToVideoPerturbation,
        usePositiveTextProjectionCache: Bool,
        teaCacheBranch: LTXTeaCacheBranch
    ) -> (video: MLXArray, audio: MLXArray) {
        transformerV2?.useTextProjectionCache(
            usePositiveTextProjectionCache ? textProjectionCache : nil
        )
        transformerV2?.useTeaCache(
            controller: teaCacheController,
            request: teaCacheController.map { _ in
                LTXTeaCacheRequest(
                    key: LTXTeaCacheKey(
                        branch: teaCacheBranch,
                        stage: teaCacheStage,
                        pipelineStage: teaCachePipelineStage
                    ),
                    stepIndex: stepIndex,
                    stepCount: teaCacheStepCount
                )
            }
        )
        return predictLTX25JointAVTokenDenoised(
            videoState: videoState,
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
            audioShape: audioShape
        )
    }

    let conditioned = predict(
        videoContext: positiveVideoContext,
        audioContext: positiveAudioContext,
        perturbation: .none,
        usePositiveTextProjectionCache: true,
        teaCacheBranch: .conditioned
    )
    let negative = needsUnconditional ? predict(
        videoContext: negativeVideoContext,
        audioContext: negativeAudioContext,
        perturbation: .none,
        usePositiveTextProjectionCache: false,
        teaCacheBranch: .unconditional
    ) : conditioned
    let perturbed = needsPerturbed ? predict(
        videoContext: positiveVideoContext,
        audioContext: positiveAudioContext,
        perturbation: .spatioTemporal(
            videoBlocks: videoGuidance.spatioTemporalBlocks,
            audioBlocks: audioGuidance.spatioTemporalBlocks
        ),
        usePositiveTextProjectionCache: true,
        teaCacheBranch: .perturbed
    ) : conditioned
    let isolated = needsIsolated ? predict(
        videoContext: positiveVideoContext,
        audioContext: positiveAudioContext,
        perturbation: .isolatedModalities,
        usePositiveTextProjectionCache: true,
        teaCacheBranch: .isolated
    ) : conditioned

    var guidedVideo = videoGuidance.shouldSkip(step: stepIndex)
        ? (lastVideo ?? conditioned.video)
        : videoGuidance.combine(
            conditioned: conditioned.video,
            negativeText: negative.video,
            perturbed: perturbed.video,
            isolatedModality: isolated.video
        )
    var guidedAudio = audioGuidance.shouldSkip(step: stepIndex)
        ? (lastAudio ?? conditioned.audio)
        : audioGuidance.combine(
            conditioned: conditioned.audio,
            negativeText: negative.audio,
            perturbed: perturbed.audio,
            isolatedModality: isolated.audio
        )
    let one = MLXArray(1).asType(dtype)
    guidedVideo = guidedVideo * videoState.denoiseMask
        + videoState.cleanLatent * (one - videoState.denoiseMask)
    if let audioConditioning {
        guidedAudio = guidedAudio * audioConditioning.denoiseMask
            + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
    }
    MLX.eval(guidedVideo, guidedAudio, negative.video, negative.audio)
    return LTXGuidedAVPrediction(
        video: guidedVideo,
        audio: guidedAudio,
        unconditionalVideo: negative.video,
        unconditionalAudio: negative.audio
    )
}

struct LTXRandomKeyStream {
    private var key: MLXArray

    init(seed: Int) {
        key = MLXRandom.key(UInt64(bitPattern: Int64(seed)))
    }

    mutating func normal(shape: [Int]) -> MLXArray {
        let (nextKey, drawKey) = MLXRandom.split(key: key)
        key = nextKey
        return MLXRandom.normal(shape, key: drawKey)
    }
}

func ltxNormalizedRes2sNoise(
    shape: [Int],
    dtype: DType,
    stream: inout LTXRandomKeyStream
) -> MLXArray {
    var noise = stream.normal(shape: shape).asType(.float32)
    let globalMean = MLX.mean(noise)
    let centered = noise - globalMean
    let elementCount = max(1, shape.reduce(1, *))
    let globalCorrection = elementCount > 1
        ? Float(elementCount) / Float(elementCount - 1)
        : 1
    let globalVariance = MLX.mean(centered * centered) * MLXArray(globalCorrection)
    noise = centered / MLX.sqrt(globalVariance + MLXArray(1e-12))
    let axes = [max(0, shape.count - 2), max(0, shape.count - 1)]
    let channelMean = MLX.mean(noise, axes: axes, keepDims: true)
    let channelCentered = noise - channelMean
    let channelElementCount = max(1, shape[axes[0]] * shape[axes[1]])
    let channelCorrection = channelElementCount > 1
        ? Float(channelElementCount) / Float(channelElementCount - 1)
        : 1
    let channelVariance = MLX.mean(
        channelCentered * channelCentered,
        axes: axes,
        keepDims: true
    ) * MLXArray(channelCorrection)
    return (channelCentered / MLX.sqrt(channelVariance + MLXArray(1e-12))).asType(dtype)
}
