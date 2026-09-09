import Foundation
import MLX
import MLXFast
import MLXNN

package typealias LTXV2CompiledBlockForward = @Sendable ([MLXArray]) -> [MLXArray]

package struct LTXV2CompiledBlockVariant: Hashable {
    package let hasVideoSelfAttentionMask: Bool
    package let hasTextProjectionCache: Bool
    package let skipsVideoSelfAttention: Bool
    package let skipsAudioSelfAttention: Bool
    package let skipsAudioToVideoCrossAttention: Bool
    package let skipsVideoToAudioCrossAttention: Bool
}

package struct LTXV2TextProjectionCache {
    package let video: [LTXAttentionProjectedContext]
    package let audio: [LTXAttentionProjectedContext]
}

package enum LTXUnifiedAVTransformerCheckpointKind {
    case ltx23
    case ltx25

    package var videoFeedForwardBias: Bool {
        self == .ltx23
    }
}

package final class LTXUnifiedAVTransformerV2: Module, LTXUnifiedAVTransformerRuntime {
    package let videoDim = 4096
    package let audioDim = 2048
    package let videoHeads = 32
    package let videoHeadDim = 128
    package let audioHeads = 32
    package let audioHeadDim = 64
    package let avCrossHeads = 32
    package let avCrossHeadDim = 64
    package let timestepScaleMultiplier: Float = 1000.0
    package let avCaTimestepScaleMultiplier: Float = 1000.0
    package var execution: LTXTransformerExecution = .eager
    private var parityForwardCount = 0
    var compiledBlockRunner: LTXUnifiedAVTransformerV2Block?
    var compiledBlockForwards: [LTXV2CompiledBlockVariant: LTXV2CompiledBlockForward] = [:]
    private var activeTextProjectionCache: LTXV2TextProjectionCache?
    private var activeTeaCacheController: LTXTeaCacheController?
    private var activeTeaCacheRequest: LTXTeaCacheRequest?
    let videoFeedForwardBias: Bool

    @ModuleInfo(key: "patchify_proj") package var patchifyProj: Linear
    @ModuleInfo(key: "keyframes_abs_pos_embedding") package var keyframesAbsPosEmbedding: MLXArray
    @ModuleInfo(key: "audio_patchify_proj") package var audioPatchifyProj: Linear
    @ModuleInfo(key: "proj_out") package var projOut: Linear
    @ModuleInfo(key: "audio_proj_out") package var audioProjOut: Linear
    @ModuleInfo(key: "scale_shift_table") package var scaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_scale_shift_table") package var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "adaln_single") package var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_adaln_single") package var audioAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "prompt_adaln_single") package var promptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_prompt_adaln_single") package var audioPromptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") package var avCaVideoScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") package var avCaAudioScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") package var avCaA2VGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") package var avCaV2AGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") package var transformerBlocks: [LTXUnifiedAVTransformerV2Block]

    @ModuleInfo(key: "norm_out") package var normOut: LayerNorm
    @ModuleInfo(key: "audio_norm_out") package var audioNormOut: LayerNorm

    package init(checkpointKind: LTXUnifiedAVTransformerCheckpointKind = .ltx25) {
        self.videoFeedForwardBias = checkpointKind.videoFeedForwardBias
        self._patchifyProj.wrappedValue = Linear(128, videoDim, bias: true)
        self._keyframesAbsPosEmbedding.wrappedValue = MLX.zeros([1, videoDim], dtype: .float32)
        self._audioPatchifyProj.wrappedValue = Linear(128, audioDim, bias: true)
        self._projOut.wrappedValue = Linear(videoDim, 128, bias: true)
        self._audioProjOut.wrappedValue = Linear(audioDim, 128, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([2, videoDim], dtype: .float32)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        self._adalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: videoDim, embeddingCoefficient: 9)
        self._audioAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: audioDim, embeddingCoefficient: 9)
        self._promptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: videoDim, embeddingCoefficient: 2)
        self._audioPromptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 2
        )
        self._avCaVideoScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: videoDim,
            embeddingCoefficient: 4
        )
        self._avCaAudioScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 4
        )
        self._avCaA2VGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: videoDim,
            embeddingCoefficient: 1
        )
        self._avCaV2AGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 1
        )
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXUnifiedAVTransformerV2Block(
                videoFeedForwardBias: checkpointKind.videoFeedForwardBias
            )
        }
        self._normOut.wrappedValue = LayerNorm(dimensions: videoDim, eps: 1e-6, affine: false)
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: audioDim, eps: 1e-6, affine: false)
        super.init()
    }

    package func prepareTextProjectionCache(
        videoContext: MLXArray,
        audioContext: MLXArray,
        timestep: MLXArray,
        audioSigma: MLXArray
    ) -> LTXV2TextProjectionCache {
        let videoSigma = timestep.asType(.bfloat16).reshaped(-1)
        let typedAudioSigma = audioSigma.asType(.bfloat16).reshaped(-1)
        let scaledVideoSigma = videoSigma * MLXArray(timestepScaleMultiplier).asType(.bfloat16)
        let scaledAudioSigma = typedAudioSigma * MLXArray(timestepScaleMultiplier).asType(.bfloat16)
        let (videoPromptParams, _) = promptAdalnSingle(
            timestep: scaledVideoSigma,
            hiddenDType: .bfloat16
        )
        let (audioPromptParams, _) = audioPromptAdalnSingle(
            timestep: scaledAudioSigma,
            hiddenDType: .bfloat16
        )
        let videoText = videoContext.asType(.bfloat16)
        let audioText = audioContext.asType(.bfloat16)
        var videoProjections: [LTXAttentionProjectedContext] = []
        var audioProjections: [LTXAttentionProjectedContext] = []
        videoProjections.reserveCapacity(transformerBlocks.count)
        audioProjections.reserveCapacity(transformerBlocks.count)
        for block in transformerBlocks {
            let projections = block.projectTextContexts(
                videoTextEmbeds: videoText,
                audioTextEmbeds: audioText,
                videoPromptAdalnParams: videoPromptParams,
                audioPromptAdalnParams: audioPromptParams
            )
            videoProjections.append(projections.video)
            audioProjections.append(projections.audio)
        }
        MLX.eval(
            videoProjections.flatMap { [$0.keys, $0.values] }
                + audioProjections.flatMap { [$0.keys, $0.values] }
        )
        return LTXV2TextProjectionCache(video: videoProjections, audio: audioProjections)
    }

    package func useTextProjectionCache(_ cache: LTXV2TextProjectionCache?) {
        activeTextProjectionCache = cache
    }

    package func useTeaCache(
        controller: LTXTeaCacheController?,
        request: LTXTeaCacheRequest?
    ) {
        activeTeaCacheController = controller
        activeTeaCacheRequest = request
    }

    package func forward(
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
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let parityIO = LTXAudioToVideoParityIO()
        let parityForwardIndex = parityForwardCount
        parityForwardCount += 1
        func paritySave(_ array: MLXArray, _ name: String) {
            try? parityIO.save(array, suffix: "forward\(parityForwardIndex)_\(name)")
        }

        let videoBatch = videoLatent.dim(0)
        let videoTokens = videoLatent.dim(1)
        let audioBatch = audioLatent.dim(0)
        let audioTokens = audioLatent.dim(1)

        var videoX = patchifyProj(videoLatent.asType(.bfloat16))
        if let videoKeyframesMask {
            let marker = videoKeyframesMask.asType(videoX.dtype)
            videoX = videoX + marker * keyframesAbsPosEmbedding.asType(videoX.dtype)
        }
        var audioX = audioPatchifyProj(audioLatent.asType(.bfloat16))
        paritySave(videoX, "patchified_video")
        paritySave(audioX, "patchified_audio")
        let videoSigma = timestep.asType(videoX.dtype).reshaped(-1)
        let audioSigma = audioSigma.asType(audioX.dtype).reshaped(-1)
        let scaledVideoSigma = videoSigma * MLXArray(timestepScaleMultiplier).asType(videoSigma.dtype)
        let scaledAudioSigma = audioSigma * MLXArray(timestepScaleMultiplier).asType(audioSigma.dtype)
        let scaledA2VGate = audioSigma * MLXArray(avCaTimestepScaleMultiplier).asType(audioSigma.dtype)
        let scaledV2AGate = videoSigma * MLXArray(avCaTimestepScaleMultiplier).asType(videoSigma.dtype)

        let (videoAdalnParams, videoEmbedded): (MLXArray, MLXArray)
        let (avCaVideoParams, _): (MLXArray, MLXArray)
        if let videoTimesteps {
            let scaled = videoTimesteps.asType(videoX.dtype) * MLXArray(timestepScaleMultiplier).asType(videoX.dtype)
            let videoFlat = scaled.reshaped(-1)
            let adaln = adalnSingle(timestep: videoFlat, hiddenDType: videoX.dtype)
            videoAdalnParams = adaln.0.reshaped(videoBatch, videoTokens, -1)
            videoEmbedded = adaln.1.reshaped(videoBatch, videoTokens, -1)
            let avCa = avCaVideoScaleShiftAdalnSingle(timestep: videoFlat, hiddenDType: videoX.dtype)
            avCaVideoParams = avCa.0.reshaped(videoBatch, videoTokens, -1)
        } else {
            (videoAdalnParams, videoEmbedded) = adalnSingle(timestep: scaledVideoSigma, hiddenDType: videoX.dtype)
            (avCaVideoParams, _) = avCaVideoScaleShiftAdalnSingle(timestep: scaledVideoSigma, hiddenDType: videoX.dtype)
        }

        let (audioAdalnParams, audioEmbedded): (MLXArray, MLXArray)
        let (avCaAudioParams, _): (MLXArray, MLXArray)
        if let audioTimesteps {
            let scaled = audioTimesteps.asType(audioX.dtype) * MLXArray(timestepScaleMultiplier).asType(audioX.dtype)
            let audioFlat = scaled.reshaped(-1)
            let adaln = audioAdalnSingle(timestep: audioFlat, hiddenDType: audioX.dtype)
            audioAdalnParams = adaln.0.reshaped(audioBatch, audioTokens, -1)
            audioEmbedded = adaln.1.reshaped(audioBatch, audioTokens, -1)
            let avCa = avCaAudioScaleShiftAdalnSingle(timestep: audioFlat, hiddenDType: audioX.dtype)
            avCaAudioParams = avCa.0.reshaped(audioBatch, audioTokens, -1)
        } else {
            (audioAdalnParams, audioEmbedded) = audioAdalnSingle(timestep: scaledAudioSigma, hiddenDType: audioX.dtype)
            (avCaAudioParams, _) = avCaAudioScaleShiftAdalnSingle(timestep: scaledAudioSigma, hiddenDType: audioX.dtype)
        }

        let (avCaA2VGateParams, _) = avCaA2VGateAdalnSingle(timestep: scaledA2VGate, hiddenDType: videoX.dtype)
        let (avCaV2AGateParams, _) = avCaV2AGateAdalnSingle(timestep: scaledV2AGate, hiddenDType: audioX.dtype)
        let (videoPromptParams, _) = promptAdalnSingle(timestep: scaledVideoSigma, hiddenDType: videoX.dtype)
        let (audioPromptParams, _) = audioPromptAdalnSingle(timestep: scaledAudioSigma, hiddenDType: audioX.dtype)
        paritySave(videoAdalnParams, "video_adaln")
        paritySave(audioAdalnParams, "audio_adaln")
        paritySave(avCaVideoParams, "av_video_adaln")
        paritySave(avCaAudioParams, "av_audio_adaln")
        paritySave(avCaA2VGateParams, "a2v_gate_adaln")
        paritySave(avCaV2AGateParams, "v2a_gate_adaln")
        paritySave(videoPromptParams, "video_prompt_adaln")
        paritySave(audioPromptParams, "audio_prompt_adaln")
        paritySave(videoRope.cos, "video_rope_cos")
        paritySave(videoRope.sin, "video_rope_sin")
        paritySave(audioRope.cos, "audio_rope_cos")
        paritySave(audioRope.sin, "audio_rope_sin")
        paritySave(videoCrossRope.cos, "video_cross_rope_cos")
        paritySave(videoCrossRope.sin, "video_cross_rope_sin")
        paritySave(audioCrossRope.cos, "audio_cross_rope_cos")
        paritySave(audioCrossRope.sin, "audio_cross_rope_sin")

        let videoSelfAttentionMask = prepareLTXSelfAttentionMask(
            videoAttentionMask,
            dtype: videoX.dtype
        )
        let blockInputVideo = videoX
        let blockInputAudio = audioX
        let teaCacheDecision: LTXTeaCacheDecision = if let activeTeaCacheController,
                                                       let activeTeaCacheRequest,
                                                       let firstBlock = transformerBlocks.first {
            activeTeaCacheController.decide(
                request: activeTeaCacheRequest,
                gate: firstBlock.teaCacheGateSignal(
                    videoHidden: videoX,
                    videoAdalnParams: videoAdalnParams
                )
            )
        } else {
            .compute
        }

        switch teaCacheDecision {
        case .reuse(let videoResidual, let audioResidual):
            videoX = blockInputVideo + videoResidual
            audioX = blockInputAudio + audioResidual
        case .compute:
            let evalEvery = Int(ProcessInfo.processInfo.environment["LTX2_DIT_EVAL_EVERY"] ?? "8") ?? 8
            for (index, block) in transformerBlocks.enumerated() {
                let skipsVideoSelfAttention = perturbation.skippedVideoSelfAttentionBlocks.contains(index)
                let skipsAudioSelfAttention = perturbation.skippedAudioSelfAttentionBlocks.contains(index)
                let videoTextProjection = activeTextProjectionCache?.video[index]
                let audioTextProjection = activeTextProjectionCache?.audio[index]
                let out = if execution == .compiled, parityIO.outputPrefix == nil {
                    compiledBlockForward(
                        block: block,
                        videoHidden: videoX,
                        audioHidden: audioX,
                        videoAdalnParams: videoAdalnParams,
                        audioAdalnParams: audioAdalnParams,
                        videoPromptAdalnParams: videoPromptParams,
                        audioPromptAdalnParams: audioPromptParams,
                        avCaVideoParams: avCaVideoParams,
                        avCaAudioParams: avCaAudioParams,
                        avCaA2VGateParams: avCaA2VGateParams,
                        avCaV2AGateParams: avCaV2AGateParams,
                        videoTextEmbeds: videoContext.asType(videoX.dtype),
                        audioTextEmbeds: audioContext.asType(audioX.dtype),
                        videoTextProjection: videoTextProjection,
                        audioTextProjection: audioTextProjection,
                        videoRope: videoRope,
                        videoSelfAttentionMask: videoSelfAttentionMask,
                        audioRope: audioRope,
                        videoCrossRope: videoCrossRope,
                        audioCrossRope: audioCrossRope,
                        skipsVideoSelfAttention: skipsVideoSelfAttention,
                        skipsAudioSelfAttention: skipsAudioSelfAttention,
                        skipsAudioToVideoCrossAttention: perturbation.skipsAudioToVideoCrossAttention,
                        skipsVideoToAudioCrossAttention: perturbation.skipsVideoToAudioCrossAttention
                    )
                } else {
                    block(
                        videoHidden: videoX,
                        audioHidden: audioX,
                        videoAdalnParams: videoAdalnParams,
                        audioAdalnParams: audioAdalnParams,
                        videoPromptAdalnParams: videoPromptParams,
                        audioPromptAdalnParams: audioPromptParams,
                        avCaVideoParams: avCaVideoParams,
                        avCaAudioParams: avCaAudioParams,
                        avCaA2VGateParams: avCaA2VGateParams,
                        avCaV2AGateParams: avCaV2AGateParams,
                        videoTextEmbeds: videoContext.asType(videoX.dtype),
                        audioTextEmbeds: audioContext.asType(audioX.dtype),
                        videoTextProjection: videoTextProjection,
                        audioTextProjection: audioTextProjection,
                        videoRope: videoRope,
                        videoSelfAttentionMask: videoSelfAttentionMask,
                        audioRope: audioRope,
                        videoCrossRope: videoCrossRope,
                        audioCrossRope: audioCrossRope,
                        skipVideoSelfAttention: skipsVideoSelfAttention,
                        skipAudioSelfAttention: skipsAudioSelfAttention,
                        skipAudioToVideoCrossAttention: perturbation.skipsAudioToVideoCrossAttention,
                        skipVideoToAudioCrossAttention: perturbation.skipsVideoToAudioCrossAttention,
                        debugSave: index == 0 ? { array, name in
                            paritySave(array, "block0_\(name)")
                        } : nil
                    )
                }
                videoX = out.video
                audioX = out.audio
                if index == 0 || index == transformerBlocks.count - 1 {
                    paritySave(videoX, "block\(index)_video")
                    paritySave(audioX, "block\(index)_audio")
                }
                if evalEvery > 0 && (index + 1).isMultiple(of: evalEvery) {
                    MLX.eval(videoX, audioX)
                }
            }
            if let activeTeaCacheController, let activeTeaCacheRequest {
                activeTeaCacheController.recordComputedResidual(
                    request: activeTeaCacheRequest,
                    videoResidual: videoX - blockInputVideo,
                    audioResidual: audioX - blockInputAudio
                )
            }
        }

        let videoVelocity = outputBlock(
            videoX,
            embeddedTimestep: videoEmbedded,
            table: scaleShiftTable,
            norm: normOut,
            projection: projOut,
            dim: videoDim
        )
        let audioVelocity = outputBlock(
            audioX,
            embeddedTimestep: audioEmbedded,
            table: audioScaleShiftTable,
            norm: audioNormOut,
            projection: audioProjOut,
            dim: audioDim
        )
        paritySave(videoVelocity, "video_velocity")
        paritySave(audioVelocity, "audio_velocity")
        return (videoVelocity, audioVelocity)
    }

    package func forwardVideoOnly(
        videoLatent: MLXArray,
        videoKeyframesMask: MLXArray?,
        videoAttentionMask: MLXArray? = nil,
        timestep: MLXArray,
        videoTimesteps: MLXArray?,
        videoContext: MLXArray,
        videoRope: LTXRope
    ) -> MLXArray {
        let batch = videoLatent.dim(0)
        let tokens = videoLatent.dim(1)
        var video = patchifyProj(videoLatent.asType(.bfloat16))
        if let videoKeyframesMask {
            video = video
                + videoKeyframesMask.asType(video.dtype)
                    * keyframesAbsPosEmbedding.asType(video.dtype)
        }
        let sigma = timestep.asType(video.dtype).reshaped(-1)
        let scaledSigma = sigma * MLXArray(timestepScaleMultiplier).asType(video.dtype)
        let adalnParams: MLXArray
        let embedded: MLXArray
        if let videoTimesteps {
            let scaled = videoTimesteps.asType(video.dtype)
                * MLXArray(timestepScaleMultiplier).asType(video.dtype)
            let values = adalnSingle(timestep: scaled.reshaped(-1), hiddenDType: video.dtype)
            adalnParams = values.0.reshaped(batch, tokens, -1)
            embedded = values.1.reshaped(batch, tokens, -1)
        } else {
            (adalnParams, embedded) = adalnSingle(
                timestep: scaledSigma,
                hiddenDType: video.dtype
            )
        }
        let (promptParams, _) = promptAdalnSingle(
            timestep: scaledSigma,
            hiddenDType: video.dtype
        )
        let videoSelfAttentionMask = prepareLTXSelfAttentionMask(
            videoAttentionMask,
            dtype: video.dtype
        )
        for (index, block) in transformerBlocks.enumerated() {
            video = block.forwardVideoOnly(
                videoHidden: video,
                videoAdalnParams: adalnParams,
                videoPromptAdalnParams: promptParams,
                videoTextEmbeds: videoContext.asType(video.dtype),
                videoRope: videoRope,
                videoSelfAttentionMask: videoSelfAttentionMask
            )
            if (index + 1).isMultiple(of: 8) {
                MLX.eval(video)
            }
        }
        return outputBlock(
            video,
            embeddedTimestep: embedded,
            table: scaleShiftTable,
            norm: normOut,
            projection: projOut,
            dim: videoDim
        )
    }

    private func outputBlock(
        _ x: MLXArray,
        embeddedTimestep: MLXArray,
        table: MLXArray,
        norm: LayerNorm,
        projection: Linear,
        dim: Int
    ) -> MLXArray {
        let embedded = embeddedTimestep.ndim == 2
            ? embeddedTimestep.expandedDimensions(axis: 1)
            : embeddedTimestep
        let scaleShift = table.reshaped(1, 1, 2, dim) + embedded.expandedDimensions(axis: 2)
        let shift = scaleShift[0..., 0..., 0, 0...]
        let scale = scaleShift[0..., 0..., 1, 0...]
        var y = norm(x)
        y = y * (MLXArray(1.0).asType(y.dtype) + scale) + shift
        return projection(y)
    }
}
