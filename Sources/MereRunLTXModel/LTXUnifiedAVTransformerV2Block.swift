import Foundation
import MLX
import MLXFast
import MLXNN

package final class LTXUnifiedAVTransformerV2Block: Module {
    package let videoDim = 4096
    package let audioDim = 2048

    @ModuleInfo(key: "attn1") package var attn1: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn1") package var audioAttn1: LTXDistilledAttention
    @ModuleInfo(key: "attn2") package var attn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn2") package var audioAttn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_to_video_attn") package var audioToVideoAttn: LTXDistilledAttention
    @ModuleInfo(key: "video_to_audio_attn") package var videoToAudioAttn: LTXDistilledAttention
    @ModuleInfo(key: "ff") package var ff: LTXDistilledFeedForward
    @ModuleInfo(key: "audio_ff") package var audioFF: LTXDistilledFeedForward
    @ModuleInfo(key: "scale_shift_table") package var scaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_scale_shift_table") package var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "prompt_scale_shift_table") package var promptScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_prompt_scale_shift_table") package var audioPromptScaleShiftTable: MLXArray
    @ModuleInfo(key: "scale_shift_table_a2v_ca_video") package var scaleShiftTableA2VCAVideo: MLXArray
    @ModuleInfo(key: "scale_shift_table_a2v_ca_audio") package var scaleShiftTableA2VCAAudio: MLXArray

    package init(videoFeedForwardBias: Bool = false) {
        self._attn1.wrappedValue = LTXDistilledAttention(
            queryDim: videoDim,
            contextDim: nil,
            heads: 32,
            headDim: 128,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._audioAttn1.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: nil,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._attn2.wrappedValue = LTXDistilledAttention(
            queryDim: videoDim,
            contextDim: videoDim,
            heads: 32,
            headDim: 128,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._audioAttn2.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: audioDim,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._audioToVideoAttn.wrappedValue = LTXDistilledAttention(
            queryDim: videoDim,
            contextDim: audioDim,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._videoToAudioAttn.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: videoDim,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._ff.wrappedValue = LTXDistilledFeedForward(
            dim: videoDim,
            dimOut: videoDim,
            mult: 4,
            bias: videoFeedForwardBias
        )
        self._audioFF.wrappedValue = LTXDistilledFeedForward(dim: audioDim, dimOut: audioDim, mult: 4)
        self._scaleShiftTable.wrappedValue = MLX.zeros([9, videoDim], dtype: .float32)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([9, audioDim], dtype: .float32)
        self._promptScaleShiftTable.wrappedValue = MLX.zeros([2, videoDim], dtype: .float32)
        self._audioPromptScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        self._scaleShiftTableA2VCAVideo.wrappedValue = MLX.zeros([5, videoDim], dtype: .float32)
        self._scaleShiftTableA2VCAAudio.wrappedValue = MLX.zeros([5, audioDim], dtype: .float32)
        super.init()
    }

    package func callAsFunction(
        videoHidden: MLXArray,
        audioHidden: MLXArray,
        videoAdalnParams: MLXArray,
        audioAdalnParams: MLXArray,
        videoPromptAdalnParams: MLXArray,
        audioPromptAdalnParams: MLXArray,
        avCaVideoParams: MLXArray,
        avCaAudioParams: MLXArray,
        avCaA2VGateParams: MLXArray,
        avCaV2AGateParams: MLXArray,
        videoTextEmbeds: MLXArray,
        audioTextEmbeds: MLXArray,
        videoTextProjection: LTXAttentionProjectedContext? = nil,
        audioTextProjection: LTXAttentionProjectedContext? = nil,
        videoRope: (cos: MLXArray, sin: MLXArray),
        videoSelfAttentionMask: MLXArray?,
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        skipVideoSelfAttention: Bool,
        skipAudioSelfAttention: Bool,
        skipAudioToVideoCrossAttention: Bool,
        skipVideoToAudioCrossAttention: Bool,
        debugSave: ((MLXArray, String) -> Void)? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        var video = videoHidden
        var audio = audioHidden

        let vAda = unpackAdaln(videoAdalnParams, table: scaleShiftTable, count: 9, dim: videoDim)
        let aAda = unpackAdaln(audioAdalnParams, table: audioScaleShiftTable, count: 9, dim: audioDim)

        if !skipVideoSelfAttention {
            var videoNorm = rmsNormNoWeight(video)
            videoNorm = videoNorm * (MLXArray(1.0).asType(videoNorm.dtype) + vAda[1]) + vAda[0]
            video = video + attn1(
                videoNorm,
                context: nil,
                mask: videoSelfAttentionMask,
                rope: videoRope
            ) * vAda[2]
        }
        debugSave?(video, "video_after_self_attention")

        if !skipAudioSelfAttention {
            var audioNorm = rmsNormNoWeight(audio)
            audioNorm = audioNorm * (MLXArray(1.0).asType(audioNorm.dtype) + aAda[1]) + aAda[0]
            audio = audio + audioAttn1(audioNorm, context: nil, mask: nil, rope: audioRope) * aAda[2]
        }
        debugSave?(audio, "audio_after_self_attention")

        let vPromptAda = unpackAdaln(videoPromptAdalnParams, table: promptScaleShiftTable, count: 2, dim: videoDim)
        let videoText = videoTextEmbeds * (MLXArray(1.0).asType(videoTextEmbeds.dtype) + vPromptAda[1]) + vPromptAda[0]
        var videoCrossNorm = rmsNormNoWeight(video)
        videoCrossNorm = videoCrossNorm * (MLXArray(1.0).asType(videoCrossNorm.dtype) + vAda[7]) + vAda[6]
        video = video + attn2(
            videoCrossNorm,
            context: videoText,
            mask: nil,
            rope: nil,
            projectedContext: videoTextProjection
        ) * vAda[8]
        debugSave?(video, "video_after_text_attention")

        let aPromptAda = unpackAdaln(
            audioPromptAdalnParams,
            table: audioPromptScaleShiftTable,
            count: 2,
            dim: audioDim
        )
        let audioText = audioTextEmbeds * (MLXArray(1.0).asType(audioTextEmbeds.dtype) + aPromptAda[1]) + aPromptAda[0]
        var audioCrossNorm = rmsNormNoWeight(audio)
        audioCrossNorm = audioCrossNorm * (MLXArray(1.0).asType(audioCrossNorm.dtype) + aAda[7]) + aAda[6]
        audio = audio + audioAttn2(
            audioCrossNorm,
            context: audioText,
            mask: nil,
            rope: nil,
            projectedContext: audioTextProjection
        ) * aAda[8]
        debugSave?(audio, "audio_after_text_attention")

        let videoNorm3 = rmsNormNoWeight(video)
        let audioNorm3 = rmsNormNoWeight(audio)
        let vAV = unpackAdaln(avCaVideoParams, table: scaleShiftTableA2VCAVideo, count: 4, dim: videoDim)
        let aAV = unpackAdaln(avCaAudioParams, table: scaleShiftTableA2VCAAudio, count: 4, dim: audioDim)
        let a2vGate = unpackAVGate(avCaA2VGateParams, table: scaleShiftTableA2VCAVideo, dim: videoDim)
        let v2aGate = unpackAVGate(avCaV2AGateParams, table: scaleShiftTableA2VCAAudio, dim: audioDim)

        if !skipAudioToVideoCrossAttention {
            let videoQA2V = videoNorm3 * (MLXArray(1.0).asType(videoNorm3.dtype) + vAV[0]) + vAV[1]
            let audioKVA2V = audioNorm3 * (MLXArray(1.0).asType(audioNorm3.dtype) + aAV[0]) + aAV[1]
            video = video + audioToVideoAttn(
                videoQA2V,
                context: audioKVA2V,
                mask: nil,
                rope: videoCrossRope,
                keyRope: audioCrossRope
            ) * a2vGate
        }
        debugSave?(video, "video_after_audio_cross_attention")

        if !skipVideoToAudioCrossAttention {
            let audioQV2A = audioNorm3 * (MLXArray(1.0).asType(audioNorm3.dtype) + aAV[2]) + aAV[3]
            let videoKVV2A = videoNorm3 * (MLXArray(1.0).asType(videoNorm3.dtype) + vAV[2]) + vAV[3]
            audio = audio + videoToAudioAttn(
                audioQV2A,
                context: videoKVV2A,
                mask: nil,
                rope: audioCrossRope,
                keyRope: videoCrossRope
            ) * v2aGate
        }
        debugSave?(audio, "audio_after_video_cross_attention")

        var videoFFNorm = rmsNormNoWeight(video)
        videoFFNorm = videoFFNorm * (MLXArray(1.0).asType(videoFFNorm.dtype) + vAda[4]) + vAda[3]
        video = video + ff(videoFFNorm) * vAda[5]
        debugSave?(video, "video_after_feed_forward")

        var audioFFNorm = rmsNormNoWeight(audio)
        audioFFNorm = audioFFNorm * (MLXArray(1.0).asType(audioFFNorm.dtype) + aAda[4]) + aAda[3]
        audio = audio + audioFF(audioFFNorm) * aAda[5]
        debugSave?(audio, "audio_after_feed_forward")

        return (video, audio)
    }

    package func projectTextContexts(
        videoTextEmbeds: MLXArray,
        audioTextEmbeds: MLXArray,
        videoPromptAdalnParams: MLXArray,
        audioPromptAdalnParams: MLXArray
    ) -> (video: LTXAttentionProjectedContext, audio: LTXAttentionProjectedContext) {
        let videoPrompt = unpackAdaln(
            videoPromptAdalnParams,
            table: promptScaleShiftTable,
            count: 2,
            dim: videoDim
        )
        let videoText = videoTextEmbeds
            * (MLXArray(1).asType(videoTextEmbeds.dtype) + videoPrompt[1])
            + videoPrompt[0]
        let audioPrompt = unpackAdaln(
            audioPromptAdalnParams,
            table: audioPromptScaleShiftTable,
            count: 2,
            dim: audioDim
        )
        let audioText = audioTextEmbeds
            * (MLXArray(1).asType(audioTextEmbeds.dtype) + audioPrompt[1])
            + audioPrompt[0]
        return (
            attn2.projectContext(videoText),
            audioAttn2.projectContext(audioText)
        )
    }

    package func teaCacheGateSignal(
        videoHidden: MLXArray,
        videoAdalnParams: MLXArray
    ) -> MLXArray {
        let adaln = unpackAdaln(
            videoAdalnParams,
            table: scaleShiftTable,
            count: 9,
            dim: videoDim
        )
        let normalized = rmsNormNoWeight(videoHidden)
        return normalized * (MLXArray(1).asType(normalized.dtype) + adaln[1]) + adaln[0]
    }

    package func forwardVideoOnly(
        videoHidden: MLXArray,
        videoAdalnParams: MLXArray,
        videoPromptAdalnParams: MLXArray,
        videoTextEmbeds: MLXArray,
        videoRope: LTXRope,
        videoSelfAttentionMask: MLXArray?
    ) -> MLXArray {
        var video = videoHidden
        let adaln = unpackAdaln(
            videoAdalnParams,
            table: scaleShiftTable,
            count: 9,
            dim: videoDim
        )
        var selfNorm = rmsNormNoWeight(video)
        selfNorm = selfNorm * (MLXArray(1).asType(selfNorm.dtype) + adaln[1]) + adaln[0]
        video = video + attn1(
            selfNorm,
            context: nil,
            mask: videoSelfAttentionMask,
            rope: videoRope
        ) * adaln[2]

        let promptAdaln = unpackAdaln(
            videoPromptAdalnParams,
            table: promptScaleShiftTable,
            count: 2,
            dim: videoDim
        )
        let text = videoTextEmbeds
            * (MLXArray(1).asType(videoTextEmbeds.dtype) + promptAdaln[1])
            + promptAdaln[0]
        var textNorm = rmsNormNoWeight(video)
        textNorm = textNorm * (MLXArray(1).asType(textNorm.dtype) + adaln[7]) + adaln[6]
        video = video + attn2(textNorm, context: text, mask: nil, rope: nil) * adaln[8]

        var feedForwardNorm = rmsNormNoWeight(video)
        feedForwardNorm = feedForwardNorm
            * (MLXArray(1).asType(feedForwardNorm.dtype) + adaln[4])
            + adaln[3]
        return video + ff(feedForwardNorm) * adaln[5]
    }

    private func unpackAdaln(_ params: MLXArray, table: MLXArray, count: Int, dim: Int) -> [MLXArray] {
        if params.ndim == 2 {
            let values = params.reshaped(-1, count, dim) + table[0..<count, 0...].reshaped(1, count, dim)
            return (0..<count).map { values[0..., $0, 0...].expandedDimensions(axis: 1) }
        }

        let batch = params.dim(0)
        let tokens = params.dim(1)
        let values = params.reshaped(batch, tokens, count, dim)
            + table[0..<count, 0...].reshaped(1, 1, count, dim)
        return (0..<count).map { values[0..., 0..., $0, 0...] }
    }

    private func unpackAVGate(_ params: MLXArray, table: MLXArray, dim: Int) -> MLXArray {
        if params.ndim == 2 {
            return (params + table[4, 0...]).expandedDimensions(axis: 1)
        }
        return params + table[4, 0...].reshaped(1, 1, dim)
    }
}
