import Foundation
import MLX
import MLXFast
import MLXNN

package final class LTXUnifiedAVTransformer: Module, LTXUnifiedAVTransformerRuntime {
    package let videoDim = 4096
    package let audioDim = 2048
    package let videoHeads = 32
    package let videoHeadDim = 128
    package let audioHeads = 32
    package let audioHeadDim = 64
    package let timestepScaleMultiplier: Float = 1000.0

    @ModuleInfo(key: "patchify_proj") package var patchifyProj: Linear
    @ModuleInfo(key: "adaln_single") package var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "caption_projection") package var captionProjection: LTXPixArtTextProjection
    @ModuleInfo(key: "scale_shift_table") package var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm_out") package var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") package var projOut: Linear

    @ModuleInfo(key: "audio_patchify_proj") package var audioPatchifyProj: Linear
    @ModuleInfo(key: "audio_adaln_single") package var audioAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_caption_projection") package var audioCaptionProjection: LTXPixArtTextProjection
    @ModuleInfo(key: "audio_scale_shift_table") package var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_norm_out") package var audioNormOut: LayerNorm
    @ModuleInfo(key: "audio_proj_out") package var audioProjOut: Linear

    @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") package var avCaVideoScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") package var avCaAudioScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") package var avCaA2VGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") package var avCaV2AGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") package var transformerBlocks: [LTXUnifiedAVTransformerBlock]

    package override init() {
        self._patchifyProj.wrappedValue = Linear(128, 4096, bias: true)
        self._adalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 6)
        self._captionProjection.wrappedValue = LTXPixArtTextProjection(inFeatures: 3840, hiddenSize: 4096, outFeatures: 4096, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([2, 4096], dtype: .float32)
        self._normOut.wrappedValue = LayerNorm(dimensions: 4096, eps: 1e-6, affine: false)
        self._projOut.wrappedValue = Linear(4096, 128, bias: true)

        self._audioPatchifyProj.wrappedValue = Linear(128, 2048, bias: true)
        self._audioAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 2048, embeddingCoefficient: 6)
        self._audioCaptionProjection.wrappedValue = LTXPixArtTextProjection(inFeatures: 3840, hiddenSize: 2048, outFeatures: 2048, bias: true)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([2, 2048], dtype: .float32)
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: 2048, eps: 1e-6, affine: false)
        self._audioProjOut.wrappedValue = Linear(2048, 128, bias: true)

        self._avCaVideoScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 4)
        self._avCaAudioScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 2048, embeddingCoefficient: 4)
        self._avCaA2VGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 1)
        self._avCaV2AGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 2048, embeddingCoefficient: 1)
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXUnifiedAVTransformerBlock()
        }
        super.init()
    }

    package func forward(
        videoLatent: MLXArray,
        videoKeyframesMask _: MLXArray?,
        videoAttentionMask _: MLXArray?,
        audioLatent: MLXArray,
        timestep _: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        audioSigma _: MLXArray,
        perturbation _: LTXAudioToVideoPerturbation
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let videoSteps = videoTimesteps ?? MLX.zeros([videoLatent.dim(0), videoLatent.dim(1)], dtype: videoLatent.dtype)
        let audioSteps = audioTimesteps ?? MLX.zeros([audioLatent.dim(0), audioLatent.dim(1)], dtype: audioLatent.dtype)
        return forward(
            videoLatent: videoLatent,
            audioLatent: audioLatent,
            videoTimesteps: videoSteps,
            audioTimesteps: audioSteps,
            videoContext: videoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope
        )
    }

    package func forward(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        videoTimesteps: MLXArray,
        audioTimesteps: MLXArray,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray)
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let videoBatch = videoLatent.dim(0)
        let videoTokens = videoLatent.dim(1)
        let audioBatch = audioLatent.dim(0)
        let audioTokens = audioLatent.dim(1)

        var videoX = patchifyProj(videoLatent)
        var audioX = audioPatchifyProj(audioLatent)

        let scaledVideoTimesteps = videoTimesteps.asType(videoX.dtype) * MLXArray(timestepScaleMultiplier).asType(videoX.dtype)
        let scaledAudioTimesteps = audioTimesteps.asType(audioX.dtype) * MLXArray(timestepScaleMultiplier).asType(audioX.dtype)

        let (videoTimeParamsFlat, videoEmbeddedFlat) = adalnSingle(timestep: scaledVideoTimesteps.reshaped(-1), hiddenDType: videoX.dtype)
        let (audioTimeParamsFlat, audioEmbeddedFlat) = audioAdalnSingle(timestep: scaledAudioTimesteps.reshaped(-1), hiddenDType: audioX.dtype)

        let videoTimeParams = videoTimeParamsFlat.reshaped(videoBatch, videoTokens, -1)
        let audioTimeParams = audioTimeParamsFlat.reshaped(audioBatch, audioTokens, -1)
        let videoEmbedded = videoEmbeddedFlat.reshaped(videoBatch, videoTokens, -1)
        let audioEmbedded = audioEmbeddedFlat.reshaped(audioBatch, audioTokens, -1)

        let projectedVideoContext = captionProjection(videoContext).reshaped(videoBatch, videoContext.dim(1), videoDim)
        let projectedAudioContext = audioCaptionProjection(audioContext).reshaped(audioBatch, audioContext.dim(1), audioDim)

        let (videoCrossScaleShiftFlat, _) = avCaVideoScaleShiftAdalnSingle(
            timestep: scaledVideoTimesteps.reshaped(-1),
            hiddenDType: videoX.dtype
        )
        let (audioCrossScaleShiftFlat, _) = avCaAudioScaleShiftAdalnSingle(
            timestep: scaledAudioTimesteps.reshaped(-1),
            hiddenDType: audioX.dtype
        )
        let (videoCrossGateFlat, _) = avCaA2VGateAdalnSingle(
            timestep: scaledVideoTimesteps.reshaped(-1),
            hiddenDType: videoX.dtype
        )
        let (audioCrossGateFlat, _) = avCaV2AGateAdalnSingle(
            timestep: scaledAudioTimesteps.reshaped(-1),
            hiddenDType: audioX.dtype
        )

        let videoCrossScaleShift = videoCrossScaleShiftFlat.reshaped(videoBatch, videoTokens, -1)
        let audioCrossScaleShift = audioCrossScaleShiftFlat.reshaped(audioBatch, audioTokens, -1)
        let videoCrossGate = videoCrossGateFlat.reshaped(videoBatch, videoTokens, -1)
        let audioCrossGate = audioCrossGateFlat.reshaped(audioBatch, audioTokens, -1)

        for block in transformerBlocks {
            let out = block(
                videoX: videoX,
                audioX: audioX,
                videoContext: projectedVideoContext,
                audioContext: projectedAudioContext,
                videoTimestepEmb: videoTimeParams,
                audioTimestepEmb: audioTimeParams,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                videoCrossScaleShiftTimestep: videoCrossScaleShift,
                audioCrossScaleShiftTimestep: audioCrossScaleShift,
                videoCrossGateTimestep: videoCrossGate,
                audioCrossGateTimestep: audioCrossGate
            )
            videoX = out.videoX
            audioX = out.audioX
        }

        let videoScaleShift = scaleShiftTable.reshaped(1, 1, 2, videoDim) + videoEmbedded.reshaped(videoBatch, videoTokens, 1, videoDim)
        let videoShift = videoScaleShift[0..., 0..., 0, 0...]
        let videoScale = videoScaleShift[0..., 0..., 1, 0...]
        videoX = normOut(videoX)
        videoX = videoX * (MLXArray(1.0).asType(videoX.dtype) + videoScale) + videoShift
        let videoVelocity = projOut(videoX)

        let audioScaleShift = audioScaleShiftTable.reshaped(1, 1, 2, audioDim) + audioEmbedded.reshaped(audioBatch, audioTokens, 1, audioDim)
        let audioShift = audioScaleShift[0..., 0..., 0, 0...]
        let audioScale = audioScaleShift[0..., 0..., 1, 0...]
        audioX = audioNormOut(audioX)
        audioX = audioX * (MLXArray(1.0).asType(audioX.dtype) + audioScale) + audioShift
        let audioVelocity = audioProjOut(audioX)

        return (videoVelocity, audioVelocity)
    }

}

package final class LTXUnifiedAVTransformerBlock: Module {
    package let videoDim = 4096
    package let audioDim = 2048

    @ModuleInfo(key: "attn1") package var attn1: LTXDistilledAttention
    @ModuleInfo(key: "attn2") package var attn2: LTXDistilledAttention
    @ModuleInfo(key: "ff") package var ff: LTXDistilledFeedForward
    @ModuleInfo(key: "scale_shift_table") package var scaleShiftTable: MLXArray

    @ModuleInfo(key: "audio_attn1") package var audioAttn1: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn2") package var audioAttn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_ff") package var audioFF: LTXDistilledFeedForward
    @ModuleInfo(key: "audio_scale_shift_table") package var audioScaleShiftTable: MLXArray

    @ModuleInfo(key: "audio_to_video_attn") package var audioToVideoAttn: LTXDistilledAttention
    @ModuleInfo(key: "video_to_audio_attn") package var videoToAudioAttn: LTXDistilledAttention
    @ModuleInfo(key: "scale_shift_table_a2v_ca_audio") package var scaleShiftTableA2VCAAudio: MLXArray
    @ModuleInfo(key: "scale_shift_table_a2v_ca_video") package var scaleShiftTableA2VCAVideo: MLXArray

    package override init() {
        self._attn1.wrappedValue = LTXDistilledAttention(queryDim: 4096, contextDim: nil, heads: 32, headDim: 128, normEps: 1e-6)
        self._attn2.wrappedValue = LTXDistilledAttention(queryDim: 4096, contextDim: 4096, heads: 32, headDim: 128, normEps: 1e-6)
        self._ff.wrappedValue = LTXDistilledFeedForward(dim: 4096, dimOut: 4096, mult: 4)
        self._scaleShiftTable.wrappedValue = MLX.zeros([6, 4096], dtype: .float32)

        self._audioAttn1.wrappedValue = LTXDistilledAttention(queryDim: 2048, contextDim: nil, heads: 32, headDim: 64, normEps: 1e-6)
        self._audioAttn2.wrappedValue = LTXDistilledAttention(queryDim: 2048, contextDim: 2048, heads: 32, headDim: 64, normEps: 1e-6)
        self._audioFF.wrappedValue = LTXDistilledFeedForward(dim: 2048, dimOut: 2048, mult: 4)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([6, 2048], dtype: .float32)

        self._audioToVideoAttn.wrappedValue = LTXDistilledAttention(queryDim: 4096, contextDim: 2048, heads: 32, headDim: 64, normEps: 1e-6)
        self._videoToAudioAttn.wrappedValue = LTXDistilledAttention(queryDim: 2048, contextDim: 4096, heads: 32, headDim: 64, normEps: 1e-6)
        self._scaleShiftTableA2VCAAudio.wrappedValue = MLX.zeros([5, 2048], dtype: .float32)
        self._scaleShiftTableA2VCAVideo.wrappedValue = MLX.zeros([5, 4096], dtype: .float32)
        super.init()
    }

    package func callAsFunction(
        videoX: MLXArray,
        audioX: MLXArray,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoTimestepEmb: MLXArray,
        audioTimestepEmb: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        videoCrossScaleShiftTimestep: MLXArray,
        audioCrossScaleShiftTimestep: MLXArray,
        videoCrossGateTimestep: MLXArray,
        audioCrossGateTimestep: MLXArray
    ) -> (videoX: MLXArray, audioX: MLXArray) {
        var vx = videoX
        var ax = audioX

        let videoAda = scaleShiftTable.reshaped(1, 1, 6, videoDim) + videoTimestepEmb.reshaped(vx.dim(0), vx.dim(1), 6, videoDim)
        let vShiftMSA = videoAda[0..., 0..., 0, 0...]
        let vScaleMSA = videoAda[0..., 0..., 1, 0...]
        let vGateMSA = videoAda[0..., 0..., 2, 0...]

        var normVX = rmsNormNoWeight(vx)
        normVX = normVX * (MLXArray(1.0).asType(normVX.dtype) + vScaleMSA) + vShiftMSA
        vx = vx + attn1(normVX, context: nil, mask: nil, rope: videoRope) * vGateMSA
        vx = vx + attn2(rmsNormNoWeight(vx), context: videoContext, mask: nil, rope: nil)

        let audioAda = audioScaleShiftTable.reshaped(1, 1, 6, audioDim) + audioTimestepEmb.reshaped(ax.dim(0), ax.dim(1), 6, audioDim)
        let aShiftMSA = audioAda[0..., 0..., 0, 0...]
        let aScaleMSA = audioAda[0..., 0..., 1, 0...]
        let aGateMSA = audioAda[0..., 0..., 2, 0...]

        var normAX = rmsNormNoWeight(ax)
        normAX = normAX * (MLXArray(1.0).asType(normAX.dtype) + aScaleMSA) + aShiftMSA
        ax = ax + audioAttn1(normAX, context: nil, mask: nil, rope: audioRope) * aGateMSA
        ax = ax + audioAttn2(rmsNormNoWeight(ax), context: audioContext, mask: nil, rope: nil)

        let a2vAudio = crossAdaValues(
            table: scaleShiftTableA2VCAAudio,
            scaleShiftTimestep: audioCrossScaleShiftTimestep,
            gateTimestep: audioCrossGateTimestep,
            dim: audioDim
        )
        let a2vVideo = crossAdaValues(
            table: scaleShiftTableA2VCAVideo,
            scaleShiftTimestep: videoCrossScaleShiftTimestep,
            gateTimestep: videoCrossGateTimestep,
            dim: videoDim
        )

        let normVX3 = rmsNormNoWeight(vx)
        let normAX3 = rmsNormNoWeight(ax)

        let vxScaledA2V = normVX3 * (MLXArray(1.0).asType(normVX3.dtype) + a2vVideo.scaleA2V) + a2vVideo.shiftA2V
        let axScaledA2V = normAX3 * (MLXArray(1.0).asType(normAX3.dtype) + a2vAudio.scaleA2V) + a2vAudio.shiftA2V
        vx = vx + audioToVideoAttn(
            vxScaledA2V,
            context: axScaledA2V,
            mask: nil,
            rope: videoCrossRope,
            keyRope: audioCrossRope
        ) * a2vVideo.gate

        let axScaledV2A = normAX3 * (MLXArray(1.0).asType(normAX3.dtype) + a2vAudio.scaleV2A) + a2vAudio.shiftV2A
        let vxScaledV2A = normVX3 * (MLXArray(1.0).asType(normVX3.dtype) + a2vVideo.scaleV2A) + a2vVideo.shiftV2A
        ax = ax + videoToAudioAttn(
            axScaledV2A,
            context: vxScaledV2A,
            mask: nil,
            rope: audioCrossRope,
            keyRope: videoCrossRope
        ) * a2vAudio.gate

        let vShiftMLP = videoAda[0..., 0..., 3, 0...]
        let vScaleMLP = videoAda[0..., 0..., 4, 0...]
        let vGateMLP = videoAda[0..., 0..., 5, 0...]
        var vMLPInput = rmsNormNoWeight(vx)
        vMLPInput = vMLPInput * (MLXArray(1.0).asType(vMLPInput.dtype) + vScaleMLP) + vShiftMLP
        vx = vx + ff(vMLPInput) * vGateMLP

        let aShiftMLP = audioAda[0..., 0..., 3, 0...]
        let aScaleMLP = audioAda[0..., 0..., 4, 0...]
        let aGateMLP = audioAda[0..., 0..., 5, 0...]
        var aMLPInput = rmsNormNoWeight(ax)
        aMLPInput = aMLPInput * (MLXArray(1.0).asType(aMLPInput.dtype) + aScaleMLP) + aShiftMLP
        ax = ax + audioFF(aMLPInput) * aGateMLP

        return (vx, ax)
    }

    private func crossAdaValues(
        table: MLXArray,
        scaleShiftTimestep: MLXArray,
        gateTimestep: MLXArray,
        dim: Int
    ) -> (scaleA2V: MLXArray, shiftA2V: MLXArray, scaleV2A: MLXArray, shiftV2A: MLXArray, gate: MLXArray) {
        let batch = scaleShiftTimestep.dim(0)
        let tokens = scaleShiftTimestep.dim(1)

        let scaleShiftValues = table[0..<4, 0...].reshaped(1, 1, 4, dim)
            + scaleShiftTimestep.reshaped(batch, tokens, 4, dim)
        let gateValues = table[4..<5, 0...].reshaped(1, 1, 1, dim)
            + gateTimestep.reshaped(batch, tokens, 1, dim)

        return (
            scaleShiftValues[0..., 0..., 0, 0...],
            scaleShiftValues[0..., 0..., 1, 0...],
            scaleShiftValues[0..., 0..., 2, 0...],
            scaleShiftValues[0..., 0..., 3, 0...],
            gateValues[0..., 0..., 0, 0...]
        )
    }
}
