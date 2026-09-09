import Foundation
import MLX
import MLXFast
import MLXNN

package final class LTXAudioOnlyTransformerV2: Module {
    package let audioDim = 2_048
    package let timestepScaleMultiplier: Float = 1_000

    @ModuleInfo(key: "audio_patchify_proj") package var audioPatchifyProj: Linear
    @ModuleInfo(key: "audio_proj_out") package var audioProjOut: Linear
    @ModuleInfo(key: "audio_scale_shift_table") package var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_adaln_single") package var audioAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_prompt_adaln_single") package var audioPromptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") package var transformerBlocks: [LTXAudioOnlyTransformerV2Block]
    @ModuleInfo(key: "audio_norm_out") package var audioNormOut: LayerNorm

    package override init() {
        self._audioPatchifyProj.wrappedValue = Linear(128, audioDim, bias: true)
        self._audioProjOut.wrappedValue = Linear(audioDim, 128, bias: true)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        self._audioAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 9
        )
        self._audioPromptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 2
        )
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXAudioOnlyTransformerV2Block()
        }
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: audioDim, eps: 1e-6, affine: false)
        super.init()
    }

    package func forward(
        audioLatent: MLXArray,
        timestep: MLXArray,
        audioTimesteps: MLXArray?,
        audioContext: MLXArray,
        audioRope: LTXRope,
        skippedSelfAttentionBlocks: Set<Int> = []
    ) -> MLXArray {
        let batch = audioLatent.dim(0)
        let tokens = audioLatent.dim(1)
        var audio = audioPatchifyProj(audioLatent.asType(.bfloat16))
        let sigma = timestep.asType(audio.dtype).reshaped(-1)
        let scaledSigma = sigma * MLXArray(timestepScaleMultiplier).asType(audio.dtype)
        let adalnParams: MLXArray
        let embedded: MLXArray
        if let audioTimesteps {
            let scaled = audioTimesteps.asType(audio.dtype)
                * MLXArray(timestepScaleMultiplier).asType(audio.dtype)
            let values = audioAdalnSingle(
                timestep: scaled.reshaped(-1),
                hiddenDType: audio.dtype
            )
            adalnParams = values.0.reshaped(batch, tokens, -1)
            embedded = values.1.reshaped(batch, tokens, -1)
        } else {
            (adalnParams, embedded) = audioAdalnSingle(
                timestep: scaledSigma,
                hiddenDType: audio.dtype
            )
        }
        let (promptParams, _) = audioPromptAdalnSingle(
            timestep: scaledSigma,
            hiddenDType: audio.dtype
        )
        for (index, block) in transformerBlocks.enumerated() {
            audio = block(
                audioHidden: audio,
                audioAdalnParams: adalnParams,
                audioPromptAdalnParams: promptParams,
                audioTextEmbeds: audioContext.asType(audio.dtype),
                audioRope: audioRope,
                skipSelfAttention: skippedSelfAttentionBlocks.contains(index)
            )
            if (index + 1).isMultiple(of: 8) {
                MLX.eval(audio)
            }
        }
        let normalizedEmbedding = embedded.ndim == 2
            ? embedded.expandedDimensions(axis: 1)
            : embedded
        let scaleShift = audioScaleShiftTable.reshaped(1, 1, 2, audioDim)
            + normalizedEmbedding.expandedDimensions(axis: 2)
        let shift = scaleShift[0..., 0..., 0, 0...]
        let scale = scaleShift[0..., 0..., 1, 0...]
        var output = audioNormOut(audio)
        output = output * (MLXArray(1).asType(output.dtype) + scale) + shift
        return audioProjOut(output)
    }
}

package final class LTXAudioOnlyTransformerV2Block: Module {
    package let audioDim = 2_048

    @ModuleInfo(key: "audio_attn1") package var audioAttn1: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn2") package var audioAttn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_ff") package var audioFF: LTXDistilledFeedForward
    @ModuleInfo(key: "audio_scale_shift_table") package var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_prompt_scale_shift_table") package var audioPromptScaleShiftTable: MLXArray

    package override init() {
        self._audioAttn1.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: nil,
            heads: 32,
            headDim: 64,
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
        self._audioFF.wrappedValue = LTXDistilledFeedForward(
            dim: audioDim,
            dimOut: audioDim,
            mult: 4
        )
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([9, audioDim], dtype: .float32)
        self._audioPromptScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        super.init()
    }

    package func callAsFunction(
        audioHidden: MLXArray,
        audioAdalnParams: MLXArray,
        audioPromptAdalnParams: MLXArray,
        audioTextEmbeds: MLXArray,
        audioRope: LTXRope,
        skipSelfAttention: Bool
    ) -> MLXArray {
        var audio = audioHidden
        let adaln = unpack(
            audioAdalnParams,
            table: audioScaleShiftTable,
            count: 9
        )
        if !skipSelfAttention {
            var selfNorm = rmsNormNoWeight(audio)
            selfNorm = selfNorm * (MLXArray(1).asType(selfNorm.dtype) + adaln[1]) + adaln[0]
            audio = audio
                + audioAttn1(selfNorm, context: nil, mask: nil, rope: audioRope) * adaln[2]
        }

        let promptAdaln = unpack(
            audioPromptAdalnParams,
            table: audioPromptScaleShiftTable,
            count: 2
        )
        let text = audioTextEmbeds
            * (MLXArray(1).asType(audioTextEmbeds.dtype) + promptAdaln[1])
            + promptAdaln[0]
        var textNorm = rmsNormNoWeight(audio)
        textNorm = textNorm * (MLXArray(1).asType(textNorm.dtype) + adaln[7]) + adaln[6]
        audio = audio + audioAttn2(textNorm, context: text, mask: nil, rope: nil) * adaln[8]

        var feedForwardNorm = rmsNormNoWeight(audio)
        feedForwardNorm = feedForwardNorm
            * (MLXArray(1).asType(feedForwardNorm.dtype) + adaln[4])
            + adaln[3]
        return audio + audioFF(feedForwardNorm) * adaln[5]
    }

    private func unpack(
        _ params: MLXArray,
        table: MLXArray,
        count: Int
    ) -> [MLXArray] {
        if params.ndim == 2 {
            let values = params.reshaped(-1, count, audioDim)
                + table[0..<count, 0...].reshaped(1, count, audioDim)
            return (0..<count).map { values[0..., $0, 0...].expandedDimensions(axis: 1) }
        }
        let values = params.reshaped(params.dim(0), params.dim(1), count, audioDim)
            + table[0..<count, 0...].reshaped(1, 1, count, audioDim)
        return (0..<count).map { values[0..., 0..., $0, 0...] }
    }
}
