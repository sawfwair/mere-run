import Foundation
import MLX
import MLXNN

final class ACEStepConditionEncoder: Module {
    @ModuleInfo(key: "text_projector") var textProjector: Linear
    @ModuleInfo(key: "lyric_encoder") var lyricEncoder: ACEStepLyricEncoder
    @ModuleInfo(key: "timbre_encoder") var timbreEncoder: ACEStepTimbreEncoder

    init(config: ACEStepConfig) {
        let encoderConfig = config.conditionEncoderConfig
        self._textProjector.wrappedValue = Linear(config.textHiddenDim, config.encoderHiddenSize, bias: false)
        self._lyricEncoder.wrappedValue = ACEStepLyricEncoder(config: encoderConfig)
        self._timbreEncoder.wrappedValue = ACEStepTimbreEncoder(config: encoderConfig)
    }

    func callAsFunction(
        textHiddenStates: MLXArray,
        textAttentionMask: MLXArray,
        lyricHiddenStates: MLXArray,
        lyricAttentionMask: MLXArray,
        referAudioAcousticHiddenStatesPacked: MLXArray,
        referAudioOrderMask: MLXArray
    ) -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        let textProjected = textProjector(textHiddenStates)
        let lyricEncoded = lyricEncoder(inputsEmbeds: lyricHiddenStates, attentionMask: lyricAttentionMask)
        let (timbreUnpacked, timbreMask) = timbreEncoder(
            referAudioAcousticHiddenStatesPacked: referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: referAudioOrderMask
        )

        let first = ACEStepSequencePacker.pack(
            hidden1: lyricEncoded,
            hidden2: timbreUnpacked,
            mask1: lyricAttentionMask,
            mask2: timbreMask
        )
        let packed = ACEStepSequencePacker.pack(
            hidden1: first.hiddenStates,
            hidden2: textProjected,
            mask1: first.attentionMask,
            mask2: textAttentionMask
        )

        return packed
    }
}
