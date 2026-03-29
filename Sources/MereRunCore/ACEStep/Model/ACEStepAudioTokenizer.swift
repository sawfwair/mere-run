import Foundation
import MLX
import MLXNN

final class ACEStepAudioTokenizer: Module {
    @ModuleInfo(key: "audio_acoustic_proj") var audioAcousticProj: Linear
    @ModuleInfo(key: "attention_pooler") var attentionPooler: ACEStepAttentionPooler
    @ModuleInfo(key: "quantizer") var quantizer: ACEStepResidualFSQ

    init(config: ACEStepConfig) {
        self._audioAcousticProj.wrappedValue = Linear(config.audioAcousticHiddenDim, config.hiddenSize, bias: true)
        self._attentionPooler.wrappedValue = ACEStepAttentionPooler(config: config)
        self._quantizer.wrappedValue = ACEStepResidualFSQ(config: config)
    }

    /// Tokenize 25Hz acoustic latents reshaped into `[B, T5, P, 64]`.
    func callAsFunction(_ x: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        var h = audioAcousticProj(x)
        h = attentionPooler(h)
        return quantizer(h)
    }
}

