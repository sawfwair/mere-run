import AudioCodecs
import AudioQwen3TTSModel
import MLX

extension Qwen3TTSSpeechTokenizer {
    /// Native reference-audio encoder path using Seanet -> Transformer -> Downsample -> Split RVQ.
    func encode(samples: [Float], sampleRate: Int) -> MLXArray {
        guard let encoderModel else {
            return MLXArray.zeros([1, 1, max(1, config.encoderValidNumQuantizers)], dtype: .int32)
        }

        let targetSampleRate = max(1, config.inputSampleRate)
        let prepared: [Float]
        if sampleRate == targetSampleRate {
            prepared = samples
        } else {
            prepared = PCMStreamConverter.resampleLinear(
                samples,
                from: sampleRate,
                to: targetSampleRate
            )
        }

        guard !prepared.isEmpty else {
            return MLXArray.zeros([1, 1, max(1, config.encoderValidNumQuantizers)], dtype: .int32)
        }

        let audio = MLXArray(prepared).reshaped(1, 1, prepared.count).asType(.float32)
        let codes = encoderModel.encode(audio) // [B, Q, T]
        let transposed = codes.transposed(0, 2, 1).asType(.int32) // [B, T, Q]
        MLX.eval(transposed)
        return transposed
    }

}
