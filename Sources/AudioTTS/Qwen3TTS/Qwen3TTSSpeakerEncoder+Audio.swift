import AudioQwen3TTSModel
import MLX

extension Qwen3TTSSpeakerEncoder {
    public func extractEmbedding(audio: [Float]) -> MLXArray {
        let mels = Qwen3TTSAudioPreprocessor.melSpectrogram(
            samples: audio,
            nMels: config.melDim,
            frameSize: 1024,
            hopSize: 256,
            sampleRate: config.sampleRate
        )
        let embedding = callAsFunction(mels)
        MLX.eval(embedding)
        return embedding
    }

}
