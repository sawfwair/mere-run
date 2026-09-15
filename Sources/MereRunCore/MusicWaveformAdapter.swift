import AudioCore
import MLX

/// Converts each backend's documented tensor layout into portable frame data.
public enum MusicWaveformAdapter {
    public enum LayoutError: Error {
        case invalidShape(expected: String, actual: [Int])
    }

    public static func aceStep(_ audio: MLXArray, sampleRate: Int = 48_000) throws -> AudioWaveform {
        guard audio.ndim == 3, audio.dim(0) == 1 else {
            throw LayoutError.invalidShape(expected: "[1, frames, channels]", actual: audio.shape)
        }
        return try interleaved(audio, channels: audio.dim(2), sampleRate: sampleRate)
    }

    public static func miniMaxMusic3(_ audio: MLXArray, sampleRate: Int) throws -> AudioWaveform {
        guard audio.ndim == 3, audio.dim(0) == 1 else {
            throw LayoutError.invalidShape(expected: "[1, channels, frames]", actual: audio.shape)
        }
        return try interleaved(audio.transposed(0, 2, 1), channels: audio.dim(1), sampleRate: sampleRate)
    }

    private static func interleaved(_ audio: MLXArray, channels: Int, sampleRate: Int) throws -> AudioWaveform {
        try Task.checkCancellation()
        let floats = audio.asType(.float32).reshaped(-1)
        MLX.eval(floats)
        return try AudioWaveform(interleaved: floats.asArray(Float.self), channels: channels, sampleRate: sampleRate)
    }
}
