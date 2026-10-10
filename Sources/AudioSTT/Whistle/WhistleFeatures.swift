import Foundation
import MLX
import AudioCodecs
import AudioCore

/// Whistle uses symmetric Hann, zero padding, natural log and per-channel sample variance.
/// The filterbank is read from the published .cact rather than approximated with Whisper's.
struct WhistleFeatures {
    let filterbank: [Float]

    func extract(_ samples: [Float]) throws -> MLXArray {
        let frames = samples.count / 160
        guard frames > 0, samples.count <= 480000, samples.allSatisfy(\.isFinite) else {
            throw SpeechTranscriptionIssue("invalid_audio", "Whistle expects finite 16 kHz audio of 10 ms to 30 seconds per window.")
        }
        let fft = try RealFFTPlan(size: 512)
        let window = (0..<400).map { 0.5 - 0.5 * cos(Float(2 * Double.pi * Double($0) / 399)) }
        var levels: [Float] = []
        for start in stride(from: 0, through: samples.count - 320, by: 320) {
            let power = samples[start..<(start + 320)].reduce(Float(0)) { $0 + $1 * $1 }
            levels.append(sqrt(power / 320))
        }
        levels.sort()
        let peak = levels.isEmpty ? 0 : levels[Int(Float(levels.count - 1) * 0.99)]
        let gain: Float = peak > 1e-6 ? 0.1 / peak : 1
        var features = [Float](repeating: 0, count: frames * 80)
        for frame in 0..<frames {
            try Task.checkCancellation()
            var padded = [Float](repeating: 0, count: 512)
            for index in 0..<400 {
                let source = frame * 160 - 200 + index
                if samples.indices.contains(source) { padded[index] = samples[source] * gain * window[index] }
            }
            // vDSP's real radix-2 transform doubles amplitudes; the scalar fallback does not.
            #if canImport(Accelerate)
            let power = fft.powerSpectrum(padded).map { $0 * 0.25 }
            #else
            let power = fft.powerSpectrum(padded)
            #endif
            for channel in 0..<80 {
                var energy: Float = 0
                for bin in 0..<257 { energy += power[bin] * filterbank[bin * 80 + channel] }
                features[frame * 80 + channel] = log(energy + Float(pow(2.0, -24.0)))
            }
        }
        for channel in 0..<80 {
            let average = (0..<frames).reduce(Float(0)) { $0 + features[$1 * 80 + channel] } / Float(frames)
            let variance = (0..<frames).reduce(Float(0)) { result, frame in
                let delta = features[frame * 80 + channel] - average
                return result + delta * delta
            } / Float(max(1, frames - 1))
            let scale = sqrt(variance) + 1e-5
            for frame in 0..<frames { features[frame * 80 + channel] = (features[frame * 80 + channel] - average) / scale }
        }
        return MLXArray(features, [frames, 80])
    }
}
