import Foundation

/// Lightweight PCM conversion helpers for realtime chunk pipelines.
public enum PCMStreamConverter {
    /// Convert interleaved PCM samples to mono by averaging channels.
    public static func downmixInterleavedToMono(
        _ samples: [Float],
        channelCount: Int
    ) -> [Float] {
        let channels = max(1, channelCount)
        if channels == 1 {
            return samples
        }

        let frameCount = samples.count / channels
        guard frameCount > 0 else { return [] }

        var mono = [Float](repeating: 0, count: frameCount)
        for frame in 0..<frameCount {
            var sum: Float = 0
            let base = frame * channels
            for channel in 0..<channels {
                sum += samples[base + channel]
            }
            mono[frame] = sum / Float(channels)
        }
        return mono
    }

    /// Linear resampling for mono PCM float samples.
    public static func resampleLinear(
        _ samples: [Float],
        from sourceSampleRate: Int,
        to targetSampleRate: Int
    ) -> [Float] {
        let sourceRate = max(1, sourceSampleRate)
        let targetRate = max(1, targetSampleRate)
        guard !samples.isEmpty else { return [] }
        guard sourceRate != targetRate else { return samples }

        let duration = Double(samples.count) / Double(sourceRate)
        let outputCount = max(1, Int((duration * Double(targetRate)).rounded(.toNearestOrAwayFromZero)))
        if outputCount == 1 {
            return [samples[0]]
        }

        let ratio = Double(sourceRate) / Double(targetRate)
        var output = [Float](repeating: 0, count: outputCount)
        let maxSourceIndex = samples.count - 1

        for i in 0..<outputCount {
            let sourcePosition = Double(i) * ratio
            let lower = min(maxSourceIndex, Int(floor(sourcePosition)))
            let upper = min(maxSourceIndex, lower + 1)
            let blend = Float(sourcePosition - Double(lower))
            let lowerValue = samples[lower]
            let upperValue = samples[upper]
            output[i] = lowerValue + ((upperValue - lowerValue) * blend)
        }

        return output
    }

    /// Convert interleaved PCM to mono and resample to target rate.
    public static func convertInterleavedToMono(
        _ samples: [Float],
        sourceSampleRate: Int,
        sourceChannelCount: Int,
        targetSampleRate: Int
    ) -> [Float] {
        let mono = downmixInterleavedToMono(samples, channelCount: sourceChannelCount)
        return resampleLinear(mono, from: sourceSampleRate, to: targetSampleRate)
    }
}
