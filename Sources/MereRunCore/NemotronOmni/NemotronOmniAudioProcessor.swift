import AudioCodecs
import Foundation
import MediaIO
import MLX

struct NemotronOmniPreparedAudio: @unchecked Sendable {
    let melFeatures: MLXArray
    let sampleCount: Int

    var validMelFrameCount: Int {
        sampleCount / 160
    }

    var languageTokenCount: Int {
        NemotronOmniPlaceholderPlanner.audioTokenCount(sampleCount: sampleCount)
    }
}

enum NemotronOmniAudioProcessor {
    private static let sampleRate = 16_000
    private static let hopLength = 160
    private static let fftSize = 512
    private static let windowLength = 400
    private static let preemphasis: Float = 0.97
    private static let logGuard: Float = 5.960_464_5e-8

    static func prepare(reference: String) throws -> NemotronOmniPreparedAudio {
        let url = try localURL(reference: reference)
        let decoded = try MediaAudioIO.decode(
            url,
            targetSampleRate: sampleRate,
            channels: 1
        )
        let maximumSamples = NemotronOmniResources.maximumAudioDurationSeconds * sampleRate
        guard decoded.samples.count <= maximumSamples else {
            throw NemotronOmniError.unsupportedMedia(
                "audio exceeds \(NemotronOmniResources.maximumAudioDurationSeconds) seconds"
            )
        }
        return NemotronOmniPreparedAudio(
            melFeatures: try logMelSpectrogram(samples: decoded.samples),
            sampleCount: decoded.samples.count
        )
    }

    static func logMelSpectrogram(samples: [Float]) throws -> MLXArray {
        guard !samples.isEmpty else {
            throw NemotronOmniError.unsupportedMedia("audio input is empty")
        }
        var emphasized = [Float](repeating: 0, count: samples.count)
        emphasized[0] = samples[0]
        if samples.count > 1 {
            for index in 1..<samples.count {
                emphasized[index] = samples[index] - preemphasis * samples[index - 1]
            }
        }

        let padding = fftSize / 2
        let padded = [Float](repeating: 0, count: padding)
            + emphasized
            + [Float](repeating: 0, count: padding)
        let frameCount = 1 + samples.count / hopLength
        let plan = try RealFFTPlan(size: fftSize)
        let window = (0..<windowLength).map { index in
            Float(0.5 * (1 - cos(2 * Double.pi * Double(index) / Double(windowLength - 1))))
        }
        let windowOffset = (fftSize - windowLength) / 2
        let melFilters = makeMelFilters(count: 128)
        var features = [Float](repeating: 0, count: frameCount * 128)
        var frame = [Float](repeating: 0, count: fftSize)
        for frameIndex in 0..<frameCount {
            frame = [Float](repeating: 0, count: fftSize)
            let start = frameIndex * hopLength
            for index in 0..<windowLength {
                frame[windowOffset + index] = padded[start + windowOffset + index] * window[index]
            }
            let power = plan.powerSpectrum(frame)
            for melIndex in 0..<128 {
                var energy: Float = 0
                for frequency in power.indices {
                    energy += melFilters[melIndex][frequency] * power[frequency]
                }
                features[frameIndex * 128 + melIndex] = log(energy + logGuard)
            }
        }

        // Parakeet's center-padded STFT produces one trailing frame that is
        // excluded by its attention mask. Match the reference feature
        // extractor by computing statistics over real frames only, then
        // zeroing that padding frame.
        let validFrameCount = max(1, samples.count / hopLength)
        for melIndex in 0..<128 {
            var mean: Float = 0
            for frameIndex in 0..<validFrameCount {
                mean += features[frameIndex * 128 + melIndex]
            }
            mean /= Float(validFrameCount)
            var variance: Float = 0
            for frameIndex in 0..<validFrameCount {
                let difference = features[frameIndex * 128 + melIndex] - mean
                variance += difference * difference
            }
            variance /= Float(max(1, validFrameCount - 1))
            let standardDeviation = sqrt(variance)
            for frameIndex in 0..<validFrameCount {
                let index = frameIndex * 128 + melIndex
                features[index] = (features[index] - mean) / (standardDeviation + 1e-5)
            }
            for frameIndex in validFrameCount..<frameCount {
                features[frameIndex * 128 + melIndex] = 0
            }
        }
        return MLXArray(features, [1, frameCount, 128]).asType(.bfloat16)
    }

    private static func makeMelFilters(count: Int) -> [[Float]] {
        let frequencyCount = fftSize / 2 + 1
        let maximumFrequency = Float(sampleRate) / 2
        let melMinimum = hzToMel(0)
        let melMaximum = hzToMel(maximumFrequency)
        let melPoints = (0..<(count + 2)).map { index in
            melMinimum + (melMaximum - melMinimum) * Float(index) / Float(count + 1)
        }
        let hzPoints = melPoints.map(melToHz)
        var filters = [[Float]](
            repeating: [Float](repeating: 0, count: frequencyCount),
            count: count
        )
        for mel in 0..<count {
            let left = hzPoints[mel]
            let center = hzPoints[mel + 1]
            let right = hzPoints[mel + 2]
            for bin in 0..<frequencyCount {
                let frequency = maximumFrequency * Float(bin) / Float(frequencyCount - 1)
                let rising = (frequency - left) / max(center - left, 1e-8)
                let falling = (right - frequency) / max(right - center, 1e-8)
                filters[mel][bin] = max(0, min(rising, falling)) * 2 / max(right - left, 1e-8)
            }
        }
        return filters
    }

    private static func hzToMel(_ frequency: Float) -> Float {
        let linearSpacing: Float = 200 / 3
        let minimumLogFrequency: Float = 1_000
        let minimumLogMel = minimumLogFrequency / linearSpacing
        let logStep = log(Float(6.4)) / 27
        if frequency >= minimumLogFrequency {
            return minimumLogMel + log(frequency / minimumLogFrequency) / logStep
        }
        return frequency / linearSpacing
    }

    private static func melToHz(_ mel: Float) -> Float {
        let linearSpacing: Float = 200 / 3
        let minimumLogFrequency: Float = 1_000
        let minimumLogMel = minimumLogFrequency / linearSpacing
        let logStep = log(Float(6.4)) / 27
        if mel >= minimumLogMel {
            return minimumLogFrequency * exp(logStep * (mel - minimumLogMel))
        }
        return mel * linearSpacing
    }

    private static func localURL(reference: String) throws -> URL {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw NemotronOmniError.unsupportedMedia("audio reference is empty")
        }
        if let parsed = URL(string: trimmed), parsed.scheme?.lowercased() == "file" {
            return parsed
        }
        if let scheme = URL(string: trimmed)?.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            throw NemotronOmniError.unsupportedMedia(
                "remote audio URLs are not fetched; use a local path or file URL"
            )
        }
        let url = URL(fileURLWithPath: trimmed).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NemotronOmniError.unsupportedMedia("audio file not found: \(url.path)")
        }
        return url
    }
}
