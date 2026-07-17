import Foundation
import MLX

/// LTX-2's 16 kHz audio-VAE frontend.
///
/// This matches `torchaudio.transforms.MelSpectrogram` as configured by the
/// upstream LTX-2 audio VAE: centered reflect padding, a periodic Hann window,
/// magnitude spectra, and a Slaney-normalized Slaney mel bank.
public final class LTXAudioMelProcessor {
    public static let sampleRate = 16_000
    public static let fftSize = 1_024
    public static let windowLength = 1_024
    public static let hopLength = 160
    public static let melBinCount = 64

    private let fftPlan: RealFFTPlan
    private let hannWindow: [Float]
    private let melFilters: [[Float]]

    public init() {
        do {
            fftPlan = try RealFFTPlan(size: Self.fftSize)
        } catch {
            preconditionFailure("Invalid LTX audio FFT size: \(Self.fftSize)")
        }

        hannWindow = (0..<Self.windowLength).map { index in
            0.5 - 0.5 * cos(2 * Float.pi * Float(index) / Float(Self.windowLength))
        }
        melFilters = Self.makeSlaneyFilterBank()
    }

    /// Converts planar audio channels to `[1, channels, frames, 64]` log-mels.
    /// Every channel must contain the same number of 16 kHz samples.
    public func extract(channels: [[Float]]) -> MLXArray {
        precondition(!channels.isEmpty, "LTX audio conditioning requires at least one channel.")
        let sampleCount = channels[0].count
        precondition(
            sampleCount > Self.fftSize / 2,
            "LTX audio conditioning requires more than \(Self.fftSize / 2) samples."
        )
        precondition(
            channels.allSatisfy { $0.count == sampleCount },
            "LTX audio conditioning channels must have equal sample counts."
        )

        let pad = Self.fftSize / 2
        let paddedCount = sampleCount + 2 * pad
        let frameCount = 1 + (paddedCount - Self.fftSize) / Self.hopLength
        let frequencyCount = Self.fftSize / 2 + 1
        var output = [Float](
            repeating: 0,
            count: channels.count * frameCount * Self.melBinCount
        )
        var frame = [Float](repeating: 0, count: Self.fftSize)

        for (channelIndex, channel) in channels.enumerated() {
            let padded = reflectPad(channel, pad: pad)
            for frameIndex in 0..<frameCount {
                let start = frameIndex * Self.hopLength
                for sampleIndex in 0..<Self.fftSize {
                    frame[sampleIndex] = padded[start + sampleIndex] * hannWindow[sampleIndex]
                }

                let power = fftPlan.powerSpectrum(frame)
                var magnitudes = [Float](repeating: 0, count: frequencyCount)
                for frequencyIndex in 0..<frequencyCount {
                    // vDSP's packed real FFT is scaled by 2 relative to torch.fft.rfft.
                    #if canImport(Accelerate)
                    let fftScale: Float = 0.5
                    #else
                    let fftScale: Float = 1
                    #endif
                    magnitudes[frequencyIndex] = fftScale * sqrt(max(power[frequencyIndex], 0))
                }

                for melIndex in 0..<Self.melBinCount {
                    var magnitude: Float = 0
                    for frequencyIndex in 0..<frequencyCount {
                        magnitude += melFilters[melIndex][frequencyIndex] * magnitudes[frequencyIndex]
                    }
                    let outputIndex = ((channelIndex * frameCount) + frameIndex) * Self.melBinCount + melIndex
                    output[outputIndex] = log(max(magnitude, 1e-5))
                }
            }
        }

        return MLXArray(output).reshaped(1, channels.count, frameCount, Self.melBinCount)
    }

    private static func makeSlaneyFilterBank() -> [[Float]] {
        let frequencyCount = fftSize / 2 + 1
        let melPoints = linspace(
            from: hzToMel(0),
            to: hzToMel(Float(sampleRate) / 2),
            count: melBinCount + 2
        )
        let filterFrequencies = melPoints.map(melToHz)
        let fftFrequencies = linspace(
            from: 0,
            to: Float(sampleRate) / 2,
            count: frequencyCount
        )
        var filters = [[Float]](
            repeating: [Float](repeating: 0, count: frequencyCount),
            count: melBinCount
        )

        for melIndex in 0..<melBinCount {
            let left = filterFrequencies[melIndex]
            let center = filterFrequencies[melIndex + 1]
            let right = filterFrequencies[melIndex + 2]
            for (frequencyIndex, frequency) in fftFrequencies.enumerated() {
                let lower = (frequency - left) / (center - left)
                let upper = (right - frequency) / (right - center)
                filters[melIndex][frequencyIndex] = max(0, min(lower, upper))
            }

            let normalization = 2 / (right - left)
            for frequencyIndex in 0..<frequencyCount {
                filters[melIndex][frequencyIndex] *= normalization
            }
        }
        return filters
    }

    private static func hzToMel(_ frequency: Float) -> Float {
        let linearScale: Float = 200 / 3
        let logThreshold: Float = 1_000
        let logThresholdMel = logThreshold / linearScale
        let logStep = log(Float(6.4)) / 27
        if frequency >= logThreshold {
            return logThresholdMel + log(frequency / logThreshold) / logStep
        }
        return frequency / linearScale
    }

    private static func melToHz(_ mel: Float) -> Float {
        let linearScale: Float = 200 / 3
        let logThreshold: Float = 1_000
        let logThresholdMel = logThreshold / linearScale
        let logStep = log(Float(6.4)) / 27
        if mel >= logThresholdMel {
            return logThreshold * exp(logStep * (mel - logThresholdMel))
        }
        return linearScale * mel
    }

    private static func linspace(from start: Float, to end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }

    private func reflectPad(_ samples: [Float], pad: Int) -> [Float] {
        var result = [Float](repeating: 0, count: samples.count + 2 * pad)
        for index in 0..<pad {
            result[index] = samples[pad - index]
        }
        result.replaceSubrange(pad..<(pad + samples.count), with: samples)
        for index in 0..<pad {
            result[pad + samples.count + index] = samples[samples.count - 2 - index]
        }
        return result
    }
}
