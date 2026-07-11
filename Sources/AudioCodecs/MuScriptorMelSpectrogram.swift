import Foundation
import MLX

/// MuScriptor's published 16 kHz, 512-bin HTK mel frontend.
///
/// This intentionally differs from the Whisper-style frontend in `MelSpectrogram`:
/// MuScriptor uses magnitude (not power), an unnormalized HTK filter bank, natural
/// logarithms, and keeps the final centered STFT frame (masked by the runtime).
public final class MuScriptorMelSpectrogram {
    public static let sampleRate = 16_000
    public static let fftSize = 2_048
    public static let hopLength = 160
    public static let melBinCount = 512
    public static let chunkSampleCount = 80_000

    private let fftPlan: RealFFTPlan
    private let hannWindow: [Float]
    private let filterBank: MLXArray

    public init() {
        do {
            self.fftPlan = try RealFFTPlan(size: Self.fftSize)
        } catch {
            preconditionFailure("Invalid MuScriptor FFT size: \(Self.fftSize)")
        }

        self.hannWindow = (0..<Self.fftSize).map { index in
            0.5 - 0.5 * cos(2 * Float.pi * Float(index) / Float(Self.fftSize))
        }
        self.filterBank = Self.makeHTKFilterBank()
    }

    /// Returns `[1, 501, 512]` log-mel frames for one zero-padded five-second chunk.
    public func extract(from input: [Float]) -> MLXArray {
        var chunk = Array(input.prefix(Self.chunkSampleCount))
        if chunk.count < Self.chunkSampleCount {
            chunk.append(contentsOf: repeatElement(0, count: Self.chunkSampleCount - chunk.count))
        }

        let pad = Self.fftSize / 2
        let padded = reflectPad(chunk, pad: pad)
        let frameCount = 1 + (padded.count - Self.fftSize) / Self.hopLength
        let frequencyCount = Self.fftSize / 2 + 1
        var spectra = [Float](repeating: 0, count: frameCount * frequencyCount)
        var frame = [Float](repeating: 0, count: Self.fftSize)

        for frameIndex in 0..<frameCount {
            let start = frameIndex * Self.hopLength
            for sampleIndex in 0..<Self.fftSize {
                frame[sampleIndex] = padded[start + sampleIndex] * hannWindow[sampleIndex]
            }
            let power = fftPlan.powerSpectrum(frame)
            for frequencyIndex in 0..<frequencyCount {
                // vDSP's packed real FFT is scaled by 2 relative to torch.fft.rfft.
                #if canImport(Accelerate)
                let fftScale: Float = 0.5
                #else
                let fftScale: Float = 1
                #endif
                spectra[frameIndex * frequencyCount + frequencyIndex] = fftScale * sqrt(max(power[frequencyIndex], 0))
            }
        }

        let magnitude = MLXArray(spectra).reshaped(frameCount, frequencyCount)
        let mel = MLX.matmul(magnitude, filterBank)
        return MLX.log(mel + MLXArray(Float(1e-6)))
            .reshaped(1, frameCount, Self.melBinCount)
    }

    private static func makeHTKFilterBank() -> MLXArray {
        let frequencyCount = fftSize / 2 + 1
        let minMel = hzToMel(0)
        let maxMel = hzToMel(Float(sampleRate) / 2)
        let melPoints = linspace(from: minMel, to: maxMel, count: melBinCount + 2)
        let frequencyPoints = melPoints.map(melToHz)
        let fftFrequencies = linspace(
            from: 0,
            to: Float(sampleRate) / 2,
            count: frequencyCount
        )

        var values = [Float](repeating: 0, count: frequencyCount * melBinCount)
        for melIndex in 0..<melBinCount {
            let left = frequencyPoints[melIndex]
            let center = frequencyPoints[melIndex + 1]
            let right = frequencyPoints[melIndex + 2]
            for (frequencyIndex, frequency) in fftFrequencies.enumerated() {
                let down = -(frequency - right) / (right - center)
                let up = (frequency - left) / (center - left)
                values[frequencyIndex * melBinCount + melIndex] = max(0, min(down, up))
            }
        }
        return MLXArray(values).reshaped(frequencyCount, melBinCount)
    }

    private static func hzToMel(_ frequency: Float) -> Float {
        2_595 * log10(1 + frequency / 700)
    }

    private static func melToHz(_ mel: Float) -> Float {
        700 * (pow(10, mel / 2_595) - 1)
    }

    private static func linspace(from start: Float, to end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }

    private func reflectPad(_ samples: [Float], pad: Int) -> [Float] {
        guard samples.count > 1, pad > 0 else { return samples }
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
