import Foundation
import AudioCodecs
import MLX

public enum Qwen3TTSAudioPreprocessor {
    public struct ProcessedReference: Sendable, Hashable {
        public let samples: [Float]
        public let sampleRate: Int

        public init(samples: [Float], sampleRate: Int) {
            self.samples = samples
            self.sampleRate = sampleRate
        }
    }

    public enum Error: LocalizedError {
        case readFailed(String)
        case emptyAudio
        case durationTooShort(Double)
        case durationTooLong(Double)

        public var errorDescription: String? {
            switch self {
            case .readFailed(let message):
                return "Failed to read audio: \(message)"
            case .emptyAudio:
                return "Reference audio is empty"
            case .durationTooShort(let seconds):
                return String(format: "Reference audio is too short (%.2fs). Minimum is 2.00s.", seconds)
            case .durationTooLong(let seconds):
                return String(
                    format: "Reference audio is too long (%.2fs). Maximum accepted is 14.00s. Audio from 12.00s to 14.00s is auto-trimmed to 12.00s.",
                    seconds
                )
            }
        }
    }

    public static func loadAndProcess(
        from url: URL,
        targetSampleRate: Int = 24_000,
        minDuration: Double = 2.0,
        maxDuration: Double = 12.0
    ) throws -> ProcessedReference {
        let samples: [Float]
        do {
            samples = try AudioReader
                .readAudioBuffer(from: url, sampleRate: targetSampleRate, channels: 1)
                .samples
        } catch {
            throw Error.readFailed(error.localizedDescription)
        }

        if samples.isEmpty {
            throw Error.emptyAudio
        }

        let trimmed = trimSilence(normalize(samples), threshold: 0.01)
        if trimmed.isEmpty {
            throw Error.emptyAudio
        }

        let duration = Double(trimmed.count) / Double(targetSampleRate)
        if duration < minDuration {
            throw Error.durationTooShort(duration)
        }

        // Treat maxDuration as the preferred cap (12s by default), but allow
        // a small grace window (up to +2s) and trim back to maxDuration.
        let hardMaxDuration = maxDuration + 2.0
        if duration > hardMaxDuration {
            throw Error.durationTooLong(duration)
        }

        let maxSampleCount = Int(maxDuration * Double(targetSampleRate))
        let processedSamples: [Float]
        if trimmed.count > maxSampleCount {
            processedSamples = Array(trimmed.prefix(maxSampleCount))
        } else {
            processedSamples = trimmed
        }

        return ProcessedReference(samples: processedSamples, sampleRate: targetSampleRate)
    }

    public static func melSpectrogram(
        samples: [Float],
        nMels: Int = 128,
        frameSize: Int = 1024,
        hopSize: Int = 256,
        sampleRate: Int = 24_000
    ) -> MLXArray {
        guard !samples.isEmpty else {
            return MLXArray.zeros([1, 1, nMels], dtype: .float32)
        }

        // Qwen3-TTS parity:
        // - manual reflect pad with padding=(n_fft-hop)/2
        // - Hann window
        // - STFT magnitude with epsilon
        // - Slaney mel filterbank (scale + norm)
        // - natural log with clamp 1e-5
        let nFFT = max(2, frameSize)
        let fftBins = (nFFT / 2) + 1
        let padding = max(0, (nFFT - hopSize) / 2)
        let paddedSamples = reflectPad(samples, padding: padding)
        guard paddedSamples.count >= nFFT else {
            return MLXArray.zeros([1, 1, nMels], dtype: .float32)
        }

        let frameCount = 1 + ((paddedSamples.count - nFFT) / max(1, hopSize))
        guard frameCount > 0 else {
            return MLXArray.zeros([1, 1, nMels], dtype: .float32)
        }

        let hann = hannWindow(size: nFFT)
        let melBasis = slaneyMelFilterBank(
            sampleRate: sampleRate,
            nFFT: nFFT,
            nMels: nMels,
            fMin: 0.0,
            fMax: Double(sampleRate) / 2.0
        ) // [nMels, fftBins]

        guard let fftPlan = try? RealFFTPlan(size: nFFT) else {
            return MLXArray.zeros([1, frameCount, nMels], dtype: .float32)
        }

        var melOutput = [Float](repeating: 0, count: frameCount * nMels)
        var frameBuffer = [Float](repeating: 0, count: nFFT)
        var magnitudes = [Float](repeating: 0, count: fftBins)

        for frameIdx in 0..<frameCount {
            let start = frameIdx * hopSize
            for i in 0..<nFFT {
                frameBuffer[i] = paddedSamples[start + i] * hann[i]
            }

            magnitudes = fftPlan.powerSpectrum(frameBuffer)

            for melIdx in 0..<nMels {
                let rowOffset = melIdx * fftBins
                var melEnergy: Float = 0
                for bin in 0..<fftBins {
                    let specMag = sqrt(magnitudes[bin] + 1e-9)
                    melEnergy += specMag * melBasis[rowOffset + bin]
                }
                melOutput[(frameIdx * nMels) + melIdx] = log(max(1e-5, melEnergy))
            }
        }

        return MLXArray(melOutput).reshaped(1, frameCount, nMels).asType(.float32)
    }

    private static func normalize(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return samples }
        var peak: Float = 0
        for sample in samples {
            peak = max(peak, abs(sample))
        }
        guard peak > 0 else { return samples }
        let gain = min(Float(0.98) / peak, Float(8.0))
        return samples.map { value in
            let scaled = value * gain
            return max(-1.0, min(1.0, scaled))
        }
    }

    private static func trimSilence(_ samples: [Float], threshold: Float) -> [Float] {
        guard !samples.isEmpty else { return samples }

        var start = 0
        while start < samples.count && abs(samples[start]) < threshold {
            start += 1
        }

        var end = samples.count - 1
        while end > start && abs(samples[end]) < threshold {
            end -= 1
        }

        guard start <= end else { return [] }
        return Array(samples[start...end])
    }

    private static func reflectPad(_ samples: [Float], padding: Int) -> [Float] {
        guard padding > 0, samples.count > 1 else { return samples }

        let usablePad = min(padding, samples.count - 1)
        var result: [Float] = []
        result.reserveCapacity(samples.count + (2 * usablePad))

        // Python parity: left = sample[1:padding+1][::-1]
        for idx in stride(from: usablePad, through: 1, by: -1) {
            result.append(samples[idx])
        }
        result.append(contentsOf: samples)
        // Python parity: right = sample[-(padding+1):-1][::-1]
        for idx in 0..<usablePad {
            result.append(samples[samples.count - 2 - idx])
        }

        return result
    }

    private static func hannWindow(size: Int) -> [Float] {
        guard size > 1 else { return [1.0] }
        let denom = Float(size - 1) // Non-periodic Hann (dsp.hanning periodic=False)
        return (0..<size).map { n in
            0.5 * (1.0 - cos((2.0 * Float.pi * Float(n)) / denom))
        }
    }

    private static func slaneyMelFilterBank(
        sampleRate: Int,
        nFFT: Int,
        nMels: Int,
        fMin: Double,
        fMax: Double
    ) -> [Float] {
        let nFreqs = (nFFT / 2) + 1
        let allFreqs: [Double] = (0..<nFreqs).map { idx in
            Double(idx) * Double(sampleRate / 2) / Double(max(1, nFreqs - 1))
        }

        let mMin = hzToMelSlaney(fMin)
        let mMax = hzToMelSlaney(fMax)
        let melPoints = linspace(start: mMin, end: mMax, count: nMels + 2)
        let freqPoints = melPoints.map(melToHzSlaney)

        var filter = [Float](repeating: 0, count: nMels * nFreqs)
        for melIdx in 0..<nMels {
            let left = freqPoints[melIdx]
            let center = freqPoints[melIdx + 1]
            let right = freqPoints[melIdx + 2]
            let denomLeft = max(1e-12, center - left)
            let denomRight = max(1e-12, right - center)
            let slaneyNorm = Float(2.0 / max(1e-12, right - left))

            let rowOffset = melIdx * nFreqs
            for (freqIdx, freq) in allFreqs.enumerated() {
                let down = (freq - left) / denomLeft
                let up = (right - freq) / denomRight
                let tri = max(0.0, min(down, up))
                filter[rowOffset + freqIdx] = Float(tri) * slaneyNorm
            }
        }

        return filter
    }

    private static func hzToMelSlaney(_ hz: Double) -> Double {
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logStep = log(6.4) / 27.0
        if hz >= minLogHz {
            return minLogMel + (log(hz / minLogHz) / logStep)
        }
        return hz / fSp
    }

    private static func melToHzSlaney(_ mel: Double) -> Double {
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logStep = log(6.4) / 27.0
        if mel >= minLogMel {
            return minLogHz * exp(logStep * (mel - minLogMel))
        }
        return fSp * mel
    }

    private static func linspace(start: Double, end: Double, count: Int) -> [Double] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Double(count - 1)
        return (0..<count).map { start + (Double($0) * step) }
    }
}
