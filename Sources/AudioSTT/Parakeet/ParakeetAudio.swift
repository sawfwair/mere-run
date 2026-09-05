import AudioCodecs
import Foundation
import MLX

#if canImport(Accelerate)
import Accelerate
#endif

public final class ParakeetAudioPreprocessor {
    public enum PreprocessorError: LocalizedError {
        case invalidFFTSize(Int)
        case fftSetupFailed(nFFT: Int)

        public var errorDescription: String? {
            switch self {
            case .invalidFFTSize(let nFFT):
                return "Parakeet preprocessor nFFT must be a positive power of two; got \(nFFT)."
            case .fftSetupFailed(let nFFT):
                return "Failed to create FFT setup for Parakeet preprocessor with nFFT=\(nFFT)."
            }
        }
    }

    public let config: ParakeetPreprocessorConfig

    private let fftPlan: RealFFTPlan
    private let fftWindow: [Float]
    private let melFilterMatrix: [Float]

    public init(config: ParakeetPreprocessorConfig) throws {
        self.config = config

        guard config.nFFT > 0 else {
            throw PreprocessorError.invalidFFTSize(config.nFFT)
        }

        do {
            self.fftPlan = try RealFFTPlan(size: config.nFFT)
        } catch {
            throw PreprocessorError.fftSetupFailed(nFFT: config.nFFT)
        }

        let window = Self.makeWindow(name: config.window, count: config.winLength)
        var fftWindow = [Float](repeating: 0, count: config.nFFT)
        for index in 0..<min(config.winLength, config.nFFT) {
            fftWindow[index] = window[index]
        }
        self.fftWindow = fftWindow
        self.melFilterMatrix = Self.createMelFilterbank(
            nMels: config.features,
            nFFT: config.nFFT,
            sampleRate: config.sampleRate,
            normalize: config.normalize
        ).flatMap { $0 }
    }

    public func logMelSpectrogram(from audio: [Float]) -> MLXArray {
        let preprocessed = applyPaddingAndPreemphasis(audio)
        let padded = reflectCenterPad(preprocessed, padding: config.nFFT / 2)
        let frames = max(1, 1 + (padded.count - config.nFFT) / max(1, config.hopLength))

        var frameBuffer = [Float](repeating: 0, count: config.nFFT)
        let frequencyCount = config.nFFT / 2 + 1
        var transposedSpectra = [Float](repeating: 0, count: frequencyCount * frames)

        for frameIndex in 0..<frames {
            let start = frameIndex * config.hopLength
            let availableSamples = min(config.nFFT, max(0, padded.count - start))
            for index in 0..<availableSamples {
                frameBuffer[index] = padded[start + index] * fftWindow[index]
            }
            if availableSamples < config.nFFT {
                for index in availableSamples..<config.nFFT {
                    frameBuffer[index] = 0
                }
            }

            let magnitudes = fftPlan.powerSpectrum(frameBuffer)
            for frequency in 0..<frequencyCount {
                transposedSpectra[frequency * frames + frameIndex] = magnitudes[frequency]
            }
        }

        var mel = [Float](repeating: 0, count: config.features * frames)
        multiplyFilterbank(
            melFilterMatrix,
            by: transposedSpectra,
            frequencyCount: frequencyCount,
            frameCount: frames,
            result: &mel
        )
        for index in mel.indices {
            mel[index] = logf(max(mel[index], 0) + 1e-5)
        }
        normalize(&mel, frameCount: frames)

        var flat = [Float](repeating: 0, count: frames * config.features)
        for t in 0..<frames {
            for f in 0..<config.features {
                flat[t * config.features + f] = mel[f * frames + t]
            }
        }

        return MLXArray(flat).reshaped(1, frames, config.features)
    }

    private func applyPaddingAndPreemphasis(_ audio: [Float]) -> [Float] {
        guard !audio.isEmpty else {
            return [Float](repeating: Float(config.padValue), count: max(config.padTo, 1))
        }

        var result = audio

        if config.padTo > 0, result.count < config.padTo {
            result += [Float](repeating: Float(config.padValue), count: config.padTo - result.count)
        }

        if config.preemph > 0, result.count > 1 {
            var emphasized = [Float](repeating: 0, count: result.count)
            emphasized[0] = result[0]
            let preemph = Float(config.preemph)
            for i in 1..<result.count {
                emphasized[i] = result[i] - preemph * result[i - 1]
            }
            result = emphasized
        }

        return result
    }

    private func reflectCenterPad(_ audio: [Float], padding: Int) -> [Float] {
        guard padding > 0 else { return audio }
        guard audio.count > 1 else { return audio }

        let prefixEnd = min(audio.count, padding + 1)
        let prefix = Array(audio[1..<prefixEnd].reversed())

        let suffixStart = max(0, audio.count - (padding + 1))
        let suffixEnd = max(suffixStart, audio.count - 1)
        let suffix = Array(audio[suffixStart..<suffixEnd].reversed())

        return prefix + audio + suffix
    }

    private func multiplyFilterbank(
        _ filters: [Float],
        by spectra: [Float],
        frequencyCount: Int,
        frameCount: Int,
        result: inout [Float]
    ) {
        #if canImport(Accelerate)
        vDSP_mmul(
            filters, 1,
            spectra, 1,
            &result, 1,
            vDSP_Length(config.features),
            vDSP_Length(frameCount),
            vDSP_Length(frequencyCount)
        )
        #else
        for feature in 0..<config.features {
            for frame in 0..<frameCount {
                var energy: Float = 0
                for frequency in 0..<frequencyCount {
                    energy += filters[feature * frequencyCount + frequency]
                        * spectra[frequency * frameCount + frame]
                }
                result[feature * frameCount + frame] = energy
            }
        }
        #endif
    }

    private func normalize(_ mel: inout [Float], frameCount: Int) {
        guard !mel.isEmpty, frameCount > 0 else { return }

        if config.normalize == "per_feature" {
            for feature in 0..<config.features {
                let start = feature * frameCount
                let end = start + frameCount
                let mean = mel[start..<end].reduce(0, +) / Float(frameCount)
                let variance = mel[start..<end].reduce(0) { partial, value in
                    let difference = value - mean
                    return partial + difference * difference
                } / Float(frameCount)
                let std = sqrt(max(variance, 0)) + 1e-5

                for index in start..<end {
                    mel[index] = (mel[index] - mean) / std
                }
            }
            return
        }

        let mean = mel.reduce(0, +) / Float(mel.count)
        let variance = mel.reduce(0) { partial, value in
            let difference = value - mean
            return partial + difference * difference
        } / Float(mel.count)
        let std = sqrt(max(variance, 0)) + 1e-5

        for index in mel.indices {
            mel[index] = (mel[index] - mean) / std
        }
    }

    private static func makeWindow(name: String, count: Int) -> [Float] {
        guard count > 0 else { return [] }

        let lower = name.lowercased()
        if count == 1 {
            return [1.0]
        }

        let denom = Float(max(1, count - 1))
        switch lower {
        case "hann", "hanning":
            return (0..<count).map { i in
                0.5 * (1.0 - cosf(2.0 * Float.pi * Float(i) / denom))
            }
        case "hamming":
            return (0..<count).map { i in
                0.54 - 0.46 * cosf(2.0 * Float.pi * Float(i) / denom)
            }
        case "blackman":
            return (0..<count).map { i in
                let theta = 2.0 * Float.pi * Float(i) / denom
                return 0.42 - 0.5 * cosf(theta) + 0.08 * cosf(2.0 * theta)
            }
        case "bartlett":
            let mid = denom / 2.0
            return (0..<count).map { i in
                1.0 - 2.0 * abs(Float(i) - mid) / denom
            }
        default:
            return (0..<count).map { i in
                0.5 * (1.0 - cosf(2.0 * Float.pi * Float(i) / denom))
            }
        }
    }

    private static func createMelFilterbank(
        nMels: Int,
        nFFT: Int,
        sampleRate: Int,
        normalize: String
    ) -> [[Float]] {
        let nFreqs = nFFT / 2 + 1
        let fMin: Float = 0
        let fMax = Float(sampleRate) / 2

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melPoints = linspace(from: melMin, to: melMax, count: nMels + 2)
        let hzPoints = melPoints.map { melToHz($0) }

        let fftFreqs = linspace(from: 0, to: fMax, count: nFreqs)

        var filters = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nMels)

        for m in 0..<nMels {
            let left = hzPoints[m]
            let center = hzPoints[m + 1]
            let right = hzPoints[m + 2]

            let leftDen = max(center - left, 1e-8)
            let rightDen = max(right - center, 1e-8)

            for f in 0..<nFreqs {
                let freq = fftFreqs[f]
                let down = (freq - left) / leftDen
                let up = (right - freq) / rightDen
                filters[m][f] = max(0, min(down, up))
            }

            if normalize == "slaney" {
                let denom = max(right - left, 1e-8)
                let enorm: Float = 2 / denom
                for f in 0..<nFreqs {
                    filters[m][f] *= enorm
                }
            }
        }

        return filters
    }

    private static func hzToMel(_ hz: Float) -> Float {
        let fMin: Float = 0.0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = (minLogHz - fMin) / fSp
        let logStep: Float = logf(6.4) / 27.0

        if hz >= minLogHz {
            return minLogMel + logf(hz / minLogHz) / logStep
        }
        return (hz - fMin) / fSp
    }

    private static func melToHz(_ mel: Float) -> Float {
        let fMin: Float = 0.0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = (minLogHz - fMin) / fSp
        let logStep: Float = logf(6.4) / 27.0

        if mel >= minLogMel {
            return minLogHz * expf(logStep * (mel - minLogMel))
        }
        return fMin + fSp * mel
    }

    private static func linspace(from: Float, to: Float, count: Int) -> [Float] {
        guard count > 1 else { return [from] }
        let step = (to - from) / Float(count - 1)
        return (0..<count).map { from + Float($0) * step }
    }
}
