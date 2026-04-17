import Accelerate
import Foundation
import MLX

public final class ParakeetAudioPreprocessor {
    public let config: ParakeetPreprocessorConfig

    private let fftSetup: vDSP.FFT<DSPSplitComplex>
    private let window: [Float]
    private let melFilters: [[Float]]

    public init(config: ParakeetPreprocessorConfig) {
        self.config = config

        let log2n = vDSP_Length(log2(Double(config.nFFT)))
        guard let fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self) else {
            fatalError("Failed to create FFT setup for Parakeet preprocessor")
        }
        self.fftSetup = fftSetup

        self.window = Self.makeWindow(name: config.window, count: config.winLength)
        self.melFilters = Self.createMelFilterbank(
            nMels: config.features,
            nFFT: config.nFFT,
            sampleRate: config.sampleRate,
            normalize: config.normalize
        )
    }

    public func logMelSpectrogram(from audio: [Float]) -> MLXArray {
        let preprocessed = applyPaddingAndPreemphasis(audio)
        let padded = reflectCenterPad(preprocessed, padding: config.nFFT / 2)
        let frames = max(1, 1 + (padded.count - config.nFFT) / max(1, config.hopLength))

        var mel = [[Float]](
            repeating: [Float](repeating: 0, count: frames),
            count: config.features
        )

        var real = [Float](repeating: 0, count: config.nFFT / 2)
        var imag = [Float](repeating: 0, count: config.nFFT / 2)
        var frameBuffer = [Float](repeating: 0, count: config.nFFT)
        var fftWindow = [Float](repeating: 0, count: config.nFFT)
        let copyCount = min(config.winLength, config.nFFT)
        if copyCount > 0 {
            for i in 0..<copyCount {
                fftWindow[i] = window[i]
            }
        }
        var magnitudes = [Float](repeating: 0, count: config.nFFT / 2 + 1)

        for frameIndex in 0..<frames {
            let start = frameIndex * config.hopLength
            frameBuffer = [Float](repeating: 0, count: config.nFFT)

            if config.nFFT > 0 {
                for i in 0..<config.nFFT {
                    let idx = start + i
                    guard idx < padded.count else { break }
                    frameBuffer[i] = padded[idx] * fftWindow[i]
                }
            }

            real = [Float](repeating: 0, count: config.nFFT / 2)
            imag = [Float](repeating: 0, count: config.nFFT / 2)

            frameBuffer.withUnsafeBufferPointer { inputPtr in
                real.withUnsafeMutableBufferPointer { realPtr in
                    imag.withUnsafeMutableBufferPointer { imagPtr in
                        var split = DSPSplitComplex(
                            realp: realPtr.baseAddress!,
                            imagp: imagPtr.baseAddress!
                        )

                        vDSP_ctoz(
                            UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                            2,
                            &split,
                            1,
                            vDSP_Length(config.nFFT / 2)
                        )

                        fftSetup.forward(input: split, output: &split)
                    }
                }
            }

            magnitudes = [Float](repeating: 0, count: config.nFFT / 2 + 1)
            magnitudes[0] = real[0] * real[0]
            for i in 1..<(config.nFFT / 2) {
                magnitudes[i] = real[i] * real[i] + imag[i] * imag[i]
            }
            magnitudes[config.nFFT / 2] = imag[0] * imag[0]

            for melIndex in 0..<config.features {
                var energy: Float = 0
                for freq in 0..<melFilters[melIndex].count {
                    energy += melFilters[melIndex][freq] * magnitudes[freq]
                }
                mel[melIndex][frameIndex] = logf(max(energy, 0) + 1e-5)
            }
        }

        normalize(&mel)

        var flat = [Float](repeating: 0, count: frames * config.features)
        for t in 0..<frames {
            for f in 0..<config.features {
                flat[t * config.features + f] = mel[f][t]
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

    private func normalize(_ mel: inout [[Float]]) {
        guard !mel.isEmpty, !mel[0].isEmpty else { return }

        if config.normalize == "per_feature" {
            for i in mel.indices {
                let values = mel[i]
                let mean = values.reduce(0, +) / Float(values.count)
                let variance = values.reduce(0) { partial, value in
                    let diff = value - mean
                    return partial + diff * diff
                } / Float(values.count)
                let std = sqrt(max(variance, 0)) + 1e-5

                for j in mel[i].indices {
                    mel[i][j] = (mel[i][j] - mean) / std
                }
            }
            return
        }

        let allValues = mel.flatMap { $0 }
        let mean = allValues.reduce(0, +) / Float(allValues.count)
        let variance = allValues.reduce(0) { partial, value in
            let diff = value - mean
            return partial + diff * diff
        } / Float(allValues.count)
        let std = sqrt(max(variance, 0)) + 1e-5

        for i in mel.indices {
            for j in mel[i].indices {
                mel[i][j] = (mel[i][j] - mean) / std
            }
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
