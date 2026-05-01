import Foundation
import Accelerate
import MLX

/// Mel spectrogram configuration.
public struct Qwen3ASRMelConfig: Sendable, Hashable {
    public let sampleRate: Int
    public let nMels: Int
    public let nFFT: Int
    public let hopLength: Int
    public let winLength: Int
    public let fMin: Float
    public let fMax: Float

    public init(
        sampleRate: Int = 16000,
        nMels: Int = 128,
        nFFT: Int = 400,
        hopLength: Int = 160,
        winLength: Int = 400,
        fMin: Float = 0.0,
        fMax: Float = 8000.0
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.fMin = fMin
        self.fMax = fMax
    }
}

/// Mel spectrogram extractor using vDSP for FFT
/// Matches Whisper/OpenAI preprocessing for ASR models
public final class MelSpectrogram {
    let nMels: Int
    let nFFT: Int
    let hopLength: Int
    let winLength: Int
    let sampleRate: Int

    private let fftSetup: vDSP.FFT<DSPSplitComplex>
    private let hannWindow: [Float]
    private let melFilters: [[Float]]
    private let log2n: vDSP_Length

    public init(config: Qwen3ASRMelConfig = Qwen3ASRMelConfig()) {
        self.nMels = config.nMels
        self.nFFT = config.nFFT
        self.hopLength = config.hopLength
        self.winLength = config.winLength
        self.sampleRate = config.sampleRate

        // FFT setup
        self.log2n = vDSP_Length(log2(Double(nFFT)))
        self.fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)!

        // Hann window (periodic, matches WhisperFeatureExtractor)
        var window = [Float](repeating: 0, count: winLength)
        let denom = Float(winLength)
        for i in 0..<winLength {
            window[i] = 0.5 - 0.5 * cosf(2.0 * Float.pi * Float(i) / denom)
        }
        self.hannWindow = window

        // Mel filterbank
        self.melFilters = Self.createMelFilterbank(
            nMels: nMels,
            nFFT: nFFT,
            sampleRate: sampleRate,
            fMin: config.fMin,
            fMax: config.fMax
        )
    }

    /// Extract mel spectrogram from audio samples
    /// - Parameter audio: Audio samples in range [-1, 1] at 16kHz
    /// - Returns: MLXArray of shape [1, nMels, numFrames]
    public func extract(from audio: [Float]) -> MLXArray {
        let paddedAudio = reflectPad(audio, pad: winLength / 2)
        let numFrames = max(1, 1 + (paddedAudio.count - winLength) / hopLength)
        var melSpec = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: nMels)

        // Buffers for FFT
        var realp = [Float](repeating: 0, count: nFFT / 2)
        var imagp = [Float](repeating: 0, count: nFFT / 2)
        var paddedFrame = [Float](repeating: 0, count: nFFT)
        var magnitudes = [Float](repeating: 0, count: nFFT / 2 + 1)

        for frameIdx in 0..<numFrames {
            let start = frameIdx * hopLength

            // Zero-pad and apply window
            paddedFrame = [Float](repeating: 0, count: nFFT)
            for i in 0..<winLength {
                let audioIdx = start + i
                if audioIdx < paddedAudio.count {
                    paddedFrame[i] = paddedAudio[audioIdx] * hannWindow[i]
                }
            }

            // Perform FFT
            realp = [Float](repeating: 0, count: nFFT / 2)
            imagp = [Float](repeating: 0, count: nFFT / 2)

            paddedFrame.withUnsafeBufferPointer { inputPtr in
                realp.withUnsafeMutableBufferPointer { realPtr in
                    imagp.withUnsafeMutableBufferPointer { imagPtr in
                        var splitComplex = DSPSplitComplex(
                            realp: realPtr.baseAddress!,
                            imagp: imagPtr.baseAddress!
                        )
                        vDSP_ctoz(
                            UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                            2,
                            &splitComplex,
                            1,
                            vDSP_Length(nFFT / 2)
                        )
                        fftSetup.forward(input: splitComplex, output: &splitComplex)
                    }
                }
            }

            // Compute magnitude spectrum
            magnitudes = [Float](repeating: 0, count: nFFT / 2 + 1)
            for i in 0..<(nFFT / 2) {
                magnitudes[i] = realp[i] * realp[i] + imagp[i] * imagp[i]
            }
            // DC and Nyquist
            magnitudes[0] = realp[0] * realp[0]
            if nFFT / 2 < magnitudes.count {
                magnitudes[nFFT / 2] = imagp[0] * imagp[0]
            }

            // Apply mel filterbank
            for melIdx in 0..<nMels {
                var energy: Float = 0
                for freqIdx in 0..<melFilters[melIdx].count {
                    energy += melFilters[melIdx][freqIdx] * magnitudes[freqIdx]
                }
                melSpec[melIdx][frameIdx] = energy
            }
        }

        // Convert to log scale (matching WhisperFeatureExtractor)
        let outFrames = max(1, numFrames - 1)  // drop last frame like Whisper
        var flatMel = [Float](repeating: 0, count: nMels * outFrames)
        var maxVal: Float = -Float.infinity

        for melIdx in 0..<nMels {
            for frameIdx in 0..<outFrames {
                let idx = melIdx * outFrames + frameIdx
                let energy = melSpec[melIdx][frameIdx]
                let logVal = log10(max(energy, 1e-10))
                flatMel[idx] = logVal
                maxVal = max(maxVal, logVal)
            }
        }

        // Clamp dynamic range and scale
        let clampFloor = maxVal - 8.0
        for i in 0..<flatMel.count {
            flatMel[i] = max(flatMel[i], clampFloor)
            flatMel[i] = (flatMel[i] + 4.0) / 4.0
        }

        // Create MLXArray [1, nMels, outFrames]
        let array = MLXArray(flatMel).reshaped(1, nMels, outFrames)
        return array
    }

    /// Create mel filterbank matrix
    private static func createMelFilterbank(
        nMels: Int,
        nFFT: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [[Float]] {
        let nFreqs = nFFT / 2 + 1
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        let melPoints = linspace(from: melMin, to: melMax, count: nMels + 2)
        let filterFreqs = melPoints.map { melToHz($0) }
        let fftFreqs = linspace(from: 0, to: Float(sampleRate) / 2.0, count: nFreqs)

        var filters = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nMels)

        for m in 0..<nMels {
            let left = filterFreqs[m]
            let center = filterFreqs[m + 1]
            let right = filterFreqs[m + 2]
            let leftDen = center - left
            let rightDen = right - center

            for f in 0..<nFreqs {
                let freq = fftFreqs[f]
                let down = leftDen == 0 ? 0 : (freq - left) / leftDen
                let up = rightDen == 0 ? 0 : (right - freq) / rightDen
                let value = max(0, min(down, up))
                filters[m][f] = value
            }
        }

        // Slaney-style normalization
        for m in 0..<nMels {
            let denom = filterFreqs[m + 2] - filterFreqs[m]
            if denom > 0 {
                let enorm = 2.0 / denom
                for f in 0..<nFreqs {
                    filters[m][f] *= enorm
                }
            }
        }

        return filters
    }

    /// Convert Hz to mel scale
    private static func hzToMel(_ hz: Float) -> Float {
        // Slaney formula (matches librosa)
        let f_min: Float = 0.0
        let f_sp: Float = 200.0 / 3.0
        let min_log_hz: Float = 1000.0
        let min_log_mel = (min_log_hz - f_min) / f_sp
        let logstep: Float = logf(6.4) / 27.0

        if hz >= min_log_hz {
            return min_log_mel + logf(hz / min_log_hz) / logstep
        } else {
            return (hz - f_min) / f_sp
        }
    }

    /// Convert mel to Hz
    private static func melToHz(_ mel: Float) -> Float {
        let f_min: Float = 0.0
        let f_sp: Float = 200.0 / 3.0
        let min_log_hz: Float = 1000.0
        let min_log_mel = (min_log_hz - f_min) / f_sp
        let logstep: Float = logf(6.4) / 27.0

        if mel >= min_log_mel {
            return min_log_hz * expf(logstep * (mel - min_log_mel))
        } else {
            return f_min + f_sp * mel
        }
    }

    private static func linspace(from: Float, to: Float, count: Int) -> [Float] {
        guard count > 1 else { return [from] }
        let step = (to - from) / Float(count - 1)
        return (0..<count).map { from + Float($0) * step }
    }

    private func reflectPad(_ audio: [Float], pad: Int) -> [Float] {
        guard pad > 0 else { return audio }
        let count = audio.count
        guard count > 0 else { return audio }
        if count == 1 {
            return [Float](repeating: audio[0], count: count + 2 * pad)
        }

        var padded = [Float](repeating: 0, count: count + 2 * pad)

        // Left pad (reflect without repeating edge)
        for i in 0..<pad {
            let src = min(pad - i, count - 1)
            padded[i] = audio[src]
        }

        // Original
        for i in 0..<count {
            padded[pad + i] = audio[i]
        }

        // Right pad (reflect without repeating edge)
        for i in 0..<pad {
            let src = max(count - 2 - i, 0)
            padded[pad + count + i] = audio[src]
        }

        return padded
    }
}
