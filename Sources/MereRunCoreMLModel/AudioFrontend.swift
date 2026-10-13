// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// Audio front end: samples -> 16 kHz mono -> log-mel frames -> the inputs of `audio_<Tb>`. The feature extraction
// re-implements the defaults of Gemma4AudioFeatureExtractor (feature_extraction_gemma4.py in Hugging Face
// transformers 5.19, Copyright 2026 Google LLC, Apache-2.0) and the HTK mel filter-bank formula of transformers'
// audio_utils.py (Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors, Apache-2.0; the
// formula as in librosa / torchaudio) exactly as python/eg2_audio_frontend.py does in numpy:
//   first 30 s (480 000 samples), right pad to a multiple of 128 samples, 160 zeros in front ("semicausal"),
//   frames of 320 samples every 160 (the window is periodic Hann), |rfft(512)| in single precision, 128 triangular
//   HTK mel filters 0-8000 Hz (no normalization) applied in double precision, ln(mel + 0.001) cast to float32;
//   a frame is valid only if its 321-sample analysis window lies in real audio (at most 2999 frames).
// The FFT is Accelerate's; its single-precision rounding differs from numpy's pocketfft by ~1e-6 (README).

import Accelerate
import Foundation

public enum AudioFrontend {
    public static let sampleRate = 16000
    public static let frameLength = 320
    public static let hopLength = 160
    public static let fftLength = 512
    public static let nMels = 128
    public static let melFloor = 1e-3
    public static let maxSamples = 480_000          // 30 s, the upstream default truncation
    public static let padMultiple = 128
    public static let maxFrames = 2999

    // MARK: - Frame counts

    /// Frames whose whole 321-sample analysis window (with the 160-sample left pad) lies in real audio.
    public static func validFrameCount(samples n: Int) -> Int {
        frameCount(samples: min(n, maxSamples))
    }

    /// (n + 160 - 321) floor-div 160 + 1, zero when the first window does not fit (n < 161).
    static func frameCount(samples n: Int) -> Int {
        let d = n + frameLength / 2 - (frameLength + 1)
        return d < 0 ? 0 : d / hopLength + 1
    }

    public static func softTokenCount(frames: Int) -> Int { ((frames + 1) / 2 + 1) / 2 }

    // MARK: - Mel filters and window (computed once, double precision like the reference)

    static let melFilters: [Double] = {   // [257 * 128], row-major (bin, mel)
        func hzToMel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
        func melToHz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }
        let nBins = fftLength / 2 + 1
        let mLo = hzToMel(0.0), mHi = hzToMel(8000.0)
        var filterFreqs = [Double](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {   // numpy.linspace: start + i * step, last = stop
            let step = (mHi - mLo) / Double(nMels + 1)
            filterFreqs[i] = melToHz(i == nMels + 1 ? mHi : mLo + Double(i) * step)
        }
        var fftFreqs = [Double](repeating: 0, count: nBins)
        for i in 0..<nBins {
            let step = Double(sampleRate / 2) / Double(nBins - 1)
            fftFreqs[i] = i == nBins - 1 ? Double(sampleRate / 2) : Double(i) * step
        }
        var diff = [Double](repeating: 0, count: nMels + 1)
        for i in 0..<(nMels + 1) { diff[i] = filterFreqs[i + 1] - filterFreqs[i] }
        var w = [Double](repeating: 0, count: nBins * nMels)
        for b in 0..<nBins {
            for m in 0..<nMels {
                let down = -(filterFreqs[m] - fftFreqs[b]) / diff[m]
                let up = (filterFreqs[m + 2] - fftFreqs[b]) / diff[m + 1]
                w[b * nMels + m] = max(0.0, min(down, up))
            }
        }
        return w
    }()

    static let hannWindow: [Float] = (0..<frameLength).map { n in
        Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(n) / Double(frameLength)))
    }

    // MARK: - Log-mel

    /// Log-mel features of a 16 kHz mono signal: [validFrames * 128] Float, row-major, at most 2999 frames.
    public static func logMel(_ wav16k: [Float], maxSamples limit: Int? = maxSamples) -> (features: [Float], frames: Int) {
        let x = limit.map { Array(wav16k.prefix($0)) } ?? wav16k
        let n = x.count
        let m = frameCount(samples: n)
        if m <= 0 { return ([], 0) }
        let nPad = (n + padMultiple - 1) / padMultiple * padMultiple
        var xp = [Float](repeating: 0, count: frameLength / 2 + nPad)
        for i in 0..<n { xp[frameLength / 2 + i] = x[i] }
        let nBins = fftLength / 2 + 1
        guard let dft = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fftLength), .FORWARD) else { return ([], 0) }
        defer { vDSP_DFT_DestroySetup(dft) }
        var inRe = [Float](repeating: 0, count: fftLength), inIm = [Float](repeating: 0, count: fftLength)
        var outRe = [Float](repeating: 0, count: fftLength), outIm = [Float](repeating: 0, count: fftLength)
        var mag = [Double](repeating: 0, count: m * nBins)
        for t in 0..<m {
            let s = t * hopLength
            for i in 0..<frameLength { inRe[i] = xp[s + i] * hannWindow[i] }
            for i in frameLength..<fftLength { inRe[i] = 0 }
            vDSP_DFT_Execute(dft, inRe, inIm, &outRe, &outIm)
            for b in 0..<nBins {
                let re = outRe[b], im = outIm[b]
                mag[t * nBins + b] = Double((re * re + im * im).squareRoot())   // magnitude in Float, like numpy complex64
            }
        }
        var mel = [Double](repeating: 0, count: m * nMels)
        melFilters.withUnsafeBufferPointer { f in
            mag.withUnsafeBufferPointer { a in
                mel.withUnsafeMutableBufferPointer { c in
                    // [m x 257] x [257 x 128] in double precision (row-major)
                    vDSP_mmulD(a.baseAddress!, 1, f.baseAddress!, 1, c.baseAddress!, 1, vDSP_Length(m), vDSP_Length(nMels), vDSP_Length(nBins))
                }
            }
        }
        var feats = [Float](repeating: 0, count: m * nMels)
        for i in 0..<(m * nMels) { feats[i] = Float(log(mel[i] + melFloor)) }
        return (feats, m)
    }

}

#endif
