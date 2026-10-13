// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// Sample-rate conversion to 16 kHz for the audio embedder: a polyphase Kaiser-windowed sinc low-pass computed in
// Double, the same design and the same arithmetic as python/eg2_audio_frontend.py (pass band 0.913 and stop band
// 1.0 of the lower Nyquist, 125 dB attenuation, unity DC gain, output length ceil(n * 16000 / rate) as librosa's
// fix=True). The upstream pipeline uses libsoxr "HQ" instead; README "Audio: decoding and resampling" gives the
// measured difference. Input already at 16 kHz is returned unchanged.

import Accelerate
import Foundation

public enum AudioResampler {
    public static let targetRate = 16000
    static let passband = 0.913
    static let stopband = 1.0
    static let attenuationDB = 125.0

    /// Modified Bessel function of order 0 by its power series (identical to the Python version).
    static func besselI0(_ x: Double) -> Double {
        var s = 1.0, t = 1.0
        var k = 1.0
        let h = x / 2.0
        while true {
            t *= (h / k) * (h / k)
            s += t
            if t < s * 1e-17 { return s }
            k += 1
        }
    }

    /// Low-pass taps (odd length, unity DC gain) and the delay (len-1)/2 for in_rate -> in_rate*up/down.
    static func sincFilter(up: Int, down: Int, inRate: Int) -> (taps: [Double], delay: Int) {
        let nyqLow = Double(min(inRate, inRate * up / down)) / 2.0
        let nyqHi = Double(inRate * up) / 2.0
        let fc = 0.5 * (passband + stopband) * nyqLow / nyqHi
        let tw = (stopband - passband) * nyqLow / nyqHi
        let beta = 0.1102 * (attenuationDB - 8.7)
        var n = Int(ceil((attenuationDB - 7.95) / (2.285 * Double.pi * tw)))
        n |= 1
        let c = (n - 1) / 2
        let i0b = besselI0(beta)
        var h = [Double](repeating: 0, count: n)
        var sum = 0.0
        for k in 0..<n {
            let m = Double(k - c)
            let sinc = (k == c) ? fc : sin(Double.pi * fc * m) / (Double.pi * m)
            let r = 2.0 * Double(k) / Double(n - 1) - 1.0
            let w = besselI0(beta * sqrt(max(0.0, 1.0 - r * r))) / i0b
            h[k] = sinc * w
            sum += h[k]
        }
        for k in 0..<n { h[k] /= sum }
        return (h, c)
    }

    /// Per-phase tap vectors: phase p (= (j*down) % up) uses h[c + p + r*up] for r = rMin(p) ... rMax(p), stored in
    /// increasing r and applied to x[q - r] (decreasing sample index), q = (j*down - p) / up.
    struct Polyphase {
        let phases: [(j0: Int, rMin: Int, taps: [Double])]   // j0 = first output index of the phase, -1 = unused
    }

    static func polyphase(up: Int, down: Int, inRate: Int) -> Polyphase {
        let (h, c) = sincFilter(up: up, down: down, inRate: inRate)
        var phases: [(j0: Int, rMin: Int, taps: [Double])] = []
        for p in 0..<up {
            var j0 = -1
            for j in 0..<up where (j * down) % up == p { j0 = j; break }
            let rMin = -((c + p) / up)
            let rMax = (h.count - 1 - c - p) / up
            // stored REVERSED (r = rMax first) so that the dot product walks the input forward from x[q - rMax]
            phases.append((j0, rMin, j0 < 0 ? [] : (rMin...rMax).reversed().map { h[c + p + $0 * up] }))
        }
        return Polyphase(phases: phases)
    }

    // guarded by `lock` (filters are computed once per input rate and shared)
    nonisolated(unsafe) private static var cache: [String: Polyphase] = [:]
    private static let lock = NSLock()

    static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

    /// Float samples at `rate` -> Float samples at 16 kHz. y[j] = up * sum_m x[m] h[c + j*down - m*up] in Double
    /// (each output is one vDSP double-precision dot product of a phase's taps with a reversed slice of the input).
    public static func resampleTo16k(_ x: [Float], rate: Int) -> [Float] {
        if rate == targetRate { return x }
        let nOut = Int(ceil(Double(x.count) * Double(targetRate) / Double(rate)))
        let g = gcd(targetRate, rate)
        let up = targetRate / g, down = rate / g
        let key = "\(up)/\(down)/\(rate)"
        lock.lock()
        var filt = cache[key]
        if filt == nil {
            filt = polyphase(up: up, down: down, inRate: rate)
            cache[key] = filt
        }
        lock.unlock()
        let pf = filt!
        let pad = (pf.phases.map { $0.taps.count }.max() ?? 0) + 1   // zero padding so no tap window leaves the buffer
        var xd = [Double](repeating: 0, count: x.count + 2 * pad)
        for i in 0..<x.count { xd[pad + i] = Double(x[i]) }
        var y = [Double](repeating: 0, count: nOut)
        let scale = Double(up)
        xd.withUnsafeBufferPointer { xb in
            for (p, ph) in pf.phases.enumerated() where ph.j0 >= 0 {
                let L = vDSP_Length(ph.taps.count)
                ph.taps.withUnsafeBufferPointer { hb in
                    let rMax = ph.rMin + Int(L) - 1
                    var j = ph.j0
                    while j < nOut {
                        let q = (j * down - p) / up
                        // taps (reversed) k = 0 ... L-1 multiply x[q - rMax + k]; the padding keeps every index in range
                        var acc = 0.0
                        vDSP_dotprD(xb.baseAddress! + (pad + q - rMax), 1, hb.baseAddress!, 1, &acc, L)
                        y[j] = acc * scale
                        j += up
                    }
                }
            }
        }
        return y.map { Float($0) }
    }
}

#endif
