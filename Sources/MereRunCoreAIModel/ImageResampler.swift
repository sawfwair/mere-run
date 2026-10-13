// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// Ported from the logic of ATen's uint8 `_upsample_bicubic2d_aa` kernel (PyTorch, aten/src/ATen/native/cpu/
// UpSampleKernel.cpp, BSD-3-Clause; itself modelled on Pillow's Resample.c, MIT-CMU/HPND). See ../../../NOTICE for
// the PyTorch and Pillow license notices.
//
// Antialiased bicubic resize on 8-bit pixels, bit-identical to the one the upstream image processor uses
// (torchvision `resize(..., BICUBIC, antialias=True)` on a uint8 tensor = ATen's `_upsample_bicubic2d_aa` uint8
// kernel): separable, WIDTH pass first then HEIGHT, taps computed in float64 and rounded to int16 with a dynamic
// precision, each pass accumulated in int32, rounded (+half, arithmetic shift) and clamped to 0...255.
// Lanczos or bilinear resampling (what Core Image / vImage offer) changes the embedding (cosine down to 0.86 for
// Lanczos, 0.96 for bilinear on the validation set), hence this port.
// Verified against torchvision 0.29.1 pixel for pixel (see tests/preprocess-cases.json and `eg2-cli pixcheck`).

import Foundation

public enum ImageResampler {
    /// Taps of one axis: for output index i the input window starts at `start[i]` and has `count[i]` taps whose
    /// int16 weights (as Int32) are `weights[i * taps ..< i * taps + count[i]]`; `precision` is the fixed-point shift.
    public struct Taps: Sendable {
        public let start: [Int32]
        public let count: [Int32]
        public let weights: [Int32]
        public let taps: Int
        public let precision: Int
    }

    @inline(__always) private static func cubic(_ v: Double) -> Double {
        let a = -0.5
        let x = abs(v)
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 }
        if x < 2.0 { return (((x - 5.0) * x + 8.0) * x - 4.0) * a }
        return 0.0
    }

    /// ATen `HelperInterpBase::_compute_indices_min_size_weights_aa` + `_compute_index_ranges_int16_weights`.
    public static func taps(inSize: Int, outSize: Int) -> Taps {
        let scale = Double(inSize) / Double(outSize)
        let support = scale >= 1.0 ? 2.0 * scale : 2.0
        let invscale = scale >= 1.0 ? 1.0 / scale : 1.0
        let taps = Int(ceil(support)) * 2 + 1
        var w = [Double](repeating: 0, count: outSize * taps)
        var start = [Int32](repeating: 0, count: outSize)
        var count = [Int32](repeating: 0, count: outSize)
        var wtMax = 0.0
        for i in 0..<outSize {
            let center = scale * (Double(i) + 0.5)
            let lo = max(Int(center - support + 0.5), 0)          // (int64_t) cast: truncation toward zero
            let size = min(Int(center + support + 0.5), inSize) - lo
            var total = 0.0
            for j in 0..<size {
                let v = cubic((Double(j + lo) - center + 0.5) * invscale)
                w[i * taps + j] = v
                total += v
            }
            if total != 0.0 {
                for j in 0..<size { w[i * taps + j] /= total }
            }
            for j in 0..<size { wtMax = max(wtMax, w[i * taps + j]) }
            start[i] = Int32(lo)
            count[i] = Int32(size)
        }
        var precision = 0
        while precision < 22 {
            if Int(0.5 + wtMax * Double(1 << (precision + 1))) >= (1 << 15) { break }
            precision += 1
        }
        let s = Double(1 << precision)
        let wi = w.map { v -> Int32 in v >= 0 ? Int32((v * s + 0.5).rounded(.down)) : Int32((v * s - 0.5).rounded(.up)) }
        return Taps(start: start, count: count, weights: wi, taps: taps, precision: precision)
    }

    /// Interleaved 8-bit image [height][width][channels] -> [outHeight][outWidth][channels].
    public static func resize(_ src: [UInt8], width: Int, height: Int, channels: Int, toWidth outW: Int, toHeight outH: Int) -> [UInt8] {
        if width == outW && height == outH { return src }
        var cur = src
        var curW = width
        if width != outW {
            cur = passWidth(cur, width: width, height: height, channels: channels, taps: taps(inSize: width, outSize: outW), outW: outW)
            curW = outW
        }
        if height != outH {
            cur = passHeight(cur, width: curW, height: height, channels: channels, taps: taps(inSize: height, outSize: outH), outH: outH)
        }
        return cur
    }

    private static func passWidth(_ src: [UInt8], width: Int, height: Int, channels: Int, taps t: Taps, outW: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: outW * height * channels)
        let half = Int32(1 << (t.precision - 1))
        let shift = Int32(t.precision)
        src.withUnsafeBufferPointer { sp in
            out.withUnsafeMutableBufferPointer { op in
                t.weights.withUnsafeBufferPointer { wp in
                    guard let s0 = sp.baseAddress, let o0 = op.baseAddress, let w0 = wp.baseAddress else { return }
                    // Each iteration writes its own output row only; the pointers outlive concurrentPerform.
                    let sb = SharedPointer(s0), ob = SharedMutablePointer(o0), wb = SharedPointer(w0)
                    DispatchQueue.concurrentPerform(iterations: height) { y in
                        let row = sb.p + y * width * channels
                        let orow = ob.p + y * outW * channels
                        for i in 0..<outW {
                            let lo = Int(t.start[i]), n = Int(t.count[i])
                            let wbase = wb.p + i * t.taps
                            for c in 0..<channels {
                                var acc = half
                                var p = row + lo * channels + c
                                for j in 0..<n {
                                    acc &+= Int32(p.pointee) &* wbase[j]
                                    p += channels
                                }
                                orow[i * channels + c] = UInt8(clamping: acc >> shift)
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    private static func passHeight(_ src: [UInt8], width: Int, height: Int, channels: Int, taps t: Taps, outH: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * outH * channels)
        let half = Int32(1 << (t.precision - 1))
        let shift = Int32(t.precision)
        let rowLen = width * channels
        src.withUnsafeBufferPointer { sp in
            out.withUnsafeMutableBufferPointer { op in
                t.weights.withUnsafeBufferPointer { wp in
                    guard let s0 = sp.baseAddress, let o0 = op.baseAddress, let w0 = wp.baseAddress else { return }
                    // Each iteration writes its own output row only; the pointers outlive concurrentPerform.
                    let sb = SharedPointer(s0), ob = SharedMutablePointer(o0), wb = SharedPointer(w0)
                    DispatchQueue.concurrentPerform(iterations: outH) { i in
                        let lo = Int(t.start[i]), n = Int(t.count[i])
                        let wbase = wb.p + i * t.taps
                        let orow = ob.p + i * rowLen
                        for x in 0..<rowLen {
                            var acc = half
                            var p = sb.p + lo * rowLen + x
                            for j in 0..<n {
                                acc &+= Int32(p.pointee) &* wbase[j]
                                p += rowLen
                            }
                            orow[x] = UInt8(clamping: acc >> shift)
                        }
                    }
                }
            }
        }
        return out
    }
}

/// Raw pointers handed to `DispatchQueue.concurrentPerform`, whose iterations read shared input and write disjoint
/// output rows; the buffers stay alive for the whole call.
private struct SharedPointer<T>: @unchecked Sendable {
    let p: UnsafePointer<T>
    init(_ p: UnsafePointer<T>) { self.p = p }
}

private struct SharedMutablePointer<T>: @unchecked Sendable {
    let p: UnsafeMutablePointer<T>
    init(_ p: UnsafeMutablePointer<T>) { self.p = p }
}

#endif
