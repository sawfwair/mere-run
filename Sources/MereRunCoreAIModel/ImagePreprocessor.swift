// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// Ports the logic of the Gemma 4 image processor (image_processing_gemma4.py in Hugging Face transformers 5.19,
// Copyright 2026 The HuggingFace Team, Apache-2.0); see ../../../NOTICE.
//
// Host-side preparation of the `vision_2520` inputs: target size (aspect-preserving, multiples of 48 px, at most
// 2520 patches of 16x16), the exact resize, 16x16x3 patches of INTEGER pixel values 0..255 (exact in Float16; the
// graph computes P - 127.5 with upstream's 2/255 scale folded into its first projection) in CELL-MAJOR order (the
// 9 patches of each 3x3 pooling cell contiguous, cells row-major), (x, y) patch positions, the axial-RoPE cos/sin
// tables and the patch mask. Mirrors python/eg2_image.py.

import Foundation

public struct VisionInputs: Sendable {
    public static let patch = 16
    public static let pool = 3
    public static let cellPx = 48
    public static let maxSoftTokens = 280
    public static let maxPatches = 2520
    public static let patchDim = 768          // 16 * 16 * 3
    public static let headDim = 64
    public static let ropeTheta = 100.0

    public var patches: [Float16]      // [2520 * 768], pixel values 0..255, zero padded
    public var posX: [Int32]           // [2520], 0 for padding
    public var posY: [Int32]           // [2520]
    public var cos: [Float16]          // [2520 * 64]
    public var sin: [Float16]          // [2520 * 64]
    public var patchMask: [Float16]    // [2520], 1 = patch, 0 = padding
    public var softTokenCount: Int     // n: valid rows of the vision output
    public var gridRows: Int           // patch rows
    public var gridCols: Int           // patch cols
    public var resizedHeight: Int
    public var resizedWidth: Int
}

public enum ImagePreprocessor {
    /// Gemma4 `get_aspect_ratio_preserving_size` (height, width), verbatim logic.
    public static func targetSize(height: Int, width: Int, maxPatches: Int = VisionInputs.maxPatches) throws -> (height: Int, width: Int) {
        let patch = VisionInputs.patch, cell = VisionInputs.cellPx
        let targetPx = Double(maxPatches * patch * patch)
        let factor = (targetPx / Double(height * width)).squareRoot()
        var th = Int((factor * Double(height) / Double(cell)).rounded(.down)) * cell
        var tw = Int((factor * Double(width) / Double(cell)).rounded(.down)) * cell
        let maxSide = (maxPatches / (VisionInputs.pool * VisionInputs.pool)) * cell
        if th == 0 && tw == 0 { throw EmbedderError(description: "image would resize to 0x0") }
        if th == 0 {
            th = cell
            tw = min(Int((Double(width) / Double(height)).rounded(.down)) * cell, maxSide)
        } else if tw == 0 {
            tw = cell
            th = min(Int((Double(height) / Double(width)).rounded(.down)) * cell, maxSide)
        }
        if Double(th * tw) > targetPx { throw EmbedderError(description: "size exceeds the patch budget") }
        return (th, tw)
    }

    /// Resized 8-bit RGB (interleaved, [height][width][3]) of a decoded image, exactly as the upstream processor
    /// produces it (bicubic antialiased on uint8, width pass then height pass).
    public static func resized(_ image: DecodedImage) throws -> DecodedImage {
        let (th, tw) = try targetSize(height: image.height, width: image.width)
        let px = ImageResampler.resize(image.rgb, width: image.width, height: image.height, channels: 3, toWidth: tw, toHeight: th)
        return DecodedImage(width: tw, height: th, rgb: px)
    }

    /// Decode (no EXIF orientation unless asked, alpha dropped, no colour management) + resize + patches.
    public static func prepare(url: URL, applyEXIFOrientation: Bool = false) throws -> VisionInputs {
        try prepare(try ImageDecoder.decodeRGB(url: url, applyEXIFOrientation: applyEXIFOrientation))
    }

    public static func prepare(_ decoded: DecodedImage) throws -> VisionInputs {
        let r = try resized(decoded)
        return patchify(resizedRGB: r.rgb, width: r.width, height: r.height)
    }

    /// `rgb`: [height][width][3] uint8 with height, width multiples of 48.
    public static func patchify(resizedRGB rgb: [UInt8], width W: Int, height H: Int) -> VisionInputs {
        let P = VisionInputs.patch, K = VisionInputs.pool, N = VisionInputs.maxPatches, D = VisionInputs.patchDim
        let ph = H / P, pw = W / P
        let kh = ph / K, kw = pw / K
        let n = kh * kw
        var patches = [Float16](repeating: 0, count: N * D)
        var posX = [Int32](repeating: 0, count: N)
        var posY = [Int32](repeating: 0, count: N)
        var mask = [Float16](repeating: 0, count: N)
        var idx = 0
        rgb.withUnsafeBufferPointer { sp in
            patches.withUnsafeMutableBufferPointer { pp in
                for ky in 0..<kh {
                    for kx in 0..<kw {
                        for dy in 0..<K {
                            for dx in 0..<K {
                                let x = kx * K + dx, y = ky * K + dy
                                posX[idx] = Int32(x)
                                posY[idx] = Int32(y)
                                mask[idx] = 1
                                let base = idx * D
                                var o = 0
                                for py in 0..<P {
                                    let row = sp.baseAddress! + ((y * P + py) * W + x * P) * 3
                                    for e in 0..<(P * 3) {
                                        pp[base + o] = Float16(row[e])   // 0...255, exact
                                        o += 1
                                    }
                                }
                                idx += 1
                            }
                        }
                    }
                }
            }
        }
        let (cos, sin) = ropeTables(posX: posX, posY: posY)
        return VisionInputs(patches: patches, posX: posX, posY: posY, cos: cos, sin: sin, patchMask: mask,
                            softTokenCount: n, gridRows: ph, gridCols: pw, resizedHeight: H, resizedWidth: W)
    }

    /// Axial RoPE tables [N, 64]: channels 0..31 rotate by x (16 frequencies, duplicated), 32..63 by y.
    /// float32 maths like upstream, cast to fp16.
    public static func ropeTables(posX: [Int32], posY: [Int32]) -> (cos: [Float16], sin: [Float16]) {
        let F = VisionInputs.headDim / 4   // 16
        var inv = [Float](repeating: 0, count: F)
        for i in 0..<F { inv[i] = Float(1.0 / pow(VisionInputs.ropeTheta, Double(2 * i) / Double(2 * F))) }
        let N = posX.count
        var cos = [Float16](repeating: 0, count: N * 64)
        var sin = [Float16](repeating: 0, count: N * 64)
        for p in 0..<N {
            let fx = Float(posX[p]), fy = Float(posY[p])
            for i in 0..<F {
                let ax = fx * inv[i], ay = fy * inv[i]
                let cx = Float16(cosf(ax)), sx = Float16(sinf(ax)), cy = Float16(cosf(ay)), sy = Float16(sinf(ay))
                let b = p * 64
                cos[b + i] = cx; cos[b + F + i] = cx; cos[b + 2 * F + i] = cy; cos[b + 3 * F + i] = cy
                sin[b + i] = sx; sin[b + F + i] = sx; sin[b + 2 * F + i] = sy; sin[b + 3 * F + i] = sy
            }
        }
        return (cos, sin)
    }
}

#endif
