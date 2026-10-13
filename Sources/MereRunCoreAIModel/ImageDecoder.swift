// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
// The CMYK -> RGB conversion is ported from Pillow's formula (MIT-CMU/HPND); see ../../../NOTICE for its notice.
//
// Decodes an image file (or a CGImage) to 8-bit RGB the way the upstream pipeline does (`decode_image` /
// Pillow `convert("RGB")`): the stored samples are used as they are — alpha is DROPPED (not composited, not
// un-premultiplied), EXIF orientation is NOT applied unless asked, no colour management (ICC profiles are
// ignored). Core Graphics drawing would premultiply, composite and colour-match, so the samples are read straight
// from the decoded CGImage buffer and unpacked by hand (RGB/RGBA/gray/gray+alpha/indexed/CMYK, 1-16 bits per
// component); anything else falls back to a Core Graphics draw into an RGB bitmap (documented tolerance).
//
// PNG and WebP decode to the same bytes as Pillow/libpng. JPEG does not: ImageIO's decoder differs from libjpeg
// by 1-3 per sample without chroma subsampling and by up to ~80 at colour edges of 4:2:0 files (different chroma
// upsampling). Embedding effect: cosine >= 0.998 on camera photos, down to ~0.98 on 4:2:0 JPEGs of charts with
// saturated colour edges (see README "Accuracy"). Decode with libjpeg if reference-exact JPEG vectors are needed.

import CoreGraphics
import Foundation
import ImageIO

public struct DecodedImage: Sendable {
    public var width: Int
    public var height: Int
    public var rgb: [UInt8]   // interleaved, row-major, width * height * 3

    public init(width: Int, height: Int, rgb: [UInt8]) {
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}

public enum ImageDecoder {
    /// Decodes the first image of a file. `applyEXIFOrientation` rotates/flips to the display orientation
    /// (the upstream reference does not; see README).
    public static func decodeRGB(url: URL, applyEXIFOrientation: Bool = false) throws -> DecodedImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw EmbedderError(description: "cannot open image \(url.path)")
        }
        guard let cg = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw EmbedderError(description: "cannot decode image \(url.path)")
        }
        var img = try decodeRGB(cgImage: cg)
        if applyEXIFOrientation,
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let o = props[kCGImagePropertyOrientation] as? UInt32, let orientation = CGImagePropertyOrientation(rawValue: o) {
            img = oriented(img, orientation)
        }
        return img
    }

    /// Raw samples of a CGImage (e.g. a screenshot) as RGB, alpha dropped. See the header for the rules.
    public static func decodeRGB(cgImage cg: CGImage) throws -> DecodedImage {
        let w = cg.width, h = cg.height
        let bpc = cg.bitsPerComponent, bpp = cg.bitsPerPixel, bpr = cg.bytesPerRow
        let model = cg.colorSpace?.model
        let byteOrder = cg.bitmapInfo.intersection(.byteOrderMask)
        let alpha = cg.alphaInfo
        let isFloat = cg.bitmapInfo.contains(.floatComponents)
        guard let data = cg.dataProvider?.data, let base = CFDataGetBytePtr(data), !isFloat, let model else {
            return try drawFallback(cg)
        }
        let n = CFDataGetLength(data)
        guard n >= bpr * h else { return try drawFallback(cg) }
        var out = [UInt8](repeating: 0, count: w * h * 3)
        let components: Int
        switch model {
        case .rgb: components = 3
        case .monochrome: components = 1
        case .cmyk: components = 4
        case .indexed: components = 1
        default: return try drawFallback(cg)
        }
        let hasAlpha = ![CGImageAlphaInfo.none, .noneSkipLast, .noneSkipFirst].contains(alpha) || (model == .rgb && bpp == 32 && bpc == 8)
        let alphaFirst = alpha == .first || alpha == .premultipliedFirst || alpha == .noneSkipFirst
        let channelsInPixel = bpp / bpc
        guard channelsInPixel == components || channelsInPixel == components + 1 else { return try drawFallback(cg) }
        let little16 = byteOrder == .byteOrder16Little
        let little32 = byteOrder == .byteOrder32Little

        // Sample accessor for 8/16-bit packed pixels: returns the 8-bit value of channel c of pixel (x, y).
        // Memory layout: big-endian (default) = logical order; 32-bit little-endian reverses the 4 bytes.
        @inline(__always) func sample8(_ row: UnsafePointer<UInt8>, _ x: Int, _ c: Int) -> UInt8 {
            var ch = c + (hasAlpha && alphaFirst ? 1 : 0)
            if bpc == 8 {
                if little32 && channelsInPixel == 4 { ch = 3 - ch }
                return row[x * channelsInPixel + ch]
            }
            // 16 bits per component: high byte (PIL's convert clips >255 to 255; upstream divides by 255 and
            // produces garbage — neither is useful, the high byte is the sensible 8-bit rendering)
            let p = row + (x * channelsInPixel + ch) * 2
            return little16 ? p[1] : p[0]
        }
        if model == .indexed {
            var table = [UInt8](repeating: 0, count: 256 * 3)
            let count = cg.colorSpace?.colorTable?.count ?? 0   // RGB triples
            if let ct = cg.colorSpace?.colorTable, count > 0 {
                for i in 0..<min(count, 256 * 3) { table[i] = ct[i] }
            } else {
                return try drawFallback(cg)
            }
            for y in 0..<h {
                let row = base + y * bpr
                for x in 0..<w {
                    let idx = Int(unpack(row, x, bitsPerComponent: bpc, channel: 0, channels: channelsInPixel))
                    let i = min(idx, 255) * 3
                    out[(y * w + x) * 3] = table[i]
                    out[(y * w + x) * 3 + 1] = table[i + 1]
                    out[(y * w + x) * 3 + 2] = table[i + 2]
                }
            }
            return DecodedImage(width: w, height: h, rgb: out)
        }
        if bpc < 8 {   // 1/2/4-bit gray (bilevel PNG): scale to 0...255 like libpng/PIL ("1" -> 0/255)
            guard model == .monochrome else { return try drawFallback(cg) }
            let maxv = (1 << bpc) - 1
            for y in 0..<h {
                let row = base + y * bpr
                for x in 0..<w {
                    let v = UInt8(Int(unpack(row, x, bitsPerComponent: bpc, channel: 0, channels: channelsInPixel)) * 255 / maxv)
                    out[(y * w + x) * 3] = v; out[(y * w + x) * 3 + 1] = v; out[(y * w + x) * 3 + 2] = v
                }
            }
            return DecodedImage(width: w, height: h, rgb: out)
        }
        guard bpc == 8 || bpc == 16 else { return try drawFallback(cg) }
        for y in 0..<h {
            let row = base + y * bpr
            let o = y * w * 3
            switch model {
            case .rgb:
                for x in 0..<w {
                    out[o + x * 3] = sample8(row, x, 0)
                    out[o + x * 3 + 1] = sample8(row, x, 1)
                    out[o + x * 3 + 2] = sample8(row, x, 2)
                }
            case .monochrome:
                for x in 0..<w {
                    let v = sample8(row, x, 0)
                    out[o + x * 3] = v; out[o + x * 3 + 1] = v; out[o + x * 3 + 2] = v
                }
            case .cmyk:   // Pillow's CMYK -> RGB: nk = 255 - k; c' = nk - c*nk/255 (MULDIV255 rounding). ImageIO's
                          // samples are the inverse of Pillow's ink values (verified on an Adobe CMYK JPEG), so invert.
                for x in 0..<w {
                    let c = 255 - Int(sample8(row, x, 0)), m = 255 - Int(sample8(row, x, 1)), yy = 255 - Int(sample8(row, x, 2)), k = 255 - Int(sample8(row, x, 3))
                    let nk = 255 - k
                    @inline(__always) func mul(_ a: Int) -> Int { let t = a * nk + 128; return ((t >> 8) + t) >> 8 }
                    out[o + x * 3] = UInt8(clamping: nk - mul(c))
                    out[o + x * 3 + 1] = UInt8(clamping: nk - mul(m))
                    out[o + x * 3 + 2] = UInt8(clamping: nk - mul(yy))
                }
            default:
                return try drawFallback(cg)
            }
        }
        return DecodedImage(width: w, height: h, rgb: out)
    }

    @inline(__always) private static func unpack(_ row: UnsafePointer<UInt8>, _ x: Int, bitsPerComponent bpc: Int, channel: Int, channels: Int) -> UInt32 {
        if bpc == 8 { return UInt32(row[x * channels + channel]) }
        if bpc == 16 { return UInt32(row[(x * channels + channel) * 2]) }
        let bit = (x * channels + channel) * bpc
        let byte = row[bit >> 3]
        let shift = 8 - bpc - (bit & 7)
        return UInt32((byte >> UInt8(shift)) & UInt8((1 << bpc) - 1))
    }

    /// Core Graphics draw (colour-managed, alpha composited over black): the fallback for exotic formats.
    private static func drawFallback(_ cg: CGImage) throws -> DecodedImage {
        let w = cg.width, h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let ok = buf.withUnsafeMutableBytes { p -> Bool in
            guard let ctx = CGContext(data: p.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            ctx.interpolationQuality = .none
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw EmbedderError(description: "cannot draw image (\(cg.bitsPerComponent) bpc, \(cg.bitsPerPixel) bpp)") }
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3] = buf[i * 4]; out[i * 3 + 1] = buf[i * 4 + 1]; out[i * 3 + 2] = buf[i * 4 + 2]
        }
        return DecodedImage(width: w, height: h, rgb: out)
    }

    /// Applies an EXIF orientation (1...8) to an RGB buffer (Pillow `ImageOps.exif_transpose`).
    public static func oriented(_ img: DecodedImage, _ o: CGImagePropertyOrientation) -> DecodedImage {
        let w = img.width, h = img.height
        let swap = [CGImagePropertyOrientation.left, .right, .leftMirrored, .rightMirrored].contains(o)
        let ow = swap ? h : w, oh = swap ? w : h
        if o == .up { return img }
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        for y in 0..<h {
            for x in 0..<w {
                let (dx, dy): (Int, Int)
                switch o {
                case .up: (dx, dy) = (x, y)
                case .upMirrored: (dx, dy) = (w - 1 - x, y)
                case .down: (dx, dy) = (w - 1 - x, h - 1 - y)
                case .downMirrored: (dx, dy) = (x, h - 1 - y)
                case .leftMirrored: (dx, dy) = (y, x)                   // transpose
                case .right: (dx, dy) = (h - 1 - y, x)                  // rotate 90 CW to display
                case .rightMirrored: (dx, dy) = (h - 1 - y, w - 1 - x)  // transverse
                case .left: (dx, dy) = (y, w - 1 - x)                   // rotate 90 CCW
                @unknown default: (dx, dy) = (x, y)
                }
                let s = (y * w + x) * 3, d = (dy * ow + dx) * 3
                out[d] = img.rgb[s]; out[d + 1] = img.rgb[s + 1]; out[d + 2] = img.rgb[s + 2]
            }
        }
        return DecodedImage(width: ow, height: oh, rgb: out)
    }
}

#endif
