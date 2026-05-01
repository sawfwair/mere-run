import Foundation
import CoreGraphics
import ImageIO
import MLX

enum QwenVLImageLoader {
    static func loadCGImage(url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    /// Default pixel constraints - nil means no constraint (matches mlx-community model configs)
    static let defaultMinPixels: Int? = nil
    static let defaultMaxPixels: Int? = nil

    static func pixelValues(
        cgImage: CGImage,
        minPixels: Int? = defaultMinPixels,
        maxPixels: Int? = defaultMaxPixels,
        patchSize: Int,
        spatialMergeSize: Int = 2
    ) throws -> MLXArray {
        // Compute target size, applying min/max pixel constraints if specified
        let (targetW, targetH) = Self.computeTargetSize(
            originalWidth: cgImage.width,
            originalHeight: cgImage.height,
            minPixels: minPixels,
            maxPixels: maxPixels,
            patchSize: patchSize,
            mergeSize: spatialMergeSize
        )

        let finalImage = Self.resizeExact(cgImage: cgImage, width: targetW, height: targetH)
        let w = finalImage.width
        let h = finalImage.height

        guard let data = Self.rgb8Pixels(cgImage: finalImage) else {
            throw NSError(domain: "QwenVLImageLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to read pixels"])
        }

        // Convert to Float32 [1, 3, H, W] with Qwen3-VL normalization
        // Qwen3-VL uses simple rescaling to [-1, 1]: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1
        var floats = [Float32](repeating: 0, count: 3 * w * h)
        floats.withUnsafeMutableBufferPointer { dst in
            data.withUnsafeBytes { src in
                let bytes = src.bindMemory(to: UInt8.self)
                for y in 0..<h {
                    for x in 0..<w {
                        let i = (y * w + x)
                        // Rescale to [-1, 1]
                        let r = Float32(bytes[i * 3 + 0]) / 127.5 - 1.0
                        let g = Float32(bytes[i * 3 + 1]) / 127.5 - 1.0
                        let b = Float32(bytes[i * 3 + 2]) / 127.5 - 1.0
                        dst[0 * w * h + i] = r
                        dst[1 * w * h + i] = g
                        dst[2 * w * h + i] = b
                    }
                }
            }
        }

        return MLXArray(floats, [1, 3, h, w])
    }

    private static func saveDebugImage(floats: [Float32], width: Int, height: Int, to url: URL) {
        // Denormalize and save as PNG for visual inspection
        // Reverse the [-1, 1] rescaling: pixel = (normalized + 1) * 127.5
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                let r = (floats[0 * width * height + i] + 1.0) * 127.5
                let g = (floats[1 * width * height + i] + 1.0) * 127.5
                let b = (floats[2 * width * height + i] + 1.0) * 127.5
                pixels[i * 4 + 0] = UInt8(max(0, min(255, r)))
                pixels[i * 4 + 1] = UInt8(max(0, min(255, g)))
                pixels[i * 4 + 2] = UInt8(max(0, min(255, b)))
                pixels[i * 4 + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        pixels.withUnsafeBytes { ptr in
            guard let ctx = CGContext(
                data: UnsafeMutableRawPointer(mutating: ptr.baseAddress),
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ), let image = ctx.makeImage() else { return }

            let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
            CGImageDestinationAddImage(dest, image, nil)
            CGImageDestinationFinalize(dest)
            print("[QwenVLImageLoader] Debug image saved to: \(url.path)")
        }
    }

    private static func rgb8Pixels(cgImage: CGImage) -> Data? {
        let w = cgImage.width
        let h = cgImage.height

        // Use RGBA format (4 bytes per pixel) for better compatibility
        let bytesPerRow = w * 4
        var data = Data(count: bytesPerRow * h)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        data.withUnsafeMutableBytes { ptr in
            guard let ctx = CGContext(
                data: ptr.baseAddress,
                width: w,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            ) else { return }

            ctx.interpolationQuality = .high
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        }

        // Convert RGBA to RGB
        var rgb = Data(count: w * h * 3)
        rgb.withUnsafeMutableBytes { dstPtr in
            data.withUnsafeBytes { srcPtr in
                let dst = dstPtr.bindMemory(to: UInt8.self)
                let src = srcPtr.bindMemory(to: UInt8.self)
                for i in 0..<(w * h) {
                    dst[i * 3 + 0] = src[i * 4 + 0]  // R
                    dst[i * 3 + 1] = src[i * 4 + 1]  // G
                    dst[i * 3 + 2] = src[i * 4 + 2]  // B
                }
            }
        }
        return rgb
    }

    /// Computes target dimensions matching Qwen2VLImageProcessor's smart_resize
    /// Grid dimensions must be divisible by merge_size, visual dimensions divisible by patch_size
    /// If minPixels/maxPixels are nil, no pixel count constraints are applied
    private static func computeTargetSize(
        originalWidth w: Int,
        originalHeight h: Int,
        minPixels: Int?,
        maxPixels: Int?,
        patchSize: Int,
        mergeSize: Int
    ) -> (width: Int, height: Int) {
        // Match Hugging Face's Qwen2/3-VL smart_resize exactly by rounding on the
        // combined patch*merge factor instead of approximating with patch alignment first.
        let factor = max(1, patchSize * mergeSize)

        func roundToFactor(_ value: Int, factor: Int) -> Int {
            max(factor, Int((Double(value) / Double(factor)).rounded()) * factor)
        }

        var targetH = roundToFactor(h, factor: factor)
        var targetW = roundToFactor(w, factor: factor)

        if let maxPixels, targetH * targetW > maxPixels {
            let beta = sqrt(Double(h * w) / Double(maxPixels))
            targetH = max(factor, Int(floor(Double(h) / beta / Double(factor))) * factor)
            targetW = max(factor, Int(floor(Double(w) / beta / Double(factor))) * factor)
        } else if let minPixels, targetH * targetW < minPixels {
            let beta = sqrt(Double(minPixels) / Double(h * w))
            targetH = Int(ceil(Double(h) * beta / Double(factor))) * factor
            targetW = Int(ceil(Double(w) * beta / Double(factor))) * factor
        }

        return (targetW, targetH)
    }

    private static func resizeExact(cgImage: CGImage, width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        // Use noneSkipLast to avoid alpha premultiplication issues
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return cgImage
        }

        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage() ?? cgImage
    }
}
