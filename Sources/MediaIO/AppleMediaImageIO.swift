import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum AppleMediaImageIO {
    static func size(of url: URL) throws -> (width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw MediaIOError.imageMetadataFailed(url)
        }
        return (width, height)
    }

    static func decode(_ url: URL) throws -> MediaImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MediaIOError.imageDecodeFailed(url)
        }
        return try mediaImage(from: image)
    }

    static func decode(data: Data) throws -> MediaImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MediaIOError.unsupportedPlatform("Failed to decode in-memory image data.")
        }
        return try mediaImage(from: image)
    }

    static func writePNG(_ image: MediaImage, to url: URL) throws {
        let cgImage = try cgImage(from: image)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw MediaIOError.imageEncodeFailed(url)
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw MediaIOError.imageEncodeFailed(url)
        }
    }

    static func mediaImage(from image: CGImage) throws -> MediaImage {
        let width = image.width
        let height = image.height
        if let rgba = straightRGBA8Bytes(of: image) {
            return try MediaImage(width: width, height: height, rgba8: rgba)
        }

        // CGContext cannot target straight alpha, so convert other formats by
        // drawing premultiplied and un-premultiplying afterwards. Lossless for
        // opaque pixels; semi-transparent RGB rounds through premultiplication.
        let bytesPerRow = width * 4
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        let succeeded = rgba.withUnsafeMutableBytes { ptr -> Bool in
            guard let base = ptr.baseAddress,
                  let context = CGContext(
                      data: base,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: bytesPerRow,
                      space: colorSpace,
                      bitmapInfo: bitmapInfo
                  ) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard succeeded else {
            throw MediaIOError.imageDecodeFailed(URL(fileURLWithPath: "<memory>"))
        }
        unpremultiplyInPlace(&rgba)
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    /// Copies samples directly when the image already uses MediaImage's 8-bit
    /// RGBA layout. ImageIO decodes RGBA PNGs this way, and the direct copy is
    /// the only path that keeps straight alpha byte-exact — the premultiplied
    /// draw fallback rounds RGB wherever alpha < 255 and zeroes it at alpha 0.
    /// Like the FFmpeg backend, this performs no color management.
    private static func straightRGBA8Bytes(of image: CGImage) -> [UInt8]? {
        let alphaInfo = image.alphaInfo
        guard alphaInfo == .last || alphaInfo == .noneSkipLast,
              image.bitsPerComponent == 8,
              image.bitsPerPixel == 32,
              image.colorSpace?.model == .rgb else {
            return nil
        }
        let byteOrder = image.bitmapInfo.intersection(.byteOrderMask)
        guard byteOrder == CGBitmapInfo() || byteOrder == .byteOrder32Big,
              let data = image.dataProvider?.data as Data? else {
            return nil
        }
        let width = image.width
        let height = image.height
        let sourceBytesPerRow = image.bytesPerRow
        let rowBytes = width * 4
        guard sourceBytesPerRow >= rowBytes,
              data.count >= (height - 1) * sourceBytesPerRow + rowBytes else {
            return nil
        }
        var rgba = [UInt8]()
        rgba.reserveCapacity(height * rowBytes)
        for row in 0..<height {
            let start = data.startIndex + row * sourceBytesPerRow
            rgba.append(contentsOf: data[start..<(start + rowBytes)])
        }
        if alphaInfo == .noneSkipLast {
            for pixel in stride(from: 3, to: rgba.count, by: 4) {
                rgba[pixel] = 255
            }
        }
        return rgba
    }

    /// Converts premultiplied RGBA to straight alpha in place, rounding to the
    /// nearest 8-bit value. Fully transparent pixels stay transparent black:
    /// premultiplication already discarded their RGB.
    private static func unpremultiplyInPlace(_ rgba: inout [UInt8]) {
        for pixelStart in stride(from: 0, to: rgba.count, by: 4) {
            let alpha = Int(rgba[pixelStart + 3])
            guard alpha > 0, alpha < 255 else { continue }
            for channel in pixelStart..<(pixelStart + 3) {
                let value = ((Int(rgba[channel]) * 255) + (alpha / 2)) / alpha
                rgba[channel] = UInt8(min(255, value))
            }
        }
    }

    static func cgImage(from image: MediaImage) throws -> CGImage {
        guard let provider = CGDataProvider(data: Data(image.rgba8) as CFData) else {
            throw MediaIOError.invalidBufferSize(expected: image.width * image.height * 4, actual: image.rgba8.count)
        }
        // MediaImage.rgba8 is straight alpha. Declaring it premultiplied makes
        // ImageIO un-premultiply on PNG encode, brightening RGB wherever
        // alpha < 255; CGImage (unlike CGContext) supports straight alpha.
        let bitmapInfo = CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let cgImage = CGImage(
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: image.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw MediaIOError.invalidBufferSize(expected: image.width * image.height * 4, actual: image.rgba8.count)
        }
        return cgImage
    }
}
#endif
