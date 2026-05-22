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
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    static func cgImage(from image: MediaImage) throws -> CGImage {
        guard let provider = CGDataProvider(data: Data(image.rgba8) as CFData) else {
            throw MediaIOError.invalidBufferSize(expected: image.width * image.height * 4, actual: image.rgba8.count)
        }
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
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
