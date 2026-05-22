import Foundation

public enum MediaImageIO {
    public static func size(of url: URL) throws -> (width: Int, height: Int) {
        #if canImport(CoreGraphics)
        try AppleMediaImageIO.size(of: url)
        #else
        try FFmpegMediaIO.imageSize(of: url)
        #endif
    }

    public static func decode(_ url: URL) throws -> MediaImage {
        #if canImport(CoreGraphics)
        try AppleMediaImageIO.decode(url)
        #else
        try FFmpegMediaIO.decodeImage(url)
        #endif
    }

    public static func decode(data: Data) throws -> MediaImage {
        #if canImport(CoreGraphics)
        try AppleMediaImageIO.decode(data: data)
        #else
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-image-\(UUID().uuidString)")
            .appendingPathExtension("img")
        try data.write(to: temp)
        defer { try? FileManager.default.removeItem(at: temp) }
        return try FFmpegMediaIO.decodeImage(temp)
        #endif
    }

    public static func writePNG(_ image: MediaImage, to url: URL) throws {
        #if canImport(CoreGraphics)
        try AppleMediaImageIO.writePNG(image, to: url)
        #else
        try FFmpegMediaIO.writePNG(image, to: url)
        #endif
    }

    public static func pngData(from image: MediaImage) throws -> Data {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-image-\(UUID().uuidString)")
            .appendingPathExtension("png")
        defer { try? FileManager.default.removeItem(at: temp) }
        try writePNG(image, to: temp)
        return try Data(contentsOf: temp)
    }

    public static func resized(_ image: MediaImage, width: Int, height: Int) throws -> MediaImage {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        if image.width == width, image.height == height {
            return image
        }

        var output = [UInt8](repeating: 255, count: width * height * 4)
        let xScale = Double(image.width) / Double(width)
        let yScale = Double(image.height) / Double(height)

        for y in 0..<height {
            let srcY = min(image.height - 1, Int((Double(y) + 0.5) * yScale))
            for x in 0..<width {
                let srcX = min(image.width - 1, Int((Double(x) + 0.5) * xScale))
                let src = ((srcY * image.width) + srcX) * 4
                let dst = ((y * width) + x) * 4
                output[dst] = image.rgba8[src]
                output[dst + 1] = image.rgba8[src + 1]
                output[dst + 2] = image.rgba8[src + 2]
                output[dst + 3] = image.rgba8[src + 3]
            }
        }
        return try MediaImage(width: width, height: height, rgba8: output)
    }

    public static func centerCropped(_ image: MediaImage, width: Int, height: Int) throws -> MediaImage {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        let scale = max(Double(width) / Double(image.width), Double(height) / Double(image.height))
        let scaledWidth = max(width, Int((Double(image.width) * scale).rounded()))
        let scaledHeight = max(height, Int((Double(image.height) * scale).rounded()))
        let resized = try resized(image, width: scaledWidth, height: scaledHeight)
        let originX = max(0, (scaledWidth - width) / 2)
        let originY = max(0, (scaledHeight - height) / 2)

        var output = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            let srcY = y + originY
            for x in 0..<width {
                let srcX = x + originX
                let src = ((srcY * scaledWidth) + srcX) * 4
                let dst = ((y * width) + x) * 4
                output[dst] = resized.rgba8[src]
                output[dst + 1] = resized.rgba8[src + 1]
                output[dst + 2] = resized.rgba8[src + 2]
                output[dst + 3] = resized.rgba8[src + 3]
            }
        }
        return try MediaImage(width: width, height: height, rgba8: output)
    }

    public static func rgbCHWFloat(_ image: MediaImage, normalizedToMinusOneToOne: Bool) -> [Float] {
        let pixelCount = image.width * image.height
        var values = [Float](repeating: 0, count: pixelCount * 3)
        for pixel in 0..<pixelCount {
            let src = pixel * 4
            let r = Float(image.rgba8[src]) / 255.0
            let g = Float(image.rgba8[src + 1]) / 255.0
            let b = Float(image.rgba8[src + 2]) / 255.0
            if normalizedToMinusOneToOne {
                values[pixel] = (r * 2.0) - 1.0
                values[pixel + pixelCount] = (g * 2.0) - 1.0
                values[pixel + (pixelCount * 2)] = (b * 2.0) - 1.0
            } else {
                values[pixel] = r
                values[pixel + pixelCount] = g
                values[pixel + (pixelCount * 2)] = b
            }
        }
        return values
    }

    public static func imageFromRGBCHW(
        _ bytes: [UInt8],
        width: Int,
        height: Int
    ) throws -> MediaImage {
        let pixelCount = width * height
        guard bytes.count == pixelCount * 3 else {
            throw MediaIOError.invalidBufferSize(expected: pixelCount * 3, actual: bytes.count)
        }
        var rgba = [UInt8](repeating: 255, count: pixelCount * 4)
        for pixel in 0..<pixelCount {
            let dst = pixel * 4
            rgba[dst] = bytes[pixel]
            rgba[dst + 1] = bytes[pixel + pixelCount]
            rgba[dst + 2] = bytes[pixel + (pixelCount * 2)]
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    public static func imageFromRGBHWC(
        _ bytes: [UInt8],
        width: Int,
        height: Int
    ) throws -> MediaImage {
        let pixelCount = width * height
        guard bytes.count == pixelCount * 3 else {
            throw MediaIOError.invalidBufferSize(expected: pixelCount * 3, actual: bytes.count)
        }
        var rgba = [UInt8](repeating: 255, count: pixelCount * 4)
        for pixel in 0..<pixelCount {
            let src = pixel * 3
            let dst = pixel * 4
            rgba[dst] = bytes[src]
            rgba[dst + 1] = bytes[src + 1]
            rgba[dst + 2] = bytes[src + 2]
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }
}
