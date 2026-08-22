import Foundation
import MediaIO
import MLX

enum SenseNovaU15ImageIO {
    struct PreparedImage {
        let pixels: MLXArray
        let gridHeight: Int
        let gridWidth: Int
    }

    static func prepareReference(
        _ url: URL,
        patchSize: Int,
        downsampleRatio: Float,
        imageCount: Int
    ) throws -> PreparedImage {
        let image = try whiteComposited(try MediaImageIO.decode(url))
        let factor = Int(Float(patchSize) / downsampleRatio)
        let maximum = min(2_048 * 2_048, (4_096 * 4_096) / max(1, imageCount))
        let resized = smartResize(
            width: image.width,
            height: image.height,
            factor: factor,
            minimumPixels: 512 * 512,
            maximumPixels: maximum,
            upscale: false
        )
        let scaled = try MediaImageIO.resized(image, width: resized.width, height: resized.height)
        let chw = MLXArray(
            MediaImageIO.rgbCHWFloat(scaled, normalizedToMinusOneToOne: false),
            [1, 3, resized.height, resized.width]
        ).asType(.bfloat16)
        var pixels = chw.transposed(0, 2, 3, 1)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).asType(.bfloat16)
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225]).asType(.bfloat16)
        pixels = (pixels - mean) / standardDeviation
        return PreparedImage(
            pixels: pixels,
            gridHeight: resized.height / patchSize,
            gridWidth: resized.width / patchSize
        )
    }

    static func save(_ pixelsNHWC: MLXArray, to url: URL) throws {
        let chw = MLX.clip((pixelsNHWC + 1) / 2, min: 0, max: 1)
            .transposed(0, 3, 1, 2)[0, 0..., 0..., 0...]
        try QwenImageIO.saveImage(array: chw, to: url)
    }

    private static func whiteComposited(_ image: MediaImage) throws -> MediaImage {
        guard image.rgba8.indices.contains(3),
              stride(from: 3, to: image.rgba8.count, by: 4).contains(where: { image.rgba8[$0] < 255 }) else {
            return image
        }
        var rgba = image.rgba8
        for pixel in 0..<(image.width * image.height) {
            let offset = pixel * 4
            let alpha = Int(rgba[offset + 3])
            for channel in 0..<3 {
                rgba[offset + channel] = UInt8((Int(rgba[offset + channel]) * alpha + 255 * (255 - alpha)) / 255)
            }
            rgba[offset + 3] = 255
        }
        return try MediaImage(width: image.width, height: image.height, rgba8: rgba)
    }

    static func smartResize(
        width: Int,
        height: Int,
        factor: Int,
        minimumPixels: Int,
        maximumPixels: Int,
        upscale _: Bool
    ) -> (width: Int, height: Int) {
        var targetHeight = max(factor, Int((Double(height) / Double(factor)).rounded()) * factor)
        var targetWidth = max(factor, Int((Double(width) / Double(factor)).rounded()) * factor)
        if targetHeight * targetWidth > maximumPixels {
            let beta = sqrt(Double(height * width) / Double(maximumPixels))
            targetHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            targetWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if targetHeight * targetWidth < minimumPixels {
            let beta = sqrt(Double(minimumPixels) / Double(height * width))
            targetHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            targetWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }
        return (targetWidth, targetHeight)
    }
}
