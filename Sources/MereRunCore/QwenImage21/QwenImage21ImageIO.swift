import Foundation
import MediaIO
import MLX

enum QwenImage21ImageIO {
    struct Reference {
        let rgba: MLXArray
        let vision: MLXArray
        let width: Int
        let height: Int
    }

    static func prepare(_ url: URL) throws -> Reference {
        let source = try MediaImageIO.decode(url)
        // The pinned pipeline uses a 1024^2 reference area independently of output dimensions.
        let ratio = Double(source.width) / Double(source.height)
        let width = max(32, Int((sqrt(1_048_576 * ratio) / 32).rounded(.toNearestOrEven)) * 32)
        let height = max(32, Int((sqrt(1_048_576 / ratio) / 32).rounded(.toNearestOrEven)) * 32)
        let image = try resized(source, width: width, height: height)
        let rgba = MLXArray(image.rgba8.map { Float($0) / 127.5 - 1 }, [1, height, width, 4]).asType(.bfloat16)
        var rgb = [Float](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            let alpha = Int(image.rgba8[pixel * 4 + 3])
            for channel in 0..<3 {
                let color = Int(image.rgba8[pixel * 4 + channel])
                let composed = (color * alpha + 255 * (255 - alpha) + 127) / 255
                rgb[channel * width * height + pixel] = Float(composed) / 127.5 - 1
            }
        }
        return Reference(rgba: rgba, vision: MLXArray(rgb, [1, 3, height, width]), width: width, height: height)
    }

    static func save(_ rgba: MLXArray, to url: URL) throws {
        let values = clip(round(rgba.asType(.float32) * 255), min: 0, max: 255).asType(.uint8)
        eval(values)
        let image = try MediaImage(width: rgba.dim(2), height: rgba.dim(1), rgba8: values.asArray(UInt8.self))
        try MediaImageIO.writePNG(image, to: url)
    }

    /// Pillow's RGBA Lanczos path resizes premultiplied bytes and converts back to straight alpha.
    static func resized(_ image: MediaImage, width: Int, height: Int) throws -> MediaImage {
        if image.width == width, image.height == height { return image }
        var bytes = image.rgba8
        for offset in stride(from: 0, to: bytes.count, by: 4) {
            let alpha = Int(bytes[offset + 3])
            for channel in 0..<3 { bytes[offset + channel] = UInt8((Int(bytes[offset + channel]) * alpha + 127) / 255) }
        }
        let horizontal = contributions(source: image.width, target: width)
        let vertical = contributions(source: image.height, target: height)
        var intermediate = [UInt8](repeating: 0, count: width * image.height * 4)
        for y in 0..<image.height {
            for x in 0..<width {
                for channel in 0..<4 {
                    var value: Int64 = 1 << 21
                    for (source, weight) in horizontal[x] { value += Int64(bytes[(y * image.width + source) * 4 + channel]) * weight }
                    intermediate[(y * width + x) * 4 + channel] = UInt8(clamping: value >> 22)
                }
            }
        }
        var output = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                for channel in 0..<4 {
                    var value: Int64 = 1 << 21
                    for (source, weight) in vertical[y] { value += Int64(intermediate[(source * width + x) * 4 + channel]) * weight }
                    output[(y * width + x) * 4 + channel] = UInt8(clamping: value >> 22)
                }
                let offset = (y * width + x) * 4, alpha = Int(output[(y * width + x) * 4 + 3])
                if alpha > 0, alpha < 255 {
                    for channel in 0..<3 { output[offset + channel] = UInt8(clamping: Int(output[offset + channel]) * 255 / alpha) }
                }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: output)
    }

    private static func contributions(source: Int, target: Int) -> [[(Int, Int64)]] {
        let scale = Double(source) / Double(target), filterScale = max(1, Double(source) / Double(target))
        func sinc(_ value: Double) -> Double { value == 0 ? 1 : sin(.pi * value) / (.pi * value) }
        return (0..<target).map { index in
            let center = (Double(index) + 0.5) * scale
            let lower = max(0, Int(center - 3 * filterScale + 0.5))
            let upper = min(source, Int(center + 3 * filterScale + 0.5))
            let taps = (lower..<upper).map { sample -> (Int, Double) in
                let distance = (Double(sample) + 0.5 - center) / filterScale
                return (sample, abs(distance) < 3 ? sinc(distance) * sinc(distance / 3) : 0)
            }
            let total = taps.reduce(0) { $0 + $1.1 }
            return taps.map { ($0.0, Int64(($0.1 / total * Double(1 << 22)).rounded())) }
        }
    }
}
