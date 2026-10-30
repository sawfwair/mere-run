import Foundation
import MediaIO

/// Pillow's separable 8-bit Lanczos path, including premultiplied RGBA conversion.
/// Fixed-point resampling follows the existing MediaIO Pillow bicubic implementation.
enum IdolImageResize {
    private struct Coefficients { let start: Int; let weights: [Int] }
    private static let precision = 22
    private static let unit = 1 << precision
    static func multiply255(_ a: UInt8, _ b: UInt8) -> UInt8 {
        let product = Int(a) * Int(b) + 128
        return UInt8((product + (product >> 8)) >> 8)
    }
    private static func coefficients(input: Int, output: Int) -> [Coefficients] {
        let scale = Double(input) / Double(output), filterScale = max(1, scale), support = 3 * filterScale
        func lanczos(_ distance: Double) -> Double {
            if distance == 0 { return 1 }
            if abs(distance) >= 3 { return 0 }
            let angle = distance * Double.pi
            return sin(angle) / angle * sin(angle / 3) / (angle / 3)
        }
        return (0..<output).map { index in
            let center = (Double(index) + 0.5) * scale
            let start = max(0, Int(center - support + 0.5)), end = min(input, Int(center + support + 0.5))
            let raw = (start..<end).map { lanczos((Double($0) - center + 0.5) / filterScale) }
            let sum = raw.reduce(0, +)
            return Coefficients(start: start, weights: raw.map { weight in
                let normalized = weight / sum
                return Int(normalized * Double(unit) + (normalized < 0 ? -0.5 : 0.5))
            })
        }
    }
    static func resize(_ image: MediaImage, width: Int, height: Int) throws -> MediaImage {
        if image.width == width && image.height == height { return image }
        let transparent = stride(from: 3, to: image.rgba8.count, by: 4).contains { image.rgba8[$0] < 255 }
        var bytes = image.rgba8
        if transparent {
            for pixel in 0..<(image.width * image.height) {
                for channel in 0..<3 { bytes[pixel * 4 + channel] = multiply255(bytes[pixel * 4 + channel], bytes[pixel * 4 + 3]) }
            }
        }
        let channels = transparent ? 4 : 3
        if image.width != width {
            let columns = coefficients(input: image.width, output: width)
            var resized = [UInt8](repeating: 255, count: image.height * width * 4)
            for y in 0..<image.height {
                for x in 0..<width {
                    let coeff = columns[x]
                    for channel in 0..<channels {
                        var total = unit / 2
                        for (offset, weight) in coeff.weights.enumerated() {
                            total += Int(bytes[(y * image.width + coeff.start + offset) * 4 + channel]) * weight
                        }
                        resized[(y * width + x) * 4 + channel] = UInt8(clamping: total >> precision)
                    }
                }
            }
            bytes = resized
        }
        if image.height != height {
            let rows = coefficients(input: image.height, output: height)
            var resized = [UInt8](repeating: 255, count: height * width * 4)
            for y in 0..<height {
                let coeff = rows[y]
                for x in 0..<width {
                    for channel in 0..<channels {
                        var total = unit / 2
                        for (offset, weight) in coeff.weights.enumerated() {
                            total += Int(bytes[((coeff.start + offset) * width + x) * 4 + channel]) * weight
                        }
                        resized[(y * width + x) * 4 + channel] = UInt8(clamping: total >> precision)
                    }
                }
            }
            bytes = resized
        }
        if transparent {
            for pixel in 0..<(width * height) {
                let alpha = Int(bytes[pixel * 4 + 3])
                if alpha > 0 && alpha < 255 {
                    for channel in 0..<3 { bytes[pixel * 4 + channel] = UInt8(clamping: 255 * Int(bytes[pixel * 4 + channel]) / alpha) }
                }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: bytes)
    }
}
