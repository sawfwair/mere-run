import Foundation
import MediaIO

// RGB resampling follows Pillow 12.3.0's separable bicubic, 8-bit path.
// See THIRD_PARTY_NOTICES.md for the Pillow MIT-CMU license and provenance.
enum FalconPerceptionRGBResize {
    private struct Coefficients {
        let start: Int
        let weights: [Int]
    }

    private static let precision = 22
    private static let unit = 1 << precision

    private static func cubic(_ distance: Double) -> Double {
        let x = abs(distance)
        if x < 1 { return ((1.5 * x - 2.5) * x * x) + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * -0.5 }
        return 0
    }

    private static func coefficients(input: Int, output: Int) -> [Coefficients] {
        let scale = Double(input) / Double(output)
        let filterScale = max(1, scale)
        let inverseFilterScale = 1 / filterScale
        let support = 2 * filterScale
        return (0..<output).map { position in
            let center = (Double(position) + 0.5) * scale
            let start = max(0, Int(center - support + 0.5))
            let end = min(input, Int(center + support + 0.5))
            let weights = (start..<end).map { source in
                cubic((Double(source) - center + 0.5) * inverseFilterScale)
            }
            let sum = weights.reduce(0, +)
            let fixed = weights.map { weight in
                let normalized = weight / sum
                return Int(normalized * Double(unit) + (normalized < 0 ? -0.5 : 0.5))
            }
            return Coefficients(start: start, weights: fixed)
        }
    }

    private static func clipped(_ value: Int) -> UInt8 {
        UInt8(clamping: value >> precision)
    }

    static func resized(_ image: MediaImage, width: Int, height: Int) throws -> MediaImage {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        var intermediate = image
        if width != image.width {
            let columns = coefficients(input: image.width, output: width)
            var rgba = [UInt8](repeating: 255, count: width * image.height * 4)
            for y in 0..<image.height {
                for x in 0..<width {
                    let coefficients = columns[x]
                    let destination = (y * width + x) * 4
                    for channel in 0..<3 {
                        var total = unit / 2
                        for (offset, weight) in coefficients.weights.enumerated() {
                            let source = (y * image.width + coefficients.start + offset) * 4 + channel
                            total += Int(image.rgba8[source]) * weight
                        }
                        rgba[destination + channel] = clipped(total)
                    }
                }
            }
            intermediate = try MediaImage(width: width, height: image.height, rgba8: rgba)
        }
        if height == intermediate.height { return intermediate }
        let rows = coefficients(input: intermediate.height, output: height)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            let coefficients = rows[y]
            for x in 0..<width {
                let destination = (y * width + x) * 4
                for channel in 0..<3 {
                    var total = unit / 2
                    for (offset, weight) in coefficients.weights.enumerated() {
                        let source = ((coefficients.start + offset) * width + x) * 4 + channel
                        total += Int(intermediate.rgba8[source]) * weight
                    }
                    rgba[destination + channel] = clipped(total)
                }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }
}
