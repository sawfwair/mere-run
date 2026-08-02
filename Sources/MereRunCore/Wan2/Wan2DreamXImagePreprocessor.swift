import Foundation
import MediaIO

/// Independent antialiased bilinear resize matching DreamX's fixed-size source-image contract.
enum Wan2DreamXImagePreprocessor {
    static func resized(_ image: MediaImage, width: Int, height: Int) throws -> MediaImage {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        if image.width == width, image.height == height { return image }

        let horizontal = contributions(inputCount: image.width, outputCount: width)
        let vertical = contributions(inputCount: image.height, outputCount: height)
        // Pillow's 8-bit resize path is separable and quantizes after each
        // axis. Retaining that intermediate quantization is required for the
        // released DreamX PIL -> torchvision Resize contract.
        var horizontalPixels = [UInt8](repeating: 0, count: width * image.height * 4)
        for y in 0..<image.height {
            for x in 0..<width {
                let destination = (y * width + x) * 4
                for channel in 0..<4 {
                    var value = Double(0)
                    for contribution in horizontal[x] {
                        let source = (y * image.width + contribution.index) * 4
                        value += Double(image.rgba8[source + channel]) * contribution.weight
                    }
                    horizontalPixels[destination + channel] = UInt8(clamping: Int(value.rounded()))
                }
            }
        }

        var output = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let destination = (y * width + x) * 4
                for channel in 0..<4 {
                    var value = Double(0)
                    for contribution in vertical[y] {
                        let source = (contribution.index * width + x) * 4
                        value += Double(horizontalPixels[source + channel]) * contribution.weight
                    }
                    output[destination + channel] = UInt8(clamping: Int(value.rounded()))
                }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: output)
    }

    private struct Contribution {
        let index: Int
        let weight: Double
    }

    private static func contributions(inputCount: Int, outputCount: Int) -> [[Contribution]] {
        let scale = Double(inputCount) / Double(outputCount)
        let filterScale = max(1, scale)
        let support = filterScale
        return (0..<outputCount).map { outputIndex in
            let center = (Double(outputIndex) + 0.5) * scale
            let lower = max(0, Int(center - support + 0.5))
            let upper = min(inputCount, Int(center + support + 0.5))
            var weights: [Contribution] = []
            for candidate in lower..<upper {
                let distance = abs((Double(candidate) + 0.5 - center) / filterScale)
                let weight = max(0, 1 - distance)
                guard weight > 0 else { continue }
                weights.append(Contribution(index: candidate, weight: weight))
            }
            let total = weights.reduce(0) { $0 + $1.weight }
            return weights.map { contribution in
                Contribution(index: contribution.index, weight: contribution.weight / total)
            }
        }
    }
}
