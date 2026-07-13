import Foundation
import MediaIO
import MLX

enum MMAudioVideoPreprocessor {
    static func sampledFrameIndices(
        sourceFrameRate: Double,
        sourceFrameCount: Int,
        targetFrameRate: Double,
        targetFrameCount: Int
    ) -> [Int] {
        precondition(sourceFrameCount > 0)
        precondition(sourceFrameRate > 0 && targetFrameRate > 0)
        precondition(targetFrameCount > 0)
        return (0..<targetFrameCount).map { targetIndex in
            let sourcePosition = Double(targetIndex) * sourceFrameRate / targetFrameRate
            return min(sourceFrameCount - 1, Int(ceil(sourcePosition - 1e-9)))
        }
    }

    static func normalizedCLIPFrame(_ image: MediaImage) -> MLXArray {
        let resized = InstantMeshPreprocessor.antialiasedBicubicResize(
            rgbHWC(image),
            sourceWidth: image.width,
            sourceHeight: image.height,
            targetWidth: 384,
            targetHeight: 384
        )
        let frame = MLXArray(resized.map(quantizedUnit)).reshaped(1, 384, 384, 3)
        let mean = MLXArray([Float(0.48145466), 0.4578275, 0.40821073])
            .reshaped(1, 1, 1, 3)
        let standardDeviation = MLXArray([Float(0.26862954), 0.26130258, 0.27577711])
            .reshaped(1, 1, 1, 3)
        return (frame - mean) / standardDeviation
    }

    static func normalizedSynchformerFrameCHW(_ image: MediaImage) -> [Float] {
        let target = 224
        let scale = Double(target) / Double(min(image.width, image.height))
        let resizedWidth = image.width <= image.height
            ? target
            : max(target, Int(Double(image.width) * scale))
        let resizedHeight = image.height <= image.width
            ? target
            : max(target, Int(Double(image.height) * scale))
        let resized = InstantMeshPreprocessor.antialiasedBicubicResize(
            rgbHWC(image),
            sourceWidth: image.width,
            sourceHeight: image.height,
            targetWidth: resizedWidth,
            targetHeight: resizedHeight
        )
        let originX = (resizedWidth - target) / 2
        let originY = (resizedHeight - target) / 2
        let pixels = target * target
        var chw = [Float](repeating: 0, count: pixels * 3)
        for y in 0..<target {
            for x in 0..<target {
                let source = ((originY + y) * resizedWidth + originX + x) * 3
                let destination = y * target + x
                chw[destination] = quantizedUnit(resized[source]) * 2 - 1
                chw[pixels + destination] = quantizedUnit(resized[source + 1]) * 2 - 1
                chw[pixels * 2 + destination] = quantizedUnit(resized[source + 2]) * 2 - 1
            }
        }
        return chw
    }

    private static func rgbHWC(_ image: MediaImage) -> [Float] {
        var rgb = [Float]()
        rgb.reserveCapacity(image.width * image.height * 3)
        for pixel in 0..<(image.width * image.height) {
            let source = pixel * 4
            rgb.append(Float(image.rgba8[source]) / 255)
            rgb.append(Float(image.rgba8[source + 1]) / 255)
            rgb.append(Float(image.rgba8[source + 2]) / 255)
        }
        return rgb
    }

    private static func quantizedUnit(_ value: Float) -> Float {
        Float(UInt8(clamping: Int((min(1, max(0, value)) * 255).rounded()))) / 255
    }
}
