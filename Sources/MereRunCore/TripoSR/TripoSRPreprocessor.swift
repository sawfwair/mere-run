import Foundation
import MediaIO
@preconcurrency import MLX

public enum TripoSRForegroundPolicy: Equatable, Sendable {
    /// Crop transparent foreground pixels, pad them to the requested ratio,
    /// and composite over the upstream 0.5-gray background. Fully opaque
    /// images are treated as already prepared and resized directly.
    case automaticTransparentAlpha(foregroundRatio: Float = 0.85)
    /// Do not crop. Alpha, if present, is still composited over 0.5 gray.
    case alreadyFramed
}

public enum TripoSRPreprocessingError: Error, Equatable, LocalizedError, Sendable {
    case emptyForeground
    case invalidForegroundRatio(Float)
    case invalidImageSize(Int)

    public var errorDescription: String? {
        switch self {
        case .emptyForeground:
            "TripoSR input alpha contains no foreground pixels."
        case .invalidForegroundRatio(let ratio):
            "TripoSR foreground ratio must be in (0, 1], found \(ratio)."
        case .invalidImageSize(let size):
            "TripoSR conditioning image size must be positive; received \(size)."
        }
    }
}

public struct TripoSRPreprocessingResult {
    /// `[1, size, size, 3]`, RGB in `[0, 1]`. ImageNet normalization remains
    /// inside the checkpoint graph's image tokenizer.
    public let image: MLXArray
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let preparedWidth: Int
    public let preparedHeight: Int
    public let croppedTransparentForeground: Bool
}

public enum TripoSRPreprocessor {
    public static func prepare(
        image: MediaImage,
        size: Int = TripoSRConfiguration.production.conditioningImageSize,
        foregroundPolicy: TripoSRForegroundPolicy = .automaticTransparentAlpha()
    ) throws -> TripoSRPreprocessingResult {
        guard size > 0 else {
            throw TripoSRPreprocessingError.invalidImageSize(size)
        }
        let prepared: PreparedRGB
        let cropped: Bool
        switch foregroundPolicy {
        case .alreadyFramed:
            prepared = compositeOnGray(image: image)
            cropped = false
        case .automaticTransparentAlpha(let foregroundRatio):
            guard foregroundRatio > 0, foregroundRatio <= 1 else {
                throw TripoSRPreprocessingError.invalidForegroundRatio(foregroundRatio)
            }
            let hasTransparency = stride(from: 3, to: image.rgba8.count, by: 4)
                .contains { image.rgba8[$0] < 255 }
            if hasTransparency {
                prepared = try cropPadAndComposite(image: image, foregroundRatio: foregroundRatio)
                cropped = true
            } else {
                prepared = compositeOnGray(image: image)
                cropped = false
            }
        }

        let resized = antialiasedBilinearResize(
            prepared.values,
            sourceWidth: prepared.width,
            sourceHeight: prepared.height,
            targetWidth: size,
            targetHeight: size
        )
        return TripoSRPreprocessingResult(
            image: MLXArray(resized).reshaped(1, size, size, 3),
            sourceWidth: image.width,
            sourceHeight: image.height,
            preparedWidth: prepared.width,
            preparedHeight: prepared.height,
            croppedTransparentForeground: cropped
        )
    }

    private struct PreparedRGB {
        let width: Int
        let height: Int
        let values: [Float]
    }

    private static func compositeOnGray(image: MediaImage) -> PreparedRGB {
        PreparedRGB(
            width: image.width,
            height: image.height,
            values: compositeRGBA(image.rgba8, width: image.width, height: image.height)
        )
    }

    private static func cropPadAndComposite(
        image: MediaImage,
        foregroundRatio: Float
    ) throws -> PreparedRGB {
        var minimumX = image.width
        var minimumY = image.height
        var maximumX = -1
        var maximumY = -1
        for y in 0..<image.height {
            for x in 0..<image.width where image.rgba8[(y * image.width + x) * 4 + 3] > 0 {
                minimumX = min(minimumX, x)
                minimumY = min(minimumY, y)
                maximumX = max(maximumX, x)
                maximumY = max(maximumY, y)
            }
        }
        guard maximumX >= minimumX, maximumY >= minimumY else {
            throw TripoSRPreprocessingError.emptyForeground
        }

        // The official `resize_foreground` uses NumPy's exclusive slice end.
        // Retain that exact behavior except for a one-pixel foreground, where
        // the upstream helper would otherwise create an empty image.
        let cropWidth = max(1, maximumX - minimumX)
        let cropHeight = max(1, maximumY - minimumY)
        let squareSize = max(cropWidth, cropHeight)
        let paddedSize = max(squareSize, Int(Float(squareSize) / foregroundRatio))
        var canvas = [UInt8](repeating: 0, count: paddedSize * paddedSize * 4)
        let destinationX = (paddedSize - cropWidth) / 2
        let destinationY = (paddedSize - cropHeight) / 2
        for y in 0..<cropHeight {
            for x in 0..<cropWidth {
                let source = ((minimumY + y) * image.width + minimumX + x) * 4
                let destination = ((destinationY + y) * paddedSize + destinationX + x) * 4
                canvas[destination] = image.rgba8[source]
                canvas[destination + 1] = image.rgba8[source + 1]
                canvas[destination + 2] = image.rgba8[source + 2]
                canvas[destination + 3] = image.rgba8[source + 3]
            }
        }
        return PreparedRGB(
            width: paddedSize,
            height: paddedSize,
            values: compositeRGBA(canvas, width: paddedSize, height: paddedSize)
        )
    }

    private static func compositeRGBA(_ rgba: [UInt8], width: Int, height: Int) -> [Float] {
        var output = [Float](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            let source = pixel * 4
            let destination = pixel * 3
            let alpha = Float(rgba[source + 3]) / 255
            for channel in 0..<3 {
                let foreground = Float(rgba[source + channel]) / 255
                let composited = foreground * alpha + 0.5 * (1 - alpha)
                // run.py converts the float composite through uint8 before
                // passing it into ImagePreprocessor.
                let quantized = UInt8(clamping: Int(composited * 255))
                output[destination + channel] = Float(quantized) / 255
            }
        }
        return output
    }

    /// Separable implementation of PyTorch bilinear interpolation with
    /// `align_corners=False, antialias=True`.
    static func antialiasedBilinearResize(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        precondition(source.count == sourceWidth * sourceHeight * 3)
        if sourceWidth == targetWidth, sourceHeight == targetHeight { return source }
        let horizontal = resamplingWeights(sourceSize: sourceWidth, targetSize: targetWidth)
        var intermediate = [Float](repeating: 0, count: sourceHeight * targetWidth * 3)
        for y in 0..<sourceHeight {
            for targetX in 0..<targetWidth {
                let weights = horizontal[targetX]
                for (sourceX, weight) in weights {
                    let sourceOffset = (y * sourceWidth + sourceX) * 3
                    let targetOffset = (y * targetWidth + targetX) * 3
                    intermediate[targetOffset] += source[sourceOffset] * weight
                    intermediate[targetOffset + 1] += source[sourceOffset + 1] * weight
                    intermediate[targetOffset + 2] += source[sourceOffset + 2] * weight
                }
            }
        }

        let vertical = resamplingWeights(sourceSize: sourceHeight, targetSize: targetHeight)
        var output = [Float](repeating: 0, count: targetHeight * targetWidth * 3)
        for targetY in 0..<targetHeight {
            for (sourceY, weight) in vertical[targetY] {
                for x in 0..<targetWidth {
                    let sourceOffset = (sourceY * targetWidth + x) * 3
                    let targetOffset = (targetY * targetWidth + x) * 3
                    output[targetOffset] += intermediate[sourceOffset] * weight
                    output[targetOffset + 1] += intermediate[sourceOffset + 1] * weight
                    output[targetOffset + 2] += intermediate[sourceOffset + 2] * weight
                }
            }
        }
        return output
    }

    private static func resamplingWeights(
        sourceSize: Int,
        targetSize: Int
    ) -> [[(index: Int, weight: Float)]] {
        let scale = Float(sourceSize) / Float(targetSize)
        let support = max(scale, 1)
        var result = [[(index: Int, weight: Float)]]()
        result.reserveCapacity(targetSize)
        for destination in 0..<targetSize {
            let center = scale * (Float(destination) + 0.5)
            let first = max(0, Int(ceil(center - support - 0.5)))
            let last = min(sourceSize - 1, Int(floor(center + support - 0.5)))
            var weights: [(Int, Float)] = []
            var total: Float = 0
            if first <= last {
                for index in first...last {
                    let distance = abs((Float(index) + 0.5 - center) / support)
                    let weight = max(0, 1 - distance)
                    if weight > 0 {
                        weights.append((index, weight))
                        total += weight
                    }
                }
            }
            if total > 0 {
                weights = weights.map { ($0.0, $0.1 / total) }
            }
            result.append(weights)
        }
        return result
    }
}
