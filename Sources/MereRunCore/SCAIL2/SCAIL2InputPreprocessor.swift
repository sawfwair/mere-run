import Foundation
import MediaIO
import MLX

public enum SCAIL2InputPreprocessor {
    /// Matches the upstream torchvision tensor path: antialiased bicubic
    /// aspect-fill resize, followed by an integer center crop and `[-1, 1]`
    /// normalization. The output is `[T, H, W, 3]`.
    public static func centerCroppedTensor(
        images: [MediaImage],
        width: Int,
        height: Int
    ) -> MLXArray {
        precondition(!images.isEmpty)
        precondition(width > 0 && height > 0)
        var output: [Float] = []
        output.reserveCapacity(images.count * width * height * 3)
        for image in images {
            output.append(contentsOf: centerCroppedRGB(image, width: width, height: height))
        }
        return MLXArray(output).reshaped(images.count, height, width, 3)
    }

    public static func centerCroppedTensor(
        image: MediaImage,
        width: Int,
        height: Int
    ) -> MLXArray {
        centerCroppedTensor(images: [image], width: width, height: height)
    }

    /// Matches `F.interpolate(..., scale_factor=0.5, mode="bilinear",
    /// align_corners=False)` for an upstream-normalized `[T, H, W, 3]` tensor.
    public static func halfResolutionBilinear(_ input: MLXArray) -> MLXArray {
        precondition(input.ndim == 4 && input.dim(3) == 3)
        precondition(input.dim(1).isMultiple(of: 2) && input.dim(2).isMultiple(of: 2))
        let frames = input.dim(0)
        let height = input.dim(1)
        let width = input.dim(2)
        let outputHeight = height / 2
        let outputWidth = width / 2
        let values = input.asType(.float32).asArray(Float.self)
        var output = [Float](repeating: 0, count: frames * outputHeight * outputWidth * 3)
        for frame in 0..<frames {
            for y in 0..<outputHeight {
                let sourceY = Float(y * 2) + 0.5
                let y0 = max(0, min(height - 1, Int(Foundation.floor(sourceY))))
                let y1 = max(0, min(height - 1, y0 + 1))
                let yFraction = sourceY - Float(y0)
                for x in 0..<outputWidth {
                    let sourceX = Float(x * 2) + 0.5
                    let x0 = max(0, min(width - 1, Int(Foundation.floor(sourceX))))
                    let x1 = max(0, min(width - 1, x0 + 1))
                    let xFraction = sourceX - Float(x0)
                    for channel in 0..<3 {
                        let topLeft = values[index(frame, y0, x0, channel, height, width)]
                        let topRight = values[index(frame, y0, x1, channel, height, width)]
                        let bottomLeft = values[index(frame, y1, x0, channel, height, width)]
                        let bottomRight = values[index(frame, y1, x1, channel, height, width)]
                        let top = topLeft + (topRight - topLeft) * xFraction
                        let bottom = bottomLeft + (bottomRight - bottomLeft) * xFraction
                        output[index(frame, y, x, channel, outputHeight, outputWidth)] =
                            top + (bottom - top) * yFraction
                    }
                }
            }
        }
        return MLXArray(output).reshaped(frames, outputHeight, outputWidth, 3)
    }

    private static func centerCroppedRGB(
        _ image: MediaImage,
        width: Int,
        height: Int
    ) -> [Float] {
        let targetRatio = Double(width) / Double(height)
        let sourceRatio = Double(image.width) / Double(image.height)
        let resizedWidth: Int
        let resizedHeight: Int
        if sourceRatio > targetRatio {
            resizedHeight = height
            resizedWidth = Int(Double(image.width) * Double(height) / Double(image.height))
        } else {
            resizedWidth = width
            resizedHeight = Int(Double(image.height) * Double(width) / Double(image.width))
        }
        let source = rgbFloat(image)
        let resized = bicubicResize(
            source,
            sourceWidth: image.width,
            sourceHeight: image.height,
            targetWidth: resizedWidth,
            targetHeight: resizedHeight
        )
        let top = (resizedHeight - height) / 2
        let left = (resizedWidth - width) / 2
        var output = [Float](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let sourceOffset = ((y + top) * resizedWidth + x + left) * 3
                let targetOffset = (y * width + x) * 3
                for channel in 0..<3 {
                    output[targetOffset + channel] = resized[sourceOffset + channel] * 2 - 1
                }
            }
        }
        return output
    }

    private static func rgbFloat(_ image: MediaImage) -> [Float] {
        var output = [Float]()
        output.reserveCapacity(image.width * image.height * 3)
        for pixel in 0..<(image.width * image.height) {
            let offset = pixel * 4
            output.append(Float(image.rgba8[offset]) / 255)
            output.append(Float(image.rgba8[offset + 1]) / 255)
            output.append(Float(image.rgba8[offset + 2]) / 255)
        }
        return output
    }

    private static func bicubicResize(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        if sourceWidth == targetWidth, sourceHeight == targetHeight { return source }
        let horizontal = bicubicWeights(sourceSize: sourceWidth, targetSize: targetWidth)
        var intermediate = [Float](repeating: 0, count: sourceHeight * targetWidth * 3)
        for y in 0..<sourceHeight {
            for targetX in 0..<targetWidth {
                let targetOffset = (y * targetWidth + targetX) * 3
                for (sourceX, weight) in horizontal[targetX] {
                    let sourceOffset = (y * sourceWidth + sourceX) * 3
                    for channel in 0..<3 {
                        intermediate[targetOffset + channel] += source[sourceOffset + channel] * weight
                    }
                }
            }
        }
        let vertical = bicubicWeights(sourceSize: sourceHeight, targetSize: targetHeight)
        var output = [Float](repeating: 0, count: targetHeight * targetWidth * 3)
        for targetY in 0..<targetHeight {
            for (sourceY, weight) in vertical[targetY] {
                for x in 0..<targetWidth {
                    let sourceOffset = (sourceY * targetWidth + x) * 3
                    let targetOffset = (targetY * targetWidth + x) * 3
                    for channel in 0..<3 {
                        output[targetOffset + channel] += intermediate[sourceOffset + channel] * weight
                    }
                }
            }
        }
        return output
    }

    private static func bicubicWeights(
        sourceSize: Int,
        targetSize: Int
    ) -> [[(index: Int, weight: Float)]] {
        let scale = Float(sourceSize) / Float(targetSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        return (0..<targetSize).map { destination in
            let center = scale * (Float(destination) + 0.5)
            let first = max(0, Int(Foundation.ceil(Double(center - support - 0.5))))
            let last = min(sourceSize - 1, Int(Foundation.floor(Double(center + support - 0.5))))
            var weights: [(Int, Float)] = []
            var total: Float = 0
            if first <= last {
                for source in first...last {
                    let distance = (Float(source) + 0.5 - center) / filterScale
                    let weight = cubicKernel(distance)
                    if weight != 0 {
                        weights.append((source, weight))
                        total += weight
                    }
                }
            }
            return total == 0 ? weights : weights.map { ($0.0, $0.1 / total) }
        }
    }

    private static func cubicKernel(_ value: Float) -> Float {
        let x = abs(value)
        let coefficient: Float = -0.5
        if x < 1 {
            return ((coefficient + 2) * x - (coefficient + 3)) * x * x + 1
        }
        if x < 2 {
            return ((coefficient * x - 5 * coefficient) * x + 8 * coefficient) * x
                - 4 * coefficient
        }
        return 0
    }

    private static func index(
        _ frame: Int,
        _ y: Int,
        _ x: Int,
        _ channel: Int,
        _ height: Int,
        _ width: Int
    ) -> Int {
        (((frame * height + y) * width + x) * 3) + channel
    }
}
