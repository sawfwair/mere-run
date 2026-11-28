import Foundation

/// Separable port of torchvision/PyTorch antialiased bicubic RGB image resize
/// (`mode="bicubic"`, `align_corners=false`, `antialias=true`, clamped to `[0, 1]`).
package enum AntialiasedBicubicResize {
    package static func resize(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        precondition(sourceWidth > 0 && sourceHeight > 0)
        precondition(targetWidth > 0 && targetHeight > 0)
        precondition(source.count == sourceWidth * sourceHeight * 3)
        if sourceWidth == targetWidth, sourceHeight == targetHeight { return source }

        let horizontal = bicubicResamplingWeights(
            sourceSize: sourceWidth,
            targetSize: targetWidth
        )
        var intermediate = [Float](repeating: 0, count: sourceHeight * targetWidth * 3)
        for y in 0..<sourceHeight {
            for targetX in 0..<targetWidth {
                let targetOffset = (y * targetWidth + targetX) * 3
                for (sourceX, weight) in horizontal[targetX] {
                    let sourceOffset = (y * sourceWidth + sourceX) * 3
                    intermediate[targetOffset] += source[sourceOffset] * weight
                    intermediate[targetOffset + 1] += source[sourceOffset + 1] * weight
                    intermediate[targetOffset + 2] += source[sourceOffset + 2] * weight
                }
            }
        }

        let vertical = bicubicResamplingWeights(
            sourceSize: sourceHeight,
            targetSize: targetHeight
        )
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
        for index in output.indices {
            output[index] = min(1, max(0, output[index]))
        }
        return output
    }

    private static func bicubicResamplingWeights(
        sourceSize: Int,
        targetSize: Int
    ) -> [[(index: Int, weight: Float)]] {
        let scale = Float(sourceSize) / Float(targetSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        var result: [[(index: Int, weight: Float)]] = []
        result.reserveCapacity(targetSize)
        for destination in 0..<targetSize {
            let center = scale * (Float(destination) + 0.5)
            let first = max(0, Int(ceil(center - support - 0.5)))
            let last = min(sourceSize - 1, Int(floor(center + support - 0.5)))
            var weights: [(Int, Float)] = []
            weights.reserveCapacity(max(0, last - first + 1))
            var total: Float = 0
            if first <= last {
                for index in first...last {
                    let distance = (Float(index) + 0.5 - center) / filterScale
                    let weight = cubicKernel(distance)
                    if weight != 0 {
                        weights.append((index, weight))
                        total += weight
                    }
                }
            }
            if total != 0 {
                weights = weights.map { ($0.0, $0.1 / total) }
            }
            result.append(weights)
        }
        return result
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
}
