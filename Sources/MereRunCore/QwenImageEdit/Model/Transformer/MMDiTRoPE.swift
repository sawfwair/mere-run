import Foundation
import MLX

/// Qwen image rotary embeddings for ordered output and reference-image token grids.
public final class MMDiTRoPE {
    public let theta: Float
    public let headDim: Int
    public let axesDims: [Int]
    private struct CacheKey: Hashable {
        let imageShapeValues: [Int]
        let textSequenceLength: Int
    }

    private var cache: [CacheKey: (image: MLXArray, text: MLXArray)] = [:]
    private var cacheOrder: [CacheKey] = []
    private let maximumCacheEntries = 8

    public init(theta: Float = 10_000, headDim: Int, axesDims: [Int]? = nil) {
        self.theta = theta
        self.headDim = headDim
        self.axesDims = axesDims ?? [16, 56, 56]
        precondition(self.axesDims.count == 3)
        precondition(self.axesDims.reduce(0, +) == headDim)
        precondition(self.axesDims.allSatisfy { $0 % 2 == 0 })
    }

    public func frequencies(
        imageShapes: [(temporal: Int, height: Int, width: Int)],
        textSequenceLength: Int
    ) -> (image: MLXArray, text: MLXArray) {
        precondition(!imageShapes.isEmpty && textSequenceLength > 0)
        let key = CacheKey(
            imageShapeValues: imageShapes.flatMap { [$0.temporal, $0.height, $0.width] },
            textSequenceLength: textSequenceLength
        )
        if let cached = cache[key] {
            return cached
        }
        var imageValues: [Float32] = []
        let imageTokenCount = imageShapes.reduce(0) { partial, shape in
            partial + shape.temporal * shape.height * shape.width
        }
        imageValues.reserveCapacity(imageTokenCount * (headDim / 2) * 2)

        var maximumImageIndex = 0
        for (imageIndex, shape) in imageShapes.enumerated() {
            maximumImageIndex = max(maximumImageIndex, shape.height / 2, shape.width / 2)
            let heightStart = -(shape.height - shape.height / 2)
            let widthStart = -(shape.width - shape.width / 2)
            for frame in 0..<shape.temporal {
                for row in 0..<shape.height {
                    for column in 0..<shape.width {
                        appendFrequencies(
                            positions: [imageIndex + frame, heightStart + row, widthStart + column],
                            to: &imageValues
                        )
                    }
                }
            }
        }

        var textValues: [Float32] = []
        textValues.reserveCapacity(textSequenceLength * (headDim / 2) * 2)
        for position in maximumImageIndex..<(maximumImageIndex + textSequenceLength) {
            appendFrequencies(positions: [position, position, position], to: &textValues)
        }

        let result = (
            image: MLXArray(imageValues).reshaped(imageTokenCount, headDim / 2, 2),
            text: MLXArray(textValues).reshaped(textSequenceLength, headDim / 2, 2)
        )
        if cache.count == maximumCacheEntries, let oldest = cacheOrder.first {
            cache.removeValue(forKey: oldest)
            cacheOrder.removeFirst()
        }
        cache[key] = result
        cacheOrder.append(key)
        return result
    }

    private func appendFrequencies(positions: [Int], to values: inout [Float32]) {
        for axis in 0..<3 {
            let dimension = axesDims[axis]
            for pair in 0..<(dimension / 2) {
                let exponent = Float(2 * pair) / Float(dimension)
                let angularFrequency = 1 / pow(theta, exponent)
                let angle = Float(positions[axis]) * angularFrequency
                values.append(cos(angle))
                values.append(sin(angle))
            }
        }
    }
}

public struct MMDiTPositionBuilder {
    public static func build3D(temporal: Int, height: Int, width: Int) -> MLXArray {
        var positions: [Int32] = []
        positions.reserveCapacity(temporal * height * width * 3)
        for frame in 0..<temporal {
            for row in 0..<height {
                for column in 0..<width {
                    positions.append(Int32(frame))
                    positions.append(Int32(row))
                    positions.append(Int32(column))
                }
            }
        }
        return MLXArray(positions).reshaped(temporal * height * width, 3)
    }
}
