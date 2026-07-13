import Foundation
import MediaIO
@preconcurrency import MLX
import MLXNN

public enum Trellis2ForegroundPolicy: Equatable, Sendable {
    /// Use non-opaque alpha to isolate and center the object. Fully opaque
    /// images require the caller to explicitly confirm that framing is ready.
    case transparentAlpha
    /// Preserve the supplied framing and composite any alpha over black.
    case alreadyFramed
}

public enum Trellis2PreprocessingError: Error, Equatable, LocalizedError, Sendable {
    case invalidImageSize(Int)
    case emptyForeground
    case backgroundRemovalRequired

    public var errorDescription: String? {
        switch self {
        case .invalidImageSize(let size):
            "TRELLIS.2 conditioning image size must be positive; received \(size)."
        case .emptyForeground:
            "TRELLIS.2 input alpha contains no foreground pixels."
        case .backgroundRemovalRequired:
            "TRELLIS.2 received a fully opaque image. Supply transparent alpha or pass the already-framed option."
        }
    }
}

public struct Trellis2PreprocessingResult {
    /// `[1, 3, size, size]`, ImageNet-normalized RGB.
    public let conditionInput: MLXArray
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let croppedTransparentForeground: Bool
}

public enum Trellis2Preprocessor {
    public static func prepare(
        image: MediaImage,
        size: Int = 512,
        foregroundPolicy: Trellis2ForegroundPolicy = .transparentAlpha
    ) throws -> Trellis2PreprocessingResult {
        guard size > 0 else { throw Trellis2PreprocessingError.invalidImageSize(size) }
        let hasTransparency = stride(from: 3, to: image.rgba8.count, by: 4)
            .contains { image.rgba8[$0] < 255 }

        let prepared: PreparedRGB
        let cropped: Bool
        switch foregroundPolicy {
        case .transparentAlpha:
            guard hasTransparency else {
                throw Trellis2PreprocessingError.backgroundRemovalRequired
            }
            prepared = try cropAndCompositeOnBlack(boundedForegroundImage(image))
            cropped = true
        case .alreadyFramed:
            prepared = compositeOnBlack(image)
            cropped = false
        }

        let resized = lanczosResize(
            prepared.values,
            sourceWidth: prepared.width,
            sourceHeight: prepared.height,
            targetWidth: size,
            targetHeight: size
        )
        let mean: [Float] = [0.485, 0.456, 0.406]
        let standardDeviation: [Float] = [0.229, 0.224, 0.225]
        var normalized = [Float](repeating: 0, count: resized.count)
        for pixel in 0..<(size * size) {
            for channel in 0..<3 {
                normalized[pixel * 3 + channel] = (
                    resized[pixel * 3 + channel] - mean[channel]
                ) / standardDeviation[channel]
            }
        }
        return Trellis2PreprocessingResult(
            conditionInput: MLXArray(normalized)
                .reshaped(1, size, size, 3)
                .transposed(0, 3, 1, 2),
            sourceWidth: image.width,
            sourceHeight: image.height,
            croppedTransparentForeground: cropped
        )
    }

    private struct PreparedRGB {
        let width: Int
        let height: Int
        let values: [Float]
    }

    private static func compositeOnBlack(_ image: MediaImage) -> PreparedRGB {
        PreparedRGB(
            width: image.width,
            height: image.height,
            values: premultipliedRGB(
                image.rgba8,
                width: image.width,
                height: image.height
            )
        )
    }

    /// Microsoft's reference pipeline bounds the source to 1024px before
    /// alpha-box discovery. Preserve that order so large inputs produce the
    /// same foreground framing before the final 512px conditioning resize.
    private static func boundedForegroundImage(_ image: MediaImage) throws -> MediaImage {
        let maximumDimension = max(image.width, image.height)
        guard maximumDimension > 1_024 else { return image }
        let scale = Float(1_024) / Float(maximumDimension)
        let width = max(1, Int(Float(image.width) * scale))
        let height = max(1, Int(Float(image.height) * scale))
        let values = image.rgba8.map { Float($0) / 255 }
        let resized = lanczosResize(
            values,
            sourceWidth: image.width,
            sourceHeight: image.height,
            targetWidth: width,
            targetHeight: height,
            channels: 4
        )
        return try MediaImage(
            width: width,
            height: height,
            rgba8: resized.map { UInt8((255 * min(max($0, 0), 1)).rounded()) }
        )
    }

    private static func cropAndCompositeOnBlack(_ image: MediaImage) throws -> PreparedRGB {
        var minimumX = image.width
        var minimumY = image.height
        var maximumX = -1
        var maximumY = -1
        for y in 0..<image.height {
            for x in 0..<image.width where image.rgba8[(y * image.width + x) * 4 + 3] > 204 {
                minimumX = min(minimumX, x)
                minimumY = min(minimumY, y)
                maximumX = max(maximumX, x)
                maximumY = max(maximumY, y)
            }
        }
        guard maximumX >= minimumX, maximumY >= minimumY else {
            throw Trellis2PreprocessingError.emptyForeground
        }

        let width = max(1, maximumX - minimumX)
        let height = max(1, maximumY - minimumY)
        let side = max(width, height)
        let centerX = (minimumX + maximumX) / 2
        let centerY = (minimumY + maximumY) / 2
        let originX = centerX - side / 2
        let originY = centerY - side / 2
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        for y in 0..<side {
            for x in 0..<side {
                let sourceX = originX + x
                let sourceY = originY + y
                guard sourceX >= 0, sourceX < image.width,
                      sourceY >= 0, sourceY < image.height else { continue }
                let source = (sourceY * image.width + sourceX) * 4
                let destination = (y * side + x) * 4
                rgba[destination] = image.rgba8[source]
                rgba[destination + 1] = image.rgba8[source + 1]
                rgba[destination + 2] = image.rgba8[source + 2]
                rgba[destination + 3] = image.rgba8[source + 3]
            }
        }
        return PreparedRGB(
            width: side,
            height: side,
            values: premultipliedRGB(rgba, width: side, height: side)
        )
    }

    private static func premultipliedRGB(
        _ rgba: [UInt8],
        width: Int,
        height: Int
    ) -> [Float] {
        var result = [Float](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            let alpha = Float(rgba[pixel * 4 + 3]) / 255
            for channel in 0..<3 {
                result[pixel * 3 + channel] = Float(rgba[pixel * 4 + channel]) / 255 * alpha
            }
        }
        return result
    }

    /// Separable Lanczos-3 resize matching PIL's high-quality conditioning
    /// resize closely while remaining deterministic and dependency-free.
    static func lanczosResize(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
        channels: Int = 3
    ) -> [Float] {
        precondition(channels > 0)
        precondition(source.count == sourceWidth * sourceHeight * channels)
        if sourceWidth == targetWidth, sourceHeight == targetHeight { return source }
        let horizontal = lanczosWeights(sourceSize: sourceWidth, targetSize: targetWidth)
        var intermediate = [Float](repeating: 0, count: sourceHeight * targetWidth * channels)
        for y in 0..<sourceHeight {
            for targetX in 0..<targetWidth {
                for (sourceX, weight) in horizontal[targetX] {
                    for channel in 0..<channels {
                        intermediate[(y * targetWidth + targetX) * channels + channel]
                            += source[(y * sourceWidth + sourceX) * channels + channel] * weight
                    }
                }
            }
        }
        let vertical = lanczosWeights(sourceSize: sourceHeight, targetSize: targetHeight)
        var output = [Float](repeating: 0, count: targetHeight * targetWidth * channels)
        for targetY in 0..<targetHeight {
            for (sourceY, weight) in vertical[targetY] {
                for x in 0..<targetWidth {
                    for channel in 0..<channels {
                        output[(targetY * targetWidth + x) * channels + channel]
                            += intermediate[(sourceY * targetWidth + x) * channels + channel] * weight
                    }
                }
            }
        }
        return output.map { min(max($0, 0), 1) }
    }

    private static func lanczosWeights(
        sourceSize: Int,
        targetSize: Int
    ) -> [[(Int, Float)]] {
        let scale = Float(sourceSize) / Float(targetSize)
        let filterScale = max(scale, 1)
        let support = 3 * filterScale
        return (0..<targetSize).map { destination in
            let center = scale * (Float(destination) + 0.5)
            let first = max(0, Int(ceil(center - support - 0.5)))
            let last = min(sourceSize - 1, Int(floor(center + support - 0.5)))
            var weights = [(Int, Float)]()
            var total: Float = 0
            if first <= last {
                for index in first...last {
                    let distance = (Float(index) + 0.5 - center) / filterScale
                    let weight = lanczos(distance)
                    if weight != 0 {
                        weights.append((index, weight))
                        total += weight
                    }
                }
            }
            return weights.map { ($0.0, $0.1 / total) }
        }
    }

    private static func lanczos(_ value: Float) -> Float {
        let magnitude = abs(value)
        if magnitude < Float.ulpOfOne { return 1 }
        if magnitude >= 3 { return 0 }
        let piValue = Float.pi * value
        return sin(piValue) / piValue * sin(piValue / 3) / (piValue / 3)
    }
}

struct Trellis2DINOv3Conditioner {
    private let model: DINOv3VisionModel

    init(checkpointURL: URL) throws {
        var configuration = DINOv3Config()
        configuration.hiddenSize = 1_024
        configuration.intermediateSize = 4_096
        configuration.numHiddenLayers = 24
        configuration.numAttentionHeads = 16
        configuration.patchSize = 16
        configuration.imageSize = 512
        configuration.numRegisterTokens = 4
        configuration.useGatedMLP = false
        configuration.queryBias = true
        configuration.keyBias = false
        configuration.valueBias = true
        configuration.projectionBias = true
        configuration.mlpBias = true

        let model = DINOv3VisionModel(config: configuration)
        let rawWeights = try MLX.loadArrays(url: checkpointURL)
        var converted = [String: MLXArray]()
        converted.reserveCapacity(rawWeights.count)
        for (key, value) in rawWeights {
            converted[key] = value.ndim == 4
                ? HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
                : value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        self.model = model
    }

    func encode(_ input: MLXArray) -> MLXArray {
        let condition = model.preNormalizedTokens(input)
        MLX.eval(condition)
        return condition
    }
}
