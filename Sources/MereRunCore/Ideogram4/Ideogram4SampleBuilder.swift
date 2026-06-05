import Foundation
import MLX

public struct Ideogram4PackedSample {
    public var llmFeatures: MLXArray
    public var x: MLXArray
    public var positionIds: MLXArray
    public var segmentIds: MLXArray
    public var indicator: MLXArray
    public var textTokenCount: Int
    public var imageTokenCount: Int
    public var imageTokenHeight: Int
    public var imageTokenWidth: Int
}

public enum Ideogram4SampleBuilder {
    public static let sequencePaddingIndicator = -1
    public static let outputImageIndicator = 2
    public static let llmTokenIndicator = 3
    public static let imagePositionOffset = 65_536
    public static let latentChannels = 32
    public static let latentPatchSize = 2
    public static let vaeScaleFactor = 8
    public static let imageTokenPixelSize = latentPatchSize * vaeScaleFactor

    public enum SampleError: LocalizedError {
        case unsupportedBatch(Int)
        case invalidImageSize(width: Int, height: Int)
        case mismatchedLatentTokenCount(expected: Int, actual: Int)
        case mismatchedLatentChannels(expected: Int, actual: Int)

        public var errorDescription: String? {
            switch self {
            case .unsupportedBatch(let batch):
                return "Ideogram 4 packing currently supports a single sample; received batch \(batch)."
            case .invalidImageSize(let width, let height):
                return "Ideogram 4 image size must be positive and divisible by \(Ideogram4SampleBuilder.imageTokenPixelSize); received \(width)x\(height)."
            case .mismatchedLatentTokenCount(let expected, let actual):
                return "Ideogram 4 latent token count mismatch: expected \(expected), received \(actual)."
            case .mismatchedLatentChannels(let expected, let actual):
                return "Ideogram 4 latent channel mismatch: expected \(expected), received \(actual)."
            }
        }
    }

    public static func zeroLatents(
        imageWidth: Int,
        imageHeight: Int,
        inChannels: Int = latentChannels * latentPatchSize * latentPatchSize,
        dtype: DType = .bfloat16
    ) throws -> MLXArray {
        let grid = try imageTokenGrid(width: imageWidth, height: imageHeight)
        return MLX.zeros([1, grid.height * grid.width, inChannels], dtype: dtype)
    }

    public static func textToImageSample(
        llmFeatures: MLXArray,
        imageWidth: Int,
        imageHeight: Int,
        inChannels: Int = latentChannels * latentPatchSize * latentPatchSize
    ) throws -> Ideogram4PackedSample {
        let latents = try zeroLatents(
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            inChannels: inChannels,
            dtype: llmFeatures.dtype
        )
        return try pack(
            llmFeatures: llmFeatures,
            imageLatents: latents,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            inChannels: inChannels
        )
    }

    public static func pack(
        llmFeatures: MLXArray,
        imageLatents: MLXArray,
        imageWidth: Int,
        imageHeight: Int,
        inChannels: Int = latentChannels * latentPatchSize * latentPatchSize
    ) throws -> Ideogram4PackedSample {
        let grid = try imageTokenGrid(width: imageWidth, height: imageHeight)
        let imageTokenCount = grid.height * grid.width
        guard llmFeatures.dim(0) == 1 else {
            throw SampleError.unsupportedBatch(llmFeatures.dim(0))
        }
        guard imageLatents.dim(0) == 1 else {
            throw SampleError.unsupportedBatch(imageLatents.dim(0))
        }
        guard imageLatents.dim(1) == imageTokenCount else {
            throw SampleError.mismatchedLatentTokenCount(expected: imageTokenCount, actual: imageLatents.dim(1))
        }
        guard imageLatents.dim(2) == inChannels else {
            throw SampleError.mismatchedLatentChannels(expected: inChannels, actual: imageLatents.dim(2))
        }

        let textTokenCount = llmFeatures.dim(1)
        let llmFeatureDim = llmFeatures.dim(2)
        let textX = MLX.zeros([1, textTokenCount, inChannels], dtype: imageLatents.dtype)
        let x = MLX.concatenated([textX, imageLatents], axis: 1)

        let imageFeatures = MLX.zeros([1, imageTokenCount, llmFeatureDim], dtype: llmFeatures.dtype)
        let packedFeatures = MLX.concatenated([llmFeatures, imageFeatures], axis: 1)

        let totalTokenCount = textTokenCount + imageTokenCount
        var indicators = [Int32](repeating: Int32(llmTokenIndicator), count: textTokenCount)
        indicators.append(contentsOf: Array(repeating: Int32(outputImageIndicator), count: imageTokenCount))

        let positionIds = buildPositionIds(
            textTokenCount: textTokenCount,
            imageTokenHeight: grid.height,
            imageTokenWidth: grid.width
        )

        return Ideogram4PackedSample(
            llmFeatures: packedFeatures,
            x: x,
            positionIds: positionIds,
            segmentIds: MLXArray([Int32](repeating: 1, count: totalTokenCount)).reshaped(1, totalTokenCount),
            indicator: MLXArray(indicators).reshaped(1, totalTokenCount),
            textTokenCount: textTokenCount,
            imageTokenCount: imageTokenCount,
            imageTokenHeight: grid.height,
            imageTokenWidth: grid.width
        )
    }

    public static func imageTokens(from transformerOutput: MLXArray, sample: Ideogram4PackedSample) -> MLXArray {
        let start = sample.textTokenCount
        let end = start + sample.imageTokenCount
        return transformerOutput[0..., start..<end, 0...]
    }

    private static func imageTokenGrid(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0,
              height > 0,
              width % imageTokenPixelSize == 0,
              height % imageTokenPixelSize == 0 else {
            throw SampleError.invalidImageSize(width: width, height: height)
        }
        return (width / imageTokenPixelSize, height / imageTokenPixelSize)
    }

    private static func buildPositionIds(
        textTokenCount: Int,
        imageTokenHeight: Int,
        imageTokenWidth: Int
    ) -> MLXArray {
        let imageTokenCount = imageTokenHeight * imageTokenWidth
        let totalTokenCount = textTokenCount + imageTokenCount
        var axes = [[Int32]](repeating: [], count: 3)
        for axis in 0..<3 {
            axes[axis].reserveCapacity(totalTokenCount)
        }

        for index in 0..<textTokenCount {
            let value = Int32(index)
            axes[0].append(value)
            axes[1].append(value)
            axes[2].append(value)
        }

        for row in 0..<imageTokenHeight {
            for column in 0..<imageTokenWidth {
                axes[0].append(Int32(imagePositionOffset))
                axes[1].append(Int32(imagePositionOffset + row))
                axes[2].append(Int32(imagePositionOffset + column))
            }
        }

        return MLXArray(axes.flatMap { $0 }).reshaped(3, 1, totalTokenCount)
    }
}
