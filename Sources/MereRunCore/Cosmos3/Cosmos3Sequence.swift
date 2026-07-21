import Foundation
import MLX

public struct Cosmos3PositionIDs: Hashable, Sendable {
    public let temporal: [Float]
    public let height: [Float]
    public let width: [Float]
    public let nextTemporalOffset: Int

    public init(temporal: [Float], height: [Float], width: [Float], nextTemporalOffset: Int) {
        precondition(temporal.count == height.count && height.count == width.count)
        self.temporal = temporal
        self.height = height
        self.width = width
        self.nextTemporalOffset = nextTemporalOffset
    }

    public var count: Int { temporal.count }

    public var mlxArray: MLXArray {
        MLX.stacked([MLXArray(temporal), MLXArray(height), MLXArray(width)], axis: 0)
    }
}

public enum Cosmos3SequenceLayout {
    public static func textPositionIDs(
        tokenCount: Int,
        temporalOffset: Float = 0
    ) -> Cosmos3PositionIDs {
        precondition(tokenCount >= 0)
        let values = (0..<tokenCount).map { temporalOffset + Float($0) }
        return Cosmos3PositionIDs(
            temporal: values,
            height: values,
            width: values,
            nextTemporalOffset: Int(ceil(temporalOffset + Float(tokenCount)))
        )
    }

    public static func vaePositionIDs(
        frames: Int,
        height: Int,
        width: Int,
        temporalOffset: Float,
        resetSpatialIndices: Bool = true,
        fps: Float? = nil,
        baseFPS: Float = 24,
        temporalCompressionFactor: Int = 4,
        baseTemporalCompressionFactor: Int? = nil,
        startFrameOffset: Int = 0
    ) -> Cosmos3PositionIDs {
        precondition(frames > 0 && height > 0 && width > 0)
        precondition(temporalCompressionFactor > 0)
        var temporal: [Float] = []
        var vertical: [Float] = []
        var horizontal: [Float] = []
        temporal.reserveCapacity(frames * height * width)
        vertical.reserveCapacity(frames * height * width)
        horizontal.reserveCapacity(frames * height * width)
        let fpsModulated = fps != nil && frames > 1
        let baseCompression = baseTemporalCompressionFactor ?? temporalCompressionFactor
        for frame in 0..<frames {
            let temporalPosition: Float
            if let fps, fpsModulated {
                let tokensPerSecond = fps / Float(temporalCompressionFactor)
                let baseTokensPerSecond = baseFPS / Float(baseCompression)
                temporalPosition = (Float(frame + startFrameOffset) / tokensPerSecond)
                    * baseTokensPerSecond + temporalOffset
            } else {
                temporalPosition = Float(frame + startFrameOffset) + temporalOffset
            }
            for row in 0..<height {
                for column in 0..<width {
                    temporal.append(temporalPosition)
                    let spatialOffset = resetSpatialIndices ? 0 : Int(temporalOffset)
                    vertical.append(Float(row + spatialOffset))
                    horizontal.append(Float(column + spatialOffset))
                }
            }
        }
        let maximum = max(
            temporal.max() ?? 0,
            vertical.max() ?? 0,
            horizontal.max() ?? 0
        )
        return Cosmos3PositionIDs(
            temporal: temporal,
            height: vertical,
            width: horizontal,
            nextTemporalOffset: Int(ceil(maximum)) + 1
        )
    }
}

public struct Cosmos3VisionPatchLayout: Hashable, Sendable {
    public let frames: Int
    public let originalHeight: Int
    public let originalWidth: Int
    public let paddedHeight: Int
    public let paddedWidth: Int
    public let patchSize: Int

    public var patchHeight: Int { paddedHeight / patchSize }
    public var patchWidth: Int { paddedWidth / patchSize }
    public var tokenCount: Int { frames * patchHeight * patchWidth }
}

public enum Cosmos3VisionPatches {
    public static func pack(
        _ latent: MLXArray,
        patchSize: Int = 2
    ) -> (tokens: MLXArray, layout: Cosmos3VisionPatchLayout) {
        precondition(latent.ndim == 4)
        precondition(patchSize > 0)
        let channels = latent.dim(0)
        let frames = latent.dim(1)
        let height = latent.dim(2)
        let width = latent.dim(3)
        let paddedHeight = ((height + patchSize - 1) / patchSize) * patchSize
        let paddedWidth = ((width + patchSize - 1) / patchSize) * patchSize
        let padded = MLX.padded(latent, widths: [
            [0, 0],
            [0, 0],
            [0, paddedHeight - height],
            [0, paddedWidth - width],
        ])
        let patchHeight = paddedHeight / patchSize
        let patchWidth = paddedWidth / patchSize
        let tokens = padded
            .reshaped(channels, frames, patchHeight, patchSize, patchWidth, patchSize)
            .transposed(1, 2, 4, 3, 5, 0)
            .reshaped(frames * patchHeight * patchWidth, patchSize * patchSize * channels)
        return (
            tokens,
            Cosmos3VisionPatchLayout(
                frames: frames,
                originalHeight: height,
                originalWidth: width,
                paddedHeight: paddedHeight,
                paddedWidth: paddedWidth,
                patchSize: patchSize
            )
        )
    }

    public static func unpack(
        _ tokens: MLXArray,
        layout: Cosmos3VisionPatchLayout,
        channels: Int
    ) -> MLXArray {
        precondition(tokens.shape == [layout.tokenCount, layout.patchSize * layout.patchSize * channels])
        return tokens
            .reshaped(
                layout.frames,
                layout.patchHeight,
                layout.patchWidth,
                layout.patchSize,
                layout.patchSize,
                channels
            )
            .transposed(5, 0, 1, 3, 2, 4)
            .reshaped(channels, layout.frames, layout.paddedHeight, layout.paddedWidth)[
                0..., 0..., 0..<layout.originalHeight, 0..<layout.originalWidth
            ]
    }
}
