import Foundation
@preconcurrency import MLX

public struct TripoSRFieldQuery {
    /// Raw decoder density, shaped like `positions.shape.dropLast() + [1]`.
    public let density: MLXArray
    /// `exp(density + densityBias)`, used for rendering and mesh extraction.
    public let activatedDensity: MLXArray
    /// Sigmoid RGB vertex/radiance color in `[0, 1]`.
    public let color: MLXArray
}

public enum TripoSRRenderer {
    /// Query the learned triplane field at object-space positions.
    ///
    /// Positions are in `[-rendererRadius, rendererRadius]`. Sampling exactly
    /// matches PyTorch `grid_sample(..., mode="bilinear", align_corners=False,
    /// padding_mode="zeros")`, including half-pixel coordinates and zero
    /// padding outside the scene planes.
    public static func query(
        model: TripoSRModel,
        sceneCode: TripoSRSceneCode,
        sceneIndex: Int = 0,
        positions: MLXArray,
        chunkSize: Int? = nil
    ) -> TripoSRFieldQuery {
        precondition(positions.ndim >= 2 && positions.dim(-1) == 3)
        precondition(sceneIndex >= 0 && sceneIndex < sceneCode.planes.dim(0))
        let inputShape = positions.shape
        let flattened = positions.reshaped(-1, 3)
        let count = flattened.dim(0)
        let effectiveChunk = min(max(1, chunkSize ?? count), max(1, count))
        var densities: [MLXArray] = []
        var activated: [MLXArray] = []
        var colors: [MLXArray] = []
        densities.reserveCapacity((count + effectiveChunk - 1) / effectiveChunk)
        activated.reserveCapacity(densities.capacity)
        colors.reserveCapacity(densities.capacity)

        let planes = sceneCode.planes[sceneIndex]
        for start in stride(from: 0, to: count, by: effectiveChunk) {
            let end = min(start + effectiveChunk, count)
            let query = queryChunk(
                decoder: model.decoder,
                configuration: model.configuration,
                planes: planes,
                positions: flattened[start..<end, 0...]
            )
            densities.append(query.density)
            activated.append(query.activatedDensity)
            colors.append(query.color)
        }

        let outputPrefix = Array(inputShape.dropLast())
        return TripoSRFieldQuery(
            density: concatenated(densities).reshaped(outputPrefix + [1]),
            activatedDensity: concatenated(activated).reshaped(outputPrefix + [1]),
            color: concatenated(colors).reshaped(outputPrefix + [3])
        )
    }

    private static func queryChunk(
        decoder: TripoSRNeRFDecoder,
        configuration: TripoSRConfiguration,
        planes: MLXArray,
        positions: MLXArray
    ) -> TripoSRFieldQuery {
        let normalized = positions / configuration.rendererRadius
        let planePairs = [(0, 1), (0, 2), (1, 2)]
        var sampled: [MLXArray] = []
        sampled.reserveCapacity(3)
        for (planeIndex, pair) in planePairs.enumerated() {
            sampled.append(bilinearGridSample(
                plane: planes[planeIndex],
                x: normalized[0..., pair.0],
                y: normalized[0..., pair.1]
            ))
        }
        let decoded = decoder(MLX.concatenated(sampled, axis: -1))
        let density = decoded[0..., 0..<1]
        let color = MLX.sigmoid(decoded[0..., 1..<4])
        return TripoSRFieldQuery(
            density: density,
            activatedDensity: MLX.exp(density + configuration.densityBias),
            color: color
        )
    }

    static func bilinearGridSample(
        plane: MLXArray,
        x: MLXArray,
        y: MLXArray
    ) -> MLXArray {
        precondition(plane.ndim == 3 && x.ndim == 1 && y.ndim == 1 && x.shape == y.shape)
        let height = plane.dim(0)
        let width = plane.dim(1)
        let channels = plane.dim(2)
        let sourceX = ((x + 1) * Float(width) - 1) / 2
        let sourceY = ((y + 1) * Float(height) - 1) / 2
        let x0 = MLX.floor(sourceX)
        let y0 = MLX.floor(sourceY)
        let x1 = x0 + 1
        let y1 = y0 + 1

        let topLeft = sample(
            plane: plane,
            width: width,
            height: height,
            channels: channels,
            x: x0,
            y: y0
        ) * ((x1 - sourceX) * (y1 - sourceY)).expandedDimensions(axis: -1)
        let topRight = sample(
            plane: plane,
            width: width,
            height: height,
            channels: channels,
            x: x1,
            y: y0
        ) * ((sourceX - x0) * (y1 - sourceY)).expandedDimensions(axis: -1)
        let bottomLeft = sample(
            plane: plane,
            width: width,
            height: height,
            channels: channels,
            x: x0,
            y: y1
        ) * ((x1 - sourceX) * (sourceY - y0)).expandedDimensions(axis: -1)
        let bottomRight = sample(
            plane: plane,
            width: width,
            height: height,
            channels: channels,
            x: x1,
            y: y1
        ) * ((sourceX - x0) * (sourceY - y0)).expandedDimensions(axis: -1)
        return topLeft + topRight + bottomLeft + bottomRight
    }

    private static func sample(
        plane: MLXArray,
        width: Int,
        height: Int,
        channels: Int,
        x: MLXArray,
        y: MLXArray
    ) -> MLXArray {
        let valid = (x .>= MLXArray(Float(0)))
            .&& (x .< MLXArray(Float(width)))
            .&& (y .>= MLXArray(Float(0)))
            .&& (y .< MLXArray(Float(height)))
        let clampedX = MLX.clip(x, min: 0, max: Float(width - 1)).asType(.int32)
        let clampedY = MLX.clip(y, min: 0, max: Float(height - 1)).asType(.int32)
        let indices = clampedY * Int32(width) + clampedX
        let gathered = plane.reshaped(height * width, channels).take(indices, axis: 0)
        return gathered * valid.asType(plane.dtype).expandedDimensions(axis: -1)
    }
}
