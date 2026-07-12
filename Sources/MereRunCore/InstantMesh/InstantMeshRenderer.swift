import Foundation
@preconcurrency import MLX

public struct InstantMeshFieldQuery {
    /// Raw learned SDF. Positive values are the reconstructed interior.
    public let signedDistance: MLXArray
    /// Raw decoder displacement before the inference-time tanh bound.
    public let rawDeformation: MLXArray
    /// `tanh(raw) / (gridResolution * deformationDivisor)`.
    public let deformation: MLXArray
    /// Upstream's clamped-sigmoid RGB, normally within `[0, 1]`.
    public let color: MLXArray
    let sampledFeatures: MLXArray
}

public enum InstantMeshRenderer {
    /// Queries the exact triplane SDF/deformation/color heads in bounded chunks.
    public static func query(
        model: InstantMeshModel,
        sceneCode: InstantMeshSceneCode,
        sceneIndex: Int = 0,
        positions: MLXArray,
        chunkSize: Int? = nil
    ) -> InstantMeshFieldQuery {
        precondition(positions.ndim >= 2 && positions.dim(-1) == 3)
        precondition(sceneIndex >= 0 && sceneIndex < sceneCode.planes.dim(0))
        let inputShape = positions.shape
        let flattened = positions.reshaped(-1, 3)
        let count = flattened.dim(0)
        let effectiveChunk = min(
            max(1, chunkSize ?? model.memoryConfiguration.fieldQueryChunkSize),
            max(1, count)
        )
        var distances: [MLXArray] = []
        var rawDeformations: [MLXArray] = []
        var deformations: [MLXArray] = []
        var colors: [MLXArray] = []
        var features: [MLXArray] = []
        let capacity = (count + effectiveChunk - 1) / effectiveChunk
        distances.reserveCapacity(capacity)
        rawDeformations.reserveCapacity(capacity)
        deformations.reserveCapacity(capacity)
        colors.reserveCapacity(capacity)
        features.reserveCapacity(capacity)

        let planes = sceneCode.planes[sceneIndex]
        for start in stride(from: 0, to: count, by: effectiveChunk) {
            let end = min(start + effectiveChunk, count)
            let query = queryChunk(
                decoder: model.decoder,
                configuration: model.configuration,
                planes: planes,
                positions: flattened[start..<end, 0...]
            )
            MLX.eval(
                query.signedDistance,
                query.rawDeformation,
                query.deformation,
                query.color,
                query.sampledFeatures
            )
            distances.append(query.signedDistance)
            rawDeformations.append(query.rawDeformation)
            deformations.append(query.deformation)
            colors.append(query.color)
            features.append(query.sampledFeatures)
        }

        let prefix = Array(inputShape.dropLast())
        return InstantMeshFieldQuery(
            signedDistance: MLX.concatenated(distances).reshaped(prefix + [1]),
            rawDeformation: MLX.concatenated(rawDeformations).reshaped(prefix + [3]),
            deformation: MLX.concatenated(deformations).reshaped(prefix + [3]),
            color: MLX.concatenated(colors).reshaped(prefix + [3]),
            sampledFeatures: MLX.concatenated(features).reshaped(prefix + [model.configuration.decoderInputSize])
        )
    }

    /// Evaluates the checkpoint's 21 FlexiCubes cell-weight logits. These
    /// learned values are exposed for parity/provenance, but mere.run's native
    /// permissive extractor does not port NVIDIA's proprietary implementation.
    public static func cellWeights(
        model: InstantMeshModel,
        sceneCode: InstantMeshSceneCode,
        sceneIndex: Int = 0,
        gridPositions: MLXArray,
        cornerIndices: MLXArray,
        chunkSize: Int? = nil
    ) -> MLXArray {
        precondition(gridPositions.ndim == 2 && gridPositions.dim(1) == 3)
        precondition(cornerIndices.ndim == 2 && cornerIndices.dim(1) == 8)
        let query = self.query(
            model: model,
            sceneCode: sceneCode,
            sceneIndex: sceneIndex,
            positions: gridPositions,
            chunkSize: chunkSize
        )
        let corners = MLX.take(
            query.sampledFeatures,
            cornerIndices.asType(.int32),
            axis: 0
        )
        return model.decoder.cellWeights(corners)
    }

    private static func queryChunk(
        decoder: InstantMeshDecoder,
        configuration: InstantMeshConfiguration,
        planes: MLXArray,
        positions: MLXArray
    ) -> InstantMeshFieldQuery {
        // The official plane bases project onto XY, XZ, and ZY.
        let planePairs = [(0, 1), (0, 2), (2, 1)]
        var sampled: [MLXArray] = []
        sampled.reserveCapacity(3)
        for (planeIndex, pair) in planePairs.enumerated() {
            sampled.append(bilinearGridSample(
                plane: planes[planeIndex],
                x: positions[0..., pair.0],
                y: positions[0..., pair.1]
            ))
        }
        let features = MLX.concatenated(sampled, axis: -1)
        let rawDeformation = decoder.deformation(features)
        let deformationScale = 1 / (
            Float(configuration.gridResolution) * configuration.deformationDivisor
        )
        return InstantMeshFieldQuery(
            signedDistance: decoder.signedDistance(features),
            rawDeformation: rawDeformation,
            deformation: MLX.tanh(rawDeformation) * deformationScale,
            color: decoder.color(features),
            sampledFeatures: features
        )
    }

    /// PyTorch `grid_sample` bilinear mode with `align_corners=false` and zero
    /// padding, for one HWC plane and a flat list of normalized coordinates.
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
            plane: plane, width: width, height: height, channels: channels, x: x0, y: y0
        ) * ((x1 - sourceX) * (y1 - sourceY)).expandedDimensions(axis: -1)
        let topRight = sample(
            plane: plane, width: width, height: height, channels: channels, x: x1, y: y0
        ) * ((sourceX - x0) * (y1 - sourceY)).expandedDimensions(axis: -1)
        let bottomLeft = sample(
            plane: plane, width: width, height: height, channels: channels, x: x0, y: y1
        ) * ((x1 - sourceX) * (sourceY - y0)).expandedDimensions(axis: -1)
        let bottomRight = sample(
            plane: plane, width: width, height: height, channels: channels, x: x1, y: y1
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
