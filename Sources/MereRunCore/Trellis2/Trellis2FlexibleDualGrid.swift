import Foundation
@preconcurrency import MLX

public struct Trellis2PBRVoxelGrid: Sendable {
    public let resolution: Int
    public let coordinates: [Trellis2VoxelCoordinate]
    /// Six row-major channels: base color RGB, metallic, roughness, alpha.
    public let attributes: [Float]

    public init(
        resolution: Int,
        coordinates: [Trellis2VoxelCoordinate],
        attributes: [Float]
    ) throws {
        guard attributes.count == coordinates.count * 6 else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[\(coordinates.count), 6] PBR attributes",
                actual: [attributes.count]
            )
        }
        self.resolution = resolution
        self.coordinates = coordinates
        self.attributes = attributes
    }
}

public struct Trellis2DecodedAsset: Sendable {
    public let mesh: MeshAsset
    public let pbrVoxels: Trellis2PBRVoxelGrid
    public let metallic: [Float]
    public let roughness: [Float]

    public init(
        mesh: MeshAsset,
        pbrVoxels: Trellis2PBRVoxelGrid,
        metallic: [Float],
        roughness: [Float]
    ) {
        self.mesh = mesh
        self.pbrVoxels = pbrVoxels
        self.metallic = metallic
        self.roughness = roughness
    }
}

enum Trellis2FlexibleDualGrid {
    private static let edgeOffsets: [[Trellis2VoxelCoordinate]] = [
        [
            .init(x: 0, y: 0, z: 0), .init(x: 0, y: 0, z: 1),
            .init(x: 0, y: 1, z: 1), .init(x: 0, y: 1, z: 0),
        ],
        [
            .init(x: 0, y: 0, z: 0), .init(x: 1, y: 0, z: 0),
            .init(x: 1, y: 0, z: 1), .init(x: 0, y: 0, z: 1),
        ],
        [
            .init(x: 0, y: 0, z: 0), .init(x: 0, y: 1, z: 0),
            .init(x: 1, y: 1, z: 0), .init(x: 1, y: 0, z: 0),
        ],
    ]
    private static let splitOne = [0, 1, 2, 0, 2, 3]
    private static let splitTwo = [0, 1, 3, 3, 1, 2]

    static func decode(
        shape: Trellis2SparseTensor,
        texture: Trellis2SparseTensor,
        resolution: Int
    ) throws -> Trellis2DecodedAsset {
        guard shape.coordinates == texture.coordinates,
              shape.channels == 7,
              texture.channels == 6 else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "matching [tokens, 7] shape and [tokens, 6] texture voxels",
                actual: [shape.count, shape.channels, texture.count, texture.channels]
            )
        }
        MLX.eval(shape.features, texture.features)
        let shapeValues = shape.features.asType(.float32).asArray(Float.self)
        let rawTextureValues = texture.features.asType(.float32).asArray(Float.self)
        let textureValues = rawTextureValues.map { 0.5 * $0 + 0.5 }
        let coordinateIndices = Dictionary(
            uniqueKeysWithValues: shape.coordinates.enumerated().map { ($1, $0) }
        )

        var vertices = [Float]()
        var dualOffsets = [Float]()
        var intersections = [Bool]()
        var splitWeights = [Float]()
        vertices.reserveCapacity(shape.count * 3)
        dualOffsets.reserveCapacity(shape.count * 3)
        intersections.reserveCapacity(shape.count * 3)
        splitWeights.reserveCapacity(shape.count)
        for index in 0..<shape.count {
            let coordinate = shape.coordinates[index]
            var rawPosition = [Float]()
            rawPosition.reserveCapacity(3)
            for axis in 0..<3 {
                let offset = 2 * sigmoid(shapeValues[index * 7 + axis]) - 0.5
                dualOffsets.append(offset)
                rawPosition.append(
                    (Float(coordinate[axis]) + offset) / Float(resolution) - 0.5
                )
                intersections.append(shapeValues[index * 7 + 3 + axis] > 0)
            }
            // O-Voxel geometry is Z-up. The shared mesh exporters and glTF
            // contract use X-right, Y-up, Z-forward, so rotate into that
            // canonical basis without changing handedness.
            vertices.append(rawPosition[0])
            vertices.append(rawPosition[2])
            vertices.append(-rawPosition[1])
            splitWeights.append(softplus(shapeValues[index * 7 + 6]))
        }

        var indices = [UInt32]()
        for voxel in 0..<shape.count {
            let coordinate = shape.coordinates[voxel]
            for axis in 0..<3 where intersections[voxel * 3 + axis] {
                let connected = Self.edgeOffsets[axis].compactMap { offset -> Int? in
                    coordinateIndices[coordinate.offset(x: offset.x, y: offset.y, z: offset.z)]
                }
                guard connected.count == 4 else { continue }
                let diagonal02 = splitWeights[connected[0]] * splitWeights[connected[2]]
                let diagonal13 = splitWeights[connected[1]] * splitWeights[connected[3]]
                let split = diagonal02 > diagonal13 ? Self.splitOne : Self.splitTwo
                indices.append(contentsOf: split.map { UInt32(connected[$0]) })
            }
        }
        guard !indices.isEmpty else { throw Trellis2ModelError.emptyExtractedMesh }

        let sampledAttributes = sampleTextureAttributes(
            coordinates: texture.coordinates,
            attributes: textureValues,
            coordinateIndices: coordinateIndices,
            dualOffsets: dualOffsets
        )
        var colors = [UInt8]()
        var metallic = [Float]()
        var roughness = [Float]()
        colors.reserveCapacity(shape.count * 4)
        metallic.reserveCapacity(shape.count)
        roughness.reserveCapacity(shape.count)
        for vertex in 0..<shape.count {
            colors.append(byte(sampledAttributes[vertex * 6]))
            colors.append(byte(sampledAttributes[vertex * 6 + 1]))
            colors.append(byte(sampledAttributes[vertex * 6 + 2]))
            metallic.append(clamp(sampledAttributes[vertex * 6 + 3]))
            roughness.append(clamp(sampledAttributes[vertex * 6 + 4]))
            colors.append(byte(sampledAttributes[vertex * 6 + 5]))
        }

        let extractedMesh = try MeshAsset(
            vertices: vertices,
            indices: indices,
            colorsRGBA8: colors,
            inferredUnseenGeometry: true
        )
        let mesh = try Trellis2MeshHoleFiller.fillSmallHoles(in: extractedMesh)
        let voxels = try Trellis2PBRVoxelGrid(
            resolution: resolution,
            coordinates: texture.coordinates,
            attributes: textureValues
        )
        return Trellis2DecodedAsset(
            mesh: mesh,
            pbrVoxels: voxels,
            metallic: metallic,
            roughness: roughness
        )
    }

    private static func sampleTextureAttributes(
        coordinates: [Trellis2VoxelCoordinate],
        attributes: [Float],
        coordinateIndices: [Trellis2VoxelCoordinate: Int],
        dualOffsets: [Float]
    ) -> [Float] {
        var result = [Float](repeating: 0, count: coordinates.count * 6)
        for vertex in 0..<coordinates.count {
            let coordinate = coordinates[vertex]
            let sample = [
                Float(coordinate.x) + dualOffsets[vertex * 3],
                Float(coordinate.y) + dualOffsets[vertex * 3 + 1],
                Float(coordinate.z) + dualOffsets[vertex * 3 + 2],
            ]
            let base = sample.map { Int32(floor($0)) }
            let fraction = zip(sample, base).map { $0 - Float($1) }
            for corner in 0..<8 {
                let x = Int32(corner & 1)
                let y = Int32((corner >> 1) & 1)
                let z = Int32((corner >> 2) & 1)
                let weight = (x == 0 ? 1 - fraction[0] : fraction[0])
                    * (y == 0 ? 1 - fraction[1] : fraction[1])
                    * (z == 0 ? 1 - fraction[2] : fraction[2])
                let neighbor = Trellis2VoxelCoordinate(
                    x: base[0] + x,
                    y: base[1] + y,
                    z: base[2] + z
                )
                guard let source = coordinateIndices[neighbor] else { continue }
                for channel in 0..<6 {
                    result[vertex * 6 + channel] += attributes[source * 6 + channel] * weight
                }
            }
        }
        return result
    }

    private static func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private static func softplus(_ value: Float) -> Float {
        value > 20 ? value : log1p(exp(value))
    }

    private static func clamp(_ value: Float) -> Float {
        min(max(value, 0), 1)
    }

    private static func byte(_ value: Float) -> UInt8 {
        UInt8((255 * clamp(value)).rounded())
    }
}
