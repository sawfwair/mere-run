import Foundation

public enum MeshCoordinateSystem: String, Codable, CaseIterable, Sendable {
    case modelXRightYUpZForward = "model-x-right-y-up-z-forward"
}

public enum MeshUnits: String, Codable, CaseIterable, Sendable {
    case normalizedObjectSpace = "normalized-object-space"
    case meters
}

public enum MeshAssetError: Error, Equatable, LocalizedError, Sendable {
    case emptyVertices
    case invalidElementCount(field: String, expected: Int, actual: Int)
    case invalidTriangleIndexCount(Int)
    case indexOutOfBounds(index: UInt32, vertexCount: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyVertices:
            "A mesh must contain at least one vertex."
        case .invalidElementCount(let field, let expected, let actual):
            "Mesh field '\(field)' expected \(expected) values but received \(actual)."
        case .invalidTriangleIndexCount(let count):
            "Mesh index count must be divisible by three; received \(count)."
        case .indexOutOfBounds(let index, let vertexCount):
            "Mesh index \(index) is outside the \(vertexCount)-vertex buffer."
        }
    }
}

public struct MeshBounds: Codable, Equatable, Sendable {
    public let minimum: [Float]
    public let maximum: [Float]

    public init(minimum: [Float], maximum: [Float]) {
        self.minimum = minimum
        self.maximum = maximum
    }
}

/// Uniform PBR scalars for an exported material. glTF viewers multiply these
/// with per-vertex color, so they carry the field-level metallic/roughness
/// that the vertex-color mesh contract cannot express per vertex.
public struct MeshPBRMaterialFactors: Codable, Equatable, Sendable {
    public let metallicFactor: Float
    public let roughnessFactor: Float

    public init(metallicFactor: Float, roughnessFactor: Float) {
        self.metallicFactor = min(max(metallicFactor, 0), 1)
        self.roughnessFactor = min(max(roughnessFactor, 0), 1)
    }
}

/// Canonical indexed triangle mesh used by native object reconstruction.
public struct MeshAsset: Sendable {
    public let vertices: [Float]
    public let indices: [UInt32]
    public let normals: [Float]?
    /// sRGB-encoded RGB with linear alpha coverage; see `VertexColorTransfer`.
    public let colorsRGBA8: [UInt8]?
    public let textureCoordinates: [Float]?
    public let coordinateSystem: MeshCoordinateSystem
    public let units: MeshUnits
    public let inferredUnseenGeometry: Bool

    public init(
        vertices: [Float],
        indices: [UInt32],
        normals: [Float]? = nil,
        colorsRGBA8: [UInt8]? = nil,
        textureCoordinates: [Float]? = nil,
        coordinateSystem: MeshCoordinateSystem = .modelXRightYUpZForward,
        units: MeshUnits = .normalizedObjectSpace,
        inferredUnseenGeometry: Bool
    ) throws {
        guard !vertices.isEmpty else { throw MeshAssetError.emptyVertices }
        guard vertices.count.isMultiple(of: 3) else {
            throw MeshAssetError.invalidElementCount(field: "vertices", expected: (vertices.count / 3) * 3, actual: vertices.count)
        }
        let vertexCount = vertices.count / 3
        guard indices.count.isMultiple(of: 3) else {
            throw MeshAssetError.invalidTriangleIndexCount(indices.count)
        }
        if let invalid = indices.first(where: { Int($0) >= vertexCount }) {
            throw MeshAssetError.indexOutOfBounds(index: invalid, vertexCount: vertexCount)
        }
        if let normals, normals.count != vertices.count {
            throw MeshAssetError.invalidElementCount(field: "normals", expected: vertices.count, actual: normals.count)
        }
        if let colorsRGBA8, colorsRGBA8.count != vertexCount * 4 {
            throw MeshAssetError.invalidElementCount(field: "colors", expected: vertexCount * 4, actual: colorsRGBA8.count)
        }
        if let textureCoordinates, textureCoordinates.count != vertexCount * 2 {
            throw MeshAssetError.invalidElementCount(field: "textureCoordinates", expected: vertexCount * 2, actual: textureCoordinates.count)
        }
        self.vertices = vertices
        self.indices = indices
        self.normals = normals
        self.colorsRGBA8 = colorsRGBA8
        self.textureCoordinates = textureCoordinates
        self.coordinateSystem = coordinateSystem
        self.units = units
        self.inferredUnseenGeometry = inferredUnseenGeometry
    }

    public var vertexCount: Int { vertices.count / 3 }
    public var triangleCount: Int { indices.count / 3 }

    public var bounds: MeshBounds {
        var minimum = [Float](repeating: .infinity, count: 3)
        var maximum = [Float](repeating: -.infinity, count: 3)
        for vertex in 0..<vertexCount {
            for axis in 0..<3 {
                let value = vertices[vertex * 3 + axis]
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
            }
        }
        return MeshBounds(minimum: minimum, maximum: maximum)
    }

    public func withGeneratedNormals() throws -> MeshAsset {
        try MeshAsset(
            vertices: vertices,
            indices: indices,
            normals: MeshNormals.generate(vertices: vertices, indices: indices),
            colorsRGBA8: colorsRGBA8,
            textureCoordinates: textureCoordinates,
            coordinateSystem: coordinateSystem,
            units: units,
            inferredUnseenGeometry: inferredUnseenGeometry
        )
    }
}

public enum MeshNormals {
    public static func generate(vertices: [Float], indices: [UInt32]) -> [Float] {
        var normals = [Float](repeating: 0, count: vertices.count)
        for triangle in stride(from: 0, to: indices.count, by: 3) {
            let a = Int(indices[triangle]) * 3
            let b = Int(indices[triangle + 1]) * 3
            let c = Int(indices[triangle + 2]) * 3
            let ab = (vertices[b] - vertices[a], vertices[b + 1] - vertices[a + 1], vertices[b + 2] - vertices[a + 2])
            let ac = (vertices[c] - vertices[a], vertices[c + 1] - vertices[a + 1], vertices[c + 2] - vertices[a + 2])
            let cross = (
                ab.1 * ac.2 - ab.2 * ac.1,
                ab.2 * ac.0 - ab.0 * ac.2,
                ab.0 * ac.1 - ab.1 * ac.0
            )
            for base in [a, b, c] {
                normals[base] += cross.0
                normals[base + 1] += cross.1
                normals[base + 2] += cross.2
            }
        }
        for vertex in 0..<(vertices.count / 3) {
            let base = vertex * 3
            let length = sqrt(
                normals[base] * normals[base]
                + normals[base + 1] * normals[base + 1]
                + normals[base + 2] * normals[base + 2]
            )
            if length > Float.ulpOfOne {
                normals[base] /= length
                normals[base + 1] /= length
                normals[base + 2] /= length
            }
        }
        return normals
    }
}
