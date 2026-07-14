import Foundation

public enum PointCloudAssetError: Error, Equatable, LocalizedError, Sendable {
    case empty
    case invalidElementCount(field: String, expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .empty:
            "A point cloud must contain at least one point."
        case .invalidElementCount(let field, let expected, let actual):
            "Point-cloud field '\(field)' expected \(expected) values but received \(actual)."
        }
    }
}

/// Canonical colored point cloud for native multi-view geometry handoffs.
public struct PointCloudAsset: Sendable {
    public let positions: [Float]
    /// sRGB-encoded RGB with linear alpha coverage; see `VertexColorTransfer`.
    public let colorsRGBA8: [UInt8]?
    public let confidence: [Float]?
    public let viewIndices: [UInt32]?
    public let coordinateSystem: GeometryCoordinateSystem
    public let units: GeometryValueUnits

    public init(
        positions: [Float],
        colorsRGBA8: [UInt8]? = nil,
        confidence: [Float]? = nil,
        viewIndices: [UInt32]? = nil,
        coordinateSystem: GeometryCoordinateSystem = .worldFromCameras,
        units: GeometryValueUnits = .relative
    ) throws {
        guard !positions.isEmpty else { throw PointCloudAssetError.empty }
        guard positions.count.isMultiple(of: 3) else {
            throw PointCloudAssetError.invalidElementCount(
                field: "positions",
                expected: positions.count / 3 * 3,
                actual: positions.count
            )
        }
        let count = positions.count / 3
        if let colorsRGBA8, colorsRGBA8.count != count * 4 {
            throw PointCloudAssetError.invalidElementCount(
                field: "colors",
                expected: count * 4,
                actual: colorsRGBA8.count
            )
        }
        if let confidence, confidence.count != count {
            throw PointCloudAssetError.invalidElementCount(
                field: "confidence",
                expected: count,
                actual: confidence.count
            )
        }
        if let viewIndices, viewIndices.count != count {
            throw PointCloudAssetError.invalidElementCount(
                field: "viewIndices",
                expected: count,
                actual: viewIndices.count
            )
        }
        self.positions = positions
        self.colorsRGBA8 = colorsRGBA8
        self.confidence = confidence
        self.viewIndices = viewIndices
        self.coordinateSystem = coordinateSystem
        self.units = units
    }

    public var pointCount: Int { positions.count / 3 }

    public var bounds: MeshBounds {
        var minimum = [Float](repeating: .infinity, count: 3)
        var maximum = [Float](repeating: -.infinity, count: 3)
        for point in 0..<pointCount {
            for axis in 0..<3 {
                let value = positions[point * 3 + axis]
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
            }
        }
        return MeshBounds(minimum: minimum, maximum: maximum)
    }
}
