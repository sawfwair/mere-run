import Foundation

/// Normalized trilinear sampler over TRELLIS.2's sparse six-channel PBR
/// field. Points are continuous voxel-space coordinates (voxel index plus
/// fractional offset). Weights renormalize over the occupied corners of the
/// containing cell so shell-boundary samples keep full magnitude; a point
/// whose cell has no occupied corner falls back to the nearest occupied
/// voxel in its 3x3x3 neighborhood.
struct Trellis2SparseFieldSampler {
    let attributes: [Float]
    let coordinateIndices: [Trellis2VoxelCoordinate: Int]

    init(coordinates: [Trellis2VoxelCoordinate], attributes: [Float]) {
        self.attributes = attributes
        self.coordinateIndices = Dictionary(
            uniqueKeysWithValues: coordinates.enumerated().map { ($1, $0) }
        )
    }

    init(attributes: [Float], coordinateIndices: [Trellis2VoxelCoordinate: Int]) {
        self.attributes = attributes
        self.coordinateIndices = coordinateIndices
    }

    /// Six channels at the point, or nil when no corner of the containing
    /// interpolation cell is occupied.
    func trilinear(x: Float, y: Float, z: Float) -> [Float]? {
        let baseX = Int32(floor(x)), baseY = Int32(floor(y)), baseZ = Int32(floor(z))
        let fractionX = x - Float(baseX), fractionY = y - Float(baseY), fractionZ = z - Float(baseZ)
        var values = [Float](repeating: 0, count: 6)
        var occupiedWeight: Float = 0
        for corner in 0..<8 {
            let dx = Int32(corner & 1)
            let dy = Int32((corner >> 1) & 1)
            let dz = Int32((corner >> 2) & 1)
            let weight = (dx == 0 ? 1 - fractionX : fractionX)
                * (dy == 0 ? 1 - fractionY : fractionY)
                * (dz == 0 ? 1 - fractionZ : fractionZ)
            guard let source = coordinateIndices[
                Trellis2VoxelCoordinate(x: baseX + dx, y: baseY + dy, z: baseZ + dz)
            ] else { continue }
            occupiedWeight += weight
            for channel in 0..<6 {
                values[channel] += attributes[source * 6 + channel] * weight
            }
        }
        guard occupiedWeight > 1e-6 else { return nil }
        for channel in 0..<6 { values[channel] /= occupiedWeight }
        return values
    }

    /// Trilinear sample with a nearest-occupied-neighbor fallback; zeros when
    /// nothing in the 3x3x3 neighborhood is occupied.
    func sample(x: Float, y: Float, z: Float) -> [Float] {
        if let values = trilinear(x: x, y: y, z: z) { return values }
        let nearestX = Int32(x.rounded()), nearestY = Int32(y.rounded()), nearestZ = Int32(z.rounded())
        var best: (distance: Float, index: Int)?
        for dx in Int32(-1)...1 {
            for dy in Int32(-1)...1 {
                for dz in Int32(-1)...1 {
                    let coordinate = Trellis2VoxelCoordinate(
                        x: nearestX + dx,
                        y: nearestY + dy,
                        z: nearestZ + dz
                    )
                    guard let index = coordinateIndices[coordinate] else { continue }
                    let deltaX = Float(coordinate.x) - x
                    let deltaY = Float(coordinate.y) - y
                    let deltaZ = Float(coordinate.z) - z
                    let distance = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ
                    if best == nil || distance < best!.distance {
                        best = (distance, index)
                    }
                }
            }
        }
        guard let best else { return [Float](repeating: 0, count: 6) }
        return (0..<6).map { attributes[best.index * 6 + $0] }
    }
}
