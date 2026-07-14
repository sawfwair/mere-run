import Foundation

public struct Trellis2RemeshConfiguration: Equatable, Sendable {
    /// Narrow-band half-width in voxel units; upstream's shipped app uses 1.
    public let band: Float
    /// Fraction to move dual vertices toward the closest crust point;
    /// upstream's shipped app uses 0 (surface stays on the band envelope).
    public let projectBack: Float
    /// Closed crust boundary loops up to this normalized perimeter are
    /// capped before the band field is built, so large clean rims (an
    /// occluded cheek pocket, for example) become sealed surface instead of
    /// tunnels through the envelope. Zero disables; colors never sample the
    /// synthetic lids because projection targets the uncapped crust.
    public let capBoundaryLoopPerimeter: Float
    /// Morphological-closing radius in voxels for the sealed inside/outside
    /// classification. Cavities whose every mouth is narrower than roughly
    /// twice this radius are classified interior even when their rims are
    /// tangled open chains no loop capping can close, and the remesh spans a
    /// membrane across the opening. Zero disables.
    public let sealRadius: Int

    public init(
        band: Float = 1,
        projectBack: Float = 0,
        capBoundaryLoopPerimeter: Float = 0.2,
        sealRadius: Int = 12
    ) {
        self.band = band
        self.projectBack = projectBack
        self.capBoundaryLoopPerimeter = capBoundaryLoopPerimeter
        self.sealRadius = sealRadius
    }
}

/// Native port of CuMesh's `remesh_narrow_band_dc` as invoked by upstream
/// TRELLIS.2's `to_glb`: the isosurface of `UDF(crust) - band` is extracted
/// with simple dual contouring (dual vertex = mean of edge crossings) over a
/// sparse octree-refined grid. The result is the closed envelope of the
/// porous crust — watertight by construction, sealing gaps narrower than
/// roughly twice the band — with quads emitted around sign-crossing grid
/// edges exactly like the flexible dual grid, so the atlas baker's quad
/// recovery applies unchanged.
enum Trellis2NarrowBandRemesher {
    static func remesh(
        mesh: MeshAsset,
        resolution: Int,
        configuration: Trellis2RemeshConfiguration = Trellis2RemeshConfiguration(),
        bvh providedBVH: Trellis2TriangleBVH? = nil,
        fieldCoordinates: [Trellis2VoxelCoordinate]? = nil
    ) throws -> MeshAsset {
        let bvh = providedBVH ?? Trellis2TriangleBVH(vertices: mesh.vertices, indices: mesh.indices)
        // Upstream inflates the domain so the band never clips the grid:
        // scale = (resolution + 3 * band) / resolution over the [-0.5, 0.5]
        // object cube, centered at the origin.
        let scale = (Float(resolution) + 3 * configuration.band) / Float(resolution)
        let eps = configuration.band * scale / Float(resolution)

        // Sealed inside/outside classification on this grid. The occupancy
        // arrives in the field's Z-up grid over the unscaled object cube;
        // convert each voxel center into a remesh-grid cell. The surface of
        // the union of the band envelope and the sealed solid spans
        // membranes across openings whose mouths are narrower than roughly
        // twice the seal radius.
        var classification: Trellis2SealedClassification?
        if let fieldCoordinates, configuration.sealRadius > 0 {
            var cells = [Int32]()
            cells.reserveCapacity(fieldCoordinates.count * 3)
            let fieldResolution = Float(resolution)
            for coordinate in fieldCoordinates {
                let rawX = (Float(coordinate.x) + 0.5) / fieldResolution - 0.5
                let rawY = (Float(coordinate.y) + 0.5) / fieldResolution - 0.5
                let rawZ = (Float(coordinate.z) + 0.5) / fieldResolution - 0.5
                // Z-up field to Y-up mesh, then into the inflated grid.
                let meshX = rawX, meshY = rawZ, meshZ = -rawY
                cells.append(Int32(((meshX / scale + 0.5) * Float(resolution)).rounded(.down)))
                cells.append(Int32(((meshY / scale + 0.5) * Float(resolution)).rounded(.down)))
                cells.append(Int32(((meshZ / scale + 0.5) * Float(resolution)).rounded(.down)))
            }
            classification = Trellis2SealedClassification(
                resolution: resolution,
                occupiedCells: cells,
                radius: configuration.sealRadius
            )
        }
        let sealMagnitude = Float(max(configuration.sealRadius, 1)) * scale / Float(resolution)

        // --- Sparse octree refinement toward the band isosurface ---
        var levelResolution = resolution
        while levelResolution > 32 {
            precondition(levelResolution.isMultiple(of: 2), "resolution must halve down to 32")
            levelResolution /= 2
        }
        var coordinates = [Int32]()
        coordinates.reserveCapacity(levelResolution * levelResolution * levelResolution * 3)
        for x in 0..<levelResolution {
            for y in 0..<levelResolution {
                for z in 0..<levelResolution {
                    coordinates.append(Int32(x))
                    coordinates.append(Int32(y))
                    coordinates.append(Int32(z))
                }
            }
        }

        while true {
            let cellSize = scale / Float(levelResolution)
            let voxelCount = coordinates.count / 3
            let inverseResolution = 1 / Float(levelResolution)
            var keep = [Bool](repeating: false, count: voxelCount)
            keep.withUnsafeMutableBufferPointer { output in
                coordinates.withUnsafeBufferPointer { input in
                    concurrentChunks(voxelCount) { range in
                        for voxel in range {
                            let cx = input[voxel * 3]
                            let cy = input[voxel * 3 + 1]
                            let cz = input[voxel * 3 + 2]
                            let x = (Float(cx) + 0.5) * inverseResolution - 0.5
                            let y = (Float(cy) + 0.5) * inverseResolution - 0.5
                            let z = (Float(cz) + 0.5) * inverseResolution - 0.5
                            let distance = bvh.unsignedDistance(x: x * scale, y: y * scale, z: z * scale) - eps
                            // 0.87 ~ sqrt(3)/2 covers the voxel's diagonal radius.
                            output[voxel] = abs(distance) < 0.87 * cellSize
                                || classification?.straddlesBoundary(
                                    x: cx, y: cy, z: cz,
                                    levelResolution: levelResolution
                                ) == true
                        }
                    }
                }
            }
            var kept = [Int32]()
            kept.reserveCapacity(coordinates.count)
            for voxel in 0..<voxelCount where keep[voxel] {
                kept.append(coordinates[voxel * 3])
                kept.append(coordinates[voxel * 3 + 1])
                kept.append(coordinates[voxel * 3 + 2])
            }
            coordinates = kept
            if levelResolution >= resolution { break }
            levelResolution *= 2
            var children = [Int32]()
            children.reserveCapacity(coordinates.count * 8)
            for voxel in 0..<(coordinates.count / 3) {
                let x = coordinates[voxel * 3] * 2
                let y = coordinates[voxel * 3 + 1] * 2
                let z = coordinates[voxel * 3 + 2] * 2
                for corner in 0..<8 {
                    children.append(x + Int32(corner & 1))
                    children.append(y + Int32((corner >> 1) & 1))
                    children.append(z + Int32((corner >> 2) & 1))
                }
            }
            coordinates = children
        }
        let voxelCount = coordinates.count / 3
        guard voxelCount > 0 else { throw Trellis2ModelError.emptyExtractedMesh }

        // --- Grid corner values (UDF - eps) ---
        let cornerStride = Int64(resolution + 1)
        func cornerKey(_ x: Int32, _ y: Int32, _ z: Int32) -> Int64 {
            (Int64(x) * cornerStride + Int64(y)) * cornerStride + Int64(z)
        }
        var cornerIndices = [Int64: Int32]()
        cornerIndices.reserveCapacity(voxelCount * 2)
        var cornerCoordinates = [Int32]()
        cornerCoordinates.reserveCapacity(voxelCount * 6)
        for voxel in 0..<voxelCount {
            let x = coordinates[voxel * 3]
            let y = coordinates[voxel * 3 + 1]
            let z = coordinates[voxel * 3 + 2]
            for corner in 0..<8 {
                let cx = x + Int32(corner & 1)
                let cy = y + Int32((corner >> 1) & 1)
                let cz = z + Int32((corner >> 2) & 1)
                let key = cornerKey(cx, cy, cz)
                if cornerIndices[key] == nil {
                    cornerIndices[key] = Int32(cornerCoordinates.count / 3)
                    cornerCoordinates.append(cx)
                    cornerCoordinates.append(cy)
                    cornerCoordinates.append(cz)
                }
            }
        }
        let cornerCount = cornerCoordinates.count / 3
        var cornerValues = [Float](repeating: 0, count: cornerCount)
        let inverseResolution = 1 / Float(resolution)
        cornerValues.withUnsafeMutableBufferPointer { output in
            cornerCoordinates.withUnsafeBufferPointer { input in
                concurrentChunks(cornerCount) { range in
                    for corner in range {
                        let gx = input[corner * 3]
                        let gy = input[corner * 3 + 1]
                        let gz = input[corner * 3 + 2]
                        let x = (Float(gx) * inverseResolution - 0.5) * scale
                        let y = (Float(gy) * inverseResolution - 0.5) * scale
                        let z = (Float(gz) * inverseResolution - 0.5) * scale
                        let envelope = bvh.unsignedDistance(x: x, y: y, z: z) - eps
                        // Union of the band envelope and the sealed solid:
                        // interior corners go negative, exterior corners keep
                        // the (clamped) envelope value.
                        if let classification {
                            output[corner] = classification.isInterior(x: gx, y: gy, z: gz)
                                ? -sealMagnitude
                                : min(envelope, sealMagnitude)
                        } else {
                            output[corner] = envelope
                        }
                    }
                }
            }
        }

        // --- Simple dual contouring: mean of edge crossings per voxel ---
        // Edges as corner-index pairs in (x | y<<1 | z<<2) encoding, grouped
        // by axis; the fourth edge of each axis group (the far edge) drives
        // the crossing flag used for quad topology, matching the kernel.
        let edgesByAxis: [[(Int, Int)]] = [
            [(0b000, 0b001), (0b100, 0b101), (0b010, 0b011), (0b110, 0b111)],
            [(0b000, 0b010), (0b001, 0b011), (0b100, 0b110), (0b101, 0b111)],
            [(0b000, 0b100), (0b010, 0b110), (0b001, 0b101), (0b011, 0b111)],
        ]
        var dualVertices = [Float](repeating: 0, count: voxelCount * 3)
        var crossings = [Int8](repeating: 0, count: voxelCount * 3)
        dualVertices.withUnsafeMutableBufferPointer { dualOut in
            crossings.withUnsafeMutableBufferPointer { crossingOut in
                concurrentChunks(voxelCount) { range in
                    var values = [Float](repeating: 0, count: 8)
                    for voxel in range {
                        let x = coordinates[voxel * 3]
                        let y = coordinates[voxel * 3 + 1]
                        let z = coordinates[voxel * 3 + 2]
                        for corner in 0..<8 {
                            let key = cornerKey(
                                x + Int32(corner & 1),
                                y + Int32((corner >> 1) & 1),
                                z + Int32((corner >> 2) & 1)
                            )
                            values[corner] = cornerValues[Int(cornerIndices[key]!)]
                        }
                        var sumX: Float = 0, sumY: Float = 0, sumZ: Float = 0
                        var count = 0
                        for axis in 0..<3 {
                            for (edgeIndex, edge) in edgesByAxis[axis].enumerated() {
                                let value1 = values[edge.0]
                                let value2 = values[edge.1]
                                let crosses = (value1 < 0 && value2 >= 0) || (value1 >= 0 && value2 < 0)
                                if crosses {
                                    let t = -value1 / (value2 - value1)
                                    var px = Float(x) + Float(edge.0 & 1)
                                    var py = Float(y) + Float((edge.0 >> 1) & 1)
                                    var pz = Float(z) + Float((edge.0 >> 2) & 1)
                                    switch axis {
                                    case 0: px += t
                                    case 1: py += t
                                    default: pz += t
                                    }
                                    sumX += px; sumY += py; sumZ += pz
                                    count += 1
                                }
                                if edgeIndex == 3 {
                                    crossingOut[voxel * 3 + axis] = !crosses
                                        ? 0
                                        : (value1 < 0 ? 1 : -1)
                                }
                            }
                        }
                        if count > 0 {
                            dualOut[voxel * 3] = sumX / Float(count)
                            dualOut[voxel * 3 + 1] = sumY / Float(count)
                            dualOut[voxel * 3 + 2] = sumZ / Float(count)
                        } else {
                            dualOut[voxel * 3] = Float(x) + 0.5
                            dualOut[voxel * 3 + 1] = Float(y) + 0.5
                            dualOut[voxel * 3 + 2] = Float(z) + 0.5
                        }
                    }
                }
            }
        }

        // --- Quad topology around crossing far edges ---
        let voxelStride = Int64(resolution)
        func voxelKey(_ x: Int32, _ y: Int32, _ z: Int32) -> Int64 {
            (Int64(x) * voxelStride + Int64(y)) * voxelStride + Int64(z)
        }
        var voxelIndices = [Int64: Int32]()
        voxelIndices.reserveCapacity(voxelCount * 2)
        for voxel in 0..<voxelCount {
            voxelIndices[voxelKey(
                coordinates[voxel * 3],
                coordinates[voxel * 3 + 1],
                coordinates[voxel * 3 + 2]
            )] = Int32(voxel)
        }
        let neighborOffsets: [[(Int32, Int32, Int32)]] = [
            [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
            [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
            [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
        ]
        // Upstream's split tables: `p` variants flip winding for positive
        // crossings; the alignment test below is ported literally.
        let splitOneNegative = [0, 1, 2, 0, 2, 3], splitOnePositive = [0, 2, 1, 0, 3, 2]
        let splitTwoNegative = [0, 1, 3, 3, 1, 2], splitTwoPositive = [0, 3, 1, 3, 2, 1]

        func position(_ vertex: Int32) -> (Float, Float, Float) {
            (dualVertices[Int(vertex) * 3], dualVertices[Int(vertex) * 3 + 1], dualVertices[Int(vertex) * 3 + 2])
        }
        func alignment(_ table: [Int], _ quad: [Int32]) -> Float {
            let t0 = position(quad[table[0]])
            let t1 = position(quad[table[1]])
            let t2 = position(quad[table[2]])
            let t3 = position(quad[table[3]])
            let n0 = cross(subtract(t1, t0), subtract(t2, t0))
            let n1 = cross(subtract(t2, t1), subtract(t3, t1))
            return abs(n0.0 * n1.0 + n0.1 * n1.1 + n0.2 * n1.2)
        }

        var indices = [UInt32]()
        indices.reserveCapacity(voxelCount * 3)
        for voxel in 0..<voxelCount {
            let x = coordinates[voxel * 3]
            let y = coordinates[voxel * 3 + 1]
            let z = coordinates[voxel * 3 + 2]
            for axis in 0..<3 {
                let sign = crossings[voxel * 3 + axis]
                guard sign != 0 else { continue }
                var quad = [Int32]()
                quad.reserveCapacity(4)
                for offset in neighborOffsets[axis] {
                    guard let neighbor = voxelIndices[voxelKey(x + offset.0, y + offset.1, z + offset.2)] else { break }
                    quad.append(neighbor)
                }
                guard quad.count == 4 else { continue }
                let one = sign > 0 ? splitOnePositive : splitOneNegative
                let two = sign > 0 ? splitTwoPositive : splitTwoNegative
                let chosen = alignment(one, quad) > alignment(two, quad) ? one : two
                for slot in chosen {
                    indices.append(UInt32(quad[slot]))
                }
            }
        }
        guard !indices.isEmpty else { throw Trellis2ModelError.emptyExtractedMesh }

        // --- Object-space vertices, optional projection onto the crust ---
        var vertices = [Float](repeating: 0, count: dualVertices.count)
        let projectFraction = configuration.projectBack
        vertices.withUnsafeMutableBufferPointer { output in
            dualVertices.withUnsafeBufferPointer { input in
                concurrentChunks(voxelCount) { range in
                    for voxel in range {
                        var x = (input[voxel * 3] * inverseResolution - 0.5) * scale
                        var y = (input[voxel * 3 + 1] * inverseResolution - 0.5) * scale
                        var z = (input[voxel * 3 + 2] * inverseResolution - 0.5) * scale
                        if projectFraction > 0 {
                            let closest = bvh.closestPoint(x: x, y: y, z: z)
                            x -= projectFraction * (x - closest.x)
                            y -= projectFraction * (y - closest.y)
                            z -= projectFraction * (z - closest.z)
                        }
                        output[voxel * 3] = x
                        output[voxel * 3 + 1] = y
                        output[voxel * 3 + 2] = z
                    }
                }
            }
        }

        // Drop dual vertices never referenced by a quad.
        var remap = [UInt32](repeating: UInt32.max, count: voxelCount)
        var compactVertices = [Float]()
        compactVertices.reserveCapacity(vertices.count)
        var compactIndices = [UInt32]()
        compactIndices.reserveCapacity(indices.count)
        for index in indices {
            if remap[Int(index)] == UInt32.max {
                remap[Int(index)] = UInt32(compactVertices.count / 3)
                compactVertices.append(contentsOf: vertices[(Int(index) * 3)..<(Int(index) * 3 + 3)])
            }
            compactIndices.append(remap[Int(index)])
        }

        return try MeshAsset(
            vertices: compactVertices,
            indices: compactIndices,
            coordinateSystem: mesh.coordinateSystem,
            units: mesh.units,
            inferredUnseenGeometry: mesh.inferredUnseenGeometry
        )
    }

    private static func concurrentChunks(_ count: Int, _ body: (Range<Int>) -> Void) {
        Trellis2Parallel.chunks(count, body)
    }

    private static func subtract(
        _ a: (Float, Float, Float),
        _ b: (Float, Float, Float)
    ) -> (Float, Float, Float) {
        (a.0 - b.0, a.1 - b.1, a.2 - b.2)
    }

    private static func cross(
        _ a: (Float, Float, Float),
        _ b: (Float, Float, Float)
    ) -> (Float, Float, Float) {
        (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
    }
}
