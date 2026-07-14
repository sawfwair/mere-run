import Foundation

/// A vertex-split mesh with per-corner UVs and baked PBR atlases, ready for
/// the textured GLB writer.
struct Trellis2BakedTexturedMesh {
    let positions: [Float]
    let normals: [Float]
    let uvs: [Float]
    let indices: [UInt32]
    let atlasWidth: Int
    let atlasHeight: Int
    /// sRGB-encoded base color, matching glTF's baseColorTexture transfer
    /// expectations. Alpha is constant 255: the material renders opaque and
    /// the field's alpha channel stays in the `.pbrvox` sidecar.
    let baseColorRGBA8: [UInt8]
    /// Linear-encoded occupancy/roughness/metallic per the glTF
    /// metallicRoughnessTexture channel layout (G = roughness, B = metallic).
    let metallicRoughnessRGBA8: [UInt8]

    var vertexCount: Int { positions.count / 3 }
}

/// Bakes TRELLIS.2's sparse PBR voxel field into per-quad atlas blocks.
///
/// The dual-grid extractor emits two triangles per grid-edge quad, so the
/// mesh partitions into quads (plus a handful of hole-fill caps). Every quad
/// gets a uniform texel block in the atlas — deterministic packing with no
/// unwrapping, injective by construction, gutters against filter bleed. Block
/// texels sample the field over the quad's bilinear patch, capturing the
/// sub-voxel gradients that per-vertex color flattens.
enum Trellis2TextureAtlasBaker {
    /// Interior texels per block edge; corners map to the centers of the
    /// interior corner texels so bilinear filtering never leaves the block.
    static let interiorTexels = 4
    static let gutterTexels = 1
    static var blockTexels: Int { interiorTexels + 2 * gutterTexels }

    private struct Cell {
        /// Original vertex indices at the block corners: 4 in cycle order
        /// (unique, shared, unique, shared) for a quad, 3 for a solo cap.
        let corners: [UInt32]
        /// Original index triples, re-emitted against the split corners.
        let triangles: [[UInt32]]
    }

    /// `projectionSurface` mirrors upstream's `to_glb` texel handling: every
    /// sample position is snapped to its closest point on the original crust
    /// before sampling the field, so remeshed envelopes (whose surface sits
    /// up to `band` voxels off the crust) sample where the field actually
    /// lives instead of falling off the occupied shell.
    static func bake(
        mesh: MeshAsset,
        field: Trellis2PBRVoxelGrid,
        projectionSurface: Trellis2TriangleBVH? = nil
    ) -> Trellis2BakedTexturedMesh {
        let normals = mesh.normals ?? MeshNormals.generate(vertices: mesh.vertices, indices: mesh.indices)
        // Mip levels average neighboring atlas blocks, so blocks must be
        // spatially coherent in BOTH atlas axes or minified views converge
        // to the global average color. Order cells by the 3D Morton code of
        // their centroid and place them along the 2D Morton curve of the
        // block grid; mip blending then approximates a surface-local
        // low-pass instead of mixing unrelated quads.
        let cells = mortonOrdered(partition(indices: mesh.indices), vertices: mesh.vertices, bounds: mesh.bounds)
        let sampler = Trellis2SparseFieldSampler(
            coordinates: field.coordinates,
            attributes: field.attributes
        )

        let block = blockTexels
        var blocksPerSide = 1
        while blocksPerSide * blocksPerSide < cells.count { blocksPerSide *= 2 }
        let width = blocksPerSide * block
        let height = blocksPerSide * block
        let span = Float(max(1, interiorTexels - 1))

        var positions = [Float]()
        var splitNormals = [Float]()
        var uvs = [Float]()
        var indices = [UInt32]()
        positions.reserveCapacity(cells.count * 4 * 3)
        splitNormals.reserveCapacity(cells.count * 4 * 3)
        uvs.reserveCapacity(cells.count * 4 * 2)
        indices.reserveCapacity(mesh.indices.count)
        var baseColor = [UInt8](repeating: 0, count: width * height * 4)
        var metallicRoughness = [UInt8](repeating: 0, count: width * height * 4)
        var colorSums: [Double] = [0, 0, 0]
        var metallicRoughnessSums: [Double] = [0, 0]
        var bakedTexels: Double = 0

        let cornerUV: [[(Float, Float)]] = [
            [(0, 0), (1, 0), (0, 1)],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        ]
        for (cellIndex, cell) in cells.enumerated() {
            let (blockX, blockY) = morton2D(cellIndex)
            let cellX = blockX * block
            let cellY = blockY * block
            let firstSplit = UInt32(positions.count / 3)
            for (slot, original) in cell.corners.enumerated() {
                let base = Int(original) * 3
                positions.append(contentsOf: mesh.vertices[base..<(base + 3)])
                splitNormals.append(contentsOf: normals[base..<(base + 3)])
                let (i, j) = cornerUV[cell.corners.count - 3][slot]
                uvs.append((Float(cellX + gutterTexels) + i * span + 0.5) / Float(width))
                uvs.append((Float(cellY + gutterTexels) + j * span + 0.5) / Float(height))
            }
            for triangle in cell.triangles {
                for original in triangle {
                    let slot = cell.corners.firstIndex(of: original)!
                    indices.append(firstSplit + UInt32(slot))
                }
            }
        }

        baseColor.withUnsafeMutableBufferPointer { basePixels in
            metallicRoughness.withUnsafeMutableBufferPointer { mrPixels in
                Trellis2Parallel.chunks(cells.count, chunk: 1_024) { range in
                    for cellIndex in range {
                        let cell = cells[cellIndex]
                        let (blockX, blockY) = morton2D(cellIndex)
                        let cellX = blockX * block
                        let cellY = blockY * block
                        let corners = cell.corners.map { index -> (Float, Float, Float) in
                            let base = Int(index) * 3
                            return (mesh.vertices[base], mesh.vertices[base + 1], mesh.vertices[base + 2])
                        }
                        for texelY in 0..<block {
                            let t = min(1, max(0, (Float(texelY) - Float(gutterTexels)) / span))
                            for texelX in 0..<block {
                                let s = min(1, max(0, (Float(texelX) - Float(gutterTexels)) / span))
                                var point = interpolate(corners: corners, s: s, t: t)
                                if let projectionSurface {
                                    let closest = projectionSurface.closestPoint(
                                        x: point.0, y: point.1, z: point.2
                                    )
                                    point = (closest.x, closest.y, closest.z)
                                }
                                // Mesh space is the Z-up field rotated to
                                // Y-up; invert the rotation, then shift into
                                // continuous voxel space.
                                let voxelX = (point.0 + 0.5) * Float(field.resolution)
                                let voxelY = (-point.2 + 0.5) * Float(field.resolution)
                                let voxelZ = (point.1 + 0.5) * Float(field.resolution)
                                let values = sampler.sample(x: voxelX, y: voxelY, z: voxelZ)
                                let pixel = ((cellY + texelY) * width + cellX + texelX) * 4
                                basePixels[pixel] = byte(values[0])
                                basePixels[pixel + 1] = byte(values[1])
                                basePixels[pixel + 2] = byte(values[2])
                                basePixels[pixel + 3] = 255
                                mrPixels[pixel] = 255
                                mrPixels[pixel + 1] = byte(values[4])
                                mrPixels[pixel + 2] = byte(values[3])
                                mrPixels[pixel + 3] = 255
                            }
                        }
                    }
                }
            }
        }
        for cellIndex in 0..<cells.count {
            let (blockX, blockY) = morton2D(cellIndex)
            for texelY in 0..<block {
                for texelX in 0..<block {
                    let pixel = ((blockY * block + texelY) * width + blockX * block + texelX) * 4
                    colorSums[0] += Double(baseColor[pixel])
                    colorSums[1] += Double(baseColor[pixel + 1])
                    colorSums[2] += Double(baseColor[pixel + 2])
                    metallicRoughnessSums[0] += Double(metallicRoughness[pixel + 1])
                    metallicRoughnessSums[1] += Double(metallicRoughness[pixel + 2])
                    bakedTexels += 1
                }
            }
        }

        // Unused blocks in the power-of-two grid take the mean baked color so
        // deep mip levels near the occupied boundary do not pull toward black.
        if cells.count < blocksPerSide * blocksPerSide, bakedTexels > 0 {
            let meanColor = colorSums.map { UInt8(($0 / bakedTexels).rounded()) }
            let meanRoughness = UInt8((metallicRoughnessSums[0] / bakedTexels).rounded())
            let meanMetallic = UInt8((metallicRoughnessSums[1] / bakedTexels).rounded())
            for blockIndex in cells.count..<(blocksPerSide * blocksPerSide) {
                let (blockX, blockY) = morton2D(blockIndex)
                for texelY in 0..<block {
                    for texelX in 0..<block {
                        let pixel = ((blockY * block + texelY) * width + blockX * block + texelX) * 4
                        baseColor[pixel] = meanColor[0]
                        baseColor[pixel + 1] = meanColor[1]
                        baseColor[pixel + 2] = meanColor[2]
                        baseColor[pixel + 3] = 255
                        metallicRoughness[pixel] = 255
                        metallicRoughness[pixel + 1] = meanRoughness
                        metallicRoughness[pixel + 2] = meanMetallic
                        metallicRoughness[pixel + 3] = 255
                    }
                }
            }
        }

        return Trellis2BakedTexturedMesh(
            positions: positions,
            normals: splitNormals,
            uvs: uvs,
            indices: indices,
            atlasWidth: width,
            atlasHeight: height,
            baseColorRGBA8: baseColor,
            metallicRoughnessRGBA8: metallicRoughness
        )
    }

    /// Splits the index buffer into quads (consecutive triangles sharing an
    /// edge — the dual-grid emission order) and solo hole-fill caps.
    private static func partition(indices: [UInt32]) -> [Cell] {
        var cells = [Cell]()
        var triangle = 0
        let triangleCount = indices.count / 3
        while triangle < triangleCount {
            let first = Array(indices[(triangle * 3)..<(triangle * 3 + 3)])
            if triangle + 1 < triangleCount {
                let second = Array(indices[(triangle * 3 + 3)..<(triangle * 3 + 6)])
                let shared = Set(first).intersection(Set(second))
                if shared.count == 2, Set(first).union(Set(second)).count == 4 {
                    let uniqueFirst = first.first { !shared.contains($0) }!
                    let uniqueSecond = second.first { !shared.contains($0) }!
                    let sharedPair = Array(shared).sorted()
                    cells.append(Cell(
                        corners: [uniqueFirst, sharedPair[0], uniqueSecond, sharedPair[1]],
                        triangles: [first, second]
                    ))
                    triangle += 2
                    continue
                }
            }
            cells.append(Cell(corners: first, triangles: [first]))
            triangle += 1
        }
        return cells
    }

    /// Cells sorted by the interleaved 3D Morton code of their centroid so
    /// consecutive cells are spatially adjacent on the surface.
    private static func mortonOrdered(
        _ cells: [Cell],
        vertices: [Float],
        bounds: MeshBounds
    ) -> [Cell] {
        let extent = (0..<3).map { max(bounds.maximum[$0] - bounds.minimum[$0], 1e-6) }
        func key(_ cell: Cell) -> UInt64 {
            var centroid: [Float] = [0, 0, 0]
            for corner in cell.corners {
                for axis in 0..<3 {
                    centroid[axis] += vertices[Int(corner) * 3 + axis]
                }
            }
            var code: UInt64 = 0
            for axis in 0..<3 {
                let normalized = (centroid[axis] / Float(cell.corners.count) - bounds.minimum[axis]) / extent[axis]
                let quantized = UInt64(min(1023, max(0, Int(normalized * 1023))))
                for bit in 0..<10 {
                    code |= ((quantized >> bit) & 1) << (bit * 3 + axis)
                }
            }
            return code
        }
        return cells
            .map { (key($0), $0) }
            .sorted { $0.0 < $1.0 }
            .map(\.1)
    }

    /// Deinterleaves a cell rank into 2D Morton block coordinates.
    private static func morton2D(_ rank: Int) -> (Int, Int) {
        var x = 0, y = 0
        for bit in 0..<16 {
            x |= ((rank >> (bit * 2)) & 1) << bit
            y |= ((rank >> (bit * 2 + 1)) & 1) << bit
        }
        return (x, y)
    }

    private static func interpolate(
        corners: [(Float, Float, Float)],
        s: Float,
        t: Float
    ) -> (Float, Float, Float) {
        if corners.count == 4 {
            let bottom = mix(corners[0], corners[1], s)
            let top = mix(corners[3], corners[2], s)
            return mix(bottom, top, t)
        }
        return (
            corners[0].0 + s * (corners[1].0 - corners[0].0) + t * (corners[2].0 - corners[0].0),
            corners[0].1 + s * (corners[1].1 - corners[0].1) + t * (corners[2].1 - corners[0].1),
            corners[0].2 + s * (corners[1].2 - corners[0].2) + t * (corners[2].2 - corners[0].2)
        )
    }

    private static func mix(
        _ a: (Float, Float, Float),
        _ b: (Float, Float, Float),
        _ fraction: Float
    ) -> (Float, Float, Float) {
        (
            a.0 + (b.0 - a.0) * fraction,
            a.1 + (b.1 - a.1) * fraction,
            a.2 + (b.2 - a.2) * fraction
        )
    }

    private static func byte(_ value: Float) -> UInt8 {
        UInt8((255 * min(max(value, 0), 1)).rounded())
    }
}
