import MediaIO
import MLX
@testable import MereRunCore
import XCTest

final class Trellis2CoreTests: MereRunCoreTestCase {
    func testProductionConfigurationsMatchPinned512Pipeline() {
        XCTAssertEqual(Trellis2FlowConfiguration.sparseStructure512.resolution, 16)
        XCTAssertEqual(Trellis2FlowConfiguration.sparseStructure512.inputChannels, 8)
        XCTAssertEqual(Trellis2FlowConfiguration.shape512.inputChannels, 32)
        XCTAssertEqual(Trellis2FlowConfiguration.texture512.inputChannels, 64)
        XCTAssertEqual(Trellis2FlowConfiguration.texture512.outputChannels, 32)
        XCTAssertEqual(Trellis2FlowConfiguration.texture512.modelChannels, 1_536)
        XCTAssertEqual(Trellis2FlowConfiguration.texture512.blockCount, 30)
        XCTAssertEqual(Trellis2FlowConfiguration.texture512.headCount, 12)
        XCTAssertEqual(Trellis2FlowConfiguration.texture512.mlpChannels, 8_192)
        XCTAssertEqual(Trellis2SamplerConfiguration.sparseStructure.steps, 12)
        XCTAssertEqual(Trellis2SamplerConfiguration.shape.steps, 12)
        XCTAssertEqual(Trellis2SamplerConfiguration.texture.steps, 12)
        XCTAssertEqual(Trellis2SamplerConfiguration.shape.timesteps.count, 13)
        XCTAssertEqual(Trellis2SamplerConfiguration.shape.timesteps.first, 1)
        XCTAssertEqual(Trellis2SamplerConfiguration.shape.timesteps.last, 0)
        XCTAssertEqual(Trellis2Normalization.shapeMean.count, 32)
        XCTAssertEqual(Trellis2Normalization.textureStandardDeviation.count, 32)
    }

    func testDenseCoordinateOrderMatchesTorchMeshgridFlattening() {
        let coordinates = Trellis2Generator.denseCoordinates(resolution: 2)
        XCTAssertEqual(coordinates.count, 8)
        XCTAssertEqual(coordinates[0], .init(x: 0, y: 0, z: 0))
        XCTAssertEqual(coordinates[1], .init(x: 0, y: 0, z: 1))
        XCTAssertEqual(coordinates[2], .init(x: 0, y: 1, z: 0))
        XCTAssertEqual(coordinates[4], .init(x: 1, y: 0, z: 0))
        XCTAssertEqual(coordinates[7], .init(x: 1, y: 1, z: 1))
    }

    func testSubmanifoldConvolutionPreservesCenterFeatures() throws {
        let coordinates = [
            Trellis2VoxelCoordinate(x: 0, y: 0, z: 0),
            Trellis2VoxelCoordinate(x: 1, y: 0, z: 0),
        ]
        let sparse = try Trellis2SparseTensor(
            features: MLXArray([Float(2), 3]).reshaped(2, 1),
            coordinates: coordinates
        )
        var kernel = [Float](repeating: 0, count: 27)
        kernel[13] = 1
        let weights = Trellis2WeightStore(values: [
            "conv.weight": MLXArray(kernel).reshaped(1, 3, 3, 3, 1),
        ])
        let output = try Trellis2SparseOperations.convolution(
            sparse,
            weights: weights,
            prefix: "conv"
        )
        MLX.eval(output.features)
        XCTAssertEqual(output.coordinates, coordinates)
        XCTAssertEqual(output.features.asArray(Float.self), [2, 3])
    }

    func testSubmanifoldConvolutionAccumulatesOnlyExistingNeighbors() throws {
        let coordinates = [
            Trellis2VoxelCoordinate(x: 0, y: 0, z: 0),
            Trellis2VoxelCoordinate(x: 1, y: 0, z: 0),
        ]
        let sparse = try Trellis2SparseTensor(
            features: MLXArray([Float(2), 3]).reshaped(2, 1),
            coordinates: coordinates
        )
        var kernel = [Float](repeating: 0, count: 27)
        kernel[4] = 10
        kernel[22] = 100
        let weights = Trellis2WeightStore(values: [
            "conv.weight": MLXArray(kernel).reshaped(1, 3, 3, 3, 1),
        ])
        let output = try Trellis2SparseOperations.convolution(
            sparse,
            weights: weights,
            prefix: "conv"
        )
        MLX.eval(output.features)
        XCTAssertEqual(output.features.asArray(Float.self), [300, 20])
    }

    func testChannelToSpatialUsesPredictedChildMaskAndFeatureRows() throws {
        let coordinate = Trellis2VoxelCoordinate(x: 3, y: 4, z: 5)
        let sparse = try Trellis2SparseTensor(
            features: MLXArray((0..<8).map(Float.init)).reshaped(1, 8),
            coordinates: [coordinate]
        )
        let subdivision = try Trellis2Subdivision(
            logits: MLXArray([Float(1), -1, -1, -1, -1, -1, -1, 1]).reshaped(1, 8),
            topology: sparse.topology
        )
        let output = try Trellis2SparseOperations.channelToSpatial(
            sparse,
            subdivision: subdivision,
            maximumTokens: 8
        )
        MLX.eval(output.features)
        XCTAssertEqual(output.coordinates, [
            .init(x: 6, y: 8, z: 10),
            .init(x: 7, y: 9, z: 11),
        ])
        XCTAssertEqual(output.features.asArray(Float.self), [0, 7])
    }

    func testDecoderResidualUsesPyTorchRepeatInterleaveChannelOrder() {
        let source = MLXArray([Float(1), 2, 3]).reshaped(1, 3)
        let repeated = MLX.repeated(source, count: 2, axis: 1)
        MLX.eval(repeated)
        XCTAssertEqual(repeated.asArray(Float.self), [1, 1, 2, 2, 3, 3])
    }

    func testFlexibleDualGridProducesIndexedVertexColoredMeshAndPBRField() throws {
        var coordinates = [Trellis2VoxelCoordinate]()
        for x in 0..<2 {
            for y in 0..<2 {
                for z in 0..<2 {
                    coordinates.append(.init(x: Int32(x), y: Int32(y), z: Int32(z)))
                }
            }
        }
        let shapeRow: [Float] = [0, 0, 0, 1, 1, 1, 0]
        let textureRow: [Float] = [1, -1, -1, 0, 0, 1]
        let shape = try Trellis2SparseTensor(
            features: MLXArray(Array(repeating: shapeRow, count: 8).flatMap { $0 }).reshaped(8, 7),
            coordinates: coordinates
        )
        let texture = try Trellis2SparseTensor(
            features: MLXArray(Array(repeating: textureRow, count: 8).flatMap { $0 }).reshaped(8, 6),
            coordinates: coordinates
        )

        let decoded = try Trellis2FlexibleDualGrid.decode(
            shape: shape,
            texture: texture,
            resolution: 2
        )
        XCTAssertEqual(decoded.mesh.vertexCount, 8)
        XCTAssertGreaterThan(decoded.mesh.triangleCount, 0)
        XCTAssertEqual(Array(decoded.mesh.vertices.prefix(3)), [-0.25, -0.25, 0.25])
        XCTAssertEqual(decoded.mesh.colorsRGBA8?.count, 32)
        XCTAssertEqual(decoded.pbrVoxels.coordinates, coordinates)
        XCTAssertEqual(decoded.pbrVoxels.attributes.count, 48)
        XCTAssertEqual(decoded.pbrVoxels.attributes[0], 1, accuracy: 1e-6)
        XCTAssertEqual(decoded.pbrVoxels.attributes[1], 0, accuracy: 1e-6)
        XCTAssertEqual(decoded.pbrVoxels.attributes[3], 0.5, accuracy: 1e-6)
        XCTAssertEqual(decoded.pbrVoxels.attributes[5], 1, accuracy: 1e-6)
    }

    func testDualGridSamplingIsUniformAcrossTheSparseShell() throws {
        // Every voxel carries the same texture row, so every dual vertex must
        // sample the identical full-magnitude value. Corner voxels of the 2^3
        // grid interpolate over cells whose other corners are unoccupied;
        // unnormalized weights would darken them (and their alpha) eightfold.
        var coordinates = [Trellis2VoxelCoordinate]()
        for x in 0..<2 {
            for y in 0..<2 {
                for z in 0..<2 {
                    coordinates.append(.init(x: Int32(x), y: Int32(y), z: Int32(z)))
                }
            }
        }
        let shapeRow: [Float] = [0, 0, 0, 1, 1, 1, 0]
        let textureRow: [Float] = [1, -1, -1, 0, 0, 1]
        let decoded = try Trellis2FlexibleDualGrid.decode(
            shape: try Trellis2SparseTensor(
                features: MLXArray(Array(repeating: shapeRow, count: 8).flatMap { $0 }).reshaped(8, 7),
                coordinates: coordinates
            ),
            texture: try Trellis2SparseTensor(
                features: MLXArray(Array(repeating: textureRow, count: 8).flatMap { $0 }).reshaped(8, 6),
                coordinates: coordinates
            ),
            resolution: 2
        )
        let colors = try XCTUnwrap(decoded.mesh.colorsRGBA8)
        for vertex in 0..<decoded.mesh.vertexCount {
            XCTAssertEqual(
                Array(colors[(vertex * 4)..<(vertex * 4 + 4)]),
                [255, 0, 0, 255],
                "vertex \(vertex) should carry the uniform field value at full magnitude"
            )
        }
        XCTAssertEqual(decoded.metallic, [Float](repeating: 0.5, count: 8))
        XCTAssertEqual(decoded.roughness, [Float](repeating: 0.5, count: 8))
    }

    func testMedianMaterialFactors() throws {
        let factors = try XCTUnwrap(Trellis2ArtifactExporter.medianMaterialFactors(
            metallic: [0.1, 0.9, 0.2],
            roughness: [0.3, 0.7, 0.5, 0.6]
        ))
        XCTAssertEqual(factors.metallicFactor, 0.2)
        XCTAssertEqual(factors.roughnessFactor, 0.55, accuracy: 1e-6)
        XCTAssertNil(Trellis2ArtifactExporter.medianMaterialFactors(metallic: [], roughness: []))
    }

    func testAtlasBakeMatchesVertexSamplesThroughUVLookup() throws {
        // Distinct red per voxel; the baked atlas texel under each split
        // corner's UV must equal the per-vertex sample of the same field.
        var coordinates = [Trellis2VoxelCoordinate]()
        for x in 0..<2 {
            for y in 0..<2 {
                for z in 0..<2 {
                    coordinates.append(.init(x: Int32(x), y: Int32(y), z: Int32(z)))
                }
            }
        }
        let shapeRow: [Float] = [0, 0, 0, 1, 1, 1, 0]
        let textureRows = (0..<8).map { [Float(-1) + 2 * Float($0) / 7, -1, 1, 0, 0, 1] }
        let decoded = try Trellis2FlexibleDualGrid.decode(
            shape: try Trellis2SparseTensor(
                features: MLXArray(Array(repeating: shapeRow, count: 8).flatMap { $0 }).reshaped(8, 7),
                coordinates: coordinates
            ),
            texture: try Trellis2SparseTensor(
                features: MLXArray(textureRows.flatMap { $0 }).reshaped(8, 6),
                coordinates: coordinates
            ),
            resolution: 2
        )
        let baked = Trellis2TextureAtlasBaker.bake(mesh: decoded.mesh, field: decoded.pbrVoxels)

        XCTAssertEqual(baked.indices.count, decoded.mesh.indices.count)
        XCTAssertEqual(baked.uvs.count, baked.vertexCount * 2)
        XCTAssertEqual(baked.normals.count, baked.positions.count)
        XCTAssertEqual(
            baked.baseColorRGBA8.count,
            baked.atlasWidth * baked.atlasHeight * 4
        )

        var originalByPosition = [[UInt32]: Int]()
        for vertex in 0..<decoded.mesh.vertexCount {
            let key = (0..<3).map { decoded.mesh.vertices[vertex * 3 + $0].bitPattern }
            originalByPosition[key] = vertex
        }
        let colors = try XCTUnwrap(decoded.mesh.colorsRGBA8)
        for vertex in 0..<baked.vertexCount {
            let key = (0..<3).map { baked.positions[vertex * 3 + $0].bitPattern }
            let original = try XCTUnwrap(originalByPosition[key], "split corner must be an original vertex")
            let texelX = Int((baked.uvs[vertex * 2] * Float(baked.atlasWidth) - 0.5).rounded())
            let texelY = Int((baked.uvs[vertex * 2 + 1] * Float(baked.atlasHeight) - 0.5).rounded())
            let pixel = (texelY * baked.atlasWidth + texelX) * 4
            for channel in 0..<4 {
                XCTAssertEqual(
                    Int(baked.baseColorRGBA8[pixel + channel]),
                    Int(colors[original * 4 + channel]),
                    accuracy: 1,
                    "vertex \(vertex) channel \(channel)"
                )
            }
            XCTAssertEqual(baked.metallicRoughnessRGBA8[pixel], 255)
            XCTAssertEqual(Int(baked.metallicRoughnessRGBA8[pixel + 1]), 128, accuracy: 1)
            XCTAssertEqual(Int(baked.metallicRoughnessRGBA8[pixel + 2]), 128, accuracy: 1)
        }
    }

    func testTexturedGLBEmbedsAtlasTexturesAndUVs() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "trellis2-textured-glb-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        var coordinates = [Trellis2VoxelCoordinate]()
        for x in 0..<2 {
            for y in 0..<2 {
                for z in 0..<2 {
                    coordinates.append(.init(x: Int32(x), y: Int32(y), z: Int32(z)))
                }
            }
        }
        let shapeRow: [Float] = [0, 0, 0, 1, 1, 1, 0]
        let textureRow: [Float] = [1, -1, -1, 0, 0, 1]
        let decoded = try Trellis2FlexibleDualGrid.decode(
            shape: try Trellis2SparseTensor(
                features: MLXArray(Array(repeating: shapeRow, count: 8).flatMap { $0 }).reshaped(8, 7),
                coordinates: coordinates
            ),
            texture: try Trellis2SparseTensor(
                features: MLXArray(Array(repeating: textureRow, count: 8).flatMap { $0 }).reshaped(8, 6),
                coordinates: coordinates
            ),
            resolution: 2
        )
        let baked = Trellis2TextureAtlasBaker.bake(mesh: decoded.mesh, field: decoded.pbrVoxels)
        let url = root.appendingPathComponent("baked.glb")
        try Trellis2TexturedGLBWriter.write(
            baked,
            coordinateSystem: decoded.mesh.coordinateSystem,
            units: decoded.mesh.units,
            inferredUnseenGeometry: decoded.mesh.inferredUnseenGeometry,
            to: url
        )

        let data = try Data(contentsOf: url)
        let jsonLength = Int(data[12]) | (Int(data[13]) << 8) | (Int(data[14]) << 16) | (Int(data[15]) << 24)
        let document = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data.subdata(in: 20..<(20 + jsonLength))) as? [String: Any]
        )
        let primitive = try XCTUnwrap(
            ((document["meshes"] as? [[String: Any]])?.first?["primitives"] as? [[String: Any]])?.first
        )
        let attributes = try XCTUnwrap(primitive["attributes"] as? [String: Any])
        XCTAssertNotNil(attributes["TEXCOORD_0"])
        XCTAssertNil(attributes["COLOR_0"], "textured GLB must not double-apply color")
        let material = try XCTUnwrap((document["materials"] as? [[String: Any]])?.first)
        let pbr = try XCTUnwrap(material["pbrMetallicRoughness"] as? [String: Any])
        XCTAssertNotNil(pbr["baseColorTexture"])
        XCTAssertNotNil(pbr["metallicRoughnessTexture"])
        let images = try XCTUnwrap(document["images"] as? [[String: Any]])
        XCTAssertEqual(images.count, 2)

        let binaryStart = 20 + jsonLength + 8
        let views = try XCTUnwrap(document["bufferViews"] as? [[String: Any]])
        let imageView = views[try XCTUnwrap(images[0]["bufferView"] as? Int)]
        let offset = binaryStart + (imageView["byteOffset"] as? Int ?? 0)
        let length = try XCTUnwrap(imageView["byteLength"] as? Int)
        let png = data.subdata(in: offset..<(offset + length))
        XCTAssertEqual(Array(png.prefix(4)), [0x89, 0x50, 0x4E, 0x47])
        let image = try MediaImageIO.decode(data: png)
        XCTAssertEqual(image.width, baked.atlasWidth)
        XCTAssertEqual(image.height, baked.atlasHeight)
    }

    func testTriangleBVHReturnsExactDistances() {
        let bvh = Trellis2TriangleBVH(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2]
        )
        XCTAssertEqual(bvh.unsignedDistance(x: 0.25, y: 0.25, z: 0.7), 0.7, accuracy: 1e-5)
        XCTAssertEqual(bvh.unsignedDistance(x: -1, y: -1, z: 0), Float(2).squareRoot(), accuracy: 1e-5)
        XCTAssertEqual(bvh.unsignedDistance(x: 2, y: 0, z: 0), 1, accuracy: 1e-5)
        XCTAssertEqual(bvh.unsignedDistance(x: 0.5, y: -0.5, z: 0), 0.5, accuracy: 1e-5)
        let closest = bvh.closestPoint(x: 0.25, y: 0.25, z: 0.7)
        XCTAssertEqual(closest.x, 0.25, accuracy: 1e-5)
        XCTAssertEqual(closest.y, 0.25, accuracy: 1e-5)
        XCTAssertEqual(closest.z, 0, accuracy: 1e-5)
    }

    func testNarrowBandRemeshClosesFullyOpenSurface() throws {
        // A lone triangle is the extreme torn crust: every edge is a
        // boundary. Its band envelope must come back watertight.
        let crust = try MeshAsset(
            vertices: [-0.2, -0.2, 0, 0.2, -0.2, 0, 0, 0.2, 0],
            indices: [0, 1, 2],
            inferredUnseenGeometry: true
        )
        let remeshed = try Trellis2NarrowBandRemesher.remesh(mesh: crust, resolution: 64)
        XCTAssertGreaterThan(remeshed.triangleCount, 0)
        XCTAssertEqual(boundaryEdgeCount(of: remeshed), 0, "envelope must be closed")
    }

    func testNarrowBandRemeshSealsSubBandGapsAndConnectsSheets() throws {
        // Two coplanar sheets separated by ~1.5 voxels at resolution 64.
        // The band envelopes overlap across the gap, so the result must be
        // one connected watertight surface.
        let gap: Float = 0.012
        let crust = try MeshAsset(
            vertices: [
                -0.3, -0.3, 0, -gap, -0.3, 0, -gap, 0.3, 0, -0.3, 0.3, 0,
                gap, -0.3, 0, 0.3, -0.3, 0, 0.3, 0.3, 0, gap, 0.3, 0,
            ],
            indices: [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
            inferredUnseenGeometry: true
        )
        let remeshed = try Trellis2NarrowBandRemesher.remesh(mesh: crust, resolution: 64)
        XCTAssertEqual(boundaryEdgeCount(of: remeshed), 0, "envelope must be closed")
        XCTAssertEqual(connectedComponentCount(of: remeshed), 1, "gap must be sealed")
    }

    private func boundaryEdgeCount(of mesh: MeshAsset) -> Int {
        var counts = [UInt64: Int]()
        for triangle in stride(from: 0, to: mesh.indices.count, by: 3) {
            let corners = [
                mesh.indices[triangle],
                mesh.indices[triangle + 1],
                mesh.indices[triangle + 2],
            ]
            for edge in 0..<3 {
                let a = UInt64(min(corners[edge], corners[(edge + 1) % 3]))
                let b = UInt64(max(corners[edge], corners[(edge + 1) % 3]))
                counts[(a << 32) | b, default: 0] += 1
            }
        }
        return counts.values.filter { $0 == 1 }.count
    }

    private func connectedComponentCount(of mesh: MeshAsset) -> Int {
        var parent = Array(0..<mesh.vertexCount)
        func find(_ vertex: Int) -> Int {
            var root = vertex
            while parent[root] != root { root = parent[root] }
            var current = vertex
            while parent[current] != root {
                let next = parent[current]
                parent[current] = root
                current = next
            }
            return root
        }
        for triangle in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = find(Int(mesh.indices[triangle]))
            let b = find(Int(mesh.indices[triangle + 1]))
            let c = find(Int(mesh.indices[triangle + 2]))
            parent[b] = a
            parent[c] = a
        }
        return Set((0..<mesh.vertexCount).map(find)).count
    }

    func testSealedClassificationClosesTornCavityOnlyWithSufficientRadius() {
        // A hollow 12-cube shell centered in a 32-grid, with a 4-cell hole
        // punched through one face. Closing with radius >= 2 must classify
        // the cavity interior; radius 0 must let the flood leak through.
        var occupied = [Int32]()
        let low: Int32 = 10, high: Int32 = 21
        for x in low...high {
            for y in low...high {
                for z in low...high {
                    let shell = x == low || x == high || y == low || y == high || z == low || z == high
                    guard shell else { continue }
                    let inHole = x == high && (14...17).contains(y) && (14...17).contains(z)
                    if !inHole {
                        occupied.append(contentsOf: [x, y, z])
                    }
                }
            }
        }
        let sealed = Trellis2SealedClassification(resolution: 32, occupiedCells: occupied, radius: 3)
        XCTAssertTrue(sealed.isInterior(x: 15, y: 15, z: 15), "cavity center must seal at radius 3")
        XCTAssertFalse(sealed.isInterior(x: 2, y: 2, z: 2), "far corner must stay exterior")
        XCTAssertFalse(sealed.isInterior(x: 29, y: 15, z: 15), "outside the holed face must stay exterior")
        let leaky = Trellis2SealedClassification(resolution: 32, occupiedCells: occupied, radius: 0)
        XCTAssertFalse(leaky.isInterior(x: 15, y: 15, z: 15), "radius 0 must leak through the hole")
    }

    func testRemeshMembraneSpansTornCavity() throws {
        // The same torn shell as a crust mesh: without sealing, the envelope
        // has a tunnel; with sealing, a membrane spans the hole. Compare
        // enclosed volumes via the divergence theorem.
        var vertices = [Float]()
        var indices = [UInt32]()
        // Build shell quads from occupied-face voxels directly: a coarse
        // axis-aligned box surface with a hole, in mesh object space.
        func addQuad(_ a: [Float], _ b: [Float], _ c: [Float], _ d: [Float]) {
            let base = UInt32(vertices.count / 3)
            vertices.append(contentsOf: a + b + c + d)
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        let half: Float = 0.2
        let step: Float = 0.05
        var tiles = 0
        for u in stride(from: -half, to: half, by: step) {
            for v in stride(from: -half, to: half, by: step) {
                let u2 = u + step, v2 = v + step
                addQuad([u, v, -half], [u2, v, -half], [u2, v2, -half], [u, v2, -half])
                let hole = abs(u + step / 2) < 0.05 && abs(v + step / 2) < 0.05
                if !hole {
                    addQuad([u, v, half], [u, v2, half], [u2, v2, half], [u2, v, half])
                }
                addQuad([u, -half, v], [u2, -half, v], [u2, -half, v2], [u, -half, v2])
                addQuad([u, half, v], [u, half, v2], [u2, half, v2], [u2, half, v])
                addQuad([-half, u, v], [-half, u, v2], [-half, u2, v2], [-half, u2, v])
                addQuad([half, u, v], [half, u2, v], [half, u2, v2], [half, u, v2])
                tiles += 1
            }
        }
        let crust = try MeshAsset(vertices: vertices, indices: indices, inferredUnseenGeometry: true)
        // Occupancy for the classification: voxelize the shell tiles into the
        // field's Z-up grid (mesh (x,y,z) -> field (x, -z, y)).
        var field = [Trellis2VoxelCoordinate]()
        let resolution = 64
        for triple in stride(from: 0, to: vertices.count, by: 9) {
            let x = vertices[triple], y = vertices[triple + 1], z = vertices[triple + 2]
            let fx = Int32((x + 0.5) * Float(resolution))
            let fy = Int32((-z + 0.5) * Float(resolution))
            let fz = Int32((y + 0.5) * Float(resolution))
            field.append(.init(x: fx, y: fy, z: fz))
        }

        func enclosedVolume(_ mesh: MeshAsset) -> Float {
            var volume: Float = 0
            for triangle in stride(from: 0, to: mesh.indices.count, by: 3) {
                let a = Int(mesh.indices[triangle]) * 3
                let b = Int(mesh.indices[triangle + 1]) * 3
                let c = Int(mesh.indices[triangle + 2]) * 3
                let ax = mesh.vertices[a], ay = mesh.vertices[a + 1], az = mesh.vertices[a + 2]
                let bx = mesh.vertices[b], by = mesh.vertices[b + 1], bz = mesh.vertices[b + 2]
                let cx = mesh.vertices[c], cy = mesh.vertices[c + 1], cz = mesh.vertices[c + 2]
                volume += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6
            }
            return abs(volume)
        }

        let unsealed = try Trellis2NarrowBandRemesher.remesh(
            mesh: crust,
            resolution: resolution,
            configuration: Trellis2RemeshConfiguration(band: 1, sealRadius: 0)
        )
        let sealedMesh = try Trellis2NarrowBandRemesher.remesh(
            mesh: crust,
            resolution: resolution,
            configuration: Trellis2RemeshConfiguration(band: 1, sealRadius: 6),
            fieldCoordinates: field
        )
        let boxVolume: Float = 0.4 * 0.4 * 0.4
        XCTAssertLessThan(enclosedVolume(unsealed), boxVolume * 0.5, "tunnel envelope must not enclose the cavity")
        XCTAssertGreaterThan(enclosedVolume(sealedMesh), boxVolume * 0.7, "sealed remesh must enclose the cavity")
        XCTAssertEqual(boundaryEdgeCount(of: sealedMesh), 0)
    }

    func testHoleFillerCapsLargeClosedRimOnlyAtRaisedThreshold() throws {
        // The open face's edges are all side*sqrt(2) diagonals, so the rim
        // perimeter is ~0.17: beyond the 0.03 reference threshold, within
        // the remesh pipeline's 0.2 rim-capping threshold.
        let side: Float = 0.04
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, side, 0, 0, 0, side, 0, 0, 0, side],
            indices: [0, 2, 1, 0, 1, 3, 0, 3, 2],
            inferredUnseenGeometry: true
        )
        let reference = try Trellis2MeshHoleFiller.fillSmallHoles(in: mesh)
        XCTAssertEqual(reference.triangleCount, 3, "reference threshold must leave the large rim open")
        let capped = try Trellis2MeshHoleFiller.fillSmallHoles(in: mesh, maximumPerimeter: 0.2)
        XCTAssertEqual(capped.triangleCount, 4, "raised threshold must cap the rim")
        XCTAssertEqual(capped.vertexCount, 4)
    }

    func testSmallHoleFillerCapsReferenceThresholdBoundaryWithoutAddingVertices() throws {
        let side: Float = 0.005
        let mesh = try MeshAsset(
            vertices: [
                0, 0, 0,
                side, 0, 0,
                0, side, 0,
                0, 0, side,
            ],
            indices: [
                0, 2, 1,
                0, 1, 3,
                0, 3, 2,
            ],
            inferredUnseenGeometry: true
        )
        let repaired = try Trellis2MeshHoleFiller.fillSmallHoles(in: mesh)
        XCTAssertEqual(repaired.vertexCount, 4)
        XCTAssertEqual(repaired.triangleCount, 4)
    }

    func testPreprocessorRequiresAlphaUnlessFramingIsExplicit() throws {
        let opaque = try MediaImage(width: 1, height: 1, rgba8: [255, 64, 0, 255])
        XCTAssertThrowsError(try Trellis2Preprocessor.prepare(image: opaque, size: 2)) {
            XCTAssertEqual($0 as? Trellis2PreprocessingError, .backgroundRemovalRequired)
        }
        let prepared = try Trellis2Preprocessor.prepare(
            image: opaque,
            size: 2,
            foregroundPolicy: .alreadyFramed
        )
        XCTAssertEqual(prepared.conditionInput.shape, [1, 3, 2, 2])
        XCTAssertFalse(prepared.croppedTransparentForeground)
        XCTAssertEqual(prepared.sourceWidth, 1)
        XCTAssertEqual(prepared.sourceHeight, 1)
    }

    func testGeneratorRejectsMissingInputBeforeResolvingGatedDependencies() async {
        let missing = URL(fileURLWithPath: "/tmp/mere-run-trellis2-missing-\(UUID().uuidString).png")
        let generator = Trellis2Generator()
        do {
            _ = try await generator.generate(
                imageURL: missing,
                outputDirectory: URL(fileURLWithPath: "/tmp/unused")
            )
            XCTFail("Expected missing input failure")
        } catch {
            XCTAssertEqual(
                error as? Trellis2GeneratorError,
                .inputImageNotFound(missing.path)
            )
        }
    }
}
