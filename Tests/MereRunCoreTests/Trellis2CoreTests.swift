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
