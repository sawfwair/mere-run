import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class InstantMeshModelTests: MereRunCoreTestCase {
    func testProductionGraphMatchesPinnedCheckpointInventory() {
        let model = InstantMeshModel()
        let parameters = model.parameters().flattened()
        XCTAssertEqual(parameters.count, InstantMeshWeights.sourceTensorCount)
        XCTAssertEqual(
            parameters.reduce(0) { $0 + $1.1.shape.reduce(1, *) },
            InstantMeshWeights.sourceScalarCount
        )
        let shapes = Dictionary(uniqueKeysWithValues: parameters.map { ($0.0, $0.1.shape) })
        XCTAssertEqual(shapes["encoder.model.embeddings.cls_token"], [1, 1, 768])
        XCTAssertEqual(
            shapes["encoder.model.embeddings.patch_embeddings.projection.weight"],
            [768, 16, 16, 3]
        )
        XCTAssertEqual(
            shapes["encoder.model.encoder.layer.11.adaLN_modulation.1.weight"],
            [3_072, 768]
        )
        XCTAssertEqual(shapes["encoder.camera_embedder.0.weight"], [768, 16])
        XCTAssertEqual(shapes["transformer.pos_embed"], [1, 3_072, 1_024])
        XCTAssertEqual(
            shapes["transformer.layers.11.cross_attn.k_proj_weight"],
            [1_024, 768]
        )
        XCTAssertEqual(
            shapes["transformer.layers.11.self_attn.in_proj_weight"],
            [3_072, 1_024]
        )
        XCTAssertEqual(shapes["transformer.deconv.weight"], [40, 2, 2, 1_024])
        XCTAssertEqual(shapes["synthesizer.decoder.net_sdf.6.weight"], [1, 64])
        XCTAssertEqual(shapes["synthesizer.decoder.net_weight.6.weight"], [21, 64])
    }

    func testMiniatureGraphRunsFourViewReconstructionAndFieldHeads() throws {
        let configuration = try miniatureConfiguration()
        let model = InstantMeshModel(configuration: configuration)
        let count = 4 * configuration.conditioningImageSize * configuration.conditioningImageSize * 3
        let images = (MLX.arange(count, dtype: .float32) / Float(count)).reshaped(
            1, 4, configuration.conditioningImageSize, configuration.conditioningImageSize, 3
        )
        let cameras = MLX.zeros([1, 4, configuration.cameraDimension], dtype: .float32)
        let scene = model(images: images, cameras: cameras)
        MLX.eval(scene.planes)
        XCTAssertEqual(scene.planes.shape, [1, 3, 4, 4, 4])
        XCTAssertTrue(scene.planes.asArray(Float.self).allSatisfy(\.isFinite))

        let positions = MLXArray([
            Float(-0.4), -0.2, 0.1,
            0, 0, 0,
            0.3, 0.2, -0.1,
            0.4, -0.3, 0.2,
            -0.2, 0.4, -0.3,
            0.1, 0.2, 0.3,
            -0.3, -0.2, -0.1,
            0.2, -0.4, 0.4,
        ]).reshaped(8, 3)
        let query = InstantMeshRenderer.query(
            model: model,
            sceneCode: scene,
            positions: positions,
            chunkSize: 3
        )
        MLX.eval(query.signedDistance, query.deformation, query.color)
        XCTAssertEqual(query.signedDistance.shape, [8, 1])
        XCTAssertEqual(query.deformation.shape, [8, 3])
        XCTAssertEqual(query.color.shape, [8, 3])
        XCTAssertTrue(query.signedDistance.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(query.color.asArray(Float.self).allSatisfy(\.isFinite))

        let corners = MLX.arange(8, dtype: .int32).reshaped(1, 8)
        let weights = InstantMeshRenderer.cellWeights(
            model: model,
            sceneCode: scene,
            gridPositions: positions,
            cornerIndices: corners
        )
        MLX.eval(weights)
        XCTAssertEqual(weights.shape, [1, 21])
        XCTAssertTrue(weights.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testWeightMapperHandlesBothConvolutionLayouts() {
        let patch = InstantMeshWeights.mapSourceTensor(
            key: "encoder.model.embeddings.patch_embeddings.projection.weight",
            value: MLX.zeros([8, 3, 2, 2])
        )
        XCTAssertEqual(patch.first?.1.shape, [8, 2, 2, 3])
        let transpose = InstantMeshWeights.mapSourceTensor(
            key: "transformer.deconv.weight",
            value: MLX.zeros([8, 5, 2, 2])
        )
        XCTAssertEqual(transpose.first?.1.shape, [5, 2, 2, 8])
    }

    func testAppleSiliconMemoryDefaultsAndChunkingPreserveMiniatureOutput() throws {
        XCTAssertEqual(InstantMeshMemoryConfiguration.appleSilicon.imageViewBatchSize, 1)
        XCTAssertEqual(InstantMeshMemoryConfiguration.appleSilicon.attentionQueryChunkSize, 256)
        XCTAssertEqual(InstantMeshMemoryConfiguration.appleSilicon.feedForwardTokenChunkSize, 512)
        XCTAssertEqual(InstantMeshMemoryConfiguration.appleSilicon.fieldQueryChunkSize, 65_536)

        let configuration = try miniatureConfiguration()
        let full = InstantMeshModel(
            configuration: configuration,
            memoryConfiguration: try InstantMeshMemoryConfiguration(
                imageViewBatchSize: 4,
                attentionQueryChunkSize: 10_000,
                feedForwardTokenChunkSize: 10_000
            )
        )
        let chunked = InstantMeshModel(
            configuration: configuration,
            memoryConfiguration: try InstantMeshMemoryConfiguration(
                imageViewBatchSize: 1,
                attentionQueryChunkSize: 3,
                feedForwardTokenChunkSize: 5
            )
        )
        try chunked.update(parameters: full.parameters(), verify: .all)
        let count = 4 * configuration.conditioningImageSize * configuration.conditioningImageSize * 3
        let images = (MLX.arange(count, dtype: .float32) / Float(count)).reshaped(
            1, 4, configuration.conditioningImageSize, configuration.conditioningImageSize, 3
        )
        let cameras = MLX.zeros([1, 4, configuration.cameraDimension], dtype: .float32)
        let expected = full(images: images, cameras: cameras).planes
        let actual = chunked(images: images, cameras: cameras).planes
        MLX.eval(expected, actual)
        let differences = zip(expected.asArray(Float.self), actual.asArray(Float.self)).map { abs($0 - $1) }
        XCTAssertLessThanOrEqual(differences.reduce(0, +) / Float(differences.count), 1e-6)
        XCTAssertLessThanOrEqual(differences.max() ?? 0, 1e-5)
    }

    func testOfficialSafetensorsParityWhenFixturesAreAvailable() throws {
        guard let weightsPath = ProcessInfo.processInfo.environment["MERERUN_TEST_INSTANTMESH_SAFETENSORS"],
              let fixturePath = ProcessInfo.processInfo.environment["MERERUN_TEST_INSTANTMESH_PARITY"] else {
            throw XCTSkip("Set MERERUN_TEST_INSTANTMESH_SAFETENSORS and MERERUN_TEST_INSTANTMESH_PARITY")
        }
        let root = URL(fileURLWithPath: fixturePath)
        let model = InstantMeshModel()
        try InstantMeshWeights.load(
            model: model,
            safetensorsURL: URL(fileURLWithPath: weightsPath),
            dtype: .float32
        )
        let input = try MLX.loadArray(url: root.appendingPathComponent("input.npy"))
        let cameras = try MLX.loadArray(url: root.appendingPathComponent("cameras.npy"))
        let diagnostics = model.forwardDiagnostics(images: input, cameras: cameras)
        MLX.eval(diagnostics.sceneCode.planes)

        assertParity(
            actual: diagnostics.imageTokens,
            expected: try MLX.loadArray(url: root.appendingPathComponent("image_tokens.npy")),
            meanTolerance: 2e-5,
            maximumTolerance: 3e-4,
            label: "image tokens"
        )
        assertParity(
            actual: diagnostics.initialTriplaneTokens,
            expected: try MLX.loadArray(url: root.appendingPathComponent("tokens_initial.npy")),
            meanTolerance: 1e-7,
            maximumTolerance: 1e-6,
            label: "initial triplane tokens"
        )
        assertParity(
            actual: diagnostics.finalTriplaneTokens,
            expected: try MLX.loadArray(url: root.appendingPathComponent("tokens_final.npy")),
            meanTolerance: 2e-4,
            maximumTolerance: 1e-2,
            label: "final triplane tokens"
        )
        assertParity(
            actual: diagnostics.sceneCode.planes,
            expected: try MLX.loadArray(
                url: root.appendingPathComponent("scene_code_bpc_hw.npy")
            ).transposed(0, 1, 3, 4, 2),
            meanTolerance: 5e-4,
            maximumTolerance: 1.5e-2,
            label: "scene code"
        )

        let positions = try MLX.loadArray(url: root.appendingPathComponent("query_positions.npy"))
        let query = InstantMeshRenderer.query(
            model: model,
            sceneCode: diagnostics.sceneCode,
            positions: positions
        )
        assertParity(
            actual: query.signedDistance,
            expected: try MLX.loadArray(url: root.appendingPathComponent("query_sdf.npy")),
            meanTolerance: 2e-5,
            maximumTolerance: 5e-5,
            label: "SDF"
        )
        assertParity(
            actual: query.rawDeformation,
            expected: try MLX.loadArray(url: root.appendingPathComponent("query_deformation_raw.npy")),
            meanTolerance: 2e-5,
            maximumTolerance: 5e-5,
            label: "raw deformation"
        )
        assertParity(
            actual: query.deformation,
            expected: try MLX.loadArray(url: root.appendingPathComponent("query_deformation.npy")),
            meanTolerance: 1e-7,
            maximumTolerance: 2e-7,
            label: "deformation"
        )
        assertParity(
            actual: query.color,
            expected: try MLX.loadArray(url: root.appendingPathComponent("query_color.npy")),
            meanTolerance: 1e-6,
            maximumTolerance: 3e-6,
            label: "color"
        )
        let weights = model.decoder.cellWeights(query.sampledFeatures.reshaped(1, 8, 120))
        assertParity(
            actual: weights,
            expected: try MLX.loadArray(url: root.appendingPathComponent("query_cell_weight.npy")),
            meanTolerance: 2e-5,
            maximumTolerance: 5e-5,
            label: "cell weights"
        )

        let mesh = try InstantMeshIsosurfaceExtractor.extractMesh(
            model: model,
            sceneCode: diagnostics.sceneCode,
            configuration: InstantMeshMeshExtractionConfiguration(
                gridResolution: 24,
                includeVertexColors: true
            )
        )
        print("InstantMesh native grid24 mesh: vertices=\(mesh.vertexCount), triangles=\(mesh.triangleCount)")
        XCTAssertGreaterThan(mesh.vertexCount, 100)
        XCTAssertGreaterThan(mesh.triangleCount, 100)
        XCTAssertEqual(mesh.colorsRGBA8?.count, mesh.vertexCount * 4)
        XCTAssertTrue(mesh.inferredUnseenGeometry)
    }

    private func assertParity(
        actual: MLXArray,
        expected: MLXArray,
        meanTolerance: Float,
        maximumTolerance: Float,
        label: String
    ) {
        MLX.eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape, label)
        guard actual.shape == expected.shape else { return }
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        let differences = zip(actualValues, expectedValues).map { abs($0 - $1) }
        let mean = differences.reduce(0, +) / Float(max(1, differences.count))
        let maximum = differences.max() ?? 0
        print("InstantMesh \(label) parity: MAE=\(mean), max=\(maximum)")
        XCTAssertLessThanOrEqual(mean, meanTolerance, "\(label) MAE \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, maximumTolerance, "\(label) MAE \(mean), max \(maximum)")
    }

    private func miniatureConfiguration() throws -> InstantMeshConfiguration {
        try InstantMeshConfiguration(
            conditioningImageSize: 8,
            imageHiddenSize: 12,
            imageLayerCount: 2,
            imageHeadCount: 3,
            imageIntermediateSize: 24,
            imagePatchSize: 2,
            imagePositionGridSize: 2,
            triplaneLowResolution: 2,
            triplaneHighResolution: 4,
            triplaneChannels: 4,
            transformerDimension: 12,
            transformerLayerCount: 2,
            transformerHeadCount: 3,
            transformerMLPMultiplier: 2,
            decoderHiddenSize: 8,
            decoderHiddenLayerCount: 2,
            gridResolution: 8,
            gridScale: 1,
            deformationDivisor: 4
        )
    }
}
