import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class TripoSRModelTests: MereRunCoreTestCase {
    func testProductionGraphMatchesPinnedCheckpointInventory() {
        let model = TripoSRModel()
        let parameters = model.parameters().flattened()
        XCTAssertEqual(parameters.count, TripoSRWeights.sourceTensorCount)
        XCTAssertEqual(
            parameters.reduce(0) { $0 + $1.1.shape.reduce(1, *) },
            TripoSRWeights.sourceScalarCount
        )

        let shapes = Dictionary(uniqueKeysWithValues: parameters.map { ($0.0, $0.1.shape) })
        XCTAssertEqual(shapes["image_tokenizer.model.embeddings.cls_token"], [1, 1, 768])
        XCTAssertEqual(shapes["image_tokenizer.model.embeddings.position_embeddings"], [1, 197, 768])
        XCTAssertEqual(
            shapes["image_tokenizer.model.embeddings.patch_embeddings.projection.weight"],
            [768, 16, 16, 3]
        )
        XCTAssertEqual(
            shapes["image_tokenizer.model.encoder.layer.11.attention.attention.query.weight"],
            [768, 768]
        )
        XCTAssertEqual(shapes["tokenizer.embeddings"], [3, 1_024, 32, 32])
        XCTAssertEqual(shapes["backbone.transformer_blocks.15.attn2.to_k.weight"], [1_024, 768])
        XCTAssertEqual(
            shapes["backbone.transformer_blocks.15.ff.net.0.proj.weight"],
            [8_192, 1_024]
        )
        XCTAssertEqual(shapes["post_processor.upsample.weight"], [40, 2, 2, 1_024])
        XCTAssertEqual(shapes["decoder.layers.18.weight"], [4, 64])
    }

    func testMiniatureGraphRunsEndToEndAndQueriesField() throws {
        let configuration = try miniatureConfiguration()
        let model = TripoSRModel(configuration: configuration)
        let elementCount = configuration.conditioningImageSize * configuration.conditioningImageSize * 3
        let input = (MLX.arange(elementCount, dtype: .float32) / Float(elementCount))
            .reshaped(1, configuration.conditioningImageSize, configuration.conditioningImageSize, 3)
        let sceneCode = model(input)
        MLX.eval(sceneCode.planes)
        XCTAssertEqual(sceneCode.planes.shape, [1, 3, 4, 4, 4])
        XCTAssertTrue(sceneCode.planes.asArray(Float.self).allSatisfy(\.isFinite))

        let positions = MLXArray([
            Float(-0.5), -0.25, 0,
            0, 0, 0,
            0.25, 0.5, -0.4,
        ]).reshaped(3, 3)
        let query = TripoSRRenderer.query(
            model: model,
            sceneCode: sceneCode,
            positions: positions,
            chunkSize: 2
        )
        MLX.eval(query.density, query.activatedDensity, query.color)
        XCTAssertEqual(query.density.shape, [3, 1])
        XCTAssertEqual(query.color.shape, [3, 3])
        XCTAssertTrue(query.activatedDensity.asArray(Float.self).allSatisfy { $0.isFinite && $0 > 0 })
        XCTAssertTrue(query.color.asArray(Float.self).allSatisfy { $0.isFinite && (0...1).contains($0) })
    }

    func testSourceWeightMapperHandlesBothConvolutionLayouts() {
        let patch = TripoSRWeights.mapSourceTensor(
            key: "image_tokenizer.model.embeddings.patch_embeddings.projection.weight",
            value: MLX.zeros([8, 3, 2, 2])
        )
        XCTAssertEqual(patch.first?.1.shape, [8, 2, 2, 3])

        let transposed = TripoSRWeights.mapSourceTensor(
            key: "post_processor.upsample.weight",
            value: MLX.zeros([8, 5, 2, 2])
        )
        XCTAssertEqual(transposed.first?.1.shape, [5, 2, 2, 8])

        let linear = TripoSRWeights.mapSourceTensor(
            key: "decoder.layers.0.weight",
            value: MLX.zeros([4, 7])
        )
        XCTAssertEqual(linear.first?.1.shape, [4, 7])
    }

    func testAppleSiliconMemoryDefaultsAndChunkingPreserveMiniatureOutput() throws {
        XCTAssertEqual(TripoSRMemoryConfiguration.appleSilicon.attentionQueryChunkSize, 1_024)
        XCTAssertEqual(TripoSRMemoryConfiguration.appleSilicon.feedForwardTokenChunkSize, 1_024)
        XCTAssertEqual(TripoSRMemoryConfiguration.appleSilicon.queryChunkSize, 8_192)
        XCTAssertEqual(TripoSRMemoryConfiguration.appleSilicon.isosurfaceChunkSize, 65_536)

        let configuration = try miniatureConfiguration()
        let full = TripoSRModel(
            configuration: configuration,
            memoryConfiguration: try TripoSRMemoryConfiguration(
                attentionQueryChunkSize: 10_000,
                feedForwardTokenChunkSize: 10_000
            )
        )
        let chunked = TripoSRModel(
            configuration: configuration,
            memoryConfiguration: try TripoSRMemoryConfiguration(
                attentionQueryChunkSize: 3,
                feedForwardTokenChunkSize: 5
            )
        )
        try chunked.update(parameters: full.parameters(), verify: .all)
        let count = configuration.conditioningImageSize * configuration.conditioningImageSize * 3
        let input = (MLX.arange(count, dtype: .float32) / Float(count)).reshaped(
            1,
            configuration.conditioningImageSize,
            configuration.conditioningImageSize,
            3
        )
        let expected = full(input).planes
        let actual = chunked(input).planes
        MLX.eval(expected, actual)
        let differences = zip(
            expected.asArray(Float.self),
            actual.asArray(Float.self)
        ).map { abs($0 - $1) }
        XCTAssertLessThanOrEqual(differences.reduce(0, +) / Float(differences.count), 1e-6)
        XCTAssertLessThanOrEqual(differences.max() ?? 0, 1e-5)
    }

    func testStrictFullCheckpointLoadWhenExplicitlyEnabled() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_TEST_TRIPOSR_LOAD"] == "1",
              let checkpointPath = ProcessInfo.processInfo.environment["MERERUN_TEST_TRIPOSR_CKPT"]
        else {
            throw XCTSkip("Set MERERUN_TEST_TRIPOSR_LOAD=1 and MERERUN_TEST_TRIPOSR_CKPT to load 1.67 GB weights")
        }
        let model = TripoSRModel()
        let archive = try PyTorchStateDictArchive(url: URL(fileURLWithPath: checkpointPath))
        try TripoSRWeights.load(model: model, archive: archive, dtype: .float32)
        let parameters = model.parameters().flattened()
        XCTAssertEqual(parameters.count, TripoSRWeights.sourceTensorCount)
        XCTAssertTrue(parameters.allSatisfy { $0.1.dtype == .float32 })
    }

    func testOfficialSafetensorsSceneAndFieldParityWhenFixturesAreAvailable() throws {
        guard let weightsPath = ProcessInfo.processInfo.environment["MERERUN_TEST_TRIPOSR_SAFETENSORS"],
              let fixturePath = ProcessInfo.processInfo.environment["MERERUN_TEST_TRIPOSR_PARITY"]
        else {
            throw XCTSkip("Set MERERUN_TEST_TRIPOSR_SAFETENSORS and MERERUN_TEST_TRIPOSR_PARITY")
        }
        let fixtureRoot = URL(fileURLWithPath: fixturePath)
        let model = TripoSRModel()
        try TripoSRWeights.load(
            model: model,
            safetensorsURL: URL(fileURLWithPath: weightsPath),
            dtype: .float32
        )
        let input = try MLX.loadArray(url: fixtureRoot.appendingPathComponent("input.npy"))
        let diagnostics = model.forwardDiagnostics(input)
        let sceneCode = diagnostics.sceneCode
        MLX.eval(sceneCode.planes)

        assertParity(
            actual: diagnostics.imageTokens,
            expected: try MLX.loadArray(
                url: fixtureRoot.appendingPathComponent("image_tokens_bcn.npy")
            ).transposed(0, 2, 1),
            meanTolerance: 1e-5,
            maximumTolerance: 1e-4,
            label: "image tokens"
        )
        assertParity(
            actual: diagnostics.initialTriplaneTokens,
            expected: try MLX.loadArray(
                url: fixtureRoot.appendingPathComponent("tokens_initial_bcn.npy")
            ),
            meanTolerance: 1e-7,
            maximumTolerance: 1e-6,
            label: "initial triplane tokens"
        )
        assertParity(
            actual: diagnostics.finalTriplaneTokens,
            expected: try MLX.loadArray(
                url: fixtureRoot.appendingPathComponent("tokens_final_bcn.npy")
            ),
            meanTolerance: 1e-4,
            maximumTolerance: 7e-3,
            label: "final triplane tokens"
        )

        let expectedScene = try MLX.loadArray(
            url: fixtureRoot.appendingPathComponent("scene_code_bpc_hw.npy")
        ).transposed(0, 1, 3, 4, 2)
        assertParity(
            actual: sceneCode.planes,
            expected: expectedScene,
            meanTolerance: 5e-4,
            maximumTolerance: 8e-3,
            label: "scene code"
        )

        let positions = try MLX.loadArray(url: fixtureRoot.appendingPathComponent("query_positions.npy"))
        let query = TripoSRRenderer.query(
            model: model,
            sceneCode: sceneCode,
            positions: positions
        )
        MLX.eval(query.density, query.activatedDensity, query.color)
        assertParity(
            actual: query.density,
            expected: try MLX.loadArray(url: fixtureRoot.appendingPathComponent("query_density.npy")),
            meanTolerance: 1e-5,
            maximumTolerance: 2e-5,
            label: "query density"
        )
        assertParity(
            actual: query.activatedDensity,
            expected: try MLX.loadArray(url: fixtureRoot.appendingPathComponent("query_density_activated.npy")),
            meanTolerance: 1e-3,
            maximumTolerance: 5e-3,
            label: "activated density"
        )
        assertParity(
            actual: query.color,
            expected: try MLX.loadArray(url: fixtureRoot.appendingPathComponent("query_color.npy")),
            meanTolerance: 1e-6,
            maximumTolerance: 2e-6,
            label: "query color"
        )
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
        let mean = differences.reduce(0, +) / Float(Swift.max(1, differences.count))
        let maximum = differences.max() ?? 0
        print("TripoSR \(label) parity: MAE=\(mean), max=\(maximum)")
        XCTAssertLessThanOrEqual(mean, meanTolerance, "\(label) MAE \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, maximumTolerance, "\(label) MAE \(mean), max \(maximum)")
    }

    private func miniatureConfiguration() throws -> TripoSRConfiguration {
        try TripoSRConfiguration(
            conditioningImageSize: 8,
            imageHiddenSize: 12,
            imageLayerCount: 2,
            imageHeadCount: 3,
            imageIntermediateSize: 24,
            imagePatchSize: 2,
            imagePositionGridSize: 2,
            planeSize: 2,
            tokenChannels: 12,
            transformerLayerCount: 2,
            transformerHeadCount: 3,
            transformerHeadDimension: 4,
            transformerGroupCount: 3,
            transformerFeedForwardMultiplier: 2,
            scenePlaneChannels: 4,
            decoderHiddenSize: 8,
            decoderHiddenLayerCount: 3,
            rendererRadius: 0.5,
            densityThreshold: 1
        )
    }
}
