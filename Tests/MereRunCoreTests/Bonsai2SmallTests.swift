import Foundation
import MLX
import MLXNN
import MediaIO
@testable import MereRunTextEncoder
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Bonsai2SmallTests: MereRunCoreTestCase {
    func testPinnedSmallPacksHaveNativeVisionRoutingAndPublishedDefaults() throws {
        let cases: [(ModelResolver.ModelID, String, String, Bool, Int)] = [
            (.bonsai2TwoB, "2", "1c6cbae21f8c38830ce337972b46fdae7b18eaaa", true, 512),
            (.bonsai2FourB, "4", "dff2caaa23fa8706164c44c10f1b58feb405f76a", true, 512),
            (.bonsai2NineB, "9", "f4e28a59531731c41b6dfac88cf271bd58f0a287", false, 4096),
        ]
        for (modelID, size, revision, tied, block) in cases {
            let id = modelID.rawValue
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
            XCTAssertEqual(spec.upstreamRepoId, "prism-ml/Ternary-Bonsai-2-\(size)B-mlx-2bit")
            XCTAssertEqual(spec.upstreamRevision, revision)
            XCTAssertEqual(spec.hubFallback?.revision, revision)
            XCTAssertEqual(Q35Resources.defaultContextLength(forModelId: id), 262_144)
            XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: id))
            let sampling = try XCTUnwrap(Q35Resources.recommendedSampling(forModelId: id))
            XCTAssertEqual(sampling.temperature, 1.0)
            XCTAssertEqual(sampling.topP, 0.95)
            XCTAssertEqual(sampling.topK, 20)
            let manifest = MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
            XCTAssertEqual(manifest.upstreamRepoId, "\(spec.upstreamRepoId!)@\(revision)")
            XCTAssertEqual(manifest.quantization?.scheme, "prism-hadamard-affine-ternary")
            XCTAssertTrue(manifest.supports?.contains(.visionChat) == true)
            let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("Fixtures/Bonsai2Small/\(size)b-config.json")
            let data = try Data(contentsOf: fixture)
            let config = try JSONDecoder().decode(Q35Config.self, from: data)
            let contract = try JSONDecoder().decode(Q35PrismConfiguration.self, from: data)
            XCTAssertEqual(config.tieWordEmbeddings, tied)
            XCTAssertEqual(contract.tieWordEmbeddings, tied)
            XCTAssertEqual(config.visionConfig?.outHiddenSize, config.textConfig.hiddenSize)
            XCTAssertEqual(Set(contract.modules.map(\.block)), [block])
            XCTAssertEqual(contract.modules.contains { $0.path == "lm_head" }, !tied)
            XCTAssertTrue(contract.modules.contains { $0.path == "model.embed_tokens" && $0.embedding })
        }
    }

    func testImagePatchesMatchIndependentPinnedPillowProcessor() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Bonsai2Small")
        for id in [Q35Resources.bonsai2TwoBModelId, Q35Resources.bonsai2FourBModelId, Q35Resources.bonsai2NineBModelId] {
            let generator = Q35Generator(modelId: id, visionMinPixels: 1_024, visionMaxPixels: 1_024)
            let difference = try await generator.bonsaiProcessorDifference(fixture: fixture)
            XCTAssertLessThanOrEqual(difference, 0.000001)
        }
    }

    func testFP16PositionsMatchIndependentUpstreamInterpolation() throws {
        struct Provenance: Decodable { let grids: [[Int]] }
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Bonsai2Small")
        let provenance = try JSONDecoder().decode(Provenance.self,
            from: Data(contentsOf: fixture.appendingPathComponent("position-provenance.json")))
        let arrays = try MLX.loadArrays(url: fixture.appendingPathComponent("position-reference.safetensors"))
        let embedding = Embedding(embeddingCount: 2_304, dimensions: 8)
        try embedding.update(parameters: ModuleParameters.unflattened([
            ("weight", try XCTUnwrap(arrays["weight"]))
        ]), verify: [.all])
        let tower = QwenVisionTower(configuration: QwenVisionConfiguration(
            depth: 0, embedDim: 8, mlpHiddenDim: 16, numHeads: 2, outHiddenDim: 8,
            numPositionEmbeddings: 2_304, useLearnedPosEmbed: true))
        tower.positionEmbeddingArithmetic = .embeddingDType
        for (index, grid) in provenance.grids.enumerated() {
            let actual = tower.fastPosEmbedInterpolate(gridThw: [(t: grid[0], h: grid[1], w: grid[2])], posEmbed: embedding)
            let expected = try XCTUnwrap(arrays["fp16_\(index)"])
            XCTAssertEqual(actual.dtype, .float16)
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertEqual(MLX.max(MLX.abs(actual - expected)).item(Float.self), 0, "grid \(grid)")
        }
    }

    func testRotaryFrequencyFormulaMatchesIndependentReference() throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Bonsai2Small/rotary-reference.safetensors")
        let arrays = try MLX.loadArrays(url: fixture)
        let device = ProcessInfo.processInfo.environment["MERERUN_TEST_MLX_DEVICE"] == "gpu" ? "gpu" : "cpu"
        for headDim in [64, 72] {
            let tower = QwenVisionTower(configuration: QwenVisionConfiguration(
                depth: 0, embedDim: headDim, mlpHiddenDim: 128, numHeads: 1, patchSize: 16,
                temporalPatchSize: 2, outHiddenDim: headDim, numPositionEmbeddings: 2_304, useLearnedPosEmbed: true))
            tower.useCheckpointRotaryFrequencies()
            let output = try tower(patchInputs: MLXArray.zeros([1, 24, 1_536]),
                grid: [QwenVisionGrid(temporal: 1, height: 4, width: 6)])
            let expected = try XCTUnwrap(arrays["\(device)_\(headDim)"])
            XCTAssertEqual(output.rotaryCos.dtype, .float32)
            XCTAssertEqual(output.rotaryCos.shape, expected.shape)
            XCTAssertEqual(MLX.max(MLX.abs(output.rotaryCos - expected)).item(Float.self), 0)
        }
    }

    func testTiedPackedHeadMatchesIndependentlyDecodedEmbeddingRows() {
        for block in [512, 4096] {
            let rows = 3
            let weights = MLXArray((0..<(rows * block / 16)).map { UInt32($0 % 2 == 0 ? 0xAAAA_AAAA : 0x5555_5555) },
                                   [rows, block / 16])
            let scales = MLXArray.full([rows, block / 128], values: MLXArray(Float(0.5))).asType(.float16)
            let biases = -MLXArray.ones([rows, block / 128]).asType(.float16)
            let signs = MLXArray((0..<block).map { Float($0 % 3 == 0 ? -1 : 1) })
            let embedding = Q35PrismEmbedding(weight: weights, scales: scales, biases: biases,
                                               signs: signs, block: block)
            // Construct dense rows independently from the packed two-bit codes and Sylvester transform.
            var dense = [Float]()
            for row in 0..<rows {
                for column in 0..<block {
                    var value: Float = 0
                    for index in 0..<block {
                        let code: Float = ((row * block / 16 + index / 16) % 2 == 0) ? 2 : 1
                        let butterfly: Float = (column & index).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                        value += (code * 0.5 - 1) * butterfly
                    }
                    dense.append(value / Float(block).squareRoot() * Float(column % 3 == 0 ? -1 : 1))
                }
            }
            let denseRows = MLXArray(dense, [rows, block])
            for count in [1, 3] {
                let hidden = MLXArray((0..<(count * block)).map { Float($0 % 19 - 9) / 32 }, [1, count, block])
                let expected = MLX.matmul(hidden, denseRows.T)
                let actual = embedding.asLinear(hidden)
                XCTAssertEqual(actual.shape, [1, count, rows])
                XCTAssertLessThan(MLX.max(MLX.abs(actual - expected)).item(Float.self), 0.0002)
            }
        }
    }
}


extension Q35Generator {
    fileprivate func bonsaiProcessorDifference(fixture: URL) throws -> Float {
        let reference = try MLX.loadArrays(url: fixture.appendingPathComponent("processor-reference.safetensors"))
        let prepared = try loadImageTensor(
            from: fixture.appendingPathComponent("processor-input.png").path,
            patchSize: 16, spatialMergeSize: 2, maximumVisionTokens: 1
        )
        XCTAssertEqual([prepared.gridTHW.0, prepared.gridTHW.1, prepared.gridTHW.2], [1, 2, 2])
        let patches = Q35VisionTower.preparePatchInputs(pixelValues: prepared.tensor,
            patchSize: 16, temporalPatchSize: 2, mergeSize: 2).reshaped(4, 1_536)
        let expected = try XCTUnwrap(reference["patches"])
        XCTAssertEqual(patches.shape, expected.shape)
        return MLX.max(MLX.abs(patches - expected)).item(Float.self)
    }
}
