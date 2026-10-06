import Foundation
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class EmbeddingGemma2LoadingTests: MLXTestCase {
    struct Fixture: Decodable { let config: EmbeddingGemma2Config }

    private func configuration() throws -> EmbeddingGemma2Config {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "synthetic", withExtension: "json", subdirectory: "Fixtures/EmbeddingGemma2"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url)).config
    }

    private func withRoot(_ operation: (EmbeddingGemma2Resources) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try operation(EmbeddingGemma2Resources(rootURL: root))
    }

    func testOfficialWeightSubtreeLoadsAndIgnoresUnusedMediaTowers() throws {
        let model = try EmbeddingGemma2TextModel(config: configuration())
        try withRoot { resources in
            var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map {
                ("language_model." + $0.0, MLXArray.ones($0.1.shape))
            })
            weights["vision_tower.unused.weight"] = MLXArray.ones([1])
            weights["audio_tower.unused.weight"] = MLXArray.ones([1])
            try MLX.save(arrays: weights, url: resources.weightsURL)
            try EmbeddingGemma2Model.loadWeights(into: model, resources: resources, dtype: .float32)
            for (_, value) in model.parameters().flattened() {
                eval(value)
                XCTAssertTrue(value.asArray(Float.self).allSatisfy { $0 == 1 })
            }
        }
    }

    func testIncompleteAndWrongShapeCheckpointsAreRejected() throws {
        let model = try EmbeddingGemma2TextModel(config: configuration())
        try withRoot { resources in
            try MLX.save(arrays: ["audio_tower.only.weight": MLXArray.ones([1])], url: resources.weightsURL)
            XCTAssertThrowsError(try EmbeddingGemma2Model.loadWeights(into: model, resources: resources, dtype: .float32))
            try MLX.save(arrays: ["language_model.embedding_projection.weight": MLXArray.ones([1, 1])], url: resources.weightsURL)
            XCTAssertThrowsError(try EmbeddingGemma2Model.loadWeights(into: model, resources: resources, dtype: .float32))
        }
    }

    func testShardedCheckpointCompletenessIsCheckedAcrossAllShards() throws {
        let model = try EmbeddingGemma2TextModel(config: configuration())
        try withRoot { resources in
            let weights = model.parameters().flattened().sorted { $0.0 < $1.0 }
            var weightMap: [String: String] = [:]
            for (index, group) in [Array(weights.prefix(weights.count / 2)), Array(weights.suffix(weights.count - weights.count / 2))].enumerated() {
                let filename = "model-\(index).safetensors"
                let shard = Dictionary(uniqueKeysWithValues: group.map { ("language_model." + $0.0, MLXArray.ones($0.1.shape)) })
                for key in shard.keys { weightMap[key] = filename }
                try MLX.save(arrays: shard, url: resources.rootURL.appending(path: filename))
            }
            let index = HFSafetensorsIndex(metadata: nil, weightMap: weightMap)
            try JSONEncoder().encode(index).write(to: resources.indexURL)
            try EmbeddingGemma2Model.loadWeights(into: model, resources: resources, dtype: .float32)
        }
    }

    func testTokenTruncationPreservesBOSAndEOS() throws {
        let text = try configuration().textConfig
        XCTAssertEqual(EmbeddingGemma2Model.tokenIDs(body: [4, 5, 6, 7], config: text, limit: 4), [2, 4, 5, 1])
        XCTAssertEqual(EmbeddingGemma2Model.tokenIDs(body: [], config: text, limit: 4), [2, 1])
        XCTAssertEqual(EmbeddingGemma2Model.tokenIDs(body: [4, 5], config: text, limit: 2), [2, 1])
    }
}
