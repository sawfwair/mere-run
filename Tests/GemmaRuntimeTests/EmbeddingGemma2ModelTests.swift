import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunGemmaModel

final class EmbeddingGemma2ModelTests: MLXTestCase {
    struct Fixture: Decodable {
        let config: EmbeddingGemma2Config
        let inputIDs: [[Int32]]
        let attentionMask: [[Int32]]
        let tokenEmbeddings: [[[Float]]]
        enum CodingKeys: String, CodingKey {
            case config, inputIDs = "input_ids", attentionMask = "attention_mask", tokenEmbeddings = "token_embeddings"
        }
    }

    private func fixtureData() throws -> Data {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try Data(contentsOf: root.appending(path: "MereRunCoreTests/Fixtures/EmbeddingGemma2/synthetic.json"))
    }

    private func fixture() throws -> Fixture { try JSONDecoder().decode(Fixture.self, from: fixtureData()) }

    private func deterministicModel(_ config: EmbeddingGemma2Config) throws -> EmbeddingGemma2TextModel {
        let model = try EmbeddingGemma2TextModel(config: config)
        let parameters = model.parameters().flattened().map { name, array in
            let phase = name.utf8.reduce(0) { $0 + Int($1) } % 7
            let values: [Float] = (0..<array.size).map { index in
                if name.hasSuffix("layer_scalar") { return 0.875 }
                if name.contains("norm") { return 1 + Float(index % 5 - 2) * 0.03125 }
                return Float(index % 13 - 6) * 0.03125 + Float(phase) * 0.0078125
            }
            return (name, MLXArray(values, array.shape))
        }
        try model.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
        return model
    }

    func testEncoderMatchesIndependentScalarReference() throws {
        let fixture = try fixture()
        let model = try deterministicModel(fixture.config)
        let output = model(inputIDs: MLXArray(fixture.inputIDs.flatMap { $0 }, [2, 5]),
                           attentionMask: MLXArray(fixture.attentionMask.flatMap { $0 }, [2, 5]))
        eval(output)
        let expected = fixture.tokenEmbeddings.flatMap { $0.flatMap { $0 } }
        let actual = output.asArray(Float.self)
        XCTAssertEqual(actual.count, expected.count)
        for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: 0.0001) }
    }

    func testBatchPaddingDoesNotChangeValidTokenEmbeddings() throws {
        let fixture = try fixture()
        let model = try deterministicModel(fixture.config)
        let padded = model(inputIDs: MLXArray(fixture.inputIDs.flatMap { $0 }, [2, 5]),
                           attentionMask: MLXArray(fixture.attentionMask.flatMap { $0 }, [2, 5]))
        let alone = model(inputIDs: MLXArray(Array(fixture.inputIDs[0].prefix(4)), [1, 4]),
                          attentionMask: MLXArray.ones([1, 4], dtype: .int32))
        eval(padded, alone)
        let batchedValid = padded[0, 0..<4, 0...].asArray(Float.self)
        for (a, b) in zip(batchedValid, alone.asArray(Float.self)) { XCTAssertEqual(a, b, accuracy: 0.0001) }
    }

    func testMasksAreBidirectionalWithInclusiveLocalBoundaryAndPadding() {
        let valid = MLXArray([Int32(1), 1, 1, 0], [1, 4])
        let local = EmbeddingGemma2TextModel.attentionMask(validTokens: valid, window: 1, dtype: .float32)
        let full = EmbeddingGemma2TextModel.attentionMask(validTokens: valid, window: nil, dtype: .float32)
        eval(local, full)
        let mask = local.asArray(Float.self)
        XCTAssertEqual(mask[1], 0) // Future token at inclusive distance one.
        XCTAssertLessThan(mask[2], -1e8)
        XCTAssertEqual(mask[4], 0) // Past token at inclusive distance one.
        XCTAssertEqual(mask[6], 0)
        for query in 0..<4 { XCTAssertLessThan(mask[query * 4 + 3], -1e8) }
        XCTAssertEqual(full.asArray(Float.self)[2], 0)
    }

    func testMeanPoolingIncludesValidTokensAndRenormalizesAfterTruncation() {
        let hidden = MLXArray([Float(3), 0, 4, 0, 0, 4, 0, 0, .nan, .nan, .nan, .nan], [1, 3, 4])
        let pooled = EmbeddingGemma2TextModel.pool(tokenEmbeddings: hidden,
                                                attentionMask: MLXArray([Int32(1), 1, 0], [1, 3]), dimensions: 2)
        eval(pooled)
        let vector = pooled.asArray(Float.self)
        XCTAssertEqual(vector[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(vector[1], 0.8, accuracy: 1e-6)
    }

    func testPerLayerGeometryAndProjectionOnlyPLE() throws {
        let fixture = try fixture()
        let model = try EmbeddingGemma2TextModel(config: fixture.config)
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        XCTAssertEqual(shapes["layers.0.self_attn.q_proj.weight"], [8, 8])
        XCTAssertEqual(shapes["layers.1.self_attn.q_proj.weight"], [16, 8])
        XCTAssertEqual(shapes["layers.1.self_attn.k_proj.weight"], [8, 8])
        XCTAssertEqual(shapes["ple.per_layer_model_projection.weight"], [8, 8])
        XCTAssertEqual(shapes["embedding_projection.weight"], [8, 8])
        XCTAssertFalse(shapes.keys.contains { $0.contains("embed_tokens_per_layer") })
    }

    func testInvalidConfigFailsAtTypedBoundary() throws {
        let data = try fixtureData()
        let source = try XCTUnwrap(String(data: data, encoding: .utf8))
        for (from, to) in [("\"embedding_gemma2\"", "\"gemma4\""),
                           ("\"head_dim\": 4", "\"head_dim\": 3"),
                           ("\"attention_bias\": false", "\"attention_bias\": true"),
                           ("\"rope_type\": \"default\"", "\"rope_type\": \"proportional\"")] {
            let changed = Data(source.replacingOccurrences(of: from, with: to).utf8)
            let fixture = try JSONDecoder().decode(Fixture.self, from: changed)
            XCTAssertThrowsError(try fixture.config.validate())
        }
    }
}
