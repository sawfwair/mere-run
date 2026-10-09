import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunQwenModel

final class PPLXEmbedV2EncoderTests: MLXTestCase {
    struct Tensor: Decodable { let shape: [Int]; let values: [Float] }
    struct Fixture: Decodable {
        let config: Q35Config
        let weights: [String: Tensor]
        let inputIDs: [Int32]
        let hidden: [[Float]]
        let imageInputIDs: [Int32]
        let pixelPatches: [[Float]]
        let vision: [[Float]]
        let imageHidden: [[Float]]
        let imagePixels: [Float]
        enum CodingKeys: String, CodingKey {
            case config, weights, hidden, vision
            case inputIDs = "input_ids", imageInputIDs = "image_input_ids", pixelPatches = "pixel_patches", imageHidden = "image_hidden", imagePixels = "image_pixels"
        }
    }

    func fixture() throws -> Fixture {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appending(path: "MereRunCoreTests/Fixtures/PPLXEmbedV2/reference.json")))
    }

    func encoder(_ fixture: Fixture) throws -> PPLXEmbedV2Encoder {
        let encoder = PPLXEmbedV2Encoder(config: fixture.config)
        let mapped = fixture.weights.compactMap { key, tensor -> (String, MLXArray)? in
            guard key.hasPrefix("language_model.") else { return nil }
            let name = String(key.dropFirst("language_model.".count))
            let array = MLXArray(tensor.values, tensor.shape)
            return (name, name.hasSuffix(".conv1d.weight") ? array.transposed(0, 2, 1) : array)
        }
        try encoder.update(parameters: ModuleParameters.unflattened(mapped), verify: [.noUnusedKeys, .shapeMismatch])
        return encoder
    }

    func assertClose(_ actual: MLXArray, _ expected: [[Float]], file: StaticString = #filePath, line: UInt = #line) {
        MLX.eval(actual)
        let flat = actual.asArray(Float.self), reference = expected.flatMap { $0 }
        XCTAssertEqual(flat.count, reference.count, file: file, line: line)
        for (value, target) in zip(flat, reference) { XCTAssertEqual(value, target, accuracy: 0.00002, file: file, line: line) }
    }

    func testHybridBidirectionalEncoderMatchesTransformers54() throws {
        let fixture = try fixture(), encoder = try encoder(fixture)
        let ids = MLXArray(fixture.inputIDs, [1, fixture.inputIDs.count])
        assertClose(encoder(inputIDs: ids), fixture.hidden)
        // A future token changes the first token after the full-attention layer.
        var changed = fixture.inputIDs; changed[changed.count - 1] = 6
        let output = encoder(inputIDs: MLXArray(changed, [1, changed.count]))
        MLX.eval(output)
        XCTAssertGreaterThan(abs(output[0, 0, 0].item(Float.self) - fixture.hidden[0][0]), 0.0001)
    }

    func testFP32ImageTowerAndMultimodalEncoderMatchTransformers54() throws {
        let fixture = try fixture(), encoder = try encoder(fixture)
        let tower = Q35VisionTower(config: fixture.config, checkpointDType: .float32)
        let mapped = Dictionary(uniqueKeysWithValues: fixture.weights.flatMap { key, tensor in
            Q35VisionTower.mapVisionWeight(key, MLXArray(tensor.values, tensor.shape))
        })
        try tower.installMappedWeights(mapped)
        let pixels = MLXArray(fixture.imagePixels, [1, 3, 8, 8])
        let visual = try tower.encodeImage(pixelValues: pixels, gridTHW: (1, 4, 4))
        assertClose(visual, fixture.vision)
        let ids = MLXArray(fixture.imageInputIDs, [1, 7])
        let embedded = encoder.embedTokens(ids)
        embedded[0, 2..<6, 0...] = visual
        let positions = MLXArray([Int32(0), 1, 2, 2, 2, 2, 4,
                                 0, 1, 2, 2, 3, 3, 4,
                                 0, 1, 2, 3, 2, 3, 4], [3, 1, 7])
        assertClose(encoder(inputIDs: ids, embeddings: embedded, positionIDs: positions), fixture.imageHidden)
    }

    func testContextualProjectionRoundsToEvenBeforeTruncationAndNormalization() {
        let hidden = MLXArray([Float(1), 0, 3, 2, 0, 4], [1, 3, 2])
        let projection = MLXArray([Float(0.001), 0.002, -0.001, 0.001], [2, 2])
        let raw = PPLXEmbedV2Encoder.contextualVectors(hidden: hidden, spans: [0..<2, 2..<3, 0..<0],
                                                      projection: projection, dimensions: 2, normalize: false)
        MLX.eval(raw)
        XCTAssertEqual(raw.asArray(Float.self), [1, 0, 1, 1, 0, 0])
        let truncated = PPLXEmbedV2Encoder.contextualVectors(hidden: hidden, spans: [0..<2],
                                                            projection: projection, dimensions: 1, normalize: true)
        XCTAssertEqual(truncated.item(Float.self), 1)
    }

    func testInactiveAttentionModulesAreAbsent() throws {
        let model = try encoder(fixture())
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertFalse(keys.contains { $0.hasPrefix("layers.0.self_attn.") || $0.hasPrefix("layers.1.linear_attn.") })
    }
}
