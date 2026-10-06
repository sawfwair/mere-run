import Foundation
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunGemmaModel

final class EmbeddingGemma2MediaTests: MLXTestCase {
    struct Tensor: Decodable { let shape: [Int]; let values: [Float] }
    struct Vision: Decodable {
        let weights: [String: Tensor]
        let pixels: [[[Float]]]
        let positions: [[Int]]
        let expected: [[Float]]
    }
    struct Audio: Decodable {
        let weights: [String: Tensor]
        let features: [[[Float]]]
        let mask: [Bool]
        let expected: [[Float]]
    }
    struct Fixture: Decodable {
        let textHiddenSize: Int
        let visionConfig: EmbeddingGemma2VisionConfig
        let audioConfig: EmbeddingGemma2AudioConfig
        let vision: Vision
        let audio: Audio
        enum CodingKeys: String, CodingKey {
            case textHiddenSize = "text_hidden_size", visionConfig = "vision_config", audioConfig = "audio_config", vision, audio
        }
    }
    private func fixture() throws -> Fixture {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appending(path: "MereRunCoreTests/Fixtures/EmbeddingGemma2/media.json")))
    }
    private func arrays(_ weights: [String: Tensor]) -> [String: MLXArray] {
        weights.mapValues { MLXArray($0.values, $0.shape) }
    }
    func testVisionMatchesUpstreamWithAxialPositionsAndPadding() throws {
        let f = try fixture()
        let model = try EmbeddingGemma2VisionModel(config: f.visionConfig, textHiddenSize: f.textHiddenSize,
            tensors: arrays(f.vision.weights), dtype: .float32)
        let result = try model(pixels: MLXArray(f.vision.pixels.flatMap { $0.flatMap { $0 } }, [1, 16, 12]), positions: f.vision.positions)
        eval(result)
        XCTAssertEqual(result.shape, [2, 8])
        for (a, b) in zip(result.asArray(Float.self), f.vision.expected.flatMap({ $0 })) { XCTAssertEqual(a, b, accuracy: 0.00002) }
    }
    func testAudioMatchesUpstreamWithClippingPartialChunksAndCausalConvolution() throws {
        let f = try fixture()
        let model = try EmbeddingGemma2AudioModel(config: f.audioConfig, textHiddenSize: f.textHiddenSize,
            tensors: arrays(f.audio.weights), dtype: .float32)
        let result = try model(features: MLXArray(f.audio.features.flatMap { $0.flatMap { $0 } }, [1, 13, 128]), validFrames: f.audio.mask)
        eval(result)
        XCTAssertEqual(result.shape, [3, 8])
        for (a, b) in zip(result.asArray(Float.self), f.audio.expected.flatMap({ $0 })) { XCTAssertEqual(a, b, accuracy: 0.00002) }
    }
    func testVisionRejectsIncompleteAndDuplicatePoolingGrids() throws {
        let f = try fixture()
        let model = try EmbeddingGemma2VisionModel(config: f.visionConfig, textHiddenSize: f.textHiddenSize,
            tensors: arrays(f.vision.weights), dtype: .float32)
        let pixels = MLXArray(f.vision.pixels.flatMap { $0.flatMap { $0 } }, [1, 16, 12])
        var positions = f.vision.positions
        positions[1] = positions[0]
        XCTAssertThrowsError(try model(pixels: pixels, positions: positions))
        positions = f.vision.positions
        positions[0] = [-1, -1]
        XCTAssertThrowsError(try model(pixels: pixels, positions: positions))
        XCTAssertThrowsError(try model(pixels: MLXArray(Float(0)), positions: positions))
    }

    func testMediaCheckpointRejectsMissingShapeMismatchedAndUnexpectedTensors() throws {
        let f = try fixture()
        var weights = arrays(f.vision.weights)
        weights.removeValue(forKey: "vision_tower.patch_embedder.input_proj.weight")
        XCTAssertThrowsError(try EmbeddingGemma2VisionModel(config: f.visionConfig, textHiddenSize: 8, tensors: weights))
        weights = arrays(f.vision.weights)
        weights["embed_vision.embedding_projection.weight"] = MLXArray.zeros([7, 8])
        XCTAssertThrowsError(try EmbeddingGemma2VisionModel(config: f.visionConfig, textHiddenSize: 8, tensors: weights))
        weights = arrays(f.vision.weights)
        weights["extra"] = MLXArray(Float(1))
        XCTAssertThrowsError(try EmbeddingGemma2VisionModel(config: f.visionConfig, textHiddenSize: 8, tensors: weights))
        XCTAssertThrowsError(try EmbeddingGemma2AudioModel(config: f.audioConfig, textHiddenSize: 8, tensors: arrays(f.audio.weights), dtype: .float16))
    }
}
