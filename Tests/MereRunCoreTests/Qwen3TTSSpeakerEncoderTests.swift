import MLX
import XCTest
@testable import MereRunCore
@testable import AudioTTS

final class Qwen3TTSSpeakerEncoderTests: MereRunCoreTestCase {
    func testSpeakerEncoderProducesFiniteEmbeddingWithExpectedShape() {
        let config = Qwen3TTSSpeakerEncoderConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        let mel = MLXArray.zeros([1, 120, config.melDim], dtype: .float32)
        let embedding = encoder(mel)
        MLX.eval(embedding)

        XCTAssertEqual(embedding.ndim, 2)
        XCTAssertEqual(embedding.dim(0), 1)
        XCTAssertEqual(embedding.dim(1), config.encDim)

        let values = embedding.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
    }
}
