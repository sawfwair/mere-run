import MLX
import XCTest
import MereRunKVCache
import MereRunMLXTestSupport
@testable import AudioQwen3TTSModel

final class Qwen3TTSSpeakerEncoderTests: MLXTestCase {
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
