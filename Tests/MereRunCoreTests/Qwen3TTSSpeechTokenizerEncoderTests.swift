import MLX
import XCTest
@testable import MereRunCore
@testable import AudioTTS

final class Qwen3TTSSpeechTokenizerEncoderTests: MereRunCoreTestCase {
    func testEncodeReferenceProducesExpectedShapeAndRange() {
        let config = Qwen3TTSTokenizerConfig()
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)

        let sampleRate = config.inputSampleRate
        let sampleCount = sampleRate * 3
        let samples = (0..<sampleCount).map { idx -> Float in
            let t = Float(idx) / Float(sampleRate)
            return sin(2.0 * Float.pi * 220.0 * t) * 0.25
        }

        let codes = tokenizer.encode(samples: samples, sampleRate: sampleRate)
        MLX.eval(codes)

        XCTAssertEqual(codes.ndim, 3)
        XCTAssertEqual(codes.dim(0), 1)
        XCTAssertEqual(codes.dim(2), config.encoderValidNumQuantizers)
        XCTAssertGreaterThan(codes.dim(1), 0)

        let values = codes.asArray(Int32.self)
        XCTAssertTrue(values.allSatisfy { $0 >= 0 && Int($0) < config.decoderConfig.codebookSize })
    }
}
