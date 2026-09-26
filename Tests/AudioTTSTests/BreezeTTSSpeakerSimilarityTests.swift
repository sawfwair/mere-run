import AudioQwen3TTSModel
import AudioTTS
import Foundation
import MLX
import MLXNN
import XCTest

/// Opt-in independent speaker embedding comparison for a real Breeze clone.
final class BreezeTTSSpeakerSimilarityTests: XCTestCase {
    func testCloneSpeakerEmbeddingAgainstReference() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let qwenRoot = environment["MERERUN_TEST_QWEN_TTS_ROOT"],
              let referencePath = environment["MERERUN_TEST_BREEZE_REFERENCE"],
              let clonePath = environment["MERERUN_TEST_BREEZE_CLONE"] else {
            throw XCTSkip("Set Qwen speaker encoder root, Breeze reference, and clone paths to compare speakers.")
        }
        let root = URL(fileURLWithPath: qwenRoot)
        let config = try Qwen3TTSModelConfig.load(from: root.appendingPathComponent("config.json"))
        let speakerConfig = try XCTUnwrap(config.speakerEncoderConfig)
        let encoder = Qwen3TTSSpeakerEncoder(config: speakerConfig)
        let weights = try MLX.loadArrays(url: root.appendingPathComponent("speaker_encoder/model.safetensors"))
        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights)
        XCTAssertFalse(sanitized.isEmpty)
        try encoder.update(parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }), verify: .none)

        func embedding(_ path: String) throws -> [Float] {
            let audio = try Qwen3TTSAudioPreprocessor.loadAndProcess(
                from: URL(fileURLWithPath: path), minDuration: 1
            )
            return encoder.extractEmbedding(audio: audio.samples).reshaped(-1).asArray(Float.self)
        }
        func cosine(_ lhs: [Float], _ rhs: [Float]) -> Float {
            let dot = zip(lhs, rhs).reduce(Float.zero) { $0 + $1.0 * $1.1 }
            let lhsNorm = sqrt(lhs.reduce(Float.zero) { $0 + $1 * $1 })
            let rhsNorm = sqrt(rhs.reduce(Float.zero) { $0 + $1 * $1 })
            return dot / (lhsNorm * rhsNorm)
        }
        let reference = try embedding(referencePath)
        let clone = try embedding(clonePath)
        XCTAssertEqual(reference.count, clone.count)
        let score = cosine(reference, clone)
        XCTAssertTrue(score.isFinite)
        print("Breeze clone/reference Qwen speaker embedding cosine: \(score)")
        if let comparisonPath = environment["MERERUN_TEST_BREEZE_COMPARISON"] {
            let comparisonScore = cosine(reference, try embedding(comparisonPath))
            XCTAssertTrue(comparisonScore.isFinite)
            print("Breeze comparison/reference Qwen speaker embedding cosine: \(comparisonScore)")
        }
    }
}
