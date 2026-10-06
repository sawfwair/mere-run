import Foundation
import MediaIO
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

/// Optional release qualification export; no weights or local paths are committed.
final class EmbeddingGemma2MediaReceiptTests: MLXTestCase {
    func testReleasedMediaPreprocessingReceipt() async throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_EMBEDDINGGEMMA2_MEDIA_RECEIPT"] else {
            throw XCTSkip("Set MERERUN_EMBEDDINGGEMMA2_MEDIA_RECEIPT to a prepared qualification directory.")
        }
        let root = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(EmbeddingGemma2ProcessorConfig.self, from: Data(contentsOf: root.appending(path: "processor_config.json")))
        let image = try EmbeddingGemma2MediaProcessor.image(root.appending(path: "pattern.png"), config: config.imageProcessor)
        let audio = try EmbeddingGemma2MediaProcessor.audio(root.appending(path: "tone.wav"), config: config.featureExtractor)
        let decoded = try MediaAudioIO.decode(root.appending(path: "tone.wav"), targetSampleRate: 16000, channels: 1)
        let frames = try MediaVideoIO.sampleFrames(from: root.appending(path: "clip.mp4"), into: root.appending(path: "native-frames"), framesPerSecond: 1, maximumFrames: 32, strategy: .frameRate)
        let video = try EmbeddingGemma2MediaProcessor.image(frames.frameURLs[0], config: config.videoProcessor)
        try MLX.save(arrays: ["image_pixels": image.pixels, "image_positions": MLXArray(image.positions.flatMap { $0.map(Int32.init) }, [image.positions.count, 2]),
                              "audio_features": audio.features, "audio_mask": MLXArray(audio.validFrames), "audio_samples": MLXArray(decoded.samples),
                              "video_pixels": video.pixels], url: root.appending(path: "native-preprocessing.safetensors"))
        XCTAssertEqual(frames.frameURLs.count, 2)
    }
}
