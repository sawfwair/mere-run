import Foundation
import MediaIO
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class EmbeddingGemma2MediaProcessorTests: MLXTestCase {
    struct AudioFixture: Decodable {
        let config: EmbeddingGemma2ProcessorConfig.Audio
        let features: [[[Float]]]
        let mask: [Bool]
    }

    func testAudioFrontendMatchesUpstreamIncludingPaddedFinalFrame() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let fixture = try JSONDecoder().decode(AudioFixture.self, from: Data(contentsOf: root.appending(path: "Fixtures/EmbeddingGemma2/audio-frontend.json")))
        let samples = (0..<1_599).map { Float(sin(Double($0) * 0.037) * 0.2) }
        let prepared = try EmbeddingGemma2MediaProcessor.audio(samples: samples, config: fixture.config)
        eval(prepared.features)
        XCTAssertEqual(prepared.validFrames, fixture.mask)
        XCTAssertEqual(prepared.softTokens, 3)
        let actual = prepared.features.asArray(Float.self), expected = fixture.features.flatMap { $0.flatMap { $0 } }
        XCTAssertEqual(actual.count, expected.count)
        for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: 0.002) }
        for samples in [[Float](repeating: 0, count: 160), [.nan, 0, 0], [Float](repeating: 0, count: 480_001)] {
            XCTAssertThrowsError(try EmbeddingGemma2MediaProcessor.audio(samples: samples, config: fixture.config))
        }
    }

    func testImagePatchesPreserveRGBAndSeparatePaddingPositions() throws {
        let source = try MediaImage(width: 9, height: 5, rgba8: Array(repeating: [UInt8(0), 64, 255, 255], count: 45).flatMap { $0 })
        let config = EmbeddingGemma2ProcessorConfig.Vision(patchSize: 2, poolingKernelSize: 2, maxSoftTokens: 4,
            doResize: true, doRescale: true, doNormalize: false, resample: 3, rescaleFactor: Float(1.0 / 255),
            fps: nil, maxFrames: nil, addTimestamps: nil, overflowStrategy: nil)
        let prepared = try EmbeddingGemma2MediaProcessor.image(source, config: config)
        XCTAssertEqual(prepared.softTokens, 2)
        XCTAssertEqual(prepared.positions.prefix(8).map { $0 }, (0..<2).flatMap { y in (0..<4).map { [$0, y] } })
        XCTAssertTrue(prepared.positions.suffix(8).allSatisfy { $0 == [-1, -1] })
        eval(prepared.pixels)
        let pixels = prepared.pixels.asArray(Float.self)
        for index in stride(from: 0, to: 8 * 12, by: 3) {
            XCTAssertEqual(pixels[index], 0)
            XCTAssertEqual(pixels[index + 1], Float(64) / 255, accuracy: 1e-7)
            XCTAssertEqual(pixels[index + 2], 1)
        }
        XCTAssertTrue(pixels.suffix(8 * 12).allSatisfy { $0 == 0 })
    }
}
