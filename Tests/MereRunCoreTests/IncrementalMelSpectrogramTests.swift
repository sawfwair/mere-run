import XCTest
import Foundation
import MLX
@testable import MereRunCore
import AudioCodecs

final class IncrementalMelSpectrogramTests: XCTestCase {
    func testAppendSnapshotAndReset() {
        var incremental = IncrementalMelSpectrogram(sampleRate: 16_000)
        incremental.append([0.1, 0.2, 0.3])
        incremental.append([0.4])

        XCTAssertEqual(incremental.sampleCount, 4)
        XCTAssertEqual(incremental.snapshotSamples(), [0.1, 0.2, 0.3, 0.4])
        XCTAssertEqual(incremental.totalAudioSeconds, 4.0 / 16_000.0, accuracy: 0.0000001)

        incremental.removeAll()
        XCTAssertEqual(incremental.sampleCount, 0)
        XCTAssertTrue(incremental.snapshotSamples().isEmpty)
    }

    func testIncrementalExtractionMatchesBatchAcrossGrowingSnapshots() {
        let extractor = MelSpectrogram()
        var incremental = IncrementalMelSpectrogram(sampleRate: 16_000)
        var samples: [Float] = []
        samples.reserveCapacity(8_000)
        for index in 0..<8_000 {
            let sampleIndex = Float(index)
            samples.append(
                (sinf(sampleIndex * 0.017) * 0.4) + (cosf(sampleIndex * 0.031) * 0.1)
            )
        }
        var previousEnd = 0

        for end in [160, 1_750, 4_933, 8_000] {
            incremental.append(Array(samples[previousEnd..<end]))
            previousEnd = end

            let streamed = incremental.extract(using: extractor)
            let batch = extractor.extract(from: Array(samples.prefix(end)))
            MLX.eval(streamed, batch)
            XCTAssertEqual(streamed.shape, batch.shape)

            let streamedValues = streamed.asArray(Float.self)
            let batchValues = batch.asArray(Float.self)
            let maximumDifference = zip(streamedValues, batchValues).reduce(Float.zero) { result, pair in
                Swift.max(result, Swift.abs(pair.0 - pair.1))
            }
            XCTAssertLessThanOrEqual(maximumDifference, 0.000_01)
        }
    }
}
