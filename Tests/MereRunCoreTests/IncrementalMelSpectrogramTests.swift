import XCTest
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
}
