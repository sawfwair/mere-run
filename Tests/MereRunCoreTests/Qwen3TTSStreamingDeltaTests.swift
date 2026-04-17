import XCTest
@testable import MereRunCore
import AudioCore

final class Qwen3TTSStreamingDeltaTests: XCTestCase {
    func testStreamingDeltaEmitsOnlyNewTailSamples() {
        let first = streamingAudioNewTail(
            fullSamples: [0.1, 0.2, 0.3],
            emittedSampleCount: 0
        )
        XCTAssertEqual(first.samples, [0.1, 0.2, 0.3])
        XCTAssertEqual(first.updatedSampleCount, 3)

        let second = streamingAudioNewTail(
            fullSamples: [0.1, 0.2, 0.3, 0.4, 0.5],
            emittedSampleCount: first.updatedSampleCount
        )
        XCTAssertEqual(second.samples, [0.4, 0.5])
        XCTAssertEqual(second.updatedSampleCount, 5)
    }

    func testStreamingDeltaDoesNotDuplicateWhenDecodeShrinks() {
        let delta = streamingAudioNewTail(
            fullSamples: [0.1, 0.2],
            emittedSampleCount: 5
        )
        XCTAssertTrue(delta.samples.isEmpty)
        XCTAssertEqual(delta.updatedSampleCount, 2)
    }
}
