import XCTest
@testable import MereRunCore
import AudioCore

final class StreamingSessionUtilitiesTests: XCTestCase {
    func testDecodeCadenceRespectsMinWindowAndInterval() {
        var cadence = StreamingDecodeCadence(
            sampleRate: 16_000,
            decodeIntervalMs: 500,
            minDecodeAudioMs: 500
        )

        XCTAssertFalse(cadence.shouldDecode(bufferedSampleCount: 7_999))
        XCTAssertTrue(cadence.shouldDecode(bufferedSampleCount: 8_000))
        XCTAssertTrue(cadence.shouldDecode(bufferedSampleCount: 8_100))

        cadence.markDecoded(sampleCount: 8_100)
        XCTAssertFalse(cadence.shouldDecode(bufferedSampleCount: 15_999))
        XCTAssertFalse(cadence.shouldDecode(bufferedSampleCount: 16_099))
        XCTAssertTrue(cadence.shouldDecode(bufferedSampleCount: 16_100))
    }

    func testDecodeCadenceForceDecodeRequiresNonEmptyBuffer() {
        let cadence = StreamingDecodeCadence(
            sampleRate: 16_000,
            decodeIntervalMs: 500,
            minDecodeAudioMs: 500
        )
        XCTAssertFalse(cadence.shouldDecode(bufferedSampleCount: 0, force: true))
        XCTAssertTrue(cadence.shouldDecode(bufferedSampleCount: 10, force: true))
    }
}
