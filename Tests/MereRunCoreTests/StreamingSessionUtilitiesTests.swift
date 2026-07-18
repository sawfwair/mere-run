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

    func testDecodeCadenceUsesMinimumForFirstDecodeThenInterval() {
        var cadence = StreamingDecodeCadence(
            sampleRate: 1_000,
            decodeIntervalMs: 2_000,
            minDecodeAudioMs: 1_600
        )

        XCTAssertFalse(cadence.shouldDecode(bufferedSampleCount: 1_599))
        XCTAssertTrue(cadence.shouldDecode(bufferedSampleCount: 1_600))
        cadence.markDecoded(sampleCount: 1_600)
        XCTAssertFalse(cadence.shouldDecode(bufferedSampleCount: 3_599))
        XCTAssertTrue(cadence.shouldDecode(bufferedSampleCount: 3_600))
    }
}
