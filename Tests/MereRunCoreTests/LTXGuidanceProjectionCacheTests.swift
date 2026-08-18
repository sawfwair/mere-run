import XCTest
@testable import MereRunCore

final class LTXGuidanceProjectionCacheTests: XCTestCase {
    private let gibibyte = UInt64(1_073_741_824)

    func testAutomaticCachesRepeatedPositivePassesWithHeadroom() {
        let decision = makeDecision(
            mode: .automatic,
            positivePredictionCount: 3,
            activeGiB: 40,
            physicalGiB: 128
        )

        XCTAssertTrue(decision.shouldCache)
        XCTAssertEqual(decision.estimatedBytes, 1_207_959_552)
        XCTAssertEqual(decision.reserveBytes, 12 * gibibyte + (4 * gibibyte / 5))
    }

    func testAutomaticSkipsSinglePositivePass() {
        let decision = makeDecision(
            mode: .automatic,
            positivePredictionCount: 1,
            activeGiB: 40,
            physicalGiB: 128
        )

        XCTAssertFalse(decision.shouldCache)
    }

    func testDisabledAlwaysSkips() {
        let decision = makeDecision(
            mode: .disabled,
            positivePredictionCount: 4,
            activeGiB: 20,
            physicalGiB: 128
        )

        XCTAssertFalse(decision.shouldCache)
    }

    func testEnabledStillFallsBackWithoutSafeReserve() {
        let decision = makeDecision(
            mode: .enabled,
            positivePredictionCount: 4,
            activeGiB: 57,
            physicalGiB: 64
        )

        XCTAssertFalse(decision.shouldCache)
        XCTAssertEqual(decision.reserveBytes, 8 * gibibyte)
    }

    func testOverflowFallsBack() {
        let decision = ltxGuidanceProjectionCacheDecision(
            mode: .enabled,
            positivePredictionCount: 4,
            batchSize: Int.max,
            videoTextTokens: Int.max,
            audioTextTokens: Int.max,
            blockCount: Int.max,
            bytesPerElement: 2,
            activeMemoryBytes: UInt64.max,
            physicalMemoryBytes: UInt64.max
        )

        XCTAssertFalse(decision.shouldCache)
    }

    private func makeDecision(
        mode: LTXGuidanceProjectionCacheMode,
        positivePredictionCount: Int,
        activeGiB: UInt64,
        physicalGiB: UInt64
    ) -> LTXGuidanceProjectionCacheDecision {
        ltxGuidanceProjectionCacheDecision(
            mode: mode,
            positivePredictionCount: positivePredictionCount,
            batchSize: 1,
            videoTextTokens: 1_024,
            audioTextTokens: 1_024,
            blockCount: 48,
            bytesPerElement: 2,
            activeMemoryBytes: activeGiB * gibibyte,
            physicalMemoryBytes: physicalGiB * gibibyte
        )
    }
}
