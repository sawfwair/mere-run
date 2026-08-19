import XCTest

final class RunPollingPolicyTests: XCTestCase {
    func testBackoffIsExponentialAndCapped() {
        let policy = RunPollingPolicy(
            regularDelayNanoseconds: 2,
            maximumDelayNanoseconds: 10,
            maximumConsecutiveFailures: 4
        )

        XCTAssertEqual(policy.delayNanoseconds(afterConsecutiveFailures: 0), 2)
        XCTAssertEqual(policy.delayNanoseconds(afterConsecutiveFailures: 1), 2)
        XCTAssertEqual(policy.delayNanoseconds(afterConsecutiveFailures: 2), 4)
        XCTAssertEqual(policy.delayNanoseconds(afterConsecutiveFailures: 3), 8)
        XCTAssertEqual(policy.delayNanoseconds(afterConsecutiveFailures: 4), 10)
    }

    func testFailureBudgetStopsPersistentPolling() {
        let policy = RunPollingPolicy(maximumConsecutiveFailures: 3)

        XCTAssertFalse(policy.shouldStop(afterConsecutiveFailures: 2))
        XCTAssertTrue(policy.shouldStop(afterConsecutiveFailures: 3))
    }
}
