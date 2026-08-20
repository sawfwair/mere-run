import XCTest

final class RuntimeResidencyStateTests: XCTestCase {
    func testReleaseWaitsForActiveInferenceThenBlocksNewWork() {
        var state = RuntimeResidencyState()

        XCTAssertTrue(state.begin(.chatting))
        XCTAssertFalse(state.requestRelease())
        XCTAssertEqual(state.activity, .chatting)
        XCTAssertTrue(state.releasePending)

        XCTAssertTrue(state.completeActivity())
        XCTAssertEqual(state.activity, .releasing)
        XCTAssertFalse(state.begin(.generatingImage))

        state.completeRelease()
        XCTAssertEqual(state.activity, .idle)
        XCTAssertTrue(state.begin(.generatingImage))
    }

    func testIdleReleaseStartsImmediately() {
        var state = RuntimeResidencyState()

        XCTAssertTrue(state.requestRelease())
        XCTAssertEqual(state.activity, .releasing)
        XCTAssertFalse(state.releasePending)
        XCTAssertFalse(state.requestRelease())
        XCTAssertFalse(state.releasePending)
    }
}
