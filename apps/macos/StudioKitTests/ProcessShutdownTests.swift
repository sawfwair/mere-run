import XCTest
@testable import StudioKit

final class ProcessShutdownTests: XCTestCase {
    func testCancellationEscalatesOnceAndWaitsForExitAndBothStreams() {
        var state = ProcessShutdown()
        XCTAssertEqual(advance(&state, cancelled: true, at: 0), [.terminate])
        XCTAssertEqual(advance(&state, cancelled: true, at: 0.19), [])
        XCTAssertEqual(advance(&state, cancelled: true, at: 0.21), [.kill])
        XCTAssertEqual(advance(&state, cancelled: true, at: 1), [])
        XCTAssertFalse(state.isComplete, "sending SIGKILL does not establish process exit")
        XCTAssertEqual(advance(&state, running: false, group: false, at: 2), [])
        XCTAssertFalse(state.isComplete, "process exit does not establish output EOF")
        XCTAssertEqual(advance(&state, running: false, group: false, eof: true, at: 2.1), [])
        XCTAssertTrue(state.isComplete)
        XCTAssertEqual(advance(&state, cancelled: true, at: 3), [], "completion is terminal")
    }

    func testParentExitCleansUpOwnedDescendantsEvenWhenStreamsAreClosed() {
        var state = ProcessShutdown()
        XCTAssertEqual(advance(&state, running: false, eof: true, at: 0), [.terminate])
        XCTAssertFalse(state.isComplete, "EOF does not establish descendant exit")
        XCTAssertEqual(advance(&state, running: false, eof: true, at: 0.21), [.kill])
        XCTAssertFalse(state.isComplete)
        XCTAssertEqual(advance(&state, running: false, group: false, eof: true, at: 0.22), [])
        XCTAssertTrue(state.isComplete)
    }

    func testDetachedPipeDrainDeadlineStartsAtParentExit() {
        var state = ProcessShutdown()
        _ = advance(&state, at: 10)
        XCTAssertEqual(advance(&state, running: false, group: false, at: 20), [])
        _ = advance(&state, running: false, group: false, at: 20.39)
        XCTAssertFalse(state.isComplete)
        XCTAssertEqual(advance(&state, running: false, group: false, at: 20.41), [])
        XCTAssertTrue(state.isComplete, "a detached pipe cannot hold completion forever")
    }

    func testClosedStreamsDoNotFinishARunningProcess() {
        var state = ProcessShutdown()
        XCTAssertEqual(advance(&state, eof: true, at: 0), [])
        XCTAssertEqual(advance(&state, eof: true, at: 10), [])
        XCTAssertFalse(state.isComplete)
    }

    func testOutputFailureInitiatesGroupTerminationWithoutUserCancellation() {
        var state = ProcessShutdown()
        XCTAssertEqual(advance(&state, failed: true, at: 0), [.terminate])
        XCTAssertEqual(advance(&state, failed: true, at: 0.21), [.kill])
        XCTAssertEqual(advance(&state, running: false, group: false, eof: true, failed: true, at: 0.3), [])
        XCTAssertTrue(state.isComplete)
    }

    private func advance(
        _ state: inout ProcessShutdown, running: Bool = true, group: Bool = true,
        eof: Bool = false, failed: Bool = false, cancelled: Bool = false, at time: TimeInterval
    ) -> [ProcessShutdown.Action] {
        state.advance(processRunning: running, ownsLiveGroup: group, streamsAtEnd: eof,
                      outputFailed: failed, terminationRequested: cancelled, now: time)
    }
}
