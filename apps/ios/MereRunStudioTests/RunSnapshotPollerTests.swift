import XCTest

@MainActor
final class RunSnapshotPollerTests: XCTestCase {
    private enum ExpectedFailure: Error { case offline }

    private let policy = RunPollingPolicy(
        regularDelayNanoseconds: 2,
        maximumDelayNanoseconds: 10,
        maximumConsecutiveFailures: 3
    )

    func testInitialFailureRecoversWithoutAnExistingJob() async {
        let poller = RunSnapshotPoller(policy: policy)
        var reads = 0
        var values: [Int] = []
        var failures = 0
        var delays: [UInt64] = []
        await poller.watch(read: {
            reads += 1
            if reads == 1 { throw ExpectedFailure.offline }
            return 42
        }, isTerminal: { _ in true }, receive: { values.append($0) }, failed: { _ in
            failures += 1
        }, sleep: { delays.append($0) })

        XCTAssertEqual(reads, 2)
        XCTAssertEqual(values, [42])
        XCTAssertEqual(failures, 1)
        XCTAssertEqual(delays, [2])
        XCTAssertEqual(poller.phase, .terminal)
    }

    func testPersistentFailuresPauseAtTheBudgetAndCanRestart() async {
        let poller = RunSnapshotPoller(policy: policy)
        var failures = 0
        var delays: [UInt64] = []
        await poller.watch(read: { () -> Int in throw ExpectedFailure.offline },
                           isTerminal: { _ in false }, receive: { _ in XCTFail("no snapshot exists") },
                           failed: { _ in failures += 1 },
                           sleep: { delays.append($0) })

        XCTAssertEqual(failures, 3)
        XCTAssertEqual(delays, [2, 4])
        XCTAssertEqual(poller.phase, .paused)

        var values: [Int] = []
        await poller.watch(read: { 1 }, isTerminal: { _ in true }, receive: { values.append($0) },
                           failed: { _ in XCTFail("restart succeeds") })
        XCTAssertEqual(values, [1])
        XCTAssertEqual(poller.phase, .terminal)
    }

    func testSuccessfulSnapshotResetsTheConsecutiveFailureBudget() async {
        let poller = RunSnapshotPoller(policy: policy)
        var reads = 0
        var values: [Int] = []
        var delays: [UInt64] = []
        await poller.watch(read: {
            reads += 1
            if [1, 2, 4].contains(reads) { throw ExpectedFailure.offline }
            return reads
        }, isTerminal: { $0 == 5 }, receive: { values.append($0) }, failed: { _ in },
                           sleep: { delays.append($0) })

        XCTAssertEqual(values, [3, 5])
        XCTAssertEqual(delays, [2, 4, 2, 2])
        XCTAssertEqual(poller.phase, .terminal)
    }

    func testTerminalSnapshotStopsAndRetryStartsAFreshWatch() async {
        let poller = RunSnapshotPoller(policy: policy)
        var values: [Int] = []
        for value in [1, 2] {
            await poller.watch(read: { value }, isTerminal: { _ in true }, receive: { values.append($0) },
                               failed: { _ in XCTFail("read succeeds") },
                               sleep: { _ in XCTFail("terminal runs must not sleep or poll again") })
            XCTAssertEqual(poller.phase, .terminal)
        }
        XCTAssertEqual(values, [1, 2])
    }

    func testStopRejectsAReadThatIgnoresCancellation() async {
        let poller = RunSnapshotPoller(policy: policy)
        let gate = SnapshotReadGate()
        let task = Task {
            await poller.watch(read: { try await gate.read() }, isTerminal: { _ in true },
                               receive: { _ in XCTFail("stopped read cannot publish") },
                               failed: { _ in XCTFail("stopped read cannot publish") })
        }
        await gate.waitUntilStarted()
        poller.stop()
        gate.finish(.success(1))
        await task.value
        XCTAssertEqual(poller.phase, .stopped)
    }

    func testSupersededReadCannotOverwriteTheNewWatchOrItsPhase() async {
        await assertSupersededReadIsIgnored(.success(1))
    }

    func testSupersededFailureCannotPauseTheNewWatch() async {
        await assertSupersededReadIsIgnored(.failure(ExpectedFailure.offline))
    }

    func testTaskCancellationDuringSleepStopsWithoutAnotherReadOrError() async {
        let poller = RunSnapshotPoller(policy: policy)
        let gate = SnapshotReadGate()
        var reads = 0
        let task = Task {
            await poller.watch(read: { reads += 1; return reads }, isTerminal: { _ in false },
                               receive: { _ in }, failed: { _ in XCTFail("cancellation is not a connection failure") },
                               sleep: { _ in _ = try await gate.read() })
        }
        await gate.waitUntilStarted()
        task.cancel()
        gate.finish(.success(1))
        await task.value
        XCTAssertEqual(reads, 1)
        XCTAssertEqual(poller.phase, .stopped)
    }

    private func assertSupersededReadIsIgnored(_ result: Result<Int, Error>) async {
        let poller = RunSnapshotPoller(policy: policy)
        let gate = SnapshotReadGate()
        let old = Task {
            await poller.watch(read: { try await gate.read() }, isTerminal: { _ in true },
                               receive: { _ in XCTFail("old snapshot cannot replace the new one") },
                               failed: { _ in XCTFail("old failure cannot replace the new one") })
        }
        await gate.waitUntilStarted()
        var values: [Int] = []
        await poller.watch(read: { 2 }, isTerminal: { _ in true }, receive: { values.append($0) },
                           failed: { _ in XCTFail("new watch succeeds") })
        gate.finish(result)
        await old.value
        XCTAssertEqual(values, [2])
        XCTAssertEqual(poller.phase, .terminal, "an old watch cannot reset the new owner's phase")
    }
}

@MainActor
private final class SnapshotReadGate {
    private var started = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var waiter: CheckedContinuation<Int, Error>?

    func read() async throws -> Int {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        return try await withCheckedThrowingContinuation { waiter = $0 }
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func finish(_ result: Result<Int, Error>) {
        waiter?.resume(with: result)
        waiter = nil
    }
}
