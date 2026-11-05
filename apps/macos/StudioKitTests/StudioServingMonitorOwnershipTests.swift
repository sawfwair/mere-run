import Foundation
import StudioTestSupport
import XCTest
@testable import StudioKit

@MainActor
final class StudioServingMonitorOwnershipTests: XCTestCase {
    func testStopAlsoPreventsARefreshFromStartingItsNextAgentProbe() async {
        let runner = RecordingProcessRunner()
        runner.launchError = StubLaunchError()
        let controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false,
                                           taskSessions: StudioTaskSessions())
        let gate = RuntimeResponseGate()
        let monitor = StudioServingMonitor(fetch: { try await gate.read($0) })
        let originalDetail = monitor.agentDetail
        let refresh = Task { await monitor.refreshNow(controller: controller) }
        await gate.waitUntilStarted()
        monitor.stop()
        await gate.resume()
        await refresh.value
        XCTAssertEqual(monitor.agentDetail, originalDetail)
        XCTAssertNotNil(runner.launchError, "the agent probe must not start after Stop")
    }

    func testStopDiscardsAnUncooperativeInFlightResponse() async {
        let controller = controller()
        let gate = RuntimeResponseGate()
        let monitor = StudioServingMonitor(fetch: { try await gate.read($0) })
        let refresh = Task { await monitor.refreshRuntimeNow(controller: controller) }
        await gate.waitUntilStarted()
        monitor.stop()
        await gate.resume()
        await refresh.value
        XCTAssertNil(monitor.runtime)
        XCTAssertNil(monitor.lastAnsweredAt)
        XCTAssertFalse(monitor.isReachable)
        XCTAssertFalse(monitor.isRefreshing)
        XCTAssertTrue(monitor.activities.isEmpty)
    }

    func testChangingCredentialsDiscardsTheOldResponseEvenBeforeAnotherRefresh() async {
        let controller = controller()
        let gate = RuntimeResponseGate()
        let monitor = StudioServingMonitor(fetch: { try await gate.read($0) })
        let refresh = Task { await monitor.refreshRuntimeNow(controller: controller) }
        await gate.waitUntilStarted()
        controller.runtimeAPIKey = "replacement-test-key"
        await gate.resume()
        await refresh.value
        XCTAssertNil(monitor.runtime)
        XCTAssertFalse(monitor.isReachable)
        XCTAssertNil(monitor.lastAnsweredAt)
    }

    func testFreshRefreshDoesNotWaitForOrPublishTheSupersededResponse() async {
        let controller = controller()
        let gate = RuntimeResponseGate()
        let freshRead = expectation(description: "new request starts before the old response returns")
        let monitor = StudioServingMonitor(fetch: { request in
            if request.value(forHTTPHeaderField: "Authorization") == nil { return try await gate.read(request) }
            freshRead.fulfill()
            return (Data(#"{"object":"new"}"#.utf8), HTTPURLResponse(url: request.url!, statusCode: 200,
                                                                   httpVersion: nil, headerFields: nil)!)
        })
        let old = Task { await monitor.refreshRuntimeNow(controller: controller) }
        await gate.waitUntilStarted()
        controller.runtimeAPIKey = "new-test-key"
        let fresh = Task { await monitor.refreshRuntimeNow(controller: controller) }
        await fulfillment(of: [freshRead], timeout: 2)
        await gate.resume()
        await old.value
        await fresh.value
        XCTAssertEqual(monitor.runtime?.object, "new")
        XCTAssertFalse(monitor.activities.contains { $0.detail?.contains("old") == true })
    }

    private func controller() -> MereRunController {
        MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false,
                          taskSessions: StudioTaskSessions())
    }
}

private actor RuntimeResponseGate {
    private var started = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var responseWaiter: CheckedContinuation<Void, Never>?

    func read(_ request: URLRequest) async throws -> (Data, URLResponse) {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        await withCheckedContinuation { responseWaiter = $0 }
        return (Data(#"{"object":"old"}"#.utf8), HTTPURLResponse(url: request.url!, statusCode: 200,
                                                               httpVersion: nil, headerFields: nil)!)
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func resume() { responseWaiter?.resume(); responseWaiter = nil }
}
