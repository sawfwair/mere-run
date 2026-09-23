@testable import StudioKit
import StudioTestSupport
import XCTest

@MainActor
final class StudioServingMonitorTests: XCTestCase {
    override func setUp() {
        super.setUp()
        URLProtocol.registerClass(StalledStatusEndpoint.self)
        StalledStatusEndpoint.requests = []
    }

    override func tearDown() {
        URLProtocol.unregisterClass(StalledStatusEndpoint.self)
        super.tearDown()
    }

    /// A server whose `/runtime/status` stalls — seen live when a model volume stalls a scan — but
    /// whose `/health` answers is up: the monitor says so, and stops piling status requests onto it.
    func testAServerThatAnswersHealthButNotStatusIsUpAndPolledGently() async {
        let controller = MereRunController(
            secretStore: InMemorySecretStore(),
            processRunner: RecordingProcessRunner(),
            resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions()
        )
        let monitor = controller.servingMonitor

        await monitor.refreshRuntimeNow(controller: controller)
        XCTAssertNotNil(monitor.lastAnsweredAt)
        XCTAssertFalse(monitor.isReachable)
        XCTAssertEqual(monitor.connectionDetail, "Up, but not reporting its status")
        XCTAssertEqual(StalledStatusEndpoint.requests, ["/runtime/status", "/health"])

        StalledStatusEndpoint.requests = []
        await monitor.refreshRuntimeNow(controller: controller)
        XCTAssertEqual(StalledStatusEndpoint.requests, ["/health"], "status is retried only now and then")
        XCTAssertNotNil(monitor.lastAnsweredAt)
    }
}

/// `/runtime/status` times out; `/health` answers 200.
private final class StalledStatusEndpoint: URLProtocol {
    nonisolated(unsafe) static var requests: [String] = []

    override class func canInit(with request: URLRequest) -> Bool {
        ["/runtime/status", "/health"].contains(request.url?.path ?? "")
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url?.path ?? ""
        Self.requests.append(path)
        if path == "/health" {
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(#"{"status":"ok"}"#.utf8))
            client?.urlProtocolDidFinishLoading(self)
        } else {
            client?.urlProtocol(self, didFailWithError: URLError(.timedOut))
        }
    }

    override func stopLoading() {}
}
