import Foundation
import XCTest
import MereRunCore
@testable import MereRunCLI

private final class ScanDouble: @unchecked Sendable {
    private let lock = NSLock()
    private var scans = 0
    private var now = Date(timeIntervalSince1970: 1_000)
    private var gate: DispatchSemaphore?

    var scanCount: Int { lock.withLock { scans } }
    var date: Date { lock.withLock { now } }

    func advance(by seconds: TimeInterval) {
        lock.withLock { now = now.addingTimeInterval(seconds) }
    }

    /// Makes the next scans block until `release()`, as a scan behind a stalled volume would.
    func block() {
        lock.withLock { gate = DispatchSemaphore(value: 0) }
    }

    func release() {
        lock.withLock { gate }?.signal()
    }

    func scan() -> RuntimeInstalledModels {
        let (count, gate) = lock.withLock {
            scans += 1
            return (scans, self.gate)
        }
        gate?.wait()
        return RuntimeInstalledModels(
            installPaths: ["text-chat-gemma4": "/models/scan-\(count)"],
            locationIssues: [ModelLocationIssue(path: "/Volumes/MODELS", problem: .unresponsive)]
        )
    }
}

final class RuntimeInstalledModelsCacheTests: XCTestCase {
    func testCallersReuseAFreshScan() async {
        let double = ScanDouble()
        let cache = RuntimeInstalledModelsCache(maxAge: 10, currentDate: { double.date }, scan: double.scan)

        let first = await cache.models()
        let second = await cache.models()
        XCTAssertEqual(first.installPaths["text-chat-gemma4"], "/models/scan-1")
        XCTAssertEqual(first.locationIssues.map(\.problem), [.unresponsive])
        XCTAssertEqual(second, first)
        XCTAssertEqual(double.scanCount, 1)
    }

    func testStaleScanIsServedWhileABlockedRefreshRuns() async throws {
        let double = ScanDouble()
        let cache = RuntimeInstalledModelsCache(maxAge: 10, currentDate: { double.date }, scan: double.scan)
        _ = await cache.models()

        double.block()
        double.advance(by: 11)
        let stale = await cache.models()
        XCTAssertEqual(stale.installPaths["text-chat-gemma4"], "/models/scan-1")
        let giveUp = Date().addingTimeInterval(5)
        while double.scanCount < 2, Date() < giveUp {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let again = await cache.models()
        XCTAssertEqual(again, stale)
        XCTAssertEqual(double.scanCount, 2, "one refresh runs at a time")

        double.release()
        var refreshed = await cache.models()
        while refreshed.installPaths["text-chat-gemma4"] == "/models/scan-1", Date() < giveUp {
            try await Task.sleep(nanoseconds: 10_000_000)
            refreshed = await cache.models()
        }
        XCTAssertEqual(refreshed.installPaths["text-chat-gemma4"], "/models/scan-2")
    }
}
