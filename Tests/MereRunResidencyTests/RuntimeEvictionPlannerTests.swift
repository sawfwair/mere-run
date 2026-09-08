import Foundation
import XCTest
import MereRunResidency

final class RuntimeEvictionPlannerTests: XCTestCase {
    func testTTLProtectsBusyQueuedPreparingPinnedAndExcludedResidents() {
        let candidates = [
            candidate("expired"), candidate("busy", active: 1), candidate("queued", queued: 1),
            candidate("preparing", ready: false), candidate("pinned", pinned: true), candidate("protected"),
            candidate("unloaded", loaded: false), candidate("no-ttl", ttl: nil)
        ]
        XCTAssertEqual(
            RuntimeEvictionPlanner.expired(candidates, now: Date(timeIntervalSince1970: 100), excluding: ["protected"]),
            ["expired"]
        )
    }

    func testPressureUsesStableLRUAndRespectsTheEvictionLimit() {
        let candidates = [candidate("b"), candidate("a"), candidate("new", lastAccess: 80), candidate("pinned", pinned: true)]
        XCTAssertEqual(RuntimeEvictionPlanner.memoryPressure(candidates, pressure: .nominal), [])
        XCTAssertEqual(RuntimeEvictionPlanner.memoryPressure(candidates, pressure: .elevated), ["a"])
        XCTAssertEqual(RuntimeEvictionPlanner.memoryPressure(candidates, pressure: .critical), ["a", "b", "new"])
        XCTAssertEqual(RuntimeEvictionPlanner.memoryPressure(candidates, pressure: .critical, excluding: ["a"]), ["b", "new"])
    }

    func testTTLDeadlineDoesNotExpireAnUnaccessedOrFreshResident() {
        let candidates = [candidate("fresh", lastAccess: 95), candidate("never-accessed", lastAccess: nil)]
        XCTAssertTrue(RuntimeEvictionPlanner.expired(candidates, now: Date(timeIntervalSince1970: 100)).isEmpty)
    }

    private func candidate(
        _ key: String, loaded: Bool = true, ready: Bool = true, lastAccess: TimeInterval? = 0,
        active: Int = 0, queued: Int = 0, pinned: Bool = false, ttl: Int? = 10
    ) -> RuntimeEvictionCandidate<String> {
        RuntimeEvictionCandidate(
            key: key, sortKey: key, loaded: loaded, ready: ready,
            lastAccess: lastAccess.map(Date.init(timeIntervalSince1970:)),
            activeRequests: active, queuedRequests: queued, pinned: pinned, ttlSeconds: ttl
        )
    }
}
