import Foundation
import XCTest
import MereRunResidency

final class ResidentRuntimeCacheTests: XCTestCase {
    func testConcurrentColdRequestsShareLoadAndWarmLeasesProtectTheResident() async throws {
        let gate = ResidencyTestGate()
        let probe = ResidencyTestProbe()
        let cache = ResidentRuntimeCache<String, Int>(unload: { await probe.recordUnload($0) })
        let first = Task { try await cache.acquire(for: "model", make: { 7 }, prepare: { _ in await gate.wait() }) }
        try await waitUntil { await gate.entered }
        let second = Task { try await cache.acquire(for: "model", make: { throw ResidencyTestFailure.unexpectedLoad }, prepare: { _ in }) }
        try await waitUntil { await cache.snapshots()["model"]?.waitingRequests == 2 }
        await gate.open()
        let lease1 = try await first.value
        let lease2 = try await second.value
        XCTAssertEqual(lease1.value, 7)
        XCTAssertEqual(lease1.generation, lease2.generation)
        let active = try await requiredSnapshot(cache, key: "model")
        XCTAssertEqual(active.activeRequests, 2)
        let evicted = await cache.evictIfIdle(key: "model", generation: active.generation, accessGeneration: active.accessGeneration)
        XCTAssertFalse(evicted)
        do {
            try await cache.unload(key: "model")
            XCTFail("Unloaded a model with active leases")
        } catch ResidentRuntimeError.activeLeases(let count) {
            XCTAssertEqual(count, 2)
        }
        async let release1: Void = lease1.release()
        async let releaseAgain: Void = lease1.release()
        _ = await (release1, releaseAgain)
        let oneActive = await cache.snapshots()["model"]?.activeRequests
        XCTAssertEqual(oneActive, 1)
        await lease2.release()

        let idle = try await requiredSnapshot(cache, key: "model")
        let warm = try await cache.acquire(for: "model", make: { throw ResidencyTestFailure.unexpectedLoad }, prepare: { _ in })
        XCTAssertEqual(warm.value, 7)
        await warm.release()
        let staleEviction = await cache.evictIfIdle(key: "model", generation: idle.generation, accessGeneration: idle.accessGeneration)
        XCTAssertFalse(staleEviction, "An older TTL/LRU decision must not ignore a completed warm request")
        let current = try await requiredSnapshot(cache, key: "model")
        let finalEviction = await cache.evictIfIdle(key: "model", generation: current.generation, accessGeneration: current.accessGeneration)
        XCTAssertTrue(finalEviction)
        let unloaded = await probe.unloaded
        XCTAssertEqual(unloaded, [7])
    }

    func testCancellingOneColdWaiterPreservesTheOtherWaiter() async throws {
        let gate = ResidencyTestGate()
        let probe = ResidencyTestProbe()
        let cache = ResidentRuntimeCache<String, Int>(unload: { await probe.recordUnload($0) })
        let cancelled = Task { try await cache.acquire(for: "model", make: { 1 }, prepare: { _ in await gate.wait() }) }
        try await waitUntil { await gate.entered }
        let survivor = Task { try await cache.acquire(for: "model", make: { 2 }, prepare: { _ in }) }
        try await waitUntil { await cache.snapshots()["model"]?.waitingRequests == 2 }
        cancelled.cancel()
        try await waitUntil { await cache.snapshots()["model"]?.waitingRequests == 1 }
        await gate.open()
        do {
            _ = try await cancelled.value
            XCTFail("Cancelled waiter acquired a lease")
        } catch is CancellationError {}
        let lease = try await survivor.value
        XCTAssertEqual(lease.value, 1)
        let unloaded = await probe.unloaded
        XCTAssertTrue(unloaded.isEmpty)
        await lease.release()
    }

    func testCancellingTheOnlyWaiterWaitsForPreparationBeforeUnload() async throws {
        let gate = ResidencyTestGate()
        let probe = ResidencyTestProbe()
        let cache = ResidentRuntimeCache<String, Int>(unload: { await probe.recordUnload($0) })
        let task = Task { try await cache.acquire(for: "model", make: { 1 }, prepare: { _ in await gate.wait() }) }
        try await waitUntil { await gate.entered }
        task.cancel()
        try await waitUntil { await cache.snapshots().isEmpty }
        let before = await probe.unloaded
        XCTAssertTrue(before.isEmpty)
        await gate.open()
        do {
            _ = try await task.value
            XCTFail("Cancelled load succeeded")
        } catch is CancellationError {}
        let after = await probe.unloaded
        XCTAssertEqual(after, [1])
    }

    func testOldColdLoadCannotReturnOrUnloadItsReplacement() async throws {
        let gate = ResidencyTestGate()
        let probe = ResidencyTestProbe()
        let cache = ResidentRuntimeCache<String, Int>(unload: { await probe.recordUnload($0) })
        let old = Task { try await cache.acquire(for: "model", make: { 1 }, prepare: { _ in await gate.wait() }) }
        try await waitUntil { await gate.entered }
        let unload = Task { try await cache.unload(key: "model") }
        try await waitUntil { await cache.snapshots().isEmpty }
        let replacement = try await cache.acquire(for: "model", make: { 2 }, prepare: { _ in })
        await gate.open()
        try await unload.value
        do {
            _ = try await old.value
            XCTFail("Old preparation returned a replacement generation")
        } catch is CancellationError {}
        let current = await cache.snapshots()["model"]
        XCTAssertEqual(current?.value, 2)
        XCTAssertEqual(current?.activeRequests, 1)
        let unloaded = await probe.unloaded
        XCTAssertEqual(unloaded, [1])
        await replacement.release()
    }

    func testPreparationFailureFinishesUnloadBeforeReturningTheError() async throws {
        let unloading = ResidencyTestGate()
        let probe = ResidencyTestProbe()
        let cache = ResidentRuntimeCache<String, Int>(unload: { value in
            await unloading.wait()
            await probe.recordUnload(value)
        })
        let task = Task {
            try await cache.acquire(for: "model", make: { 1 }, prepare: { _ in throw ResidencyTestFailure.expected })
        }
        try await waitUntil { await unloading.entered }
        let residents = await cache.snapshots()
        XCTAssertTrue(residents.isEmpty)
        await unloading.open()
        do {
            _ = try await task.value
            XCTFail("Preparation failure succeeded")
        } catch ResidencyTestFailure.expected {}
        let unloaded = await probe.unloaded
        XCTAssertEqual(unloaded, [1])
    }

    func testScopedLeaseReleasesAfterFailureAndCancellation() async throws {
        let cache = ResidentRuntimeCache<String, Int>(unload: { _ in })
        for cancelled in [false, true] {
            let lease = try await cache.acquire(for: "model", make: { 1 }, prepare: { _ in })
            do {
                try await withResidentRuntimeLease(using: lease) { _ in
                    if cancelled { throw CancellationError() }
                    throw ResidencyTestFailure.expected
                }
                XCTFail("Expected operation to fail")
            } catch is CancellationError {
                XCTAssertTrue(cancelled)
            } catch ResidencyTestFailure.expected {
                XCTAssertFalse(cancelled)
            }
            let active = await cache.snapshots()["model"]?.activeRequests
            XCTAssertEqual(active, 0)
        }
    }

    func testCancelledWarmRequestAndColdCoordinatorDoNotAcquireLeases() async throws {
        let cache = ResidentRuntimeCache<String, Int>(unload: { _ in })
        let initial = try await cache.acquire(for: "model", make: { 1 }, prepare: { _ in })
        await initial.release()
        let gate = ResidencyTestGate()
        let coordinator = RuntimeOperationCoordinator()
        let cancelled = Task {
            await gate.wait()
            do {
                _ = try await cache.acquire(for: "model", make: { 2 }, prepare: { _ in })
                XCTFail("Cancelled warm request acquired a lease")
            } catch is CancellationError {}
            do {
                _ = try await coordinator.acquire(.cold)
                XCTFail("Cancelled operation acquired the cold lane")
            } catch is CancellationError {}
        }
        try await waitUntil { await gate.entered }
        cancelled.cancel()
        await gate.open()
        try await cancelled.value
        let active = await cache.snapshots()["model"]?.activeRequests
        XCTAssertEqual(active, 0)
        let cold = try await coordinator.acquire(.cold)
        await cold.release()
    }

    private func requiredSnapshot(_ cache: ResidentRuntimeCache<String, Int>, key: String) async throws -> ResidentRuntimeSnapshot<Int> {
        let snapshot = await cache.snapshots()[key]
        return try XCTUnwrap(snapshot)
    }

    private func waitUntil(_ condition: () async -> Bool) async throws {
        let start = ContinuousClock.now
        while !(await condition()) {
            guard start.duration(to: .now) < .seconds(3) else { throw ResidencyTestFailure.timeout }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
}

private enum ResidencyTestFailure: Error { case expected, unexpectedLoad, timeout }

private actor ResidencyTestGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var entered = false

    func wait() async {
        entered = true
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

private actor ResidencyTestProbe {
    private(set) var unloaded: [Int] = []
    func recordUnload(_ value: Int) { unloaded.append(value) }
}
