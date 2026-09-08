import Foundation
import XCTest
@testable import MereRunAdmission

final class RuntimeRequestAdmissionTests: XCTestCase {
    func testCancelledRequestDoesNotTakeAvailableSlot() async throws {
        let start = ControlledPressure()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let request = Task {
            _ = await start.sample()
            do {
                let lease = try await admission.acquire()
                await lease.release()
                XCTFail("An already-cancelled request was admitted")
            } catch is CancellationError {
                // Cancellation is the expected terminal result.
            } catch {
                XCTFail("Unexpected error: \(error)")
            }
        }
        try await waitUntil { await start.entered }
        request.cancel()
        await start.resume()
        await request.value
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 0)
    }

    func testCancellationDuringPressureSampleDoesNotAdmitRequest() async throws {
        let pressure = ControlledPressure()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1, pressureProvider: { await pressure.sample() })
        let request = Task {
            do {
                let lease = try await admission.acquire()
                await lease.release()
                XCTFail("A request cancelled during pressure sampling was admitted")
            } catch is CancellationError {
                // Cancellation is the expected terminal result.
            } catch {
                XCTFail("Unexpected error: \(error)")
            }
        }
        try await waitUntil { await pressure.entered }
        request.cancel()
        await pressure.resume()
        await request.value
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 0)
    }

    func testCancelledMaintenanceDoesNotTakeAvailableSlot() async throws {
        let pressure = ControlledPressure()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1, pressureProvider: { await pressure.sample() })
        let maintenance = Task {
            if let lease = await admission.tryAcquire() {
                await lease.release()
                XCTFail("Cancelled maintenance acquired a request slot")
            }
        }
        try await waitUntil { await pressure.entered }
        maintenance.cancel()
        await pressure.resume()
        await maintenance.value
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 0)
    }

    func testQueuedCancellationNeverRunsScopedOperation() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let active = try await admission.acquire()
        let queued = Task {
            try await withRuntimeRequestAdmission(using: admission) {
                XCTFail("Cancelled queued operation ran")
            }
        }
        try await waitUntil { await admission.snapshot().queuedRequests == 1 }
        queued.cancel()
        do {
            try await queued.value
            XCTFail("Expected queued cancellation")
        } catch is CancellationError {
            // The queue owns cancellation until it grants a lease.
        }
        var snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.queuedRequests, 0)
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.totalCancelledRequests, 1)
        await active.release()
        snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
        XCTAssertEqual(snapshot.totalCompletedRequests, 1)
    }

    private func waitUntil(_ condition: () async -> Bool) async throws {
        let start = ContinuousClock.now
        while !(await condition()) {
            guard start.duration(to: .now) < .seconds(2) else {
                XCTFail("Timed out waiting for the pressure sample")
                return
            }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
    }
}

private actor ControlledPressure {
    private var blocked = true
    private var continuation: CheckedContinuation<RuntimeMemoryPressureLevel, Never>?
    private(set) var entered = false

    func sample() async -> RuntimeMemoryPressureLevel {
        entered = true
        guard blocked else { return .nominal }
        return await withCheckedContinuation { continuation = $0 }
    }

    func resume() {
        blocked = false
        continuation?.resume(returning: .nominal)
        continuation = nil
    }
}
