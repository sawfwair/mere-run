import Foundation
import XCTest
import MereRunAdmission

final class ScopedAdmissionTests: XCTestCase {
    private enum ProbeFailure: Error { case expected }
    private enum Outcome: CaseIterable { case success, failure, cancellation }

    func testRequestScopeReleasesAfterSuccessFailureAndCancellation() async throws {
        for outcome in Outcome.allCases {
            let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
            do {
                let result: Int = try await withRuntimeRequestAdmission(using: admission) {
                    let active = await admission.snapshot()
                    XCTAssertEqual(active.activeRequests, 1)
                    switch outcome {
                    case .success: return 42
                    case .failure: throw ProbeFailure.expected
                    case .cancellation: throw CancellationError()
                    }
                }
                XCTAssertEqual(result, 42)
                XCTAssertEqual(outcome, .success)
            } catch is CancellationError {
                XCTAssertEqual(outcome, .cancellation)
            } catch ProbeFailure.expected {
                XCTAssertEqual(outcome, .failure)
            }
            let snapshot = await admission.snapshot()
            XCTAssertEqual(snapshot.activeRequests, 0)
            XCTAssertEqual(snapshot.queuedRequests, 0)
            XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
            XCTAssertEqual(snapshot.totalCompletedRequests, outcome == .cancellation ? 0 : 1)
            XCTAssertEqual(snapshot.totalCancelledRequests, outcome == .cancellation ? 1 : 0)
        }
    }

    func testConcurrentExplicitReleaseIsCountedOnce() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let lease = try await admission.acquire()
        async let first: Void = lease.release()
        async let second: Void = lease.release()
        _ = await (first, second)
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalCompletedRequests, 1)
    }

    func testMachineScopeReleasesAfterEveryOutcome() async throws {
        let coordinator = try coordinator()
        for outcome in Outcome.allCases {
            do {
                let _: Int = try await withMachineInferenceAdmission(
                    using: coordinator, request: .init(label: "operation", resourceClass: .standard)
                ) {
                    XCTAssertEqual(try coordinator.snapshot().activePermits, 2)
                    switch outcome {
                    case .success: return 42
                    case .failure: throw ProbeFailure.expected
                    case .cancellation: throw CancellationError()
                    }
                }
                XCTAssertEqual(outcome, .success)
            } catch is CancellationError {
                XCTAssertEqual(outcome, .cancellation)
            } catch ProbeFailure.expected {
                XCTAssertEqual(outcome, .failure)
            }
            let snapshot = try coordinator.snapshot()
            XCTAssertEqual(snapshot.activePermits, 0)
            XCTAssertTrue(snapshot.active.isEmpty)
            XCTAssertTrue(snapshot.queued.isEmpty)
        }
    }

    func testRequestConcurrencyStaysInsideOneServerReservation() async throws {
        let coordinator = try coordinator()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 2)
        try await withMachineInferenceAdmission(
            using: coordinator, request: .init(label: "server", resourceClass: .large)
        ) {
            let first = try await admission.acquire()
            let second = try await admission.acquire()
            let requests = await admission.snapshot()
            let machine = try coordinator.snapshot()
            XCTAssertEqual(requests.activeRequests, 2)
            XCTAssertEqual(machine.active.count, 1)
            XCTAssertEqual(machine.activePermits, machine.capacityPermits)
            await first.release()
            XCTAssertEqual(try coordinator.snapshot().active.count, 1)
            await second.release()
            XCTAssertEqual(try coordinator.snapshot().active.count, 1)
        }
        XCTAssertEqual(try coordinator.snapshot().activePermits, 0)
    }

    func testResolvedModelClassificationPreservesCostFloors() {
        let gibibyte = Int64(1_073_741_824)
        XCTAssertEqual(MachineInferenceClass.forModel(estimatedBytes: nil, minimum: .small), .standard)
        XCTAssertEqual(MachineInferenceClass.forModel(estimatedBytes: 16 * gibibyte, minimum: .small), .small)
        XCTAssertEqual(MachineInferenceClass.forModel(estimatedBytes: 16 * gibibyte + 1, minimum: .small), .standard)
        XCTAssertEqual(MachineInferenceClass.forModel(estimatedBytes: 48 * gibibyte - 1), .standard)
        XCTAssertEqual(MachineInferenceClass.forModel(estimatedBytes: 48 * gibibyte), .large)
        XCTAssertEqual(MachineInferenceClass.forModel(estimatedBytes: 1, minimum: .large), .large)
        XCTAssertEqual(MachineInferenceClass.forModel(estimatedBytes: nil, requiresExclusive: true), .large)
        XCTAssertEqual(MachineInferenceRequest(label: "image", estimatedModelBytes: 1).resourceClass, .standard)
    }

    private func coordinator() throws -> MachineInferenceCoordinator {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let gibibyte = UInt64(1_073_741_824)
        return MachineInferenceCoordinator(
            stateDirectory: directory, processID: 101, bootSessionID: "test-boot",
            hostSnapshot: {
                .init(physicalMemoryBytes: 128 * gibibyte, availableMemoryBytes: 96 * gibibyte,
                      memoryPressure: .nominal, availableDiskBytes: 96 * gibibyte)
            },
            processIsAlive: { _ in true }
        )
    }
}
