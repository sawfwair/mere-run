import XCTest
@testable import MereRunCLI

final class MachineInferenceAdmissionTests: XCTestCase {
    private let gibibyte = UInt64(1_073_741_824)

    func testCapacityScalesWithoutMakingLargeWorkConcurrent() {
        XCTAssertEqual(MachineInferenceCoordinator.capacityPermits(physicalMemoryBytes: 32 * gibibyte), 1)
        XCTAssertEqual(MachineInferenceCoordinator.capacityPermits(physicalMemoryBytes: 64 * gibibyte), 2)
        XCTAssertEqual(MachineInferenceCoordinator.capacityPermits(physicalMemoryBytes: 128 * gibibyte), 4)
        XCTAssertEqual(MachineInferenceCoordinator.capacityPermits(physicalMemoryBytes: 256 * gibibyte), 6)

        XCTAssertEqual(MachineInferenceClass.small.permits(capacity: 4), 1)
        XCTAssertEqual(MachineInferenceClass.standard.permits(capacity: 4), 2)
        XCTAssertEqual(MachineInferenceClass.large.permits(capacity: 4), 4)
    }

    func testWeightedAdmissionAllowsSafeConcurrencyAndQueuesOverflow() async throws {
        let directory = try temporaryDirectory()
        let coordinator = makeCoordinator(directory: directory, processID: 101)
        let standard = MachineInferenceRequest(label: "image generate", resourceClass: .standard)
        let small = MachineInferenceRequest(label: "speech synthesize", resourceClass: .small)

        let first = try await coordinator.acquire(standard)
        let second = try await coordinator.acquire(standard)
        var snapshot = try coordinator.snapshot()
        XCTAssertEqual(snapshot.activePermits, 4)
        XCTAssertEqual(snapshot.active.count, 2)

        let queued = Task {
            try await coordinator.acquire(small)
        }
        try await waitUntil {
            try coordinator.snapshot().queued.count == 1
        }

        first.release()
        let third = try await queued.value
        snapshot = try coordinator.snapshot()
        XCTAssertEqual(snapshot.activePermits, 3)
        XCTAssertEqual(snapshot.active.count, 2)
        XCTAssertEqual(snapshot.queued.count, 0)

        second.release()
        third.release()
        XCTAssertEqual(try coordinator.snapshot().activePermits, 0)
    }

    func testLargeAdmissionWaitsForExclusiveCapacity() async throws {
        let directory = try temporaryDirectory()
        let coordinator = makeCoordinator(directory: directory, processID: 102)
        let small = try await coordinator.acquire(
            MachineInferenceRequest(label: "speech synthesize", resourceClass: .small)
        )
        let queuedLarge = Task {
            try await coordinator.acquire(
                MachineInferenceRequest(label: "video generate", resourceClass: .large)
            )
        }
        try await waitUntil {
            try coordinator.snapshot().queued.count == 1
        }

        XCTAssertEqual(try coordinator.snapshot().activePermits, 1)
        small.release()
        let large = try await queuedLarge.value
        let snapshot = try coordinator.snapshot()
        XCTAssertEqual(snapshot.activePermits, snapshot.capacityPermits)
        XCTAssertEqual(snapshot.active.first?.label, "video generate")
        large.release()
    }

    func testCancellationRemovesQueuedTicket() async throws {
        let directory = try temporaryDirectory()
        let coordinator = makeCoordinator(directory: directory, processID: 103)
        let active = try await coordinator.acquire(
            MachineInferenceRequest(label: "video generate", resourceClass: .large)
        )
        let queued = Task {
            try await coordinator.acquire(
                MachineInferenceRequest(label: "speech synthesize", resourceClass: .small)
            )
        }
        try await waitUntil {
            try coordinator.snapshot().queued.count == 1
        }

        queued.cancel()
        do {
            _ = try await queued.value
            XCTFail("Expected queued admission to be cancelled")
        } catch is CancellationError {
            // Expected.
        }
        XCTAssertEqual(try coordinator.snapshot().queued.count, 0)
        active.release()
    }

    func testDeadProcessTicketIsPrunedBeforeNewAdmission() async throws {
        let directory = try temporaryDirectory()
        let firstCoordinator = makeCoordinator(
            directory: directory,
            processID: 1_001,
            processIsAlive: { $0 == 1_001 }
        )
        let abandoned = try await firstCoordinator.acquire(
            MachineInferenceRequest(label: "image generate", resourceClass: .standard)
        )
        XCTAssertEqual(try firstCoordinator.snapshot().active.count, 1)

        let recoveringCoordinator = makeCoordinator(
            directory: directory,
            processID: 1_002,
            processIsAlive: { $0 == 1_002 }
        )
        XCTAssertEqual(try recoveringCoordinator.snapshot().active.count, 0)
        let recovered = try await recoveringCoordinator.acquire(
            MachineInferenceRequest(label: "video generate", resourceClass: .large)
        )
        XCTAssertEqual(try recoveringCoordinator.snapshot().activePermits, 4)

        abandoned.release()
        recovered.release()
    }

    func testPreviousBootTicketIsPrunedEvenWhenPIDWasReused() async throws {
        let directory = try temporaryDirectory()
        let previousBoot = makeCoordinator(
            directory: directory,
            processID: 1_101,
            bootSessionID: "previous-boot"
        )
        let abandoned = try await previousBoot.acquire(
            MachineInferenceRequest(label: "video generate", resourceClass: .large)
        )

        let currentBoot = makeCoordinator(
            directory: directory,
            processID: 1_101,
            bootSessionID: "current-boot"
        )
        XCTAssertEqual(try currentBoot.snapshot().active.count, 0)
        let recovered = try await currentBoot.acquire(
            MachineInferenceRequest(label: "image generate", resourceClass: .standard)
        )
        XCTAssertEqual(try currentBoot.snapshot().activePermits, 2)

        abandoned.release()
        recovered.release()
    }

    func testQueuedLargeWorkPreventsNewSmallWorkFromJumpingTheFIFO() async throws {
        let directory = try temporaryDirectory()
        let coordinator = makeCoordinator(directory: directory, processID: 106)
        let active = try await coordinator.acquire(
            MachineInferenceRequest(label: "image generate", resourceClass: .standard)
        )
        let largeTask = Task {
            try await coordinator.acquire(
                MachineInferenceRequest(label: "video generate", resourceClass: .large)
            )
        }
        try await waitUntil {
            try coordinator.snapshot().queued.count == 1
        }
        let smallTask = Task {
            try await coordinator.acquire(
                MachineInferenceRequest(label: "speech synthesize", resourceClass: .small)
            )
        }
        try await waitUntil {
            try coordinator.snapshot().queued.count == 2
        }

        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertEqual(try coordinator.snapshot().active.count, 1)
        active.release()
        let large = try await largeTask.value
        XCTAssertEqual(try coordinator.snapshot().active.first?.label, "video generate")
        XCTAssertEqual(try coordinator.snapshot().queued.first?.label, "speech synthesize")

        large.release()
        let small = try await smallTask.value
        small.release()
    }

    func testLowDiskRejectsBeforeRegisteringTicket() async throws {
        let directory = try temporaryDirectory()
        let coordinator = makeCoordinator(
            directory: directory,
            processID: 104,
            availableDiskBytes: gibibyte
        )
        do {
            _ = try await coordinator.acquire(
                MachineInferenceRequest(label: "image generate", resourceClass: .standard)
            )
            XCTFail("Expected low disk to block admission")
        } catch let error as MachineInferenceAdmissionError {
            guard case .insufficientDisk(let available, let required) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(available, gibibyte)
            XCTAssertEqual(required, 16 * gibibyte)
        }
        XCTAssertEqual(try coordinator.snapshot().active.count, 0)
        XCTAssertEqual(try coordinator.snapshot().queued.count, 0)
    }

    func testDiskCapacityFallsBackWhenImportantUsageProbeReturnsFalseZero() {
        let available = 188 * gibibyte

        XCTAssertEqual(
            MachineInferenceCoordinator.reconciledAvailableDiskBytes(
                importantUsageCapacity: 0,
                fileSystemFreeBytes: available
            ),
            available
        )
    }

    func testDiskCapacityUsesConservativePositiveProbe() {
        XCTAssertEqual(
            MachineInferenceCoordinator.reconciledAvailableDiskBytes(
                importantUsageCapacity: Int64(80 * gibibyte),
                fileSystemFreeBytes: 100 * gibibyte
            ),
            80 * gibibyte
        )
        XCTAssertEqual(
            MachineInferenceCoordinator.reconciledAvailableDiskBytes(
                importantUsageCapacity: Int64(120 * gibibyte),
                fileSystemFreeBytes: 100 * gibibyte
            ),
            100 * gibibyte
        )
    }

    func testDiskCapacityPreservesRealZeroWhenNoPositiveProbeExists() {
        XCTAssertEqual(
            MachineInferenceCoordinator.reconciledAvailableDiskBytes(
                importantUsageCapacity: 0,
                fileSystemFreeBytes: 0
            ),
            0
        )
        XCTAssertNil(
            MachineInferenceCoordinator.reconciledAvailableDiskBytes(
                importantUsageCapacity: nil,
                fileSystemFreeBytes: nil
            )
        )
    }

    func testLowMemoryRejectsWhenNoActiveWorkCanReleaseHeadroom() async throws {
        let directory = try temporaryDirectory()
        let coordinator = makeCoordinator(
            directory: directory,
            processID: 105,
            availableMemoryBytes: 8 * gibibyte
        )
        do {
            _ = try await coordinator.acquire(
                MachineInferenceRequest(label: "image generate", resourceClass: .standard)
            )
            XCTFail("Expected low memory to block admission")
        } catch let error as MachineInferenceAdmissionError {
            guard case .insufficientMemory(let available, let required) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(available, 8 * gibibyte)
            XCTAssertEqual(required, 16 * gibibyte)
        }
    }

    func testCLIClassifierProtectsMediaAndLeavesLightweightCommandsFree() {
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "image", "generate", "--prompt", "test"]
            ),
            MachineInferenceRequest(label: "image generate", resourceClass: .standard)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "speech", "synthesize", "hello"]
            ),
            MachineInferenceRequest(label: "speech synthesize", resourceClass: .small)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "video", "generate", "test"]
            ),
            MachineInferenceRequest(label: "video generate", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "geo", "fire", "input.safetensors"]
            ),
            MachineInferenceRequest(label: "geo fire", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "geo", "olmoearth", "input.safetensors"]
            ),
            MachineInferenceRequest(label: "geo olmoearth", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "text", "chat", "--model", "text-chat-deepseek-v4-flash"]
            ),
            MachineInferenceRequest(label: "text chat", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "text", "chat", "--model=text-chat-inkling-small"]
            ),
            MachineInferenceRequest(label: "text chat", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "text", "chat", "--model", "text-chat-laguna-xs-2-1"]
            ),
            MachineInferenceRequest(label: "text chat", resourceClass: .standard)
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "status"])
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "model", "list"])
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "image", "generate", "--help"])
        )
    }

    func testAPIServerKeepsInternalConcurrencyInsideWeightedReservation() {
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(engine: .textChatGemma4),
            MachineInferenceRequest(label: "api serve text-chat-gemma4", resourceClass: .standard)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(engine: .textChatDeepseekV4Flash),
            MachineInferenceRequest(label: "api serve text-chat-deepseek-v4-flash", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(
                engine: .textChatLaguna,
                modelID: "text-chat-laguna-s-2-1"
            ),
            MachineInferenceRequest(label: "api serve text-chat-laguna", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(
                engine: .textChatLaguna,
                modelID: "text-chat-laguna-xs-2-1"
            ),
            MachineInferenceRequest(label: "api serve text-chat-laguna", resourceClass: .standard)
        )
    }

    private func makeCoordinator(
        directory: URL,
        processID: Int32,
        bootSessionID: String = "test-boot",
        availableMemoryBytes: UInt64? = 96 * 1_073_741_824,
        availableDiskBytes: UInt64? = 100 * 1_073_741_824,
        processIsAlive: @escaping @Sendable (Int32) -> Bool = { _ in true }
    ) -> MachineInferenceCoordinator {
        MachineInferenceCoordinator(
            stateDirectory: directory,
            processID: processID,
            bootSessionID: bootSessionID,
            currentDate: { Date() },
            hostSnapshot: {
                MachineInferenceHostSnapshot(
                    physicalMemoryBytes: 128 * 1_073_741_824,
                    availableMemoryBytes: availableMemoryBytes,
                    memoryPressure: .nominal,
                    availableDiskBytes: availableDiskBytes
                )
            },
            processIsAlive: processIsAlive
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-machine-admission-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }

    private func waitUntil(
        timeoutNanoseconds: UInt64 = 2_000_000_000,
        condition: () throws -> Bool
    ) async throws {
        let started = ContinuousClock.now
        while try !condition() {
            if started.duration(to: .now) > .nanoseconds(Int64(timeoutNanoseconds)) {
                XCTFail("Timed out waiting for admission state")
                return
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
    }
}
