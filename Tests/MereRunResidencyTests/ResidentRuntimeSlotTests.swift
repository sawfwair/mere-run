import Foundation
import XCTest
import MereRunResidency

final class ResidentRuntimeSlotTests: XCTestCase {
    func testResidentSlotReusesMatchingKeyAndUnloadsBeforeReplacement() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()

        let first = try await slot.withValue(
            for: "image-a",
            make: { await probe.makeValue(1) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        let repeated = try await slot.withValue(
            for: "image-a",
            make: { await probe.makeValue(2) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        let replacement = try await slot.withValue(
            for: "image-b",
            make: { await probe.makeValue(3) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )

        XCTAssertEqual(first, 1)
        XCTAssertEqual(repeated, 1)
        XCTAssertEqual(replacement, 3)
        let madeValues = await probe.madeValues()
        let unloadedValues = await probe.unloadedValues()
        let residentKey = await slot.residentKey()
        XCTAssertEqual(madeValues, [1, 3])
        XCTAssertEqual(unloadedValues, [1])
        XCTAssertEqual(residentKey, "image-b")
    }

    func testResidentSlotSerializesConcurrentOperationsAndDeduplicatesLoad() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()

        async let first = slot.withValue(
            for: "speech",
            make: { await probe.makeValue(7) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value) }
        )
        async let second = slot.withValue(
            for: "speech",
            make: { await probe.makeValue(7) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value) }
        )

        let values = try await [first, second]
        let madeValues = await probe.madeValues()
        let maximumConcurrentUses = await probe.maximumConcurrentUses()
        XCTAssertEqual(values, [7, 7])
        XCTAssertEqual(madeValues, [7])
        XCTAssertEqual(maximumConcurrentUses, 1)
    }

    func testColdSidecarOperationsAreExclusiveWhileWarmLanesOverlap() async throws {
        let probe = APISidecarSlotProbe()
        let coordinator = RuntimeOperationCoordinator()
        let image = ResidentRuntimeSlot<String, Int>()
        let speech = ResidentRuntimeSlot<String, Int>()

        async let coldImage = image.withValue(
            for: "image",
            operationCoordinator: coordinator,
            make: { await probe.makeValue(1) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value, nanoseconds: 60_000_000) }
        )
        async let coldSpeech = speech.withValue(
            for: "speech",
            operationCoordinator: coordinator,
            make: { await probe.makeValue(2) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value, nanoseconds: 60_000_000) }
        )

        let coldValues = try await [coldImage, coldSpeech]
        let coldMaximum = await probe.maximumConcurrentUses()
        XCTAssertEqual(coldValues, [1, 2])
        XCTAssertEqual(coldMaximum, 1)

        await probe.resetMaximumConcurrentUses()
        async let warmImage = image.withValue(
            for: "image",
            operationCoordinator: coordinator,
            make: { await probe.makeValue(3) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value, nanoseconds: 60_000_000) }
        )
        async let warmSpeech = speech.withValue(
            for: "speech",
            operationCoordinator: coordinator,
            make: { await probe.makeValue(4) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value, nanoseconds: 60_000_000) }
        )

        let warmValues = try await [warmImage, warmSpeech]
        let warmMaximum = await probe.maximumConcurrentUses()
        XCTAssertEqual(warmValues, [1, 2])
        XCTAssertEqual(warmMaximum, 2)
    }

    func testForcedColdWarmImageSkipsResidentReloadClassification() async throws {
        let preparation = APISidecarColdPreparationProbe()
        let coordinator = RuntimeOperationCoordinator()
        let slot = ResidentRuntimeSlot<String, Int>()

        for _ in 0..<2 {
            _ = try await slot.withValue(
                for: "image",
                operationCoordinator: coordinator,
                forceColdOperation: true,
                prepareForColdOperation: { residentNeedsLoad in
                    await preparation.record(residentNeedsLoad)
                },
                make: { 4 },
                unload: { _ in },
                operation: { $0 }
            )
        }

        let classifications = await preparation.classifications()
        XCTAssertEqual(classifications, [true, false])
    }

    func testColdReplacementUnloadsOutgoingResidentBeforeHeadroomPreparation() async throws {
        let slotProbe = APISidecarSlotProbe()
        let preparation = APISidecarReplacementPreparationProbe()
        let coordinator = RuntimeOperationCoordinator()
        let slot = ResidentRuntimeSlot<String, Int>()

        for (key, value) in [("image-a", 1), ("image-b", 2)] {
            _ = try await slot.withValue(
                for: key,
                operationCoordinator: coordinator,
                prepareForColdOperation: { _ in
                    await preparation.record(
                        unloadedValues: await slotProbe.unloadedValues()
                    )
                },
                make: { await slotProbe.makeValue(value) },
                unload: { unloaded in await slotProbe.unload(unloaded) },
                operation: { $0 }
            )
        }

        let observations = await preparation.observations()
        XCTAssertEqual(observations, [[], [1]])
    }

    func testCancellationDuringColdPreparationDoesNotConstructResident() async throws {
        let preparation = APISidecarColdPreparationGate()
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()
        let task = Task {
            try await slot.withValue(
                for: "image",
                operationCoordinator: RuntimeOperationCoordinator(),
                prepareForColdOperation: { _ in await preparation.wait() },
                make: { await probe.makeValue(1) },
                unload: { value in await probe.unload(value) },
                operation: { $0 }
            )
        }

        for _ in 0..<50 {
            if await preparation.hasStarted() { break }
            try await Task.sleep(for: .milliseconds(2))
        }
        let preparationStarted = await preparation.hasStarted()
        XCTAssertTrue(preparationStarted)
        task.cancel()
        await preparation.release()

        do {
            _ = try await task.value
            XCTFail("Expected cancellation during cold preparation")
        } catch is CancellationError {
            // Expected.
        }
        let state = await slot.state()
        let madeValues = await probe.madeValues()
        XCTAssertNil(state.residentKey)
        XCTAssertEqual(state.activeRequests, 0)
        XCTAssertTrue(madeValues.isEmpty)
    }

    func testIdleEvictionRecordsDiagnosticsAndUnloadsResident() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()
        _ = try await slot.withValue(
            for: "image",
            make: { await probe.makeValue(9) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )

        let evicted = await slot.evictIfIdle(
            expectedKey: "image",
            reason: .ttl,
            using: { value in await probe.unload(value) }
        )
        let state = await slot.state()
        let unloadedValues = await probe.unloadedValues()

        XCTAssertTrue(evicted)
        XCTAssertNil(state.residentKey)
        XCTAssertEqual(state.lastKey, "image")
        XCTAssertEqual(state.lastEvictionReason, .ttl)
        XCTAssertEqual(state.evictionCount, 1)
        XCTAssertEqual(unloadedValues, [9])
    }

    func testFailedNotReadyResidentUnloadsImmediatelyEvenWhenPinned() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()
        do {
            let _: Int = try await slot.withValue(
                for: "image",
                idleTTL: .seconds(300),
                pinned: true,
                make: { await probe.makeValue(5) },
                unload: { value in await probe.unload(value) },
                operation: { _ in throw APISidecarSlotTestError.loadFailed }
            )
            XCTFail("Expected the first operation to fail")
        } catch APISidecarSlotTestError.loadFailed {
            // Expected.
        }
        var state = await slot.state()
        XCTAssertNil(state.residentKey)
        XCTAssertFalse(state.ready)
        let firstUnloads = await probe.unloadedValues()
        XCTAssertEqual(firstUnloads, [5])

        _ = try await slot.withValue(
            for: "image",
            make: { await probe.makeValue(6) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        state = await slot.state()
        XCTAssertTrue(state.ready)
        let madeValues = await probe.madeValues()
        XCTAssertEqual(madeValues, [5, 6])
    }

    func testIdleTTLEvictsAutonomouslyWithoutStatusOrAnotherRequest() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()
        _ = try await slot.withValue(
            for: "image",
            idleTTL: .milliseconds(25),
            make: { await probe.makeValue(9) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )

        for _ in 0..<50 {
            if await probe.unloadedValues() == [9] { break }
            try await Task.sleep(for: .milliseconds(5))
        }

        let state = await slot.state()
        XCTAssertNil(state.residentKey)
        XCTAssertEqual(state.lastEvictionReason, .ttl)
        XCTAssertEqual(state.evictionCount, 1)
        let unloadedValues = await probe.unloadedValues()
        XCTAssertEqual(unloadedValues, [9])
    }

    func testNewAccessInvalidatesEarlierIdleTTLGeneration() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()
        _ = try await slot.withValue(
            for: "speech",
            idleTTL: .seconds(60),
            make: { await probe.makeValue(7) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        let originalGeneration = await slot.state().idleEvictionGeneration

        _ = try await slot.withValue(
            for: "speech",
            idleTTL: .seconds(60),
            make: { await probe.makeValue(8) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        let refreshedGeneration = await slot.state().idleEvictionGeneration
        XCTAssertNotEqual(refreshedGeneration, originalGeneration)

        let staleEviction = await slot.evictIfIdle(
            expectedKey: "speech",
            expectedGeneration: originalGeneration,
            reason: .ttl,
            using: { value in await probe.unload(value) }
        )

        let residentAfterStaleEviction = await slot.residentKey()
        let earlyUnloads = await probe.unloadedValues()
        XCTAssertFalse(staleEviction)
        XCTAssertEqual(residentAfterStaleEviction, "speech")
        XCTAssertTrue(earlyUnloads.isEmpty)

        let currentEviction = await slot.evictIfIdle(
            expectedKey: "speech",
            expectedGeneration: refreshedGeneration,
            reason: .ttl,
            using: { value in await probe.unload(value) }
        )
        let finalResident = await slot.residentKey()
        XCTAssertTrue(currentEviction)
        XCTAssertNil(finalResident)
        let finalUnloads = await probe.unloadedValues()
        XCTAssertEqual(finalUnloads, [7])
    }

    func testPinnedResidentDoesNotScheduleIdleTTLEviction() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()
        _ = try await slot.withValue(
            for: "transcription",
            idleTTL: .milliseconds(10),
            pinned: true,
            make: { await probe.makeValue(4) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        try await Task.sleep(for: .milliseconds(40))

        let residentKey = await slot.residentKey()
        let unloadedValues = await probe.unloadedValues()
        XCTAssertEqual(residentKey, "transcription")
        XCTAssertTrue(unloadedValues.isEmpty)
    }

    func testIdleTTLRechecksPinBeforeEviction() async throws {
        let probe = APISidecarSlotProbe()
        let policy = APISidecarIdlePolicyProbe(pinned: false, ttl: .milliseconds(25))
        let slot = ResidentRuntimeSlot<String, Int>()
        _ = try await slot.withValue(
            for: "image",
            idleTTL: .milliseconds(25),
            currentIdlePolicy: { _ in policy.current() },
            idlePolicyPollInterval: .milliseconds(5),
            make: { await probe.makeValue(3) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        policy.setPinned(true)
        try await Task.sleep(for: .milliseconds(50))

        let residentKey = await slot.residentKey()
        let unloadedValues = await probe.unloadedValues()
        XCTAssertEqual(residentKey, "image")
        XCTAssertTrue(unloadedValues.isEmpty)
    }

    func testIdleTTLReschedulesWhenConfiguredDeadlineChanges() async throws {
        let probe = APISidecarSlotProbe()
        let policy = APISidecarIdlePolicyProbe(pinned: false, ttl: .milliseconds(200))
        let slot = ResidentRuntimeSlot<String, Int>()
        _ = try await slot.withValue(
            for: "speech",
            idleTTL: .milliseconds(200),
            currentIdlePolicy: { _ in policy.current() },
            idlePolicyPollInterval: .milliseconds(5),
            make: { await probe.makeValue(6) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        try await Task.sleep(for: .milliseconds(20))
        policy.setTTL(.milliseconds(30))

        for _ in 0..<30 {
            if await probe.unloadedValues() == [6] { break }
            try await Task.sleep(for: .milliseconds(5))
        }

        let residentKey = await slot.residentKey()
        let unloadedValues = await probe.unloadedValues()
        XCTAssertNil(residentKey)
        XCTAssertEqual(unloadedValues, [6])
    }

    func testEvictionNeverTakesAnActiveLease() async throws {
        let probe = APISidecarSlotProbe()
        let slot = ResidentRuntimeSlot<String, Int>()
        let request = Task {
            try await slot.withValue(
                for: "speech",
                make: { await probe.makeValue(4) },
                unload: { value in await probe.unload(value) },
                operation: { value in try await probe.use(value, nanoseconds: 100_000_000) }
            )
        }
        for _ in 0..<50 {
            if await probe.activeUseCount() == 1 { break }
            try await Task.sleep(nanoseconds: 1_000_000)
        }

        let evicted = await slot.evictIfIdle(
            expectedKey: "speech",
            reason: .memoryPressure,
            using: { value in await probe.unload(value) }
        )
        let result = try await request.value
        let residentKey = await slot.residentKey()
        let unloadedValues = await probe.unloadedValues()

        XCTAssertFalse(evicted)
        XCTAssertEqual(result, 4)
        XCTAssertEqual(residentKey, "speech")
        XCTAssertTrue(unloadedValues.isEmpty)
    }

}

private enum APISidecarSlotTestError: Error {
    case loadFailed
}

private actor APISidecarSlotProbe {
    private var made: [Int] = []
    private var unloaded: [Int] = []
    private var activeUses = 0
    private var maxActiveUses = 0

    func makeValue(_ value: Int) -> Int {
        made.append(value)
        return value
    }

    func unload(_ value: Int) {
        unloaded.append(value)
    }

    func use(_ value: Int, nanoseconds: UInt64 = 20_000_000) async throws -> Int {
        activeUses += 1
        maxActiveUses = max(maxActiveUses, activeUses)
        try await Task.sleep(nanoseconds: nanoseconds)
        activeUses -= 1
        return value
    }

    func madeValues() -> [Int] {
        made
    }

    func unloadedValues() -> [Int] {
        unloaded
    }

    func maximumConcurrentUses() -> Int {
        maxActiveUses
    }

    func resetMaximumConcurrentUses() {
        maxActiveUses = activeUses
    }

    func activeUseCount() -> Int {
        activeUses
    }
}

private actor APISidecarColdPreparationProbe {
    private var values: [Bool] = []

    func record(_ residentNeedsLoad: Bool) {
        values.append(residentNeedsLoad)
    }

    func classifications() -> [Bool] {
        values
    }
}

private actor APISidecarColdPreparationGate {
    private var started = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        started = true
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func hasStarted() -> Bool {
        started
    }

    func release() {
        continuation?.resume()
        continuation = nil
    }
}

private actor APISidecarReplacementPreparationProbe {
    private var values: [[Int]] = []

    func record(unloadedValues: [Int]) {
        values.append(unloadedValues)
    }

    func observations() -> [[Int]] {
        values
    }
}

private final class APISidecarIdlePolicyProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var pinned: Bool
    private var ttl: Duration

    init(pinned: Bool, ttl: Duration) {
        self.pinned = pinned
        self.ttl = ttl
    }

    func setPinned(_ value: Bool) {
        lock.lock()
        defer { lock.unlock() }
        pinned = value
    }

    func setTTL(_ value: Duration) {
        lock.lock()
        defer { lock.unlock() }
        ttl = value
    }

    func current() -> ResidentIdlePolicy {
        lock.lock()
        defer { lock.unlock() }
        return ResidentIdlePolicy(pinned: pinned, ttl: ttl)
    }
}
