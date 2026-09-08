import Foundation
import MereRunAdmission

/// A bounded, exclusive resident slot for mutable inference runtimes.
///
/// The slot keeps exactly one value resident. Requests for the same key reuse
/// it, while a different key unloads the previous value before constructing the
/// replacement. The execution admission remains held for the complete
/// operation because model generators contain mutable caches and are not
/// generally safe to re-enter while an inference is suspended.
public struct ResidentRuntimeSlotState<Key: Equatable & Sendable>: Sendable {
    public let residentKey: Key?
    public let ready: Bool
    public let lastKey: Key?
    public let loadedAt: Date?
    public let lastAccess: Date?
    public let lastEvictedAt: Date?
    public let lastEvictionReason: RuntimeEvictionReason?
    public let activeRequests: Int
    public let queuedRequests: Int
    public let loadCount: Int
    public let replacementCount: Int
    public let evictionCount: Int
    public let completedRequests: Int
    public let failedRequests: Int
    public let idleEvictionGeneration: UInt64
}

public struct ResidentIdlePolicy: Sendable {
    public let pinned: Bool
    public let ttl: Duration

    public init(pinned: Bool, ttl: Duration) {
        self.pinned = pinned
        self.ttl = ttl
    }
}

public actor ResidentRuntimeSlot<Key: Equatable & Sendable, Value: Sendable> {
    private struct Resident: Sendable {
        let key: Key
        let value: Value
        let loadedAt: Date
        var lastAccess: Date
        var ready: Bool
    }

    private let execution = RuntimeRequestAdmission(maxActiveRequests: 1)
    private let currentDate: @Sendable () -> Date
    private var resident: Resident?
    private var lastKey: Key?
    private var lastLoadedAt: Date?
    private var lastAccess: Date?
    private var lastEvictedAt: Date?
    private var lastEvictionReason: RuntimeEvictionReason?
    private var activeRequests = 0
    private var queuedRequests = 0
    private var loadCount = 0
    private var replacementCount = 0
    private var evictionCount = 0
    private var completedRequests = 0
    private var failedRequests = 0
    private var idleEvictionGeneration: UInt64 = 0
    private var idleEvictionTask: Task<Void, Never>?

    public init(currentDate: @escaping @Sendable () -> Date = { Date() }) {
        self.currentDate = currentDate
    }

    public func withValue<Result: Sendable>(
        for key: Key,
        idleTTL: Duration? = nil,
        pinned: Bool = false,
        currentIdlePolicy: (@Sendable (Key) -> ResidentIdlePolicy)? = nil,
        idlePolicyPollInterval: Duration = .seconds(1),
        operationCoordinator: RuntimeOperationCoordinator? = nil,
        forceColdOperation: Bool = false,
        prepareForColdOperation: (@Sendable (_ residentNeedsLoad: Bool) async throws -> Void)? = nil,
        make: @Sendable () async throws -> Value,
        unload: @escaping @Sendable (Value) async -> Void,
        operation: @Sendable (Value) async throws -> Result
    ) async throws -> Result {
        precondition(idlePolicyPollInterval > .zero, "Idle policy poll interval must be positive")
        queuedRequests += 1
        let lease: RuntimeRequestAdmissionLease
        do {
            lease = try await execution.acquire()
        } catch {
            queuedRequests = max(0, queuedRequests - 1)
            throw error
        }
        queuedRequests = max(0, queuedRequests - 1)
        activeRequests += 1
        let residentNeedsLoad = resident?.key != key || resident?.ready != true
        let mode: RuntimeOperationMode = !forceColdOperation
            && !residentNeedsLoad
            ? .warm
            : .cold
        var operationLease: RuntimeOperationLease?
        do {
            operationLease = try await operationCoordinator?.acquire(mode)
            try Task.checkCancellation()
            if mode == .cold {
                if residentNeedsLoad {
                    await unloadResidentBeforeColdLoad(replacingWith: key, unload: unload)
                }
                try Task.checkCancellation()
                try await prepareForColdOperation?(residentNeedsLoad)
                try Task.checkCancellation()
            }
            let value = try await value(for: key, make: make, unload: unload)
            try Task.checkCancellation()
            let result = try await operation(value)
            completedRequests += 1
            markReady(key: key)
            let idleGeneration = touch(key: key)
            activeRequests = max(0, activeRequests - 1)
            await lease.release()
            scheduleIdleEviction(
                expectedKey: key,
                expectedGeneration: idleGeneration,
                idleTTL: idleTTL,
                pinned: pinned,
                currentIdlePolicy: currentIdlePolicy,
                pollInterval: idlePolicyPollInterval,
                unload: unload
            )
            await operationLease?.release()
            return result
        } catch {
            failedRequests += 1
            _ = await unloadResidentIfNotReady(for: key, unload: unload)
            let idleGeneration = touch(key: key)
            activeRequests = max(0, activeRequests - 1)
            await lease.release()
            scheduleIdleEviction(
                expectedKey: key,
                expectedGeneration: idleGeneration,
                idleTTL: idleTTL,
                pinned: pinned,
                currentIdlePolicy: currentIdlePolicy,
                pollInterval: idlePolicyPollInterval,
                unload: unload
            )
            await operationLease?.release()
            throw error
        }
    }

    public func evictIfIdle(
        expectedKey: Key,
        expectedGeneration: UInt64? = nil,
        reason: RuntimeEvictionReason,
        using unload: @Sendable (Value) async -> Void
    ) async -> Bool {
        guard activeRequests == 0,
              queuedRequests == 0,
              expectedGeneration.map({ $0 == idleEvictionGeneration }) ?? true,
              let lease = await execution.tryAcquire() else {
            return false
        }
        guard activeRequests == 0,
              queuedRequests == 0,
              expectedGeneration.map({ $0 == idleEvictionGeneration }) ?? true,
              let resident,
              resident.key == expectedKey else {
            await lease.release()
            return false
        }

        self.resident = nil
        invalidateIdleEviction()
        lastKey = resident.key
        lastLoadedAt = resident.loadedAt
        lastAccess = resident.lastAccess
        lastEvictedAt = currentDate()
        lastEvictionReason = reason
        evictionCount += 1
        await unload(resident.value)
        await lease.release()
        return true
    }

    public func residentKey() -> Key? {
        resident?.key
    }

    public func state() -> ResidentRuntimeSlotState<Key> {
        ResidentRuntimeSlotState(
            residentKey: resident?.key,
            ready: resident?.ready ?? false,
            lastKey: lastKey,
            loadedAt: resident?.loadedAt ?? lastLoadedAt,
            lastAccess: resident?.lastAccess ?? lastAccess,
            lastEvictedAt: lastEvictedAt,
            lastEvictionReason: lastEvictionReason,
            activeRequests: activeRequests,
            queuedRequests: queuedRequests,
            loadCount: loadCount,
            replacementCount: replacementCount,
            evictionCount: evictionCount,
            completedRequests: completedRequests,
            failedRequests: failedRequests,
            idleEvictionGeneration: idleEvictionGeneration
        )
    }

    private func value(
        for key: Key,
        make: @Sendable () async throws -> Value,
        unload: @Sendable (Value) async -> Void
    ) async throws -> Value {
        if let resident, resident.key == key {
            return resident.value
        }
        if let resident {
            self.resident = nil
            invalidateIdleEviction()
            lastKey = resident.key
            lastLoadedAt = resident.loadedAt
            lastAccess = resident.lastAccess
            replacementCount += 1
            await unload(resident.value)
        }
        let value = try await make()
        let now = currentDate()
        resident = Resident(
            key: key,
            value: value,
            loadedAt: now,
            lastAccess: now,
            ready: false
        )
        invalidateIdleEviction()
        lastKey = key
        lastLoadedAt = now
        lastAccess = now
        loadCount += 1
        return value
    }

    private func unloadResidentBeforeColdLoad(
        replacingWith key: Key,
        unload: @Sendable (Value) async -> Void
    ) async {
        guard let resident else { return }
        self.resident = nil
        invalidateIdleEviction()
        lastKey = resident.key
        lastLoadedAt = resident.loadedAt
        lastAccess = resident.lastAccess
        if resident.key != key {
            replacementCount += 1
        }
        await unload(resident.value)
    }

    private func unloadResidentIfNotReady(
        for key: Key,
        unload: @Sendable (Value) async -> Void
    ) async -> Bool {
        guard let resident,
              resident.key == key,
              !resident.ready else {
            return false
        }
        self.resident = nil
        invalidateIdleEviction()
        lastKey = resident.key
        lastLoadedAt = resident.loadedAt
        lastAccess = resident.lastAccess
        await unload(resident.value)
        return true
    }

    private func touch(key: Key) -> UInt64? {
        guard var resident, resident.key == key else { return nil }
        let now = currentDate()
        invalidateIdleEviction()
        lastKey = key
        lastAccess = now
        resident.lastAccess = now
        self.resident = resident
        return idleEvictionGeneration
    }

    private func markReady(key: Key) {
        guard var resident, resident.key == key else { return }
        resident.ready = true
        self.resident = resident
    }

    private func scheduleIdleEviction(
        expectedKey: Key,
        expectedGeneration: UInt64?,
        idleTTL: Duration?,
        pinned: Bool,
        currentIdlePolicy: (@Sendable (Key) -> ResidentIdlePolicy)?,
        pollInterval: Duration,
        unload: @escaping @Sendable (Value) async -> Void
    ) {
        guard let idleTTL,
              (!pinned || currentIdlePolicy != nil),
              let expectedGeneration,
              expectedGeneration == idleEvictionGeneration,
              resident?.key == expectedKey else {
            return
        }
        idleEvictionTask = Task { [weak self] in
            let clock = ContinuousClock()
            let idleStart = clock.now
            var effectiveTTL = idleTTL
            while !Task.isCancelled {
                if let currentIdlePolicy {
                    let policy = currentIdlePolicy(expectedKey)
                    effectiveTTL = policy.ttl
                    if policy.pinned {
                        do {
                            try await Task.sleep(for: pollInterval)
                        } catch {
                            return
                        }
                        continue
                    }
                }

                let elapsed = idleStart.duration(to: clock.now)
                if elapsed >= effectiveTTL {
                    break
                }
                let remaining = effectiveTTL - elapsed
                let delay = remaining < pollInterval ? remaining : pollInterval
                do {
                    try await Task.sleep(for: delay)
                } catch {
                    return
                }
            }
            guard !Task.isCancelled, let self else { return }
            _ = await self.evictIfIdle(
                expectedKey: expectedKey,
                expectedGeneration: expectedGeneration,
                reason: .ttl,
                using: unload
            )
        }
    }

    private func invalidateIdleEviction() {
        idleEvictionGeneration &+= 1
        idleEvictionTask?.cancel()
        idleEvictionTask = nil
    }
}
