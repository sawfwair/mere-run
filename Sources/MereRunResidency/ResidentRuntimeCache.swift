import Foundation

/// Describes one generation, including preparation and active model leases.
public struct ResidentRuntimeSnapshot<Value: Sendable>: Sendable {
    public let value: Value
    public let generation: UUID
    public let ready: Bool
    public let activeRequests: Int
    public let accessGeneration: UInt64
    public let waitingRequests: Int
    public let lastAccess: Date?
}

public enum ResidentRuntimeError: Error, Equatable, Sendable {
    case activeLeases(Int)
}

/// Keeps concurrently usable runtimes warm and shares each cold preparation.
/// Runtime adapters supply construction, preparation, and unloading. A lease
/// protects its generation until the caller finishes, including stream production.
public actor ResidentRuntimeCache<Key: Hashable & Sendable, Value: Sendable> {
    private struct Preparation {
        let token: UUID
        let task: Task<Value, Error>
        var waiterIDs: Set<UUID> = []
    }

    private struct Resident {
        let value: Value
        let token: UUID
        var activeRequests = 0
        var accessGeneration: UInt64 = 0
        var lastAccess: Date?
    }

    private let currentDate: @Sendable () -> Date
    private let unload: @Sendable (Value) async -> Void
    private var residents: [Key: Resident] = [:]
    private var preparations: [Key: Preparation] = [:]
    private var cleanupTasks: [UUID: Task<Void, Never>] = [:]

    public init(
        currentDate: @escaping @Sendable () -> Date = Date.init,
        unload: @escaping @Sendable (Value) async -> Void
    ) {
        self.currentDate = currentDate
        self.unload = unload
    }

    public func acquire(
        for key: Key,
        make: @Sendable () throws -> Value,
        prepare: @escaping @Sendable (Value) async throws -> Void
    ) async throws -> ResidentRuntimeLease<Key, Value> {
        try Task.checkCancellation()
        let token: UUID
        if let resident = residents[key], preparations[key] == nil {
            token = resident.token
        } else if let preparation = preparations[key] {
            token = try await awaitPreparation(preparation, key: key)
        } else {
            let value = try make()
            let generation = UUID()
            residents[key] = Resident(value: value, token: generation)
            let task = Task<Value, Error> {
                try await prepare(value)
                return value
            }
            let preparation = Preparation(token: generation, task: task)
            preparations[key] = preparation
            token = try await awaitPreparation(preparation, key: key)
        }
        try Task.checkCancellation()
        // Finalization can suspend for old-generation cleanup. Grant a lease
        // only if this exact generation still owns the resident entry.
        guard var resident = residents[key], resident.token == token,
              preparations[key] == nil else {
            throw CancellationError()
        }
        resident.activeRequests += 1
        resident.accessGeneration &+= 1
        resident.lastAccess = currentDate()
        residents[key] = resident
        return ResidentRuntimeLease(key: key, generation: token, value: resident.value, cache: self)
    }

    /// Cancels an unfinished load or unloads an idle generation. Active leases
    /// must finish before an explicit unload can proceed.
    public func unload(key: Key) async throws {
        guard let resident = residents[key] else { return }
        guard resident.activeRequests == 0 else {
            throw ResidentRuntimeError.activeLeases(resident.activeRequests)
        }
        if let preparation = preparations[key] {
            await invalidate(preparation, key: key)
        } else {
            residents.removeValue(forKey: key)
            await unload(resident.value)
        }
    }

    /// Revalidates an eviction decision after actor suspension. A stale
    /// decision cannot unload a replacement, an active lease, or a cold load.
    public func evictIfIdle(key: Key, generation: UUID, accessGeneration: UInt64) async -> Bool {
        guard let resident = residents[key], resident.token == generation,
              resident.accessGeneration == accessGeneration,
              resident.activeRequests == 0, preparations[key] == nil else {
            return false
        }
        residents.removeValue(forKey: key)
        await unload(resident.value)
        return true
    }

    public func snapshots() -> [Key: ResidentRuntimeSnapshot<Value>] {
        Dictionary(uniqueKeysWithValues: residents.map { key, resident in
            let preparation = preparations[key]
            return (key, ResidentRuntimeSnapshot(
                value: resident.value,
                generation: resident.token,
                ready: preparation == nil,
                activeRequests: resident.activeRequests,
                accessGeneration: resident.accessGeneration,
                waitingRequests: preparation?.waiterIDs.count ?? 0,
                lastAccess: resident.lastAccess
            ))
        })
    }

    fileprivate func release(key: Key, generation: UUID) {
        guard var resident = residents[key], resident.token == generation else { return }
        resident.activeRequests -= 1
        resident.accessGeneration &+= 1
        resident.lastAccess = currentDate()
        residents[key] = resident
    }

    private func awaitPreparation(_ preparation: Preparation, key: Key) async throws -> UUID {
        let waiterID = UUID()
        guard var current = preparations[key], current.token == preparation.token else {
            return try await finalize(key: key, token: preparation.token)
        }
        current.waiterIDs.insert(waiterID)
        preparations[key] = current
        return try await withTaskCancellationHandler {
            do {
                _ = try await preparation.task.value
                try Task.checkCancellation()
                return try await finalize(key: key, token: preparation.token)
            } catch {
                if Task.isCancelled {
                    await cancelWaiter(key: key, token: preparation.token, waiterID: waiterID)
                } else {
                    await invalidate(preparation, key: key)
                }
                throw error
            }
        } onCancel: {
            Task { await self.cancelWaiter(key: key, token: preparation.token, waiterID: waiterID) }
        }
    }

    private func finalize(key: Key, token: UUID) async throws -> UUID {
        if cleanupTasks[token] != nil {
            await awaitCleanup(token: token)
            throw CancellationError()
        }
        guard residents[key]?.token == token else { throw CancellationError() }
        if preparations[key]?.token == token {
            preparations.removeValue(forKey: key)
        }
        return token
    }

    private func cancelWaiter(key: Key, token: UUID, waiterID: UUID) async {
        guard var preparation = preparations[key], preparation.token == token,
              preparation.waiterIDs.remove(waiterID) != nil else {
            await awaitCleanup(token: token)
            return
        }
        guard preparation.waiterIDs.isEmpty else {
            preparations[key] = preparation
            return
        }
        await invalidate(preparation, key: key)
    }

    private func invalidate(_ preparation: Preparation, key: Key) async {
        if preparations[key]?.token == preparation.token {
            preparations.removeValue(forKey: key)
            preparation.task.cancel()
            if residents[key]?.token == preparation.token,
               let resident = residents.removeValue(forKey: key) {
                let unload = self.unload
                cleanupTasks[preparation.token] = Task {
                    // Preparation must settle before unload: a resumed loader
                    // must not repopulate a generation after its cleanup.
                    _ = await preparation.task.result
                    await unload(resident.value)
                }
            }
        }
        await awaitCleanup(token: preparation.token)
    }

    private func awaitCleanup(token: UUID) async {
        guard let cleanup = cleanupTasks[token] else { return }
        await cleanup.value
        cleanupTasks.removeValue(forKey: token)
    }

    #if DEBUG
    public func seedForTesting(key: Key, value: Value, lastAccess: Date, activeRequests: Int = 0) {
        residents[key] = Resident(value: value, token: UUID(), activeRequests: activeRequests, lastAccess: lastAccess)
        preparations.removeValue(forKey: key)
    }
    #endif
}

/// Releases an acquired generation once, explicitly or when its owner deinitializes.
public final class ResidentRuntimeLease<Key: Hashable & Sendable, Value: Sendable>: @unchecked Sendable {
    public let key: Key
    public let generation: UUID
    public let value: Value
    private let cache: ResidentRuntimeCache<Key, Value>
    private let lock = NSLock()
    private var released = false

    fileprivate init(key: Key, generation: UUID, value: Value, cache: ResidentRuntimeCache<Key, Value>) {
        self.key = key
        self.generation = generation
        self.value = value
        self.cache = cache
    }

    deinit {
        guard markReleased() else { return }
        let cache = cache
        let key = key
        let generation = generation
        Task { await cache.release(key: key, generation: generation) }
    }

    public func release() async {
        guard markReleased() else { return }
        await cache.release(key: key, generation: generation)
    }

    private func markReleased() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !released else { return false }
        released = true
        return true
    }
}
