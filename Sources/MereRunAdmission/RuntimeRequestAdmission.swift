import Foundation

/// Runtime-neutral progress used by request admission telemetry.
public struct RuntimeRequestProgress: Sendable {
    public enum Stage: Sendable { case loadingModel, encoding, generating }
    public let stage: Stage
    public let message: String?

    public init(stage: Stage, message: String? = nil) {
        self.stage = stage
        self.message = message
    }
}

public struct RuntimeRequestAdmissionSnapshot: Codable, Equatable, Sendable {
    public let maxActiveRequests: Int
    public let activeRequests: Int
    public let queuedRequests: Int
    public let totalAdmittedRequests: Int
    public let totalCompletedRequests: Int
    public let totalCancelledRequests: Int
    public let admissionPaused: Bool?
    public let pressure: String?
    public var activeRequestDetails: [RuntimeActiveRequestSnapshot]? = nil
    public var lastClientDisconnectAt: Date? = nil
    public var lastCancellationAt: Date? = nil
    public var lastSlotReleaseAt: Date? = nil

    public init(
        maxActiveRequests: Int,
        activeRequests: Int,
        queuedRequests: Int,
        totalAdmittedRequests: Int,
        totalCompletedRequests: Int,
        totalCancelledRequests: Int,
        admissionPaused: Bool? = nil,
        pressure: String? = nil,
        activeRequestDetails: [RuntimeActiveRequestSnapshot]? = nil,
        lastClientDisconnectAt: Date? = nil,
        lastCancellationAt: Date? = nil,
        lastSlotReleaseAt: Date? = nil
    ) {
        self.maxActiveRequests = maxActiveRequests
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
        self.totalAdmittedRequests = totalAdmittedRequests
        self.totalCompletedRequests = totalCompletedRequests
        self.totalCancelledRequests = totalCancelledRequests
        self.admissionPaused = admissionPaused
        self.pressure = pressure
        self.activeRequestDetails = activeRequestDetails
        self.lastClientDisconnectAt = lastClientDisconnectAt
        self.lastCancellationAt = lastCancellationAt
        self.lastSlotReleaseAt = lastSlotReleaseAt
    }
}

public struct RuntimeActiveRequestSnapshot: Codable, Equatable, Sendable {
    public let id: UUID
    public let admittedAt: Date
    public let modelID: String?
    public let streaming: Bool?
    public let requestedMaxTokens: Int?
    public let toolCount: Int?
    public let phase: String
    public let phaseDetail: String?
    public let firstTokenAt: Date?
    public let generatedTokenUpdates: Int
}

public actor RuntimeRequestAdmission {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<RuntimeRequestAdmissionLease, Error>
    }

    private enum WaiterState {
        case pending
        case waiting
        case cancelled
    }

    private struct ActiveRequestState {
        let id: UUID
        let admittedAt: Date
        var modelID: String?
        var streaming: Bool?
        var requestedMaxTokens: Int?
        var toolCount: Int?
        var phase = "admitted"
        var phaseDetail: String?
        var firstTokenAt: Date?
        var generatedTokenUpdates = 0
        var lastEventSequence: UInt64 = 0

        var snapshot: RuntimeActiveRequestSnapshot {
            RuntimeActiveRequestSnapshot(
                id: id,
                admittedAt: admittedAt,
                modelID: modelID,
                streaming: streaming,
                requestedMaxTokens: requestedMaxTokens,
                toolCount: toolCount,
                phase: phase,
                phaseDetail: phaseDetail,
                firstTokenAt: firstTokenAt,
                generatedTokenUpdates: generatedTokenUpdates
            )
        }
    }

    private let maxActiveRequests: Int
    private let pressureProvider: @Sendable () async -> RuntimeMemoryPressureLevel
    private var activeRequests = 0
    private var waiters: [Waiter] = []
    private var waiterStates: [UUID: WaiterState] = [:]
    private var totalAdmittedRequests = 0
    private var totalCompletedRequests = 0
    private var totalCancelledRequests = 0
    private var activeRequestStates: [UUID: ActiveRequestState] = [:]
    private var lastClientDisconnectAt: Date?
    private var lastCancellationAt: Date?
    private var lastSlotReleaseAt: Date?

    public init(
        maxActiveRequests: Int,
        pressureProvider: @escaping @Sendable () async -> RuntimeMemoryPressureLevel = { .nominal }
    ) {
        precondition(maxActiveRequests > 0, "maxActiveRequests must be positive")
        self.maxActiveRequests = maxActiveRequests
        self.pressureProvider = pressureProvider
    }

    public func acquire() async throws -> RuntimeRequestAdmissionLease {
        try Task.checkCancellation()
        await drain()
        let canAdmit = await canAdmitNow(requireEmptyQueue: true)
        try Task.checkCancellation()
        if canAdmit {
            return admit()
        }

        let waiterID = UUID()
        waiterStates[waiterID] = .pending
        let lease = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                enqueueWaiter(id: waiterID, continuation: continuation)
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelWaiter(id: waiterID)
            }
        }
        if Task.isCancelled {
            await lease.release(cancelled: true)
            throw CancellationError()
        }
        return lease
    }

    /// Acquires immediately without joining the FIFO, or returns nil when an
    /// active/queued request already owns the capacity. Eviction uses this to
    /// avoid waiting behind and then unloading a runtime that was active when
    /// maintenance began.
    public func tryAcquire() async -> RuntimeRequestAdmissionLease? {
        guard !Task.isCancelled, waiters.isEmpty, activeRequests < maxActiveRequests else {
            return nil
        }
        let pressure = await pressureProvider()
        guard !Task.isCancelled, waiters.isEmpty,
              activeRequests < maxActiveRequests,
              !admissionPaused(for: pressure) || activeRequests == 0 else {
            return nil
        }
        return admit()
    }

    fileprivate func configureLease(
        id: UUID,
        modelID: String,
        streaming: Bool,
        requestedMaxTokens: Int,
        toolCount: Int
    ) {
        guard var state = activeRequestStates[id] else { return }
        state.modelID = modelID
        state.streaming = streaming
        state.requestedMaxTokens = requestedMaxTokens
        state.toolCount = toolCount
        activeRequestStates[id] = state
    }

    public func recordProgress(
        id: UUID,
        sequence: UInt64,
        progress: RuntimeRequestProgress,
        observedAt: Date = Date()
    ) {
        guard var state = activeRequestStates[id] else { return }
        if progress.stage == .generating, progress.message?.isEmpty == false {
            if let firstTokenAt = state.firstTokenAt {
                state.firstTokenAt = min(firstTokenAt, observedAt)
            } else {
                state.firstTokenAt = observedAt
            }
            state.generatedTokenUpdates += 1
        }
        guard state.phase != "cancelling", sequence > state.lastEventSequence else {
            activeRequestStates[id] = state
            return
        }
        state.lastEventSequence = sequence
        switch progress.stage {
        case .loadingModel:
            state.phase = "loading_model"
            state.phaseDetail = progress.message
        case .encoding:
            state.phase = "prefill"
            state.phaseDetail = progress.message
        case .generating:
            state.phase = "decode"
            state.phaseDetail = nil
        }
        activeRequestStates[id] = state
    }

    public func recordClientDisconnect(id: UUID, sequence: UInt64, observedAt: Date = Date()) {
        guard var state = activeRequestStates[id] else { return }
        state.phase = "cancelling"
        state.phaseDetail = "Client disconnected"
        state.lastEventSequence = max(state.lastEventSequence, sequence)
        activeRequestStates[id] = state
        lastClientDisconnectAt = observedAt
    }

    fileprivate func releaseLease(id: UUID, cancelled: Bool) async {
        activeRequests = max(0, activeRequests - 1)
        activeRequestStates.removeValue(forKey: id)
        lastSlotReleaseAt = Date()
        if cancelled {
            totalCancelledRequests += 1
            lastCancellationAt = Date()
        } else {
            totalCompletedRequests += 1
        }
        await drain()
    }

    public func snapshot() async -> RuntimeRequestAdmissionSnapshot {
        let pressure = await pressureProvider()
        return RuntimeRequestAdmissionSnapshot(
            maxActiveRequests: maxActiveRequests,
            activeRequests: activeRequests,
            queuedRequests: waiters.count,
            totalAdmittedRequests: totalAdmittedRequests,
            totalCompletedRequests: totalCompletedRequests,
            totalCancelledRequests: totalCancelledRequests,
            admissionPaused: admissionPaused(for: pressure),
            pressure: pressure.rawValue,
            activeRequestDetails: activeRequestStates.values
                .map(\.snapshot)
                .sorted { $0.admittedAt < $1.admittedAt },
            lastClientDisconnectAt: lastClientDisconnectAt,
            lastCancellationAt: lastCancellationAt,
            lastSlotReleaseAt: lastSlotReleaseAt
        )
    }

    private func enqueueWaiter(
        id: UUID,
        continuation: CheckedContinuation<RuntimeRequestAdmissionLease, Error>
    ) {
        if waiterStates[id] == .cancelled {
            waiterStates.removeValue(forKey: id)
            continuation.resume(throwing: CancellationError())
            return
        }

        waiterStates[id] = .waiting
        waiters.append(Waiter(id: id, continuation: continuation))
    }

    private func cancelWaiter(id: UUID) {
        switch waiterStates[id] {
        case .pending:
            waiterStates[id] = .cancelled
            totalCancelledRequests += 1
        case .waiting:
            guard let index = waiters.firstIndex(where: { $0.id == id }) else {
                waiterStates.removeValue(forKey: id)
                return
            }
            let waiter = waiters.remove(at: index)
            waiterStates.removeValue(forKey: id)
            totalCancelledRequests += 1
            waiter.continuation.resume(throwing: CancellationError())
        case .cancelled, nil:
            return
        }
    }

    private func drain() async {
        while activeRequests < maxActiveRequests, !waiters.isEmpty {
            guard await canAdmitNow() else {
                return
            }
            // Pressure sampling yields the actor. A concurrent drain may have
            // consumed capacity or cancellation may have emptied the FIFO, so
            // revalidate both before removing its first waiter.
            guard activeRequests < maxActiveRequests, !waiters.isEmpty else {
                return
            }
            let waiter = waiters.removeFirst()
            guard waiterStates[waiter.id] != .cancelled else {
                waiterStates.removeValue(forKey: waiter.id)
                waiter.continuation.resume(throwing: CancellationError())
                continue
            }
            waiterStates.removeValue(forKey: waiter.id)
            waiter.continuation.resume(returning: admit())
        }
    }

    private func admit() -> RuntimeRequestAdmissionLease {
        let id = UUID()
        activeRequests += 1
        totalAdmittedRequests += 1
        activeRequestStates[id] = ActiveRequestState(id: id, admittedAt: Date())
        return RuntimeRequestAdmissionLease(id: id, admission: self)
    }

    private func canAdmitNow(requireEmptyQueue: Bool = false) async -> Bool {
        guard activeRequests < maxActiveRequests,
              !requireEmptyQueue || waiters.isEmpty else {
            return false
        }
        let pressure = await pressureProvider()
        // `pressureProvider` is an actor reentrancy point. Never rely on the
        // capacity/FIFO snapshot taken before it suspended.
        guard activeRequests < maxActiveRequests,
              !requireEmptyQueue || waiters.isEmpty else {
            return false
        }
        guard admissionPaused(for: pressure) else {
            return true
        }
        return activeRequests == 0
    }

    private func admissionPaused(for pressure: RuntimeMemoryPressureLevel) -> Bool {
        switch pressure {
        case .elevated, .critical:
            return true
        case .disabled, .unknown, .nominal:
            return false
        }
    }
}

public final class RuntimeRequestAdmissionLease: @unchecked Sendable {
    private let id: UUID
    private let admission: RuntimeRequestAdmission
    private let lock = NSLock()
    private var released = false
    private var clientDisconnected = false
    private var nextEventSequence: UInt64 = 0

    fileprivate init(id: UUID, admission: RuntimeRequestAdmission) {
        self.id = id
        self.admission = admission
    }

    public var requestID: UUID { id }

    deinit {
        guard markReleased() else { return }
        let admission = admission
        let id = id
        Task {
            await admission.releaseLease(id: id, cancelled: false)
        }
    }

    public func configure(
        modelID: String,
        streaming: Bool,
        requestedMaxTokens: Int,
        toolCount: Int
    ) async {
        await admission.configureLease(
            id: id,
            modelID: modelID,
            streaming: streaming,
            requestedMaxTokens: requestedMaxTokens,
            toolCount: toolCount
        )
    }

    public func observe(_ progress: RuntimeRequestProgress) {
        guard let sequence = reserveEventSequence() else { return }
        let admission = admission
        let id = id
        let observedAt = Date()
        Task {
            await admission.recordProgress(
                id: id,
                sequence: sequence,
                progress: progress,
                observedAt: observedAt
            )
        }
    }

    public func observeClientDisconnect() {
        guard let sequence = reserveDisconnectSequence() else { return }
        let admission = admission
        let id = id
        let observedAt = Date()
        Task {
            await admission.recordClientDisconnect(
                id: id,
                sequence: sequence,
                observedAt: observedAt
            )
        }
    }

    public func release(cancelled: Bool = false) async {
        guard markReleased() else { return }
        await admission.releaseLease(id: id, cancelled: cancelled)
    }

    private func markReleased() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !released else { return false }
        released = true
        return true
    }

    private func reserveEventSequence() -> UInt64? {
        lock.lock()
        defer { lock.unlock() }
        guard !released, !clientDisconnected else { return nil }
        nextEventSequence += 1
        return nextEventSequence
    }

    private func reserveDisconnectSequence() -> UInt64? {
        lock.lock()
        defer { lock.unlock() }
        guard !released, !clientDisconnected else { return nil }
        clientDisconnected = true
        nextEventSequence += 1
        return nextEventSequence
    }
}
