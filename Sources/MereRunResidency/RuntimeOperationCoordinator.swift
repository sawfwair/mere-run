import Foundation

public enum RuntimeOperationMode: Equatable, Sendable {
    case warm
    case cold
}

/// A fair async reader/writer gate for resident runtime lanes.
///
/// Warm operations may overlap across lanes. A cold operation is exclusive:
/// it waits for in-flight warm work and blocks later warm requests so model
/// loading cannot overlap another sidecar's inference or a second cold load.
public actor RuntimeOperationCoordinator {
    public init() {}

    private struct Waiter {
        let id: UUID
        let mode: RuntimeOperationMode
        let continuation: CheckedContinuation<RuntimeOperationLease, Error>
    }

    private enum WaiterState {
        case pending
        case waiting
        case cancelled
    }

    private var activeWarmOperations = 0
    private var coldOperationActive = false
    private var waiters: [Waiter] = []
    private var waiterStates: [UUID: WaiterState] = [:]

    public func acquire(_ mode: RuntimeOperationMode) async throws -> RuntimeOperationLease {
        try Task.checkCancellation()
        if waiters.isEmpty, canAdmit(mode) {
            return admit(mode)
        }

        let waiterID = UUID()
        waiterStates[waiterID] = .pending
        let lease = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                enqueueWaiter(id: waiterID, mode: mode, continuation: continuation)
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelWaiter(id: waiterID)
            }
        }
        if Task.isCancelled {
            await lease.release()
            throw CancellationError()
        }
        return lease
    }

    fileprivate func release(_ mode: RuntimeOperationMode) {
        switch mode {
        case .warm:
            activeWarmOperations = max(0, activeWarmOperations - 1)
        case .cold:
            coldOperationActive = false
        }
        drain()
    }

    private func enqueueWaiter(
        id: UUID,
        mode: RuntimeOperationMode,
        continuation: CheckedContinuation<RuntimeOperationLease, Error>
    ) {
        if waiterStates[id] == .cancelled {
            waiterStates.removeValue(forKey: id)
            continuation.resume(throwing: CancellationError())
            return
        }
        waiterStates[id] = .waiting
        waiters.append(Waiter(id: id, mode: mode, continuation: continuation))
        drain()
    }

    private func cancelWaiter(id: UUID) {
        switch waiterStates[id] {
        case .pending:
            waiterStates[id] = .cancelled
        case .waiting:
            guard let index = waiters.firstIndex(where: { $0.id == id }) else {
                waiterStates.removeValue(forKey: id)
                return
            }
            let waiter = waiters.remove(at: index)
            waiterStates.removeValue(forKey: id)
            waiter.continuation.resume(throwing: CancellationError())
            drain()
        case .cancelled, nil:
            return
        }
    }

    private func drain() {
        guard !coldOperationActive, !waiters.isEmpty else { return }
        if activeWarmOperations > 0 {
            admitLeadingWarmWaiters()
            return
        }

        if waiters[0].mode == .cold {
            resumeFirstWaiter()
        } else {
            admitLeadingWarmWaiters()
        }
    }

    private func admitLeadingWarmWaiters() {
        while let waiter = waiters.first, waiter.mode == .warm, !coldOperationActive {
            resumeFirstWaiter()
        }
    }

    private func resumeFirstWaiter() {
        let waiter = waiters.removeFirst()
        guard waiterStates[waiter.id] != .cancelled else {
            waiterStates.removeValue(forKey: waiter.id)
            waiter.continuation.resume(throwing: CancellationError())
            drain()
            return
        }
        waiterStates.removeValue(forKey: waiter.id)
        waiter.continuation.resume(returning: admit(waiter.mode))
    }

    private func canAdmit(_ mode: RuntimeOperationMode) -> Bool {
        guard !coldOperationActive else { return false }
        switch mode {
        case .warm:
            return true
        case .cold:
            return activeWarmOperations == 0
        }
    }

    private func admit(_ mode: RuntimeOperationMode) -> RuntimeOperationLease {
        switch mode {
        case .warm:
            activeWarmOperations += 1
        case .cold:
            precondition(activeWarmOperations == 0 && !coldOperationActive)
            coldOperationActive = true
        }
        return RuntimeOperationLease(coordinator: self, mode: mode)
    }
}

public final class RuntimeOperationLease: @unchecked Sendable {
    private let coordinator: RuntimeOperationCoordinator
    private let mode: RuntimeOperationMode
    private let lock = NSLock()
    private var released = false

    fileprivate init(coordinator: RuntimeOperationCoordinator, mode: RuntimeOperationMode) {
        self.coordinator = coordinator
        self.mode = mode
    }

    deinit {
        guard markReleased() else { return }
        let coordinator = coordinator
        let mode = mode
        Task {
            await coordinator.release(mode)
        }
    }

    public func release() async {
        guard markReleased() else { return }
        await coordinator.release(mode)
    }

    private func markReleased() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !released else { return false }
        released = true
        return true
    }
}
