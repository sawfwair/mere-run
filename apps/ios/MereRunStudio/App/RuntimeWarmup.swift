import Foundation

/// Owns the preparation task until it actually settles. Cancellation is a request; draining
/// still waits for native preparation that ignores cancellation before allowing unload/reuse.
@MainActor
final class RuntimeWarmup {
    private struct Work {
        let lease: RuntimeResidencyState.WarmupLease
        let task: Task<Bool, Never>
    }

    private var work: Work?
    var isRunning: Bool { work != nil }

    func start(
        _ lease: RuntimeResidencyState.WarmupLease,
        prepare: @escaping @MainActor () async throws -> Void,
        completed: @escaping @MainActor (RuntimeResidencyState.WarmupLease, Bool) -> Void
    ) {
        precondition(work == nil, "Drain the previous preparation before starting another.")
        let task = Task { [weak self] in
            let succeeded: Bool
            do {
                try Task.checkCancellation()
                try await prepare()
                succeeded = !Task.isCancelled
            } catch {
                succeeded = false
            }
            if self?.work?.lease == lease { self?.work = nil }
            completed(lease, succeeded)
            return succeeded
        }
        work = Work(lease: lease, task: task)
    }

    func cancel() { work?.task.cancel() }

    /// Nil means no preparation was outstanding. A failed/cancelled handoff lets chat discard
    /// partial preparation before loading again, while a successful handoff reuses its weights.
    @discardableResult
    func drain(cancel: Bool) async -> Bool? {
        guard let current = work else { return nil }
        if cancel { current.task.cancel() }
        let succeeded = await current.task.value
        if work?.lease == current.lease { work = nil }
        return succeeded
    }
}
