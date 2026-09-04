import Combine
import Foundation

/// Publishes the structural changes of a `JobStore` — a job starting or finishing — so a view
/// that derives *which* jobs exist (the feed's cards, a pull's progress row) re-renders exactly
/// then, while each running card observes its own `Job` for progress and log updates. Attached
/// once to the controller's store; never mutates it.
@MainActor
package final class StudioJobMonitor: ObservableObject {
    /// Bumped on every start and finish. Views read it through `objectWillChange`.
    @Published package private(set) var generation = 0
    private var store: JobStore?
    private var subscription: AnyCancellable?

    package init() {}

    /// Starts observing `store`. Attaching the same store twice is a no-op.
    package func attach(_ store: JobStore) {
        guard self.store !== store else { return }
        self.store = store
        subscription = store.events.sink { [weak self] event in
            switch event {
            case .started, .finished:
                self?.generation += 1
            case .changed, .output:
                break
            }
        }
        generation += 1
    }

    /// The job for a durable Studio request id, preferring one still active.
    package func job(requestID: UUID) -> Job? {
        store?.job(requestID: requestID)
    }

    /// Whether any inference job for one of `requestIDs` is running.
    package func isRunning(anyOf requestIDs: [UUID]) -> Bool {
        guard let store else { return false }
        let wanted = Set(requestIDs)
        return store.running(in: .inference).contains {
            $0.request.requestID.map(wanted.contains) ?? false
        }
    }

    /// The active `model pull` job for `modelID`, if one is downloading it right now.
    package func pullJob(for modelID: String) -> Job? {
        guard let store, !modelID.isEmpty else { return nil }
        return (store.running(in: .inference) + store.queued(in: .inference)).first {
            $0.request.templateID == .modelPull && $0.request.draft?.model == modelID
        }
    }

    /// The store's inference lane has a free slot, so a new run starts rather than queues.
    package var hasInferenceCapacity: Bool {
        store?.hasCapacity(in: .inference) ?? true
    }

    @discardableResult
    package func cancel(_ job: Job) -> Bool {
        store?.cancel(job.id) ?? false
    }
}
