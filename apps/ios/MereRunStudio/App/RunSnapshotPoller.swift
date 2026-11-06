import Combine
import Foundation

/// Owns one snapshot watch. Stopping or replacing a watch invalidates reads that
/// ignore cancellation, including reads started before a retry or reconnect.
@MainActor
final class RunSnapshotPoller: ObservableObject {
    enum Phase: Equatable {
        case stopped
        case polling(UUID, consecutiveFailures: Int)
        case terminal
        case paused
    }

    @Published private(set) var phase: Phase = .stopped
    private let policy: RunPollingPolicy

    init(policy: RunPollingPolicy = RunPollingPolicy()) {
        self.policy = policy
    }

    func stop() { phase = .stopped }

    func watch<Value>(
        read: @MainActor () async throws -> Value,
        isTerminal: @MainActor (Value) -> Bool,
        receive: @MainActor (Value) -> Void,
        failed: @MainActor (Error) -> Void,
        sleep: @MainActor (UInt64) async throws -> Void = { try await Task.sleep(nanoseconds: $0) }
    ) async {
        guard !Task.isCancelled else { return }
        let ticket = UUID()
        phase = .polling(ticket, consecutiveFailures: 0)
        defer {
            if isCurrent(ticket) { phase = .stopped }
        }
        var failures = 0
        while isCurrent(ticket), !Task.isCancelled {
            do {
                let value = try await read()
                guard isCurrent(ticket), !Task.isCancelled else { return }
                failures = 0
                let terminal = isTerminal(value)
                phase = terminal ? .terminal : .polling(ticket, consecutiveFailures: 0)
                receive(value)
                if terminal { return }
            } catch is CancellationError {
                return
            } catch {
                guard isCurrent(ticket), !Task.isCancelled else { return }
                failures += 1
                phase = .polling(ticket, consecutiveFailures: failures)
                failed(error)
                guard isCurrent(ticket) else { return }
                if policy.shouldStop(afterConsecutiveFailures: failures) {
                    phase = .paused
                    return
                }
            }
            guard isCurrent(ticket), !Task.isCancelled else { return }
            do {
                try await sleep(policy.delayNanoseconds(afterConsecutiveFailures: failures))
            } catch {
                return
            }
        }
    }

    private func isCurrent(_ ticket: UUID) -> Bool {
        guard case .polling(let current, _) = phase else { return false }
        return current == ticket
    }
}
