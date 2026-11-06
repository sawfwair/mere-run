import Foundation

/// Process exit and group termination are independent: the parent may have exited while its
/// descendants still own the pipes. Completion requires drainage, with a bound for detached
/// descendants that retain a pipe after leaving the owned process group.
struct ProcessShutdown {
    enum Action: Equatable {
        case terminate
        case kill
    }

    private enum StopPhase {
        case running
        case terminating(since: TimeInterval)
        case killing
    }

    private enum OutputPhase {
        case running
        case draining(since: TimeInterval)
        case complete
    }

    private var stop: StopPhase = .running
    private var output: OutputPhase = .running

    var isComplete: Bool {
        if case .complete = output { return true }
        return false
    }

    mutating func advance(
        processRunning: Bool,
        ownsLiveGroup: Bool,
        streamsAtEnd: Bool,
        outputFailed: Bool,
        terminationRequested: Bool,
        now: TimeInterval
    ) -> [Action] {
        guard !isComplete else { return [] }
        var actions: [Action] = []
        switch stop {
        case .running where terminationRequested || outputFailed || (!processRunning && ownsLiveGroup):
            stop = .terminating(since: now)
            actions.append(.terminate)
        case .terminating(let since) where now - since >= 0.2:
            stop = .killing
            actions.append(.kill)
        default:
            break
        }

        if !processRunning {
            if case .running = output { output = .draining(since: now) }
            if streamsAtEnd && !ownsLiveGroup {
                output = .complete
            } else if case .draining(let since) = output, now - since >= 0.4 {
                output = .complete
            }
        }
        return actions
    }
}
