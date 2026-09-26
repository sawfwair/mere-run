import Combine
import Foundation

/// One long-lived server command — `api serve`, `vision serve`, `music serve` — as Studio runs it:
/// in the `.service` lane, where it holds no generation slot and is not a Library run. It owns the
/// job it started, adopts one the Command Console started, and gives Start, Stop, and Restart to
/// every surface that shows the server, so none of them keeps its own copy of the process.
@MainActor
package final class StudioServiceProcess: ObservableObject {
    /// What the server process is doing.
    package enum State: Equatable {
        /// Studio has not started this server, or its last one exited when asked to.
        case none(exitedAt: Date?)
        /// `since` is when the process launched: an endpoint answer from before then was
        /// another server's.
        case running(stopRequested: Bool, since: Date)
        /// The last server exited on its own, or never launched; the message is why.
        case failed(String, exitedAt: Date)

        package var isRunning: Bool {
            if case .running = self { return true }
            return false
        }

        package var isStopping: Bool {
            if case .running(stopRequested: true, _) = self { return true }
            return false
        }
    }

    package let templateID: CommandTemplateID
    @Published package private(set) var state = State.none(exitedAt: nil)

    /// "API server", "Vision server": how menus, alerts, and notifications name it.
    package var title: String {
        switch templateID {
        case .apiServe: return "API server"
        case .visionServe: return "Vision server"
        case .musicServe: return "Music server"
        case .worldServe: return "World server"
        default: return CommandCatalog.template(id: templateID)?.title ?? "Server"
        }
    }

    private weak var controller: MereRunController?
    private var jobID: JobID?
    private var subscription: AnyCancellable?

    package init(templateID: CommandTemplateID, controller: MereRunController) {
        self.templateID = templateID
        self.controller = controller
        jobID = controller.jobs.running.first {
            $0.request.templateID == templateID && $0.request.draft?.preflight != true
        }?.id
        subscription = controller.jobs.events.sink { [weak self] event in
            guard let self else { return }
            switch event {
            case .started(let job) where job.request.templateID == templateID && job.request.draft?.preflight != true:
                // A server the Command Console started is still this Mac's server: adopt it so
                // Stop and Restart reach it, unless Studio already owns a live one. A preflight
                // run checks and exits; it is not a server.
                if !(self.job?.state.isActive ?? false) { self.jobID = job.id }
                self.refresh()
            case .changed(let job) where job.id == self.jobID, .finished(let job, _) where job.id == self.jobID:
                self.refresh()
            default:
                break
            }
        }
        refresh()
    }

    /// The job running (or last run for) this server, for its log.
    package var job: Job? {
        jobID.flatMap { controller?.jobs.job($0) }
    }

    /// Launches the server with `draft`, or with the command its task's Command view edited.
    /// `pinned` options keep their values even when that command edited them. Does nothing while
    /// Studio's server is still running.
    package func start(draft: CommandDraft, pinned: [(flag: String, value: String)] = []) {
        guard let controller, let template = CommandCatalog.template(id: templateID), !state.isRunning else { return }
        let request = controller.taskSessions.resolving(StudioRunRequest(
            mode: templateID.studioTask.mode ?? .chat,
            templateID: templateID,
            template: template,
            draft: draft
        ), source: controller.scopeSource)
        let arguments = request.execution.map { Self.pinning(pinned, in: $0.arguments) }
        jobID = controller.startService(template: template, draft: request.draft, arguments: arguments)
        refresh()
    }

    /// `arguments` with each pinned option set to its value, added when absent.
    static func pinning(_ pinned: [(flag: String, value: String)], in arguments: [String]) -> [String] {
        var arguments = arguments
        for option in pinned {
            if let index = arguments.firstIndex(of: option.flag), index + 1 < arguments.count {
                arguments[index + 1] = option.value
            } else {
                arguments += [option.flag, option.value]
            }
        }
        return arguments
    }

    /// Asks Studio's server to exit (SIGTERM). Returns false when Studio has none running.
    @discardableResult
    package func stop() -> Bool {
        guard let controller, let jobID, controller.jobs.cancel(jobID) else { return false }
        refresh()
        return true
    }

    /// Stops Studio's server, waits for the process to exit so its port is free, then starts
    /// `draft`. Returns false when Studio has no server running to restart.
    package func restart(draft: CommandDraft, pinned: [(flag: String, value: String)] = []) async -> Bool {
        guard let controller, let jobID, state.isRunning else { return false }
        controller.jobs.cancel(jobID)
        _ = await controller.jobs.result(for: jobID)
        start(draft: draft, pinned: pinned)
        return true
    }

    /// Runs `draft` once with `--preflight`, which validates the model and port and exits, and
    /// returns the failure to show, or nil when it passed.
    package func preflight(draft: CommandDraft) async -> String? {
        guard let controller, let template = CommandCatalog.template(id: templateID) else {
            return "The server command is unavailable."
        }
        var preflight = draft
        preflight.preflight = true
        preflight.json = true
        let result = await controller.utilityCommandResult(
            args: template.arguments(from: preflight, source: controller.scopeSource),
            environmentOverrides: CommandLaunchEnvironment.overrides(templateID: templateID, draft: preflight)
        )
        return result.exitCode == 0 ? nil : StudioActivitySanitizer.sanitize(result.outputText)
    }

    private func refresh() {
        let next = Self.state(of: job)
        guard state != next else { return }
        if case .running(stopRequested: false, _) = state, case .failed(let reason, _) = next {
            controller?.notifyServerStopped(title, reason: reason)
        }
        state = next
    }

    private static func state(of job: Job?) -> State {
        guard let job else { return .none(exitedAt: nil) }
        switch job.state {
        case .queued, .running:
            return .running(stopRequested: job.cancelRequested, since: job.startedAt ?? job.submittedAt)
        case .cancelled(_, let at):
            return .none(exitedAt: at)
        case .finished(let exit, let at):
            return exit == 0 && job.cancelRequested
                ? .none(exitedAt: at)
                : .failed(failureMessage(for: job, exit: exit), exitedAt: at)
        case .preflightFailed(let failure):
            return .failed(failure.message, exitedAt: job.result?.completedAt ?? job.submittedAt)
        }
    }

    /// The server's last error on stderr, or its exit status. A server that dies without saying why
    /// — killed, or crashed — leaves progress and startup lines last, which are not the reason.
    static func failureMessage(for job: Job, exit: Int32) -> String {
        let lastError = job.log.lines.last { $0.stream == .stderr && describesFailure($0.text) }
        guard let lastError else {
            return exit == 0 ? "The server exited." : "The server exited with status \(exit)."
        }
        return StudioActivitySanitizer.sanitize(lastError.text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Whether a log line states a failure, the way the CLI and the runtimes word one.
    static func describesFailure(_ line: String) -> Bool {
        let text = line.lowercased()
        return ["error", "failed", "fatal", "in use", "refused", "denied", "not found", "cannot", "unable"]
            .contains { text.contains($0) }
    }
}
