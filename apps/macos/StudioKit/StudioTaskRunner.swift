import Foundation

/// Why a request cannot run yet, in the words the composer's banner shows.
package struct StudioValidationError: LocalizedError, Equatable {
    package let message: String

    package init(message: String) {
        self.message = message
    }

    package var errorDescription: String? { message }
}

/// The one way a task's run reaches the Library and the job store. The prompt controller's
/// `runTask`, the shared task workspace's Run, and — until the last page moves —
/// `StudioSpecialistRunner.submit` all come through here, so every run is named, validated,
/// prepared, attributed, recorded, and remembered for Stop the same way.
@MainActor
package final class StudioTaskRunner {
    package let controller: MereRunController
    package let library: StudioLibraryStore

    package init(controller: MereRunController, library: StudioLibraryStore) {
        self.controller = controller
        self.library = library
    }

    private var sessions: StudioTaskSessions { controller.taskSessions }

    /// What every submission does to a request before it may change history or create output
    /// directories: apply the task's Command edits, validate the effective command, and make its
    /// destination real (or move it to App Outputs and say why). Static so the live-acceptance
    /// tests build exactly what the app runs.
    ///
    /// `validating: false` skips the throw and lets job admission fail the run instead, which
    /// records a failed Library row with the reason — what the legacy pages show, since they have
    /// no banner of their own to put a thrown message in.
    package static func prepare(
        _ base: StudioRunRequest,
        sessions: StudioTaskSessions,
        validating: Bool = true,
        fileManager: FileManager = .default
    ) throws -> (request: StudioRunRequest, fallbackReason: String?) {
        let resolved = sessions.resolving(base)
        if validating, let message = resolved.template.validationMessage(for: resolved.draft, execution: resolved.execution) {
            throw StudioValidationError(message: message)
        }
        return StudioOutputLocation.preparing(resolved, fileManager: fileManager)
    }

    /// The request a task draft runs: its destination named, then prepared. Throws before
    /// anything is created when the command is incomplete.
    package func request(for draft: StudioTaskDraft, task: StudioTask) throws -> StudioRunRequest {
        guard let base = StudioOutputLocation.destination(for: draft).request() else {
            throw StudioValidationError(message: "This command can't run from Studio.")
        }
        let prepared = try Self.prepare(base, sessions: sessions)
        if let reason = prepared.fallbackReason { controller.noteOutputFallback(reason) }
        return prepared.request
    }

    /// Runs a task draft: the readiness gate the composer shows, then the prepared request into
    /// the Library and the inference lane, remembered as the task's current job for Stop.
    @discardableResult
    package func run(_ draft: StudioTaskDraft, task: StudioTask) throws -> StudioRunRequest {
        let readiness = controller.readiness(for: task)
        if readiness.blocksRun { throw StudioValidationError(message: readiness.message(titles: controller.modelStore.titles)) }
        // The contract leaves an input optional when the command has another mode without one
        // (`music transcribe --list-instruments`); the task's surface says whether a run needs
        // the well filled, so an empty well never launches the CLI to fail on its own.
        if let slot = StudioTaskSchema.primarySlot(for: draft.templateID),
           task.presentation.attaching(slot).requiresAttachment, draft.primaryInputPath.isBlank {
            throw StudioValidationError(message: "Attach \(slot.label.lowercased()) first.")
        }
        // A typed input is the run's whole subject; `text anonymize` would otherwise launch and
        // wait on a stdin nobody can type into.
        if task.analyzeArchetype?.inputKind(for: draft.templateID) == .text, draft.prompt.isBlank {
            throw StudioValidationError(message: "Type the text first.")
        }
        let request = try self.request(for: draft, task: task)
        submit(request, task: task)
        return request
    }

    /// Runs a request a page built from its own `CommandDraft` (the legacy pages and the Command
    /// view's Run on them), prepared the same way. The Command view validates first and shows
    /// the reason in its banner; a legacy page's own Run passes `validating: false` so an
    /// incomplete command becomes a failed row in its result view, as it always did.
    @discardableResult
    package func run(request base: StudioRunRequest, task: StudioTask, validating: Bool = true) throws -> StudioRunRequest {
        let prepared = try Self.prepare(base, sessions: sessions, validating: validating)
        if let reason = prepared.fallbackReason { controller.noteOutputFallback(reason) }
        submit(prepared.request, task: task)
        return prepared.request
    }

    /// Records the run and launches it. The Library row starts as running or queued by whether
    /// the inference lane has a slot, the same reading the feed's cards make.
    private func submit(_ request: StudioRunRequest, task: StudioTask) {
        sessions.set(Optional(request.id), for: task.rawValue + ".requestID")
        let arguments = request.execution?.arguments ?? request.template.arguments(from: request.draft)
        library.start(request: request, commandPreview: controller.commandPreview(arguments: arguments, masksSecrets: true),
                      status: controller.jobs.hasCapacity(in: .inference) ? .running : .queued)
        // `run(studio:)` keeps the camera-access gate in front of `vision track-live`.
        _ = controller.run(studio: request)
    }

    /// The job Stop acts on: the run this task last submitted while it is alive, else the newest
    /// live run of any of the task's templates.
    package func currentJob(for task: StudioTask) -> Job? {
        let remembered = sessions.value(for: task.rawValue + ".requestID", default: Optional<UUID>.none)
        if let remembered, let job = controller.jobs.job(requestID: remembered), job.state.isActive { return job }
        return controller.jobs.all.last { $0.state.isActive && $0.request.templateID?.studioTask == task }
    }

    package func stop(task: StudioTask) {
        if let job = currentJob(for: task) { controller.jobs.cancel(job.id) }
    }
}
