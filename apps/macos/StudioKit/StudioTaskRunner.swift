import AVFoundation
import Foundation

/// Why a request cannot run yet, in the words the composer's banner shows.
package struct StudioValidationError: LocalizedError, Equatable {
    package let message: String

    package init(message: String) {
        self.message = message
    }

    package var errorDescription: String? { message }
}

/// The one way a task's run reaches the Library and the job store. The prompt controller,
/// shared task workspace, and project, session, and manage pages all come through here, so
/// every run is prepared, attributed, recorded, and remembered for Stop the same way.
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
    /// records a failed Library row with the reason for pages that show errors in their own pane.
    package static func prepare(
        _ base: StudioRunRequest,
        sessions: StudioTaskSessions,
        source: StudioScopeSource,
        validating: Bool = true,
        fileManager: FileManager = .default
    ) throws -> (request: StudioRunRequest, fallbackReason: String?) {
        let resolved = sessions.resolving(base, source: source)
        if validating, let message = resolved.template.validationMessage(
            for: resolved.draft, execution: resolved.execution, source: source
        ) {
            throw StudioValidationError(message: message)
        }
        return StudioOutputLocation.preparing(resolved, fileManager: fileManager)
    }

    /// The draft as a run launches it: the launch-time defaults a task's page applies (a Klein
    /// base gets its checkpoint and preview cadence; Audio ▸ Live's commands print events only,
    /// as JSON lines where the transcript reads them). The runner and the Command view's
    /// "Will run" both go through this, so a Run from either surface and the preview agree;
    /// every other task's draft is left as it is.
    package static func launching(_ draft: StudioTaskDraft, source: StudioScopeSource) -> StudioTaskDraft {
        switch draft.templateID {
        case .speechListen, .speechDiarizeLive: return draft.liveListenLaunch()
        default: return StudioTrainingRun.launchDraft(draft, source: source)
        }
    }

    /// The draft a Run launches, destinations and all: `launching`, then named by
    /// `StudioOutputLocation.destination(for:)`. The Command view's "Will run" reads this, so it
    /// shows the files the run will write rather than the draft's blank destination.
    package static func launchPreview(_ draft: StudioTaskDraft, source: StudioScopeSource) -> StudioTaskDraft {
        StudioOutputLocation.destination(for: launching(draft, source: source), source: source)
    }

    /// The request a task draft runs: its launch-time defaults applied, its destination named,
    /// validated and prepared, then a camera draft the inspector wrote copied beside the output
    /// (`StudioCameraDocuments`) so the run's folder carries its own file. Static so the
    /// live-acceptance tests build exactly what the app runs. Throws before anything is created
    /// when the command is incomplete.
    package static func prepare(
        draft: StudioTaskDraft,
        sessions: StudioTaskSessions,
        source: StudioScopeSource,
        fileManager: FileManager = .default
    ) throws -> (request: StudioRunRequest, fallbackReason: String?) {
        let named = StudioOutputLocation.destination(for: launching(draft, source: source), source: source, fileManager: fileManager)
        guard let base = named.request(source: source) else {
            throw StudioValidationError(message: "This command can't run from Studio.")
        }
        var prepared = try prepare(base, sessions: sessions, source: source, fileManager: fileManager)
        if prepared.fallbackReason == nil, let page = StudioCameraDocuments.draftPage(for: draft.templateID) {
            let placed = try StudioCameraDocuments.placingDraft(of: named, page: page, fileManager: fileManager)
            if placed != named, let request = placed.request(source: source) {
                prepared = try prepare(request, sessions: sessions, source: source, fileManager: fileManager)
            }
        }
        return prepared
    }

    package func request(for draft: StudioTaskDraft, task: StudioTask) throws -> StudioRunRequest {
        let prepared = try Self.prepare(draft: draft, sessions: sessions, source: controller.scopeSource)
        if let reason = prepared.fallbackReason { controller.noteOutputFallback(reason) }
        return prepared.request
    }

    /// Runs a task draft: the readiness gate the composer shows, then the prepared request into
    /// the Library and the inference lane, remembered as the task's current job for Stop.
    @discardableResult
    package func run(_ draft: StudioTaskDraft, task: StudioTask) throws -> StudioRunRequest {
        try ensureRunnable(draft, task: task)
        let request = try self.request(for: draft, task: task)
        submit(request, task: task)
        return request
    }

    /// The gates a task draft clears before any of its runs is prepared.
    private func ensureRunnable(_ draft: StudioTaskDraft, task: StudioTask) throws {
        let readiness = controller.readiness(for: task)
        if readiness.blocksRun { throw StudioValidationError(message: readiness.message(titles: controller.modelStore.titles)) }
        // The contract leaves an input optional when the command has another mode without one
        // (`music transcribe --list-instruments`) or another way to take it (Faces ▸ Batch's
        // pictures or its `--input-list` file); the task's surface says whether a run needs the
        // well filled, so an empty well never launches the CLI to fail on its own. Where the
        // contract itself requires the first slot, only that slot fills the well.
        if let slot = StudioTaskSchema.primarySlot(for: draft.templateID),
           task.presentation.attaching(slot).requiresAttachment,
           slot.isRequired
            ? draft.primaryInputPath.isBlank
            : draft.slots(source: controller.scopeSource).allSatisfy({ $0.paths(in: draft).isEmpty }) {
            throw StudioValidationError(message: "Attach \(slot.label.lowercased()) first.")
        }
        // A typed input is the run's whole subject; `text anonymize` would otherwise launch and
        // wait on a stdin nobody can type into.
        if task.analyzeArchetype?.inputKind(for: draft.templateID) == .text, draft.prompt.isBlank {
            throw StudioValidationError(message: "Type the text first.")
        }
    }

    // MARK: Variations

    /// The composer's "Run variations" on a shared task workspace: the draft once per seed,
    /// each prepared the way Run prepares it, then submitted as one group.
    @discardableResult
    package func runVariations(_ draft: StudioTaskDraft, task: StudioTask, seeds: [String]) throws -> [StudioRunRequest] {
        try ensureRunnable(draft, task: task)
        let requests = try seeds.map { try request(for: StudioVariations.seeded(draft, seed: $0), task: task) }
        submitGroup(requests, task: task)
        return requests
    }

    /// Submits requests that differ only in their seed — a prompt composer's variations, or a
    /// Library row's replays — as one variation group: each destination made real, each run
    /// recorded and launched like any other, then the rows filed under one group id.
    @discardableResult
    package func submitVariations(_ requests: [StudioRunRequest], task: StudioTask) -> StudioVariationSubmission {
        var fallbackReason: String?
        let prepared = requests.map { request in
            let prepared = StudioOutputLocation.preparing(request)
            if let reason = prepared.fallbackReason {
                controller.noteOutputFallback(reason)
                fallbackReason = reason
            }
            return prepared.request
        }
        submitGroup(prepared, task: task)
        return StudioVariationSubmission(requests: prepared, outputFallbackReason: fallbackReason)
    }

    /// "Run variations" on a Library row: its recorded command once per seed, never the task's
    /// current Command edits, submitted from the task that owns the row.
    @discardableResult
    package func replayVariations(of item: StudioLibraryItem, seeds: [String]) throws -> [StudioRunRequest] {
        let requests = try StudioVariations.replayRequests(for: item, seeds: seeds, source: controller.scopeSource)
        return submitVariations(requests, task: StudioVariations.submittingTask(for: item)).requests
    }

    private func submitGroup(_ requests: [StudioRunRequest], task: StudioTask) {
        for request in requests { submit(request, task: task) }
        // Stop acts on the run that starts first, not on the last one queued behind it.
        if let first = requests.first { sessions.set(Optional(first.id), for: task.rawValue + ".requestID") }
        library.assignVariationGroup(UUID(), to: requests.map(\.id))
    }

    /// Runs a request a project, session, or manage page built from its own `CommandDraft`,
    /// prepared the same way. The Command view validates first and shows the reason in its
    /// banner; pages with their own result pane pass `validating: false` so an incomplete
    /// command becomes a failed row with the reason.
    @discardableResult
    package func run(
        request base: StudioRunRequest,
        task: StudioTask,
        validating: Bool = true,
        onLaunchRefused: (() -> Void)? = nil
    ) throws -> StudioRunRequest {
        let prepared = try Self.prepare(base, sessions: sessions, source: controller.scopeSource, validating: validating)
        if let reason = prepared.fallbackReason { controller.noteOutputFallback(reason) }
        if !submit(prepared.request, task: task) { onLaunchRefused?() }
        return prepared.request
    }

    /// Records the run and launches it. The Library row starts as running or queued by whether
    /// the inference lane has a slot, the same reading the feed's cards make.
    @discardableResult
    func submit(_ request: StudioRunRequest, task: StudioTask) -> Bool {
        sessions.set(Optional(request.id), for: task.rawValue + ".requestID")
        sessions.noteSubmission(request.id, from: task)
        let arguments = request.execution?.arguments ?? request.template.arguments(from: request.draft, source: controller.scopeSource)
        let preview = controller.commandPreview(arguments: arguments, masksSecrets: true)
        library.start(request: request, commandPreview: preview,
                      status: controller.jobs.hasCapacity(in: .inference) ? .running : .queued,
                      source: controller.scopeSource)
        // `run(studio:)` keeps the camera-access gate in front of `vision track-live`. While the
        // system is still asking, the controller retries once the answer comes — unless Stop
        // cancelled the row in the meantime.
        let launched = controller.run(studio: request, stillWanted: { [controller, library] in
            Self.isAwaitingLaunch(request.id, controller: controller, library: library)
        })
        guard !launched else { return true }
        if request.template.id == .visionTrackLive,
           AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined { return false }
        // A synchronous refusal already has a terminal job result. Record it here as well as
        // through the Library observer, so this row cannot remain queued or running.
        if let row = library.items.first(where: { $0.id == request.id }),
           row.status == .running || row.status == .queued {
            let result = controller.jobs.job(requestID: request.id)?.result
            let reason = request.template.id == .visionTrackLive
                && AVCaptureDevice.authorizationStatus(for: .video) != .authorized
                ? Self.cameraDeniedMessage
                : (result?.outputText ?? "The run could not be started.")
            library.complete(
                id: request.id, exitCode: result?.exitCode ?? 1, outputURL: result?.outputURL,
                outputText: reason, commandPreview: preview,
                artifactURLs: result?.artifactURLs ?? [], artifactRoles: result?.artifactRoles ?? [:]
            )
        }
        return false
    }

    /// What a Live row says when the Mac will not give mere.run the camera.
    package static let cameraDeniedMessage =
        "Camera access is off for mere.run. Turn it on in System Settings ▸ Privacy & Security ▸ Camera, then start again."

    /// The job Stop acts on: the run this task last submitted while it is alive, else the newest
    /// live run that belongs to the task — submitted from it, or, for a run no task submitted
    /// (the Command Console), one of the commands the task owns. Audio ▸ Separate runs
    /// Music ▸ Separate's command, so the command alone never makes another task's run its own.
    package func currentJob(for task: StudioTask) -> Job? {
        let remembered = sessions.value(for: task.rawValue + ".requestID", default: Optional<UUID>.none)
        if let remembered, let job = controller.jobs.job(requestID: remembered), job.state.isActive { return job }
        return controller.jobs.all.last { job in
            guard job.state.isActive else { return false }
            let owner = job.request.requestID.flatMap(sessions.submittingTask(of:)) ?? job.request.templateID?.studioTask
            return owner == task
        }
    }

    /// How long a Session task's Stop waits for the CLI to finish on SIGINT before terminating it.
    package static let sessionStopGrace: Duration = .seconds(4)

    /// Stops the task's current job: a session the way Ctrl-C does, so the CLI flushes what it
    /// has (then terminated after `sessionStopGrace`); anything else terminated at once. A run
    /// submitted but not launched yet — Vision ▸ Live while macOS asks for the camera — has no
    /// job to stop; its row is cancelled, so the launch waiting on the answer never happens.
    package func stop(task: StudioTask) {
        guard let job = currentJob(for: task) else {
            let remembered = sessions.value(for: task.rawValue + ".requestID", default: Optional<UUID>.none)
            if let remembered, isAwaitingLaunch(remembered) { library.setStatus(.cancelled, id: remembered) }
            return
        }
        if task.archetype == .session {
            controller.jobs.interruptThenCancel(job.id, after: Self.sessionStopGrace)
        } else {
            controller.jobs.cancel(job.id)
        }
    }

    /// Whether a submitted run is still waiting to launch: its row is running or queued and no
    /// job exists for it yet.
    package func isAwaitingLaunch(_ requestID: UUID) -> Bool {
        Self.isAwaitingLaunch(requestID, controller: controller, library: library)
    }

    private static func isAwaitingLaunch(_ requestID: UUID, controller: MereRunController, library: StudioLibraryStore) -> Bool {
        guard controller.jobs.job(requestID: requestID) == nil,
              let row = library.items.first(where: { $0.id == requestID }) else { return false }
        return row.status == .running || row.status == .queued
    }
}
