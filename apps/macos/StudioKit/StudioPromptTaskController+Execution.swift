import Foundation

extension StudioPromptTaskController {
    package struct Submission {
        package let request: StudioRunRequest
        package let outputFallbackReason: String?
    }

    package typealias ValidationError = StudioValidationError

    /// Resolves Command edits and validates before a request can change history or create output
    /// directories. The existing adapter, contract, and sessions own those decisions.
    func preparedRequest(mode: StudioMode, draft: StudioDraft, conversationID: UUID? = nil) throws -> StudioRunRequest {
        let base = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft, conversationID: conversationID)
        let request = sessions.resolving(base)
        try validate(request)
        return request
    }

    private func validate(_ request: StudioRunRequest) throws {
        let message = request.template.validationMessage(for: request.draft, execution: request.execution)
        if let message { throw ValidationError(message: message) }
    }

    /// The model and readiness gates every submission clears before it may change history:
    /// send and retry share them so a retry on a missing or unsupported model keeps its reply.
    func ensureRunnable(mode: StudioMode, draft: StudioDraft) throws {
        switch StudioCommandAdapter.capabilityRequirement(for: mode, draft: draft) {
        case .unavailable(let message): throw ValidationError(message: message)
        case .managedModel(let modelID):
            if let message = controller.modelCapabilitiesByID[modelID]?.unavailableMessage(titles: controller.modelStore.titles) {
                throw ValidationError(message: message)
            }
        case nil: break
        }
        let readiness = controller.readinessByMode[mode] ?? .notChecked
        if readiness.blocksRun { throw ValidationError(message: readiness.message(titles: controller.modelStore.titles)) }
    }

    package func runPrompt(
        inventory: [StudioModelInventoryRow],
        prepareOutput: (CommandDraft) -> StudioOutputLocation.Preparation = { StudioOutputLocation.preparingDestination(of: $0) }
    ) throws -> Submission? {
        guard let mode = activatedMode else { return nil }
        try ensureRunnable(mode: mode, draft: draft)
        if mode.isConversational {
            return try sendConversationTurn(inventory: inventory).map { Submission(request: $0, outputFallbackReason: nil) }
        }
        let request = try preparedRequest(mode: mode, draft: draft)
        let prepared = prepareOutput(request.draft)
        let effective = StudioOutputLocation.request(request, preparedAs: prepared)
        submitLibraryRequest(effective)
        return Submission(request: effective, outputFallbackReason: prepared.fallbackReason)
    }

    /// Specialist forms use the same stored Command override as their Command panel. The task
    /// runner does the work; this stays so the shell has one call for either kind of task.
    package func runTask(_ base: StudioRunRequest, task: StudioTask) throws -> StudioRunRequest {
        try runner.run(request: base, task: task)
    }

    /// Historical replay uses its recorded command, never the current task's Command edits.
    package func replay(_ item: StudioLibraryItem, variationSeed: String? = nil) throws -> StudioRunRequest {
        guard let request = StudioLibraryReplay.request(for: item, variationSeed: variationSeed) else {
            throw ValidationError(message: "This older Library item does not include a replayable command.")
        }
        try validate(request)
        submitLibraryRequest(request)
        return request
    }

    private func submitLibraryRequest(_ request: StudioRunRequest) {
        let arguments = request.execution?.arguments ?? request.template.arguments(from: request.draft)
        library.start(request: request, commandPreview: controller.commandPreview(arguments: arguments, masksSecrets: true),
                      status: controller.jobs.hasCapacity(in: .inference) ? .running : .queued)
        controller.run(studio: request)
    }

    /// Selection determines Stop, even when another task submitted a newer job.
    package func currentJob(for task: StudioTask) -> Job? {
        if task.mode?.isConversational == true {
            guard let id = activeConversationID else { return nil }
            return controller.jobs.all.first { $0.state.isActive && $0.request.conversationID == id }
        }
        return runner.currentJob(for: task)
    }

    /// A conversation turn is cancelled; every other task stops the way its own page does
    /// (`StudioTaskRunner.stop`), so ⌘. on Audio ▸ Live interrupts the session first and lets
    /// the CLI flush its last events.
    package func stop(task: StudioTask) {
        guard task.mode?.isConversational == true else { return runner.stop(task: task) }
        if let job = currentJob(for: task) { controller.jobs.cancel(job.id) }
    }
}
