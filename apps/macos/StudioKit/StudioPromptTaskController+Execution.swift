import Foundation

extension StudioPromptTaskController {
    package struct Submission {
        package let request: StudioRunRequest
        package let outputFallbackReason: String?
    }

    package struct ValidationError: LocalizedError, Equatable {
        package let message: String
        package var errorDescription: String? { message }
    }

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

    package func runPrompt(
        inventory: [StudioModelInventoryRow],
        prepareOutput: (CommandDraft) -> StudioOutputLocation.Preparation = { StudioOutputLocation.preparingDestination(of: $0) }
    ) throws -> Submission? {
        guard let mode = activatedMode else { return nil }
        switch StudioCommandAdapter.capabilityRequirement(for: mode, draft: draft) {
        case .unavailable(let message): throw ValidationError(message: message)
        case .managedModel(let modelID):
            if let message = controller.modelCapabilitiesByID[modelID]?.unavailableMessage {
                throw ValidationError(message: message)
            }
        case nil: break
        }
        let readiness = controller.readinessByMode[mode] ?? .unknown("Readiness has not been checked yet.")
        if readiness.blocksRun { throw ValidationError(message: readiness.message) }
        if mode.isConversational {
            return try sendConversationTurn(inventory: inventory).map { Submission(request: $0, outputFallbackReason: nil) }
        }
        let request = try preparedRequest(mode: mode, draft: draft)
        let prepared = prepareOutput(request.draft)
        let effective = Self.replacingDestination(of: request, with: prepared.draft)
        submitLibraryRequest(effective)
        return Submission(request: effective, outputFallbackReason: prepared.fallbackReason)
    }

    /// Specialist forms use the same stored Command override as their Command panel.
    package func runTask(_ base: StudioRunRequest, task: StudioTask) throws -> StudioRunRequest {
        let request = sessions.resolving(base)
        try validate(request)
        sessions.set(Optional(request.id), for: task.rawValue + ".requestID")
        submitLibraryRequest(request)
        return request
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

    private static func replacingDestination(of request: StudioRunRequest, with draft: CommandDraft) -> StudioRunRequest {
        guard draft != request.draft else { return request }
        return StudioRunRequest(id: request.id, mode: request.mode, templateID: request.templateID,
            template: request.template, draft: draft, createdAt: request.createdAt,
            conversationID: request.conversationID,
            execution: request.execution?.replacing(request.templateID.capability?.output.flag ?? "--output", with: draft.outputPath),
            parentID: request.parentID)
    }

    /// Selection determines Stop, even when another task submitted a newer job.
    package func currentJob(for task: StudioTask) -> Job? {
        if task.mode?.isConversational == true {
            guard let id = activeConversationID else { return nil }
            return controller.jobs.all.first { $0.state.isActive && $0.request.conversationID == id }
        }
        let remembered = sessions.value(for: task.rawValue + ".requestID", default: Optional<UUID>.none)
        if let remembered, let job = controller.jobs.job(requestID: remembered), job.state.isActive { return job }
        return controller.jobs.all.last { $0.state.isActive && $0.request.templateID?.studioTask == task }
    }

    package func stop(task: StudioTask) {
        if let job = currentJob(for: task) { controller.jobs.cancel(job.id) }
    }
}
