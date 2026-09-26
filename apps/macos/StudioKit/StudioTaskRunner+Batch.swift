import Foundation

// A batch runs through the same runner as one file: readiness is checked once, every file is
// checked on its own before anything is submitted, then each file is prepared and submitted in
// order — named by `StudioOutputLocation`, recorded in the Library, queued in the inference lane —
// and the rows are filed under one `batchGroup`, which is what stopping and following it read.

extension StudioTaskRunner {
    /// Checks a task draft's batch: the readiness gate once (it throws, as a single Run's does),
    /// then each file — on disk, of a kind the slot takes, and a command the contract accepts —
    /// without creating a folder or a row. Nil when the draft holds no batch.
    package func reviewBatch(_ draft: StudioTaskDraft, task: StudioTask) throws -> StudioBatchReview? {
        guard let slot = StudioTaskSchema.primarySlot(for: draft.templateID), slot.isBatched(in: draft) else { return nil }
        let readiness = controller.readiness(for: task)
        if readiness.blocksRun { throw StudioValidationError(message: readiness.message(titles: controller.modelStore.titles)) }
        let source = controller.scopeSource
        return StudioBatchReview(files: slot.runPaths(in: draft).map { path in
            StudioBatchFileCheck(
                path: path,
                problem: StudioInputBatch.fileProblem(path, slot: slot)
                    ?? Self.validationMessage(for: draft.running(path, in: slot), sessions: controller.taskSessions, source: source)
            )
        })
    }

    /// Runs `paths` of a task draft's batch, one run per file in order, the way `run(_:task:)`
    /// prepares a single one. Readiness and the files were checked by `reviewBatch`.
    @discardableResult
    package func runBatch(_ draft: StudioTaskDraft, task: StudioTask, paths: [String]) -> StudioBatchSubmission {
        let source = controller.scopeSource
        let slot = StudioTaskSchema.primarySlot(for: draft.templateID)
        return submitBatch(paths, task: task) { path in
            let single = slot.map { draft.running(path, in: $0) } ?? draft
            return try Self.prepare(draft: single, sessions: controller.taskSessions, source: source)
        }
    }

    /// Prepares and submits each file in order, then files the rows under one new group. A file
    /// whose preparation throws is reported and the rest still run.
    package func submitBatch(
        _ paths: [String],
        task: StudioTask,
        prepare: (String) throws -> (request: StudioRunRequest, fallbackReason: String?)
    ) -> StudioBatchSubmission {
        var requests: [StudioRunRequest] = []
        var failures: [StudioBatchFileCheck] = []
        var fallbackReason: String?
        for path in paths {
            do {
                let prepared = try prepare(path)
                fallbackReason = fallbackReason ?? prepared.fallbackReason
                submit(prepared.request, task: task)
                requests.append(prepared.request)
            } catch {
                failures.append(StudioBatchFileCheck(path: path, problem: error.localizedDescription))
            }
        }
        let group = UUID()
        library.assignBatchGroup(group, to: requests.map(\.id))
        if let fallbackReason { controller.noteOutputFallback(fallbackReason) }
        return StudioBatchSubmission(group: group, requests: requests, failures: failures)
    }

    /// Why one run of a task draft cannot start, as `prepare(draft:)` would refuse it, without
    /// creating its folder: the destination named, the command resolved, the contract checked.
    static func validationMessage(for draft: StudioTaskDraft, sessions: StudioTaskSessions, source: StudioScopeSource) -> String? {
        let named = StudioOutputLocation.destination(for: launching(draft, source: source), source: source)
        guard let base = named.request(source: source) else { return "This command can't run from Studio." }
        let resolved = sessions.resolving(base, source: source)
        return resolved.template.validationMessage(for: resolved.draft, execution: resolved.execution, source: source)
    }

    // MARK: Following and stopping

    /// Every batch in the Library, newest first, each attributed to the task it was run from.
    package func batches() -> [StudioBatchProgress] {
        let sessions = controller.taskSessions
        return StudioBatchProgress.all(in: library.items) { item in
            sessions.submittingTask(of: item.id) ?? item.templateID?.studioTask
        }
    }

    /// The batches with runs still running or waiting.
    package func activeBatches() -> [StudioBatchProgress] {
        batches().filter(\.isActive)
    }

    /// The newest batch run from `task` that still has work in flight.
    package func activeBatch(for task: StudioTask) -> StudioBatchProgress? {
        activeBatches().first { $0.task == task }
    }

    /// Stops every run of a batch still in flight: the waiting ones leave the queue first, so the
    /// lane never starts the next file while the running one stops, then the running ones are
    /// terminated. A row submitted but not launched yet is marked cancelled. Finished runs keep
    /// their results.
    package func stopBatch(_ group: UUID) {
        let rows = library.items.filter { $0.batchGroup == group && ($0.status == .running || $0.status == .queued) }
        let jobs = rows.compactMap { controller.jobs.job(requestID: $0.id) }.filter(\.state.isActive)
        for job in jobs.filter(\.state.isQueued) + jobs.filter({ !$0.state.isQueued }) {
            controller.jobs.cancel(job.id)
        }
        for row in rows where isAwaitingLaunch(row.id) {
            library.setStatus(.cancelled, id: row.id)
        }
    }
}

extension StudioPromptTaskController {
    /// The prompt draft's batch checked the way `reviewBatch` checks a task draft's: the model
    /// and readiness gates once, then each file as its own request. Nil when there is no batch.
    package func reviewPromptBatch() throws -> StudioBatchReview? {
        guard let mode = activatedMode, let slot = batchSlot(for: mode) else { return nil }
        try ensureRunnable(mode: mode, draft: draft)
        return StudioBatchReview(files: slot.runPaths(in: draft).map { path in
            let problem: String?
            if let fileProblem = StudioInputBatch.fileProblem(path, slot: slot) {
                problem = fileProblem
            } else {
                do {
                    _ = try preparedRequest(mode: mode, draft: draft.running(path, in: slot))
                    problem = nil
                } catch {
                    problem = error.localizedDescription
                }
            }
            return StudioBatchFileCheck(path: path, problem: problem)
        })
    }

    /// Runs `paths` of the prompt draft's batch through the task runner, one run per file.
    @discardableResult
    package func runPromptBatch(paths: [String]) -> StudioBatchSubmission? {
        guard let mode = activatedMode, let slot = batchSlot(for: mode) else { return nil }
        let draft = draft
        let source = controller.scopeSource
        return runner.submitBatch(paths, task: mode.task) { [sessions] path in
            let base = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft.running(path, in: slot), source: source)
            return try StudioTaskRunner.prepare(base, sessions: sessions, source: source)
        }
    }

    private func batchSlot(for mode: StudioMode) -> StudioAttachmentSlot? {
        guard !mode.isConversational,
              let slot = mode.attachmentSlots(for: draft, source: controller.scopeSource).first(where: \.batches),
              slot.isBatched(in: draft) else { return nil }
        return slot
    }
}
