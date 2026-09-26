import Foundation

extension StudioPromptTaskController {
    /// Whether the open prompt task's composer offers "Run variations": its draft's command
    /// takes a seed for the model it runs.
    package var offersVariations: Bool {
        guard let mode = activatedMode else { return false }
        return StudioVariations.applies(mode: mode, draft: draft, source: controller.scopeSource)
    }

    /// The composer's "Run variations": the draft once per seed through the same gates, Command
    /// edits, and validation as Run, then submitted through the task runner as one group. The
    /// draft itself keeps the seed the user chose.
    @discardableResult
    package func runPromptVariations(seeds: [String]) throws -> [StudioRunRequest] {
        guard let mode = activatedMode, !mode.isConversational else { return [] }
        try ensureRunnable(mode: mode, draft: draft)
        let requests = try seeds.map { seed in
            var seeded = draft
            seeded.seed = seed
            return StudioVariations.seeded(try preparedRequest(mode: mode, draft: seeded), seed: seed)
        }
        return runner.submitVariations(requests, task: mode.task)
    }
}
