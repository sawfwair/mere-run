import Foundation

public struct TextLoRATrainingOutcome: Sendable {
    public let manifest: TextLoRATrainingManifest
    public let report: TextLoRATrainingReport?
}

/// Owns training execution and adapter-manifest publication. Existing optimizer
/// checkpoints remain the source of resumable training state.
public enum TextLoRATrainingOperation {
    public typealias Trainer = @Sendable (
        TextLoRATrainingPlan, (@Sendable (ChatProgress) -> Void)?,
        (@Sendable (TextLoRATrainingProgress) -> Void)?
    ) async throws -> TextLoRATrainingReport

    public static func execute(
        _ plan: TextLoRATrainingPlan,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)? = nil,
        trainer: Trainer = train
    ) async throws -> TextLoRATrainingOutcome {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: plan.outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let report: TextLoRATrainingReport?
        do {
            report = plan.options.dryRun ? nil : try await trainer(plan, progressHandler, trainingProgressHandler)
            try Task.checkCancellation()
        } catch {
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            throw error
        }
        let manifest = plan.options.makeManifest(
            family: plan.family, outputURL: plan.outputURL, datasetSummary: plan.dataset.summary,
            evalPromptCount: plan.evalPromptCount, status: plan.options.dryRun ? "prepared" : "trained"
        )
        try manifest.write(nextTo: plan.outputURL)
        return TextLoRATrainingOutcome(manifest: manifest, report: report)
    }
}
