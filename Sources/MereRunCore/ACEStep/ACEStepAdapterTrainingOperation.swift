import Foundation

/// Owns execution and terminal training events. Callers own presentation and
/// machine admission, as they do for the other native training operations.
public enum ACEStepAdapterTrainingOperation {
    public typealias Trainer = @Sendable (
        ACEStepAdapterTrainingPlan,
        (@Sendable (ACEStepAdapterTrainingProgress) -> Void)?
    ) async throws -> ACEStepAdapterTrainingReport

    public static func execute(
        _ plan: ACEStepAdapterTrainingPlan,
        progress: (@Sendable (ACEStepAdapterTrainingProgress) -> Void)? = nil,
        trainer: Trainer = train
    ) async throws -> ACEStepAdapterTrainingReport {
        try Task.checkCancellation()
        let configuration = plan.options.configuration
        let logger = try LoRATrainingEventLogger(baseOutputURL: plan.outputURL)
        try logger.record(
            type: "run_started", stage: "training",
            message: "ACE-Step \(configuration.kind.rawValue) training started.",
            step: 0, totalSteps: configuration.trainingSteps, fraction: 0,
            path: plan.outputURL.path,
            metadata: ["dataset": plan.manifestURL.path, "model": plan.options.model]
        )
        do {
            let report = try await trainer(plan) { update in
                try? logger.record(
                    type: "progress", stage: "training", step: update.step,
                    totalSteps: update.totalSteps, loss: update.loss,
                    fraction: Float(update.step) / Float(max(update.totalSteps, 1)),
                    path: plan.outputURL.path
                )
                progress?(update)
            }
            try Task.checkCancellation()
            try logger.record(
                type: "run_finished", stage: "finished",
                message: "ACE-Step adapter training finished.",
                step: configuration.trainingSteps, totalSteps: configuration.trainingSteps,
                loss: report.finalLoss, fraction: 1, path: plan.outputURL.path,
                metadata: ["sha256": report.outputSHA256]
            )
            return report
        } catch {
            let cancelled = error is CancellationError || Task.isCancelled
            try? logger.record(
                type: "run_failed", stage: "failed",
                message: cancelled ? "ACE-Step adapter training cancelled." : error.localizedDescription,
                path: plan.outputURL.path
            )
            if cancelled { throw CancellationError() }
            throw error
        }
    }
}
