import Foundation

public enum ImageLoRATrainingProgress: Sendable {
    case krea(Krea2LoRATrainingProgress)
    case klein(Flux2KleinLoRATrainingProgress)
    case sampleSaved(step: Int, url: URL, checkpoint: URL)
    case sampleFailed(step: Int, message: String, checkpoint: URL)
}

public enum ImageLoRATrainingOutcome: Sendable, Equatable {
    case saved(URL)
    case benchmark
}

/// Executes a prepared family-specific plan. The caller owns admission and any
/// dashboard lifecycle; the native trainer owns optimizer state and artifacts.
public enum ImageLoRATrainingOperation {
    public typealias Trainer = @Sendable (
        ImageLoRATrainingPlan, (@Sendable (ImageLoRATrainingProgress) -> Void)?
    ) async throws -> Void

    public static func execute(
        _ plan: ImageLoRATrainingPlan,
        progress: (@Sendable (ImageLoRATrainingProgress) -> Void)? = nil,
        trainer: Trainer = train
    ) async throws -> ImageLoRATrainingOutcome {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: plan.outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        do {
            try await trainer(plan, progress)
            try Task.checkCancellation()
        } catch {
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            throw error
        }
        return plan.isBenchmark ? .benchmark : .saved(plan.outputURL)
    }
}
