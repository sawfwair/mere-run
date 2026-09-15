import Foundation
import MLX

/// Effective generation inputs. Lazy tensors stay within the owning music runtime.
public struct ACEStepGenerationPlan {
    public let request: ACEStepSessionRequest
    public let quality: ACEStepQualityPreset
    public let task: ACEStepTask
    public let candidateCount: Int
    public let conditioningMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata
}

/// Owns preparation, language-model planning, and ranked generation for both entry points.
public final class ACEStepGenerationOperation {
    public let session: ACEStepGenerationSession
    private let variant: ACEStepCheckpointVariant
    private let languageModelAvailable: Bool

    public init(resources: ACEStepModelResources, variant: ACEStepCheckpointVariant) throws {
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources, vaeResources: resources.vaeResources,
            lmResources: resources.lmResources, textEncoderResources: resources.textEncoderResources
        )
        self.session = ACEStepGenerationSession(pipeline: pipeline)
        self.variant = variant
        self.languageModelAvailable = resources.lmResources != nil
    }

    public func generate(_ plan: ACEStepGenerationPlan) throws -> ACEStepRankedGeneration {
        try Task.checkCancellation()
        return try session.generateBest(plan.request, candidateCount: plan.candidateCount)
    }

    public func prepare(_ options: ACEStepGenerationOptions) throws -> ACEStepGenerationPlan {
        try ACEStepGenerationPreparation.prepare(
            options, variant: variant, languageModelAvailable: languageModelAvailable, planner: session.pipeline
        )
    }
}
