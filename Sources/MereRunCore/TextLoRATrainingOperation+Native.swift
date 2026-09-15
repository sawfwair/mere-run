import Foundation

extension TextLoRATrainingOperation {
    /// Dispatches to the existing native pipelines without changing model math,
    /// checkpoint loading or machine admission.
    public static func train(
        _ plan: TextLoRATrainingPlan,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)?
    ) async throws -> TextLoRATrainingReport {
        let options = plan.options
        let config = TextLoRATrainingConfig(
            trainingSteps: options.trainingSteps, batchSize: options.batchSize,
            learningRate: options.learningRate, seed: options.seed,
            resumeFrom: options.resumeFrom.map { URL(fileURLWithPath: $0).standardizedFileURL },
            resumeStep: options.resumeStep
        )
        var metadata = ["adapter_name": options.adapterName, "dataset_fingerprint": plan.dataset.summary.fingerprint]
        if let imageFingerprint = plan.dataset.summary.imageFingerprint {
            metadata["dataset_image_fingerprint"] = imageFingerprint
        }
        switch plan.family {
        case .gemma4:
            return try await Gemma4TextLoRATrainingPipeline.train(
                Gemma4TextLoRATrainingPipelineRequest(
                    modelId: options.model,
                    modelPath: options.modelPath,
                    examples: plan.dataset.examples,
                    evaluationExamples: plan.evaluationDataset?.examples ?? [],
                    outputURL: plan.outputURL,
                    trainingConfig: config,
                    maxSequenceLength: options.maxSequenceLength,
                    rank: options.rank,
                    alpha: options.alpha ?? Float(options.rank),
                    targetSuffixes: options.resolvedTargetModules(),
                    metadata: metadata
                ),
                progressHandler: progressHandler,
                trainingProgressHandler: trainingProgressHandler
            )
        case .gemma4VLM:
            return try await Gemma4VLMLoRATrainingPipeline.train(
                Gemma4VLMLoRATrainingPipelineRequest(
                    modelId: options.model,
                    modelPath: options.modelPath,
                    examples: plan.dataset.examples,
                    evaluationExamples: plan.evaluationDataset?.examples ?? [],
                    trainingImageDigestsByPath: plan.dataset.imageDigestsByPath,
                    evaluationImageDigestsByPath: plan.evaluationDataset?
                        .imageDigestsByPath ?? [:],
                    outputURL: plan.outputURL,
                    trainingConfig: config,
                    maxSequenceLength: options.maxSequenceLength,
                    rank: options.rank,
                    alpha: options.alpha ?? Float(options.rank),
                    targetSuffixes: options.resolvedTargetModules(),
                    metadata: metadata
                ),
                progressHandler: progressHandler,
                trainingProgressHandler: trainingProgressHandler
            )
        case .lagunaXS:
            return try await LagunaTextLoRATrainingPipeline.train(
                LagunaTextLoRATrainingPipelineRequest(
                    modelId: options.model,
                    modelPath: options.modelPath,
                    examples: plan.dataset.examples,
                    evaluationExamples: plan.evaluationDataset?.examples ?? [],
                    outputURL: plan.outputURL,
                    trainingConfig: config,
                    maxSequenceLength: options.maxSequenceLength,
                    rank: options.rank,
                    alpha: options.alpha ?? Float(options.rank),
                    targetSuffixes: options.resolvedTargetModules(),
                    metadata: metadata
                ),
                progressHandler: progressHandler,
                trainingProgressHandler: trainingProgressHandler
            )
        case .inkling:
            return try await InklingTextLoRATrainingPipeline.train(
                InklingTextLoRATrainingPipelineRequest(
                    modelId: options.model,
                    modelPath: options.modelPath,
                    examples: plan.dataset.examples,
                    evaluationExamples: plan.evaluationDataset?.examples ?? [],
                    outputURL: plan.outputURL,
                    trainingConfig: config,
                    maxSequenceLength: options.maxSequenceLength,
                    reasoningEffort: options.reasoningEffort,
                    rank: options.rank,
                    alpha: options.alpha ?? Float(options.rank),
                    targetSuffixes: options.resolvedTargetModules(),
                    metadata: metadata
                ),
                progressHandler: progressHandler,
                trainingProgressHandler: trainingProgressHandler
            )
        case .lfm2A1B:
            return try await LFM2TextLoRATrainingPipeline.train(
                LFM2TextLoRATrainingPipelineRequest(
                    modelId: options.model,
                    modelPath: options.modelPath,
                    examples: plan.dataset.examples,
                    evaluationExamples: plan.evaluationDataset?.examples ?? [],
                    outputURL: plan.outputURL,
                    trainingConfig: config,
                    maxSequenceLength: options.maxSequenceLength,
                    rank: options.rank,
                    alpha: options.alpha ?? Float(options.rank),
                    targetSuffixes: options.resolvedTargetModules(),
                    metadata: metadata
                ),
                progressHandler: progressHandler,
                trainingProgressHandler: trainingProgressHandler
            )
        }
    }
}
