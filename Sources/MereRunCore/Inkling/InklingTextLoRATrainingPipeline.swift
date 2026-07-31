import Foundation
import MLX

public struct InklingTextLoRATrainingPipelineRequest: Sendable {
    public let modelId: String
    public let modelPath: String?
    public let examples: [TextSFTExample]
    public let evaluationExamples: [TextSFTExample]
    public let outputURL: URL
    public let trainingConfig: TextLoRATrainingConfig
    public let maxSequenceLength: Int
    public let reasoningEffort: Double
    public let rank: Int
    public let alpha: Float
    public let targetSuffixes: [String]
    public let metadata: [String: String]

    public init(
        modelId: String,
        modelPath: String? = nil,
        examples: [TextSFTExample],
        evaluationExamples: [TextSFTExample] = [],
        outputURL: URL,
        trainingConfig: TextLoRATrainingConfig,
        maxSequenceLength: Int,
        reasoningEffort: Double = 0.9,
        rank: Int,
        alpha: Float,
        targetSuffixes: [String],
        metadata: [String: String] = [:]
    ) {
        self.modelId = modelId
        self.modelPath = modelPath
        self.examples = examples
        self.evaluationExamples = evaluationExamples
        self.outputURL = outputURL
        self.trainingConfig = trainingConfig
        self.maxSequenceLength = maxSequenceLength
        self.reasoningEffort = reasoningEffort
        self.rank = rank
        self.alpha = alpha
        self.targetSuffixes = targetSuffixes
        self.metadata = metadata
    }
}

public enum InklingTextLoRATrainingPipeline {
    public static func train(
        _ request: InklingTextLoRATrainingPipelineRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)? = nil
    ) async throws -> TextLoRATrainingReport {
        try await Stream.withNewDefaultStream {
            try await trainOnCurrentStream(
                request,
                progressHandler: progressHandler,
                trainingProgressHandler: trainingProgressHandler
            )
        }
    }

    private static func trainOnCurrentStream(
        _ request: InklingTextLoRATrainingPipelineRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)?
    ) async throws -> TextLoRATrainingReport {
        guard InklingResources.handles(modelSpec: request.modelId) else {
            throw InklingError.generationFailed(
                "Native Inkling text LoRA training supports \(InklingResources.modelID)."
            )
        }
        let loaded = try await InklingTextModelLoader.load(
            modelId: request.modelId,
            modelPath: request.modelPath,
            maxContextLength: request.maxSequenceLength,
            progressHandler: progressHandler
        )
        progressHandler?(ChatProgress(stage: .encoding, message: "Tokenizing Inkling SFT examples"))
        let tokenized = try InklingTextSFTTokenizer.tokenize(
            request.examples,
            tokenizerAndTemplate: loaded.tokenizerAndTemplate,
            maxSequenceLength: request.maxSequenceLength,
            reasoningEffort: request.reasoningEffort
        )
        let tokenizedEvaluation = try InklingTextSFTTokenizer.tokenize(
            request.evaluationExamples,
            tokenizerAndTemplate: loaded.tokenizerAndTemplate,
            maxSequenceLength: request.maxSequenceLength,
            reasoningEffort: request.reasoningEffort
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Injecting Inkling LoRA layers"))
        let layers = try InklingTextLoRAInjector.inject(
            into: loaded.model,
            rank: request.rank,
            alpha: request.alpha,
            targetSuffixes: request.targetSuffixes
        )
        var metadata = request.metadata
        metadata["base_model"] = request.modelId
        metadata["model_root"] = loaded.rootURL.path
        metadata["format"] = TextLoRATrainingManifest.inklingFormat
        metadata["max_sequence_length"] = String(request.maxSequenceLength)
        metadata["reasoning_effort"] = String(request.reasoningEffort)

        progressHandler?(ChatProgress(stage: .generating, message: "Training Inkling LoRA adapter"))
        return try TextLoRATrainer.train(
            model: loaded.model,
            loraLayers: layers,
            examples: tokenized,
            evaluationExamples: tokenizedEvaluation,
            config: request.trainingConfig,
            outputURL: request.outputURL,
            metadata: metadata,
            progressHandler: trainingProgressHandler,
            gatheredForward: { model, inputIDs, targetPositions in
                model.trainingLogits(
                    inputIDs: inputIDs,
                    flatTargetPositions: targetPositions
                )
            }
        ) { model, inputIDs in
            model.trainingForward(inputIDs)
        }
    }
}
