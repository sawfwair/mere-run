import Foundation
import MLX

public struct Gemma4TextLoRATrainingPipelineRequest: Sendable {
    public let modelId: String
    public let modelPath: String?
    public let examples: [TextSFTExample]
    public let evaluationExamples: [TextSFTExample]
    public let outputURL: URL
    public let trainingConfig: TextLoRATrainingConfig
    public let maxSequenceLength: Int
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
        self.rank = rank
        self.alpha = alpha
        self.targetSuffixes = targetSuffixes
        self.metadata = metadata
    }
}

public enum Gemma4TextLoRATrainingPipeline {
    public static func train(
        _ request: Gemma4TextLoRATrainingPipelineRequest,
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
        _ request: Gemma4TextLoRATrainingPipelineRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)?
    ) async throws -> TextLoRATrainingReport {
        let loaded = try await Gemma4TextModelLoader.load(
            modelId: request.modelId,
            modelPath: request.modelPath,
            maxContextLength: request.maxSequenceLength,
            progressHandler: progressHandler
        )

        progressHandler?(ChatProgress(stage: .encoding, message: "Tokenizing SFT examples"))
        let tokenized = try Gemma4TextSFTTokenizer.tokenize(
            request.examples,
            tokenizerAndTemplate: loaded.tokenizerAndTemplate,
            maxSequenceLength: request.maxSequenceLength
        )
        let tokenizedEvaluation = try Gemma4TextSFTTokenizer.tokenize(
            request.evaluationExamples,
            tokenizerAndTemplate: loaded.tokenizerAndTemplate,
            maxSequenceLength: request.maxSequenceLength
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Injecting Gemma4 LoRA layers"))
        let loraLayers = try Gemma4TextLoRAInjector.inject(
            into: loaded.model,
            rank: request.rank,
            alpha: request.alpha,
            targetSuffixes: request.targetSuffixes
        )

        var metadata = request.metadata
        metadata["base_model"] = request.modelId
        metadata["model_root"] = loaded.rootURL.path
        metadata["format"] = "mererun.gemma4.text-lora"
        metadata["max_sequence_length"] = String(request.maxSequenceLength)

        progressHandler?(ChatProgress(stage: .generating, message: "Training Gemma4 LoRA adapter"))
        return try TextLoRATrainer.train(
            model: loaded.model,
            loraLayers: loraLayers,
            examples: tokenized,
            evaluationExamples: tokenizedEvaluation,
            config: request.trainingConfig,
            outputURL: request.outputURL,
            metadata: metadata,
            progressHandler: trainingProgressHandler,
            gatheredForward: { model, inputIds, targetPositions in
                model.trainingLogits(inputIds: inputIds, flatTargetPositions: targetPositions)
            }
        ) { model, inputIds in
            model(inputIds)
        }
    }
}
