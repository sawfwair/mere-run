import Foundation
import MLX

public struct LagunaTextLoRATrainingPipelineRequest: Sendable {
    public let modelId: String
    public let modelPath: String?
    public let examples: [TextSFTExample]
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
        self.outputURL = outputURL
        self.trainingConfig = trainingConfig
        self.maxSequenceLength = maxSequenceLength
        self.rank = rank
        self.alpha = alpha
        self.targetSuffixes = targetSuffixes
        self.metadata = metadata
    }
}

public enum LagunaTextLoRATrainingPipeline {
    public static func train(
        _ request: LagunaTextLoRATrainingPipelineRequest,
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
        _ request: LagunaTextLoRATrainingPipelineRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)?
    ) async throws -> TextLoRATrainingReport {
        guard LagunaResources.managedModelID(for: request.modelId)
            == LagunaResources.xsModelID else {
            throw LagunaError.generationFailed(
                "Native Laguna text LoRA training currently supports Laguna XS 2.1."
            )
        }
        let loaded = try await LagunaTextModelLoader.load(
            modelId: request.modelId,
            modelPath: request.modelPath,
            maxContextLength: request.maxSequenceLength,
            progressHandler: progressHandler
        )
        progressHandler?(ChatProgress(stage: .encoding, message: "Tokenizing Laguna SFT examples"))
        let tokenized = try LagunaTextSFTTokenizer.tokenize(
            request.examples,
            tokenizerAndTemplate: loaded.tokenizerAndTemplate,
            maxSequenceLength: request.maxSequenceLength
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Injecting Laguna LoRA layers"))
        let layers = try LagunaTextLoRAInjector.inject(
            into: loaded.model,
            rank: request.rank,
            alpha: request.alpha,
            targetSuffixes: request.targetSuffixes
        )
        var metadata = request.metadata
        metadata["base_model"] = request.modelId
        metadata["model_root"] = loaded.rootURL.path
        metadata["format"] = TextLoRATrainingManifest.lagunaFormat
        metadata["max_sequence_length"] = String(request.maxSequenceLength)

        progressHandler?(ChatProgress(stage: .generating, message: "Training Laguna LoRA adapter"))
        return try TextLoRATrainer.train(
            model: loaded.model,
            loraLayers: layers,
            examples: tokenized,
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
            model(inputIDs)
        }
    }
}
