import Foundation
import MLX

public struct Gemma4VLMLoRATrainingPipelineRequest: Sendable {
    public let modelId: String
    public let modelPath: String?
    public let examples: [TextSFTExample]
    public let evaluationExamples: [TextSFTExample]
    public let trainingImageDigestsByPath: [String: String]
    public let evaluationImageDigestsByPath: [String: String]
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
        trainingImageDigestsByPath: [String: String] = [:],
        evaluationImageDigestsByPath: [String: String] = [:],
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
        self.trainingImageDigestsByPath = trainingImageDigestsByPath
        self.evaluationImageDigestsByPath = evaluationImageDigestsByPath
        self.outputURL = outputURL
        self.trainingConfig = trainingConfig
        self.maxSequenceLength = maxSequenceLength
        self.rank = rank
        self.alpha = alpha
        self.targetSuffixes = targetSuffixes
        self.metadata = metadata
    }
}

public enum Gemma4VLMLoRATrainingPipeline {
    public static func train(
        _ request: Gemma4VLMLoRATrainingPipelineRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)? = nil
    ) async throws -> TextLoRATrainingReport {
        guard request.trainingConfig.batchSize == 1 else {
            throw Gemma4VLMLoRATrainingError.batchSizeMustBeOne(
                request.trainingConfig.batchSize
            )
        }
        return try await Stream.withNewDefaultStream {
            try await trainOnCurrentStream(
                request,
                progressHandler: progressHandler,
                trainingProgressHandler: trainingProgressHandler
            )
        }
    }

    private static func trainOnCurrentStream(
        _ request: Gemma4VLMLoRATrainingPipelineRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        trainingProgressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)?
    ) async throws -> TextLoRATrainingReport {
        let loaded = try await Gemma4UnifiedModelLoader.load(
            modelId: request.modelId,
            modelPath: request.modelPath,
            maxContextLength: request.maxSequenceLength,
            progressHandler: progressHandler
        )
        guard let visionConfig = loaded.config.visionConfig,
              let imageTokenId = loaded.config.imageTokenId else {
            throw Gemma4Error.unsupportedConfiguration(
                "Gemma4 VLM LoRA training requires vision and image-token configuration."
            )
        }

        progressHandler?(ChatProgress(
            stage: .encoding,
            message: "Preparing image-conditioned SFT examples"
        ))
        let tokenized = try Gemma4VLMSFTTokenizer.tokenize(
            request.examples,
            tokenizerAndTemplate: loaded.tokenizerAndTemplate,
            config: loaded.config,
            maxSequenceLength: request.maxSequenceLength,
            expectedImageDigestsByPath: request.trainingImageDigestsByPath
        )
        let tokenizedEvaluation = try Gemma4VLMSFTTokenizer.tokenize(
            request.evaluationExamples,
            tokenizerAndTemplate: loaded.tokenizerAndTemplate,
            config: loaded.config,
            maxSequenceLength: request.maxSequenceLength,
            expectedImageDigestsByPath: request.evaluationImageDigestsByPath
        )

        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Injecting Gemma4 VLM language LoRA layers"
        ))
        let loraLayers = try Gemma4TextLoRAInjector.inject(
            into: loaded.model,
            rank: request.rank,
            alpha: request.alpha,
            targetSuffixes: request.targetSuffixes
        )

        var metadata = request.metadata
        metadata["base_model"] = request.modelId
        metadata["model_root"] = loaded.rootURL.path
        metadata["format"] = TextLoRATrainingManifest.gemma4VLMFormat
        metadata["max_sequence_length"] = String(request.maxSequenceLength)
        metadata["modality"] = "image"
        metadata["training_scope"] = "language_attention"

        progressHandler?(ChatProgress(
            stage: .generating,
            message: "Training image-conditioned Gemma4 LoRA adapter"
        ))
        return try TextLoRATrainer.train(
            model: loaded.model,
            loraLayers: loraLayers,
            examples: tokenized,
            evaluationExamples: tokenizedEvaluation,
            config: request.trainingConfig,
            outputURL: request.outputURL,
            metadata: metadata,
            progressHandler: trainingProgressHandler,
            multimodalBatchBuilder: { examples in
                try makeTrainingBatch(
                    examples,
                    visionConfig: visionConfig,
                    imageTokenId: imageTokenId
                )
            },
            multimodalGatheredForward: { model, inputs in
                model.trainingLogits(
                    inputIds: inputs[0],
                    pixelValues: inputs[1],
                    imagePositionIds: inputs[2],
                    softTokenCounts: inputs[3],
                    mmTokenTypeIds: inputs[4],
                    flatTargetPositions: inputs[5]
                )
            }
        ) { model, inputIds in
            model.forward(inputIds: inputIds)
        }
    }

    static func makeTrainingBatch(
        _ examples: [TextSFTTokenizedExample],
        visionConfig: Gemma4UnifiedVisionConfig,
        imageTokenId: Int
    ) throws -> TextLoRAMultimodalTrainingBatch {
        guard examples.count == 1, let example = examples.first else {
            throw Gemma4VLMLoRATrainingError.batchSizeMustBeOne(examples.count)
        }
        guard let multimodal = example.multimodalInputs else {
            throw Gemma4VLMLoRATrainingError.missingMultimodalInputs
        }
        guard multimodal.imageReferences.count == multimodal.imageSHA256.count else {
            throw Gemma4VLMLoRATrainingError.invalidImageProvenance
        }
        for (reference, expectedDigest) in zip(
            multimodal.imageReferences,
            multimodal.imageSHA256
        ) {
            let digest = try TextSFTDataset.fileDigest(
                URL(fileURLWithPath: reference).standardizedFileURL
            )
            guard digest == expectedDigest else {
                throw Gemma4VLMLoRATrainingError.imageContentChanged
            }
        }
        let expectedImageTokenCount = multimodal.softTokenCounts.reduce(0, +)
        guard example.inputTokenIds.filter({ $0 == imageTokenId }).count
            == expectedImageTokenCount else {
            throw Gemma4VLMLoRATrainingError.imageTokenCountMismatch
        }
        guard multimodal.mmTokenTypeIds.count == example.inputTokenIds.count,
              multimodal.mmTokenTypeShape == [1, example.inputTokenIds.count] else {
            throw Gemma4VLMLoRATrainingError.invalidMultimodalTokenTypes
        }
        let imageStorage = try Gemma4UnifiedImageProcessor.makeStorage(
            imageReferences: multimodal.imageReferences,
            visionConfig: visionConfig
        )
        guard imageStorage.softTokenCounts == multimodal.softTokenCounts else {
            throw Gemma4VLMLoRATrainingError.imageShapeChanged
        }
        let gathered = try TextSFTTrainingBatchBuilder.makeGatheredBatch(examples)
        return TextLoRAMultimodalTrainingBatch(
            modelInputs: [
                gathered.inputIds,
                MLXArray(imageStorage.pixelValues, imageStorage.pixelShape),
                MLXArray(imageStorage.imagePositionIds, imageStorage.imagePositionShape),
                MLXArray(multimodal.softTokenCounts.map(Int32.init)),
                MLXArray(multimodal.mmTokenTypeIds, multimodal.mmTokenTypeShape),
                gathered.targetPositions,
            ],
            targetLabels: gathered.targetLabels
        )
    }
}

public enum Gemma4VLMLoRATrainingError: Error, LocalizedError, Sendable {
    case batchSizeMustBeOne(Int)
    case missingMultimodalInputs
    case imageShapeChanged
    case invalidImageProvenance
    case imageContentChanged
    case imageTokenCountMismatch
    case invalidMultimodalTokenTypes

    public var errorDescription: String? {
        switch self {
        case .batchSizeMustBeOne(let value):
            return "Gemma4 VLM LoRA training currently requires --batch-size 1 (got \(value))."
        case .missingMultimodalInputs:
            return "Gemma4 VLM LoRA training received an example without prepared image inputs."
        case .imageShapeChanged:
            return "A VLM SFT image changed dimensions after dataset validation."
        case .invalidImageProvenance:
            return "Gemma4 VLM LoRA training received incomplete image provenance."
        case .imageContentChanged:
            return "A VLM SFT image changed contents after dataset validation."
        case .imageTokenCountMismatch:
            return "Gemma4 VLM LoRA training received mismatched image tokens."
        case .invalidMultimodalTokenTypes:
            return "Gemma4 VLM LoRA training received invalid multimodal token types."
        }
    }
}
