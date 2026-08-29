import Foundation
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

/// Owns the public Qwen image-edit entrypoint and high-level orchestration.
/// Model loading and semantic/image encoding live in companion files so the
/// edit pipeline reads in the same order it executes.
public actor QwenImageEditGenerator: ImageGenerator {
    public enum GeneratorError: LocalizedError {
        case inputImageRequired
        case tooManyReferenceImages(Int)
        case inputImageNotFound(URL)
        case inputImageDecodeFailed(URL)
        case tokenizerMissing(URL)
        case invalidOutputDirectory(URL)
        case modelLoadFailed(String)
        case checkpointCoverage(component: String, missing: [String], unexpected: [String])
        case lightningRequiresFourSteps(Int)
        case lightningDoesNotSupportCFG(Double)
        case unsupportedPlatform

        public var errorDescription: String? {
            switch self {
            case .inputImageRequired:
                return "Qwen-Image-Edit requires at least one input or reference image."
            case .tooManyReferenceImages(let count):
                return "Qwen-Image-Edit supports up to 3 ordered images; received \(count)."
            case .inputImageNotFound(let url):
                return "Input image not found: \(url.path)"
            case .inputImageDecodeFailed(let url):
                return "Failed to decode input image: \(url.lastPathComponent)"
            case .tokenizerMissing(let url):
                return "Tokenizer folder missing: \(url.path)"
            case .invalidOutputDirectory(let url):
                return "Output directory does not exist: \(url.deletingLastPathComponent().path)"
            case .modelLoadFailed(let message):
                return "Failed to load model: \(message)"
            case .checkpointCoverage(let component, let missing, let unexpected):
                let missingSummary = missing.prefix(8).joined(separator: ", ")
                let unexpectedSummary = unexpected.prefix(8).joined(separator: ", ")
                return "\(component) checkpoint coverage failed: missing [\(missingSummary)]; "
                    + "unexpected [\(unexpectedSummary)]."
            case .lightningRequiresFourSteps(let steps):
                return "Qwen Image Edit 2511 Lightning requires exactly 4 steps; received \(steps)."
            case .lightningDoesNotSupportCFG(let scale):
                return "Qwen Image Edit 2511 Lightning requires CFG disabled (scale 1.0); received \(scale)."
            case .unsupportedPlatform:
                return "Image editing is not supported on this platform."
            }
        }
    }

    struct LoadedModel {
        let modelSpec: String
        let runtimeModelID: String?
        let rootURL: URL
        let resources: QwenImageEditResources
        let configs: QwenImageEditModelConfigs
        let tokenizer: Qwen25VLTokenizer
        let quantConfig: QuantizationConfig?
        var encoder: Qwen25VLEncoder?
        var transformer: MMDiT?
        var vae: QwenImageEditVAE?

        var isPreQuantized: Bool { quantConfig != nil }
        var usesPinned2511BF16: Bool {
            runtimeModelID == QwenImageEditRepository.model2511Id
                || runtimeModelID == QwenImageEditRepository.lightning2511Id
        }
        var isLightning2511: Bool {
            runtimeModelID == QwenImageEditRepository.lightning2511Id
        }
    }

    var loaded: LoadedModel?

    public init() {}

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        let referenceURLs = [request.inputImage].compactMap { $0 } + request.referenceImages
        guard !referenceURLs.isEmpty else {
            throw GeneratorError.inputImageRequired
        }
        guard referenceURLs.count <= QwenImageEditConditioningPlan.maximumReferenceCount else {
            throw GeneratorError.tooManyReferenceImages(referenceURLs.count)
        }

        try ensureOutputDirectory(request.outputURL)

        let modelSpec = request.model ?? QwenImageEditRepository.id
        var model = try await loadBaseModelIfNeeded(modelSpec: modelSpec, progressHandler: progressHandler)
        if model.isLightning2511 {
            guard request.steps == 4 else {
                throw GeneratorError.lightningRequiresFourSteps(request.steps)
            }
            guard request.guidanceScale == 1 else {
                throw GeneratorError.lightningDoesNotSupportCFG(request.guidanceScale)
            }
        }
        try ensureVAELoaded(model: &model, progressHandler: progressHandler)
        try ensureEncoderLoaded(model: &model, progressHandler: progressHandler)
        loaded = model

        let inferenceConfig = QwenImageEditInferenceConfig(
            width: request.width,
            height: request.height,
            numInferenceSteps: request.steps,
            guidanceScale: Float(request.guidanceScale),
            seed: request.seed
        )
        let scheduler = FlowMatchEulerScheduler(
            config: model.configs.scheduler,
            numInferenceSteps: inferenceConfig.numInferenceSteps,
            imageSeqLen: inferenceConfig.imageSeqLen
        )

        let usesCFG = inferenceConfig.guidanceScale > 1.0
        let encodingTotalSteps = usesCFG ? 4 : 3
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: encodingTotalSteps))

        guard let vae = model.vae else {
            throw GeneratorError.modelLoadFailed("VAE was not loaded.")
        }

        let encodedReferences = try await encodeReferenceImages(
            urls: referenceURLs,
            vae: vae,
            outputWidth: inferenceConfig.width,
            outputHeight: inferenceConfig.height
        )

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: encodingTotalSteps))
        MLX.eval(encodedReferences.appearanceLatents)
        Memory.clearCache()

        let semanticEmbeds = try encodeSemanticEmbeddingsForRequest(
            request: request,
            inputImageTensors: encodedReferences.semanticImages,
            tokenizer: model.tokenizer,
            encoder: model.encoder,
            guidanceScale: inferenceConfig.guidanceScale,
            progressHandler: progressHandler,
            totalSteps: encodingTotalSteps
        )

        Memory.clearCache()
        model.encoder = nil
        loaded = model
        Memory.clearCache()

        try ensureTransformerLoaded(model: &model, progressHandler: progressHandler)
        loaded = model

        let seed = request.seed ?? UInt64.random(in: 0..<UInt64.max)
        let noise = QwenImageEditLatentCreator.createNoise(
            batchSize: 1,
            height: inferenceConfig.height,
            width: inferenceConfig.width,
            latentChannels: model.configs.vae.latentChannels,
            vaeScaleFactor: model.configs.vae.vaeScaleFactor,
            seed: seed
        )

        var latents = QwenImageEditLatentCreator.packLatents(noise)

        guard let transformer = model.transformer else {
            throw GeneratorError.modelLoadFailed("Transformer was not loaded.")
        }
        let machine = MereRunMachineProfile.current
        let useBatchedCFG = usesCFG && QwenImageEditCFGExecution.shouldBatch(
            mode: QwenImageEditCFGExecutionMode.current,
            width: inferenceConfig.width,
            height: inferenceConfig.height,
            physicalMemoryBytes: machine.physicalMemoryBytes,
            activeMemoryBytes: Memory.activeMemory,
            cacheMemoryBytes: Memory.cacheMemory,
            isUnifiedMemory: machine.isAppleSiliconMac
        )
        for stepIndex in 0..<inferenceConfig.numInferenceSteps {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(
                stage: .denoising,
                stepIndex: stepIndex,
                totalSteps: inferenceConfig.numInferenceSteps
            ))

            let timestepBatch1 = scheduler.modelTimestep(at: stepIndex).expandedDimensions(axis: 0)

            let finalNoisePred: MLXArray
            if inferenceConfig.guidanceScale > 1.0 {
                if useBatchedCFG {
                    let predictions = transformer.forwardEdit(
                        latents: QwenImageEditCFGExecution.duplicateBatch(latents),
                        timestep: QwenImageEditCFGExecution.duplicateBatch(timestepBatch1),
                        semanticEmbeds: semanticEmbeds.embeddings,
                        semanticMask: semanticEmbeds.attentionMask,
                        appearanceLatents: encodedReferences.appearanceLatents.map { reference in
                            QwenImageEditCFGExecution.duplicateBatch(reference)
                        },
                        imageShapes: encodedReferences.plan.transformerImageShapes
                    )
                    finalNoisePred = QwenImageEditCFGExecution.combineQwenImagePredictions(
                        predictions,
                        guidanceScale: inferenceConfig.guidanceScale
                    )
                } else {
                    let uncondEmbeds = semanticEmbeds.embeddings[0..<1, 0..., 0...]
                    let condEmbeds = semanticEmbeds.embeddings[1..<2, 0..., 0...]
                    let uncondMask = semanticEmbeds.attentionMask[0..<1, 0...]
                    let condMask = semanticEmbeds.attentionMask[1..<2, 0...]

                    let noisePredUncond = transformer.forwardEdit(
                        latents: latents,
                        timestep: timestepBatch1,
                        semanticEmbeds: uncondEmbeds,
                        semanticMask: uncondMask,
                        appearanceLatents: encodedReferences.appearanceLatents,
                        imageShapes: encodedReferences.plan.transformerImageShapes
                    )
                    MLX.eval(noisePredUncond)
                    Memory.clearCache()

                    let noisePredCond = transformer.forwardEdit(
                        latents: latents,
                        timestep: timestepBatch1,
                        semanticEmbeds: condEmbeds,
                        semanticMask: condMask,
                        appearanceLatents: encodedReferences.appearanceLatents,
                        imageShapes: encodedReferences.plan.transformerImageShapes
                    )

                    finalNoisePred = QwenImageEditCFGExecution.combineQwenImagePredictions(
                        MLX.concatenated([noisePredUncond, noisePredCond], axis: 0),
                        guidanceScale: inferenceConfig.guidanceScale
                    )
                }
            } else {
                finalNoisePred = transformer.forwardEdit(
                    latents: latents,
                    timestep: timestepBatch1,
                    semanticEmbeds: semanticEmbeds.embeddings,
                    semanticMask: semanticEmbeds.attentionMask,
                    appearanceLatents: encodedReferences.appearanceLatents,
                    imageShapes: encodedReferences.plan.transformerImageShapes
                )
            }

            latents = scheduler.step(
                modelOutput: finalNoisePred,
                timestepIndex: stepIndex,
                sample: latents
            )

            MLX.eval(latents)
            Memory.clearCache()
        }

        progressHandler?(GenerationProgress(
            stage: .decoding,
            stepIndex: inferenceConfig.numInferenceSteps,
            totalSteps: inferenceConfig.numInferenceSteps
        ))
        let unpackedLatents = QwenImageEditLatentCreator.unpackLatents(
            latents,
            height: inferenceConfig.latentHeight,
            width: inferenceConfig.latentWidth,
            channels: model.configs.vae.latentChannels
        )
        let decoded = decodeLatents(
            unpackedLatents,
            vae: vae,
            height: inferenceConfig.height,
            width: inferenceConfig.width
        )

        progressHandler?(GenerationProgress(
            stage: .saving,
            stepIndex: inferenceConfig.numInferenceSteps,
            totalSteps: inferenceConfig.numInferenceSteps
        ))
        try QwenImageIO.saveImage(array: decoded, to: request.outputURL)

        Memory.clearCache()
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    func ensureOutputDirectory(_ outputURL: URL) throws {
        let dir = outputURL.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw GeneratorError.invalidOutputDirectory(outputURL)
        }
    }

    public func clearCache() {
        loaded = nil
        Memory.clearCache()
    }
}
