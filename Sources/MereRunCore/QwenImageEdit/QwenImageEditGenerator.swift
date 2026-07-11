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
        case inputImageNotFound(URL)
        case inputImageDecodeFailed(URL)
        case tokenizerMissing(URL)
        case invalidOutputDirectory(URL)
        case modelLoadFailed(String)
        case unsupportedPlatform

        public var errorDescription: String? {
            switch self {
            case .inputImageRequired:
                return "Qwen-Image-Edit requires an input image for editing."
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
            case .unsupportedPlatform:
                return "Image editing is not supported on this platform."
            }
        }
    }

    struct LoadedModel {
        let modelSpec: String
        let rootURL: URL
        let resources: QwenImageEditResources
        let configs: QwenImageEditModelConfigs
        let tokenizer: Qwen25VLTokenizer
        let quantConfig: QuantizationConfig?
        var encoder: Qwen25VLEncoder?
        var transformer: MMDiT?
        var vae: QwenImageEditVAE?

        var isPreQuantized: Bool { quantConfig != nil }
    }

    var loaded: LoadedModel?

    public init() {}

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        guard let inputImageURL = request.inputImage else {
            throw GeneratorError.inputImageRequired
        }

        try ensureOutputDirectory(request.outputURL)

        let modelSpec = request.model ?? QwenImageEditRepository.id
        var model = try await loadBaseModelIfNeeded(modelSpec: modelSpec, progressHandler: progressHandler)
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

        let (appearanceLatents, inputImageTensor) = try await encodeInputImage(
            url: inputImageURL,
            vae: vae,
            targetWidth: inferenceConfig.width,
            targetHeight: inferenceConfig.height
        )

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: encodingTotalSteps))
        MLX.eval(appearanceLatents)
        Memory.clearCache()

        let semanticEmbeds = try encodeSemanticEmbeddingsForRequest(
            request: request,
            inputImageTensor: inputImageTensor,
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

        var latents = QwenImageEditLatentCreator.blendWithNoise(
            sourceLatents: appearanceLatents,
            noise: noise,
            sigma: scheduler.sigma(at: 0)
        )

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

            let timestepBatch1 = scheduler.timestep(at: stepIndex).expandedDimensions(axis: 0)

            let finalNoisePred: MLXArray
            if inferenceConfig.guidanceScale > 1.0 {
                if useBatchedCFG {
                    let predictions = transformer.forwardEdit(
                        latents: QwenImageEditCFGExecution.duplicateBatch(latents),
                        timestep: QwenImageEditCFGExecution.duplicateBatch(timestepBatch1),
                        semanticEmbeds: semanticEmbeds,
                        appearanceLatents: QwenImageEditCFGExecution.duplicateBatch(appearanceLatents)
                    )
                    finalNoisePred = QwenImageEditCFGExecution.combinePredictions(
                        predictions,
                        guidanceScale: inferenceConfig.guidanceScale
                    )
                } else {
                    let uncondEmbeds = semanticEmbeds[0..<1, 0..., 0...]
                    let condEmbeds = semanticEmbeds[1..<2, 0..., 0...]

                    let noisePredUncond = transformer.forwardEdit(
                        latents: latents,
                        timestep: timestepBatch1,
                        semanticEmbeds: uncondEmbeds,
                        appearanceLatents: appearanceLatents
                    )
                    MLX.eval(noisePredUncond)
                    Memory.clearCache()

                    let noisePredCond = transformer.forwardEdit(
                        latents: latents,
                        timestep: timestepBatch1,
                        semanticEmbeds: condEmbeds,
                        appearanceLatents: appearanceLatents
                    )

                    finalNoisePred = noisePredUncond
                        + (noisePredCond - noisePredUncond) * MLXArray(inferenceConfig.guidanceScale)
                }
            } else {
                finalNoisePred = transformer.forwardEdit(
                    latents: latents,
                    timestep: timestepBatch1,
                    semanticEmbeds: semanticEmbeds,
                    appearanceLatents: appearanceLatents
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
        let decoded = decodeLatents(
            latents,
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
