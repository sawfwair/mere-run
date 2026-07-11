import Foundation
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

/// Owns the public Z-Image Turbo inference entrypoints and runtime state.
/// Model loading, LoRA application, and prompt/latent helpers live in
/// companion files so readers can follow the request flow first.
public actor ZImageTurboGenerator: ImageGenerator, ChatGenerator {
    public enum GeneratorError: LocalizedError {
        case inputImageUnsupportedPlatform
        case inputImageNotFound(URL)
        case inputImageDecodeFailed(URL)
        case tokenizerMissing(URL)
        case invalidOutputDirectory(URL)

        public var errorDescription: String? {
            switch self {
            case .inputImageUnsupportedPlatform:
                return "Image-to-image is unavailable on this platform build."
            case .inputImageNotFound(let url):
                return "Input image not found: \(url.path)"
            case .inputImageDecodeFailed(let url):
                return "Failed to decode input image: \(url.lastPathComponent)"
            case .tokenizerMissing(let url):
                return "Tokenizer folder missing: \(url.path)"
            case .invalidOutputDirectory(let url):
                return "Output directory does not exist: \(url.deletingLastPathComponent().path)"
            }
        }
    }

    struct LoadedModel {
        let modelSpec: String
        let rootURL: URL
        let manifest: MereRunModelManifest?
        let resources: ZImageTurboResources
        let configs: ZImageTurboModelConfigs
        let textEncoderQuantization: ModelWeightsLoader.QuantizationParams?
        let transformerQuantization: ModelWeightsLoader.QuantizationParams?
        let tokenizer: QwenTokenizer
        let textEncoder: QwenTextEncoder
        let transformer: ZImageTransformer2DModel
        let vae: AutoencoderKL
    }

    var loaded: LoadedModel?
    var currentTextLoRA: LoRA?
    var currentTransformerLoRA: LoRA?
    var transformerLoRALayers: [String: TrainableLoRALayer]?
    var transformerLoRARank: Int?
    static let loraDebugEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_LORA_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    public init() {}

    public func unload() {
        loaded = nil
        currentTextLoRA = nil
        currentTransformerLoRA = nil
        transformerLoRALayers = nil
        transformerLoRARank = nil
        clearGPUMemory(synchronize: true)
    }

    func clearGPUMemory(synchronize: Bool = true) {
        if synchronize {
            Stream.gpu.synchronize()
        }
        MLX.eval(MLXArray([]))
        Memory.clearCache()
    }

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        try ensureOutputDirectory(request.outputURL)

        let modelSpec = request.model ?? ZImageTurboRepository.defaultModelSpec
        var model = try await loadModelIfNeeded(modelSpec: modelSpec, progressHandler: progressHandler)
        model = try await applyTransformerLoRAIfNeeded(request.lora, model: model, progressHandler: progressHandler)
        let outputURL = request.outputURL
        let seed = request.seed ?? UInt64.random(in: 0..<UInt64.max)

        do {
            do {
                let inferenceConfig = ZImageTurboInferenceConfig(
                    width: request.width,
                    height: request.height,
                    numInferenceSteps: request.steps,
                    imageStrength: request.inputImage != nil ? request.strength : nil
                )

                let scheduler = ZImageTurboLinearScheduler(
                    config: inferenceConfig,
                    requiresSigmaShift: true,
                    sigmaShift: request.sigmaShift
                )

                progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
                let promptEmbeds = try encodePrompt(
                    request.prompt,
                    tokenizer: model.tokenizer,
                    textEncoder: model.textEncoder,
                    maxLength: request.maxSequenceLength
                )

                let cfgScale = Float(request.guidanceScale)
                let negativePromptEmbeds: MLXArray? = cfgScale > 1 ? try encodePrompt(
                    request.negativePrompt ?? "",
                    tokenizer: model.tokenizer,
                    textEncoder: model.textEncoder,
                    maxLength: request.maxSequenceLength
                ) : nil

                let latentShape = [1, 16, max(inferenceConfig.height, 8) / 8, max(inferenceConfig.width, 8) / 8]
                let noise = MLXRandom.normal(latentShape, key: MLXRandom.key(seed)).asType(.float32)

                let startStep = inferenceConfig.initTimeStep
                var latents: MLXArray
                if let inputImageURL = request.inputImage {
                    latents = try await prepareImg2ImgLatents(
                        inputImageURL: inputImageURL,
                        noise: noise,
                        sigma: scheduler.sigmas[startStep],
                        vae: model.vae,
                        height: inferenceConfig.height,
                        width: inferenceConfig.width
                    )
                } else {
                    latents = noise
                }

                let machine = MereRunMachineProfile.current
                let useBatchedCFG = negativePromptEmbeds.map { negativeEmbeds in
                    DiffusionCFGExecution.canPair(negativeEmbeds, promptEmbeds)
                        && DiffusionCFGExecution.shouldBatch(
                            mode: DiffusionCFGExecutionMode.current(
                                modelEnvironmentKey: "MERERUN_ZIMAGE_BATCHED_CFG"
                            ),
                            width: inferenceConfig.width,
                            height: inferenceConfig.height,
                            physicalMemoryBytes: machine.physicalMemoryBytes,
                            activeMemoryBytes: Memory.activeMemory,
                            cacheMemoryBytes: Memory.cacheMemory,
                            isUnifiedMemory: machine.isAppleSiliconMac,
                            baseReserveBytes: 6 * DiffusionCFGExecution.gibibyte,
                            activationBytesPerPixel: 4_096
                        )
                } ?? false

                for stepIndex in startStep..<inferenceConfig.numInferenceSteps {
                    try Task.checkCancellation()
                    progressHandler?(GenerationProgress(
                        stage: .denoising,
                        stepIndex: stepIndex,
                        totalSteps: inferenceConfig.numInferenceSteps
                    ))

                    let tInput = (MLXArray([Float(1.0)]) - scheduler.sigmas[stepIndex].asType(.float32)).asType(.float32)
                    let latentsModelInput = latents.asType(.bfloat16)

                    let noisePred: MLXArray
                    if let negativePromptEmbeds, cfgScale > 1 {
                        if useBatchedCFG {
                            let predictions = model.transformer.forward(
                                latents: DiffusionCFGExecution.duplicateBatch(latentsModelInput),
                                timestep: DiffusionCFGExecution.duplicateBatch(tInput),
                                promptEmbeds: DiffusionCFGExecution.paired(negativePromptEmbeds, promptEmbeds)
                            )
                            noisePred = DiffusionCFGExecution.combinePositiveAnchoredPredictions(
                                -predictions.asType(.float32),
                                guidanceScale: cfgScale
                            )
                        } else {
                            let predictedPos = model.transformer.forward(
                                latents: latentsModelInput,
                                timestep: tInput,
                                promptEmbeds: promptEmbeds
                            )
                            let noisePos = (-predictedPos).asType(.float32)
                            let predictedNeg = model.transformer.forward(
                                latents: latentsModelInput,
                                timestep: tInput,
                                promptEmbeds: negativePromptEmbeds
                            )
                            let noiseNeg = (-predictedNeg).asType(.float32)
                            noisePred = noisePos + (noisePos - noiseNeg) * MLXArray(cfgScale)
                        }
                    } else {
                        let predictedPos = model.transformer.forward(
                            latents: latentsModelInput,
                            timestep: tInput,
                            promptEmbeds: promptEmbeds
                        )
                        noisePred = (-predictedPos).asType(.float32)
                    }

                    latents = scheduler.step(noise: noisePred, timestep: stepIndex, latents: latents)
                    MLX.eval(latents)
                }

                progressHandler?(GenerationProgress(
                    stage: .decoding,
                    stepIndex: inferenceConfig.numInferenceSteps,
                    totalSteps: inferenceConfig.numInferenceSteps
                ))

                let decoded = decodeLatents(
                    latents,
                    vae: model.vae,
                    height: inferenceConfig.height,
                    width: inferenceConfig.width
                )

                progressHandler?(GenerationProgress(
                    stage: .saving,
                    stepIndex: inferenceConfig.numInferenceSteps,
                    totalSteps: inferenceConfig.numInferenceSteps
                ))
                try QwenImageIO.saveImage(array: decoded, to: outputURL)
            }

            model.transformer.clearCache()
            clearGPUMemory()
            return GenerationResult(outputURL: outputURL, seed: seed)
        } catch {
            model.transformer.clearCache()
            clearGPUMemory()
            throw error
        }
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let debugLog = MereRunRuntimeDebug.logger(keys: ["MERERUN_ZIMAGE_DEBUG"], prefix: "[ZImageTurboGenerator]")
        debugLog?("chat: starting")
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading model"))

        let modelSpec = ZImageTurboRepository.defaultModelSpec
        debugLog?("chat: loading model \(modelSpec)")
        let model = try await loadModelIfNeeded(modelSpec: modelSpec, progressHandler: nil)
        debugLog?("chat: model loaded, applying LoRA if needed")
        try await applyTextLoRAIfNeeded(request.lora, model: model, progressHandler: progressHandler)
        debugLog?("chat: LoRA applied")

        do {
            let response: ChatResponse = try {
                debugLog?("chat: encoding messages")
                progressHandler?(ChatProgress(stage: .encoding, message: "Encoding messages"))

                let messages: [[String: Any]] = request.messages.map { msg in
                    var dict: [String: Any] = [
                        "role": msg.role.rawValue,
                        "content": msg.content
                    ]
                    if let imageUrl = msg.imageUrl {
                        dict["image_url"] = imageUrl
                    }
                    return dict
                }

                debugLog?("chat: tokenizing \(messages.count) messages")
                let tokens = try model.tokenizer.encodeChatForGeneration(
                    messages: messages,
                    maxLength: model.tokenizer.maxLength
                )
                debugLog?("chat: got \(tokens.count) tokens")
                let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)

                debugLog?("chat: starting generation")
                progressHandler?(ChatProgress(stage: .generating, message: "Generating response"))

                let config = PromptEnhanceConfig(
                    maxNewTokens: request.maxTokens,
                    temperature: Float(request.temperature),
                    topP: 0.9,
                    repetitionPenalty: 1.05,
                    repetitionContextSize: 20,
                    eosTokenId: model.tokenizer.eosTokenId ?? 151645,
                    stopTokenIds: Set([model.tokenizer.eosTokenId ?? 151645, 151643])
                )

                debugLog?("chat: calling generate with maxTokens=\(request.maxTokens)")
                let generatedTokens = model.textEncoder.generate(inputIds: inputIds, config: config)
                debugLog?("chat: generated \(generatedTokens.count) tokens")

                let decoded = model.tokenizer.decode(tokens: generatedTokens)
                let cleanedResponse = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
                debugLog?("chat: decoded response length=\(cleanedResponse.count)")

                return ChatResponse(
                    generatedText: cleanedResponse,
                    tokensGenerated: generatedTokens.count,
                    showThinking: request.showThinking
                )
            }()

            model.transformer.clearCache()
            clearGPUMemory()
            return response
        } catch {
            model.transformer.clearCache()
            clearGPUMemory()
            throw error
        }
    }
}
