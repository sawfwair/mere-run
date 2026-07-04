import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Hummingbird
import NIOCore
import AudioCore
import AudioSTT
import AudioTTS
import MediaIO
import MereRunCore

struct APIServe: AsyncParsableCommand {
    private static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start an OpenAI-compatible API server for local chat and embedding models.",
        discussion: """
        Runs an HTTP server that exposes an OpenAI-compatible API for the selected local runtime.
        Compatible with any OpenAI client (VS Code extensions, Continue, Cursor, etc.).

        Endpoints:
          GET  /health              - Health check
          GET  /v1/models           - List available models
          POST /v1/chat/completions - Chat completions (streaming supported)
          POST /v1/embeddings       - Native Qwen3 text embeddings
          POST /v1/images/generations - Native image generation
          POST /v1/images/edits       - Native image editing
          POST /v1/audio/speech      - Native text to speech
          POST /v1/audio/transcriptions - Native speech to text

        Example:
          # Start with the default local code model
          mere.run api serve

          # Start with a specific GGUF model
          mere.run api serve -m ~/models/Qwen3-Coder-Next-Q4_K_M.gguf

          # Start a Gemma 4 text-chat server
          mere.run api serve --engine text-chat-gemma4

          # Start a Qwen3.6 text-chat server with an explicit model root
          mere.run api serve --engine text-chat-q36 -m ~/Models/text-chat-q36-nano

          # Start an LFM2.5 text-chat server
          mere.run api serve --engine text-chat-lfm2

          # Start the DeepSeek V4 Flash OpenAI-compatible server
          mere.run api serve --engine text-chat-deepseek-v4-flash

          # Custom host/port (non-loopback binds require an API key)
          export MERERUN_API_KEY=change-me
          mere.run api serve --host 0.0.0.0 --port 11434 --api-key "$MERERUN_API_KEY"

          # Check server health and the served model from another terminal
          mere.run status --port 11434

          # Test chat with curl (request.json contains an OpenAI chat payload)
          curl http://localhost:8080/v1/chat/completions \\
            -H "Content-Type: application/json" \\
            --data @request.json

          # Test embeddings with curl
          curl http://localhost:8080/v1/embeddings \\
            -H "Content-Type: application/json" \\
            --data '{"model":"text-embed-qwen3-0.6b","input":"hello"}'

          # Generate an image as base64 PNG JSON
          curl http://localhost:8080/v1/images/generations \\
            -H "Content-Type: application/json" \\
            --data '{"model":"image-zimage-nano","prompt":"a tiny workstation in morning light","size":"1024x1024"}'

          # Edit an image as base64 PNG JSON
          curl http://localhost:8080/v1/images/edits \\
            -F model=image-zimage-nano \\
            -F prompt='make the workstation dusk-lit' \\
            -F image=@input.png

          # Generate speech
          curl http://localhost:8080/v1/audio/speech \\
            -H "Content-Type: application/json" \\
            --output speech.wav \\
            --data '{"model":"speech-tts-qwen3-nano","input":"mere.run is online","voice":"nova","response_format":"wav"}'

          # Transcribe audio
          curl http://localhost:8080/v1/audio/transcriptions \\
            -F model=speech-asr-parakeet \\
            -F file=@speech.wav
        """
    )

    @Option(name: [.short, .long], help: "Port to listen on.")
    var port: Int = 8080

    @Option(name: [.long], help: "Host to bind to.")
    var host: String = "127.0.0.1"

    @Option(name: [.customShort("m"), .long, .customLong("model-path")], help: "Model path. For --engine text-code, pass a GGUF file. For --engine text-chat-klein, pass a Klein-root text chat model. For --engine text-chat-gemma4, pass a Gemma 4 model root or repo ID. For --engine text-chat-q36, pass a Qwen3.6 text chat model root. For --engine text-chat-lfm2, pass an LFM2 MLX model root or repo ID. For --engine text-chat-deepseek-v4-flash, pass a DS4 GGUF file or managed model root.")
    var model: String?

    @Option(name: [.long], help: "Serving engine: text-chat-q36 (default; serves text-chat-q36-nano), text-code, text-chat-klein, text-chat-gemma4, text-chat-lfm2, or text-chat-deepseek-v4-flash.")
    var engine: APIEngine = .textChatQ36

    @Option(name: [.long], help: "Default LoRA adapter path for all requests.")
    var lora: String?

    @Option(name: [.long], help: "Bearer token required by API endpoints. Also read from MERERUN_API_KEY.")
    var apiKey: String?

    @Option(name: [.long], help: "Global request limit for /v1/chat/completions and /v1/embeddings per rolling minute.")
    var rateLimitPerMinute: Int = 60

    @Option(name: [.long], help: "Maximum chat completions admitted at once. Defaults to 1 for serialized local inference.")
    var maxActiveRequests: Int = 1

    @Option(name: [.long], help: "Runtime memory guard tier: off, safe, balanced, aggressive, or custom.")
    var memoryGuard: RuntimeMemoryGuardTier = .default

    @Option(name: [.long], help: "Custom memory guard ceiling in GiB. Requires --memory-guard custom.")
    var memoryGuardCustomCeilingGB: Double?

    @Option(name: [.long], help: "Context size (default: 32768).")
    var contextSize: Int = 32768

    @Option(name: [.long], help: "Quantize the Gemma4 KV cache to this many bits. Supports integer widths for uniform/polar and integer/.5 widths for turboquant.")
    var kvBits: Double?

    @Option(name: [.long], help: "Gemma4 KV cache quantization backend: uniform, polar, or turboquant.")
    var kvQuantScheme: String?

    @Option(name: [.long], help: "Gemma4 KV cache quantization group size.")
    var kvGroupSize: Int?

    @Option(name: [.long], help: "Gemma4 token offset at which KV cache quantization begins.")
    var quantizedKVStart: Int?

    func run() async throws {
        let resolvedAPIKey = resolveAPIKey()
        try validateServerSecurity(apiKey: resolvedAPIKey)
        let resolvedModelPath = try resolveModelPath()
        let gemma4KVCacheQuantization = try resolveGemma4KVCacheQuantization()
        let memoryPressurePolicy = try resolveMemoryPressurePolicy()
        let server = try await CodeGenServer(
            defaultModelID: defaultRuntimeModelID(modelPath: resolvedModelPath),
            modelPath: resolvedModelPath,
            fallbackLoraPath: lora,
            apiKey: resolvedAPIKey,
            rateLimitPerMinute: rateLimitPerMinute,
            maxActiveRequests: maxActiveRequests,
            engine: engine,
            contextSize: contextSize,
            gemma4KVCacheQuantization: gemma4KVCacheQuantization,
            memoryPressurePolicy: memoryPressurePolicy
        )
        try await server.run(host: host, port: port)
    }

    private func defaultRuntimeModelID(modelPath: String?) -> String {
        if let requested = model?.trimmingCharacters(in: .whitespacesAndNewlines),
           let spec = ManagedModelCatalog.spec(for: requested),
           spec.defaultRuntimeServingEngine == engine.runtimeServingEngine {
            return spec.id
        }
        if let modelPath, model != nil {
            return URL(fileURLWithPath: modelPath).lastPathComponent
        }
        switch engine {
        case .textChatKlein:
            return ModelResolver.ModelID.mebot.rawValue
        case .textChatGemma4:
            return ModelResolver.ModelID.gemma4.rawValue
        case .textChatQ36, .textChatQ35:
            return ModelResolver.ModelID.q36Nano.rawValue
        case .textChatLFM2:
            return LFM2Resources.defaultModelId
        case .textCode:
            return CodeGenResources.defaultModelId
        case .textChatDeepseekV4Flash:
            return DeepseekV4FlashResources.defaultModelId
        }
    }

    private func resolveModelPath() throws -> String? {
        switch engine {
        case .textCode:
            return model
        case .textChatKlein:
            if let explicit = model {
                return explicit
            }
            if let mebotPath = MeBotModelCatalog.resolveModelPath() {
                return mebotPath
            }
            throw ValidationError("Model 'text-chat-mebot' is not installed.")
        case .textChatGemma4:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.gemma4) {
                return resolved.rootURL.path
            }
            return nil
        case .textChatQ36, .textChatQ35:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.q36Nano) {
                return resolved.rootURL.path
            }
            // Allow the Qwen-family generator to auto-download from Hugging Face when model path is omitted.
            return nil
        case .textChatLFM2:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.lfm25A1B8Bit) {
                return resolved.rootURL.path
            }
            // Allow LFM2Generator to auto-download from Hugging Face when model path is omitted.
            return nil
        case .textChatDeepseekV4Flash:
            // DeepseekV4FlashGenerator resolves and (if needed) downloads its own GGUF.
            // Honor an explicit --model path if provided.
            return model
        }
    }

    func resolveGemma4KVCacheQuantization() throws -> Gemma4KVCacheQuantization {
        let scheme = try resolveGemma4KVQuantizationScheme()
        return Gemma4KVCacheQuantization(
            bits: resolvedGemma4KVBits,
            scheme: scheme,
            groupSize: kvGroupSize ?? Gemma4Resources.defaultKVGroupSize,
            quantizedStart: resolvedGemma4QuantizedKVStart
        )
    }

    private var requestedGemma4ModelSpec: String {
        model?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? Gemma4Resources.defaultModelId
    }

    private var usesGemma4TurboKVDefaults: Bool {
        Gemma4Resources.usesTurboDefaults(modelSpec: requestedGemma4ModelSpec)
            && Gemma4Resources.supportsDefaultTurboKVQuantization
    }

    private var resolvedGemma4KVBits: Double? {
        kvBits ?? (usesGemma4TurboKVDefaults ? Gemma4Resources.defaultTurboKVBits : nil)
    }

    private func resolveGemma4KVQuantizationScheme() throws -> Gemma4KVQuantizationScheme {
        let raw = kvQuantScheme
            ?? (usesGemma4TurboKVDefaults
                ? Gemma4Resources.defaultTurboKVQuantizationScheme.rawValue
                : Gemma4Resources.defaultKVQuantizationScheme.rawValue)
        guard let scheme = Gemma4KVQuantizationScheme(
            rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        ) else {
            throw ValidationError("Unsupported --kv-quant-scheme '\(raw)'. Expected 'uniform', 'polar', or 'turboquant'.")
        }
        return scheme
    }

    private var resolvedGemma4QuantizedKVStart: Int {
        quantizedKVStart ?? (usesGemma4TurboKVDefaults
            ? Gemma4Resources.defaultTurboQuantizedKVStart
            : Gemma4Resources.defaultQuantizedKVStart)
    }

    private func resolveAPIKey() -> String? {
        if let apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !apiKey.isEmpty {
            return apiKey
        }
        if let apiKey = ProcessInfo.processInfo.environment[Self.apiKeyEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        return nil
    }

    private func validateServerSecurity(apiKey: String?) throws {
        guard rateLimitPerMinute > 0 else {
            throw ValidationError("--rate-limit-per-minute must be greater than zero.")
        }
        guard maxActiveRequests > 0 else {
            throw ValidationError("--max-active-requests must be greater than zero.")
        }
        if let memoryGuardCustomCeilingGB {
            guard memoryGuard == .custom else {
                throw ValidationError("--memory-guard-custom-ceiling-gb requires --memory-guard custom.")
            }
            guard memoryGuardCustomCeilingGB.isFinite, memoryGuardCustomCeilingGB > 0 else {
                throw ValidationError("--memory-guard-custom-ceiling-gb must be greater than zero.")
            }
        }
        guard (1...Int(Int32.max)).contains(contextSize) else {
            throw ValidationError("--context-size must be between 1 and \(Int(Int32.max)).")
        }
        guard Self.isLoopbackHost(host) || apiKey != nil else {
            throw ValidationError("Binding to non-loopback hosts requires --api-key or MERERUN_API_KEY.")
        }
    }

    private func resolveMemoryPressurePolicy() throws -> RuntimeMemoryPressurePolicy {
        let gib = Double(1024 * 1024 * 1024)
        let customCeilingBytes = memoryGuardCustomCeilingGB.map { UInt64(($0 * gib).rounded(.down)) }
        if memoryGuard == .custom, customCeilingBytes == nil {
            throw ValidationError("--memory-guard custom requires --memory-guard-custom-ceiling-gb.")
        }
        return RuntimeMemoryPressurePolicy(
            tier: memoryGuard,
            customCeilingBytes: customCeilingBytes
        )
    }

    private static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "127.0.0.1" || normalized == "localhost" || normalized == "::1"
    }
}

extension RuntimeMemoryGuardTier: ExpressibleByArgument {}

enum APIEngine: String, ExpressibleByArgument {
    case textCode = "text-code"
    case textChatKlein = "text-chat-klein"
    case textChatGemma4 = "text-chat-gemma4"
    case textChatQ36 = "text-chat-q36"
    case textChatQ35 = "text-chat-q35"
    case textChatLFM2 = "text-chat-lfm2"
    case textChatDeepseekV4Flash = "text-chat-deepseek-v4-flash"

    var runtimeServingEngine: RuntimeServingEngine {
        switch self {
        case .textCode:
            return .textCode
        case .textChatKlein:
            return .textChatKlein
        case .textChatGemma4:
            return .textChatGemma4
        case .textChatQ36:
            return .textChatQ36
        case .textChatQ35:
            return .textChatQ36
        case .textChatLFM2:
            return .textChatLFM2
        case .textChatDeepseekV4Flash:
            return .textChatDeepseekV4Flash
        }
    }

    var openAICompatibility: APIEngineCapabilities {
        runtimeServingEngine.openAICompatibility
    }
}

struct APIEngineCapabilities: Equatable, Sendable {
    var supportsRawProxy: Bool = false
    var supportsTools: Bool = false
    var supportsToolChoice: Bool = false
    var supportsDeveloperRole: Bool = true
    var supportsStructuredOutputs: Bool = false
    var supportsReasoningEffort: Bool = false
    var supportsMaxCompletionTokens: Bool = true
    var supportsUsageInStreaming: Bool = true
    var supportsVisionContentParts: Bool = false
    var supportsStrictMode: Bool = false
    var supportsStopSequences: Bool = false
    var supportsSeed: Bool = false
    var supportsPenalties: Bool = false
    var supportsLogprobs: Bool = false
    var supportsProviderThinkingControls: Bool = false

    static let localText = APIEngineCapabilities()

    static let localTextWithStopSequences = APIEngineCapabilities(
        supportsStopSequences: true
    )

    static let localTextWithStructuredJSON = APIEngineCapabilities(
        supportsStructuredOutputs: true
    )

    static let localTextWithTools = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true
    )

    static let localTextWithToolsAndStructuredJSON = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true,
        supportsStructuredOutputs: true
    )

    static let localTextWithToolsAndVision = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true,
        supportsVisionContentParts: true
    )

    static let rawProxy = APIEngineCapabilities(
        supportsRawProxy: true,
        supportsTools: true,
        supportsToolChoice: true,
        supportsDeveloperRole: true,
        supportsStructuredOutputs: true,
        supportsReasoningEffort: true,
        supportsMaxCompletionTokens: true,
        supportsUsageInStreaming: true,
        supportsVisionContentParts: true,
        supportsStrictMode: false,
        supportsStopSequences: true,
        supportsSeed: true,
        supportsPenalties: true,
        supportsLogprobs: true,
        supportsProviderThinkingControls: true
    )
}

struct APIHealthStatus: Codable, Equatable, Sendable {
    let status: String
}

enum APIServerContract {
    static let defaultMaxTokens = 2048
    static let defaultImageModelID = ModelResolver.ModelID.zetaNano.rawValue
    static let defaultSpeechModelID = Qwen3TTSResources.defaultModelId
    static let defaultTranscriptionModelID = ParakeetResources.defaultModelId

    struct ImageGenerationPlan: Equatable, Sendable {
        let modelID: String
        let prompt: String
        let width: Int
        let height: Int
        let responseFormat: String
        let seed: UInt64?
        let negativePrompt: String?
        let steps: Int?
        let guidanceScale: Double?
        let inputImage: URL?
        let additionalInputImages: [URL]
        let maskImage: URL?
        let strength: Double?

        init(
            modelID: String,
            prompt: String,
            width: Int,
            height: Int,
            responseFormat: String,
            seed: UInt64?,
            negativePrompt: String?,
            steps: Int?,
            guidanceScale: Double?,
            inputImage: URL?,
            additionalInputImages: [URL] = [],
            maskImage: URL? = nil,
            strength: Double?
        ) {
            self.modelID = modelID
            self.prompt = prompt
            self.width = width
            self.height = height
            self.responseFormat = responseFormat
            self.seed = seed
            self.negativePrompt = negativePrompt
            self.steps = steps
            self.guidanceScale = guidanceScale
            self.inputImage = inputImage
            self.additionalInputImages = additionalInputImages
            self.maskImage = maskImage
            self.strength = strength
        }
    }

    struct SpeechPlan: Equatable, Sendable {
        let modelID: String
        let input: String
        let voiceDescription: String
        let responseFormat: String
        let speed: Float
        let temperature: Float
    }

    struct TranscriptionPlan: Equatable, Sendable {
        let modelID: String
        let language: String?
        let responseFormat: String
        let task: ASRTask
        let maxTokens: Int
    }

    static func healthStatus() -> APIHealthStatus {
        APIHealthStatus(status: "ok")
    }

    static func modelsResponse(modelId: String, createdAt: Date = Date()) -> OpenAIModelsResponse {
        modelsResponse(modelIds: [modelId], createdAt: createdAt)
    }

    static func modelsResponse(modelIds: [String], createdAt: Date = Date()) -> OpenAIModelsResponse {
        OpenAIModelsResponse(
            object: "list",
            data: modelIds.map {
                OpenAIModel(
                    id: $0,
                    object: "model",
                    created: Int(createdAt.timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            }
        )
    }

    static func embeddingTexts(from request: OpenAIEmbeddingRequest) throws -> [String] {
        guard !request.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw APIRequestValidationError.invalidField("model", "must not be empty")
        }
        if let encodingFormat = request.encoding_format?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
           !encodingFormat.isEmpty,
           encodingFormat != "float" {
            throw APIRequestValidationError.invalidField(
                "encoding_format",
                "only float embeddings are supported"
            )
        }
        if request.dimensions != nil {
            throw APIRequestValidationError.invalidField(
                "dimensions",
                "dimension overrides are not supported by this embedding model"
            )
        }

        let texts = request.input.texts
        guard !texts.isEmpty else {
            throw APIRequestValidationError.invalidField("input", "must contain at least one text")
        }
        return texts
    }

    static func embeddingResponse(
        modelId: String,
        embeddings: [[Float]],
        tokenCounts: [Int]
    ) -> OpenAIEmbeddingResponse {
        let promptTokens = tokenCounts.reduce(0, +)
        return OpenAIEmbeddingResponse(
            model: modelId,
            data: embeddings.enumerated().map { index, vector in
                OpenAIEmbeddingDatum(index: index, embedding: vector)
            },
            usage: OpenAIEmbeddingUsage(
                prompt_tokens: promptTokens,
                total_tokens: promptTokens
            )
        )
    }

    static func companionModelIDs(
        fileManager: FileManager = .default,
        installedModelIDs: Set<String>? = nil
    ) -> [String] {
        let categories: Set<ManagedModelCategory> = [.image, .speechTTS, .speechASR, .textEmbed]
        let ids = ManagedModelCatalog.allSpecs
            .filter { categories.contains($0.category) }
            .filter { isCompanionModelInstalled($0, fileManager: fileManager, installedModelIDs: installedModelIDs) }
            .map(\.id)
        var uniqueIDs = Set(ids)
        if isQwenImageEditInstalled(fileManager: fileManager, installedModelIDs: installedModelIDs) {
            uniqueIDs.insert(QwenImageEditRepository.modelId)
        }
        return Array(uniqueIDs).sorted()
    }

    static func imageGenerationPlan(
        from request: OpenAIImageGenerationRequest
    ) throws -> ImageGenerationPlan {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        if let steps = request.steps, steps <= 0 {
            throw APIRequestValidationError.invalidField("steps", "must be greater than zero")
        }
        if let guidanceScale = request.guidance_scale,
           (!guidanceScale.isFinite || guidanceScale < 0) {
            throw APIRequestValidationError.invalidField("guidance_scale", "must be a positive number")
        }
        let size = try imageSize(from: request.size)
        let responseFormat = try imageResponseFormat(request.response_format)
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(request.model),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: responseFormat,
            seed: request.seed,
            negativePrompt: normalizedOptional(request.negative_prompt),
            steps: request.steps,
            guidanceScale: request.guidance_scale,
            inputImage: nil,
            strength: nil
        )
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURL: URL
    ) throws -> ImageGenerationPlan {
        try imageEditPlan(from: form, inputImageURLs: [inputImageURL], maskImageURL: nil)
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURLs: [URL],
        maskImageURL: URL?
    ) throws -> ImageGenerationPlan {
        guard let inputImageURL = inputImageURLs.first else {
            throw APIRequestValidationError.invalidField("image", "image file is required")
        }
        let prompt = normalizedOptional(form.field("prompt")) ?? ""
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let rawN = normalizedOptional(form.field("n")) {
            guard let n = Int(rawN), n == 1 else {
                throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
            }
        }
        let steps = try optionalPositiveIntField(form.field("steps"), field: "steps")
        let guidanceScale = try optionalPositiveDoubleField(form.field("guidance_scale"), field: "guidance_scale")
        let strength = try optionalUnitDoubleField(form.field("strength"), field: "strength")
        let size = try imageSize(from: form.field("size"))
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(form.field("model")),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: try imageResponseFormat(form.field("response_format")),
            seed: try optionalUInt64Field(form.field("seed")),
            negativePrompt: normalizedOptional(form.field("negative_prompt")),
            steps: steps,
            guidanceScale: guidanceScale,
            inputImage: inputImageURL,
            additionalInputImages: Array(inputImageURLs.dropFirst()),
            maskImage: maskImageURL,
            strength: strength
        )
    }

    static func imageResponse(
        outputURL: URL,
        plan: ImageGenerationPlan,
        createdAt: Date = Date()
    ) throws -> OpenAIImageGenerationResponse {
        let datum: OpenAIImageGenerationData
        switch plan.responseFormat {
        case "url":
            datum = OpenAIImageGenerationData(url: outputURL.absoluteString, revised_prompt: plan.prompt)
        case "b64_json":
            let data = try Data(contentsOf: outputURL)
            datum = OpenAIImageGenerationData(
                b64_json: data.base64EncodedString(),
                revised_prompt: plan.prompt
            )
        default:
            throw APIRequestValidationError.invalidField("response_format", "unsupported response format")
        }
        return OpenAIImageGenerationResponse(
            created: Int(createdAt.timeIntervalSince1970),
            data: [datum]
        )
    }

    static func speechPlan(from request: OpenAIAudioSpeechRequest) throws -> SpeechPlan {
        let input = request.input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else {
            throw APIRequestValidationError.invalidField("input", "must not be empty")
        }
        let responseFormat = try speechResponseFormat(request.response_format)
        let speed = try speechSpeed(request.speed)
        let temperature = request.temperature ?? 0.6
        guard temperature.isFinite, (0...2).contains(temperature) else {
            throw APIRequestValidationError.invalidField("temperature", "must be between 0 and 2")
        }
        return SpeechPlan(
            modelID: normalizedSpeechModelID(request.model),
            input: input,
            voiceDescription: voiceDescription(for: request.voice, instructions: request.instructions),
            responseFormat: responseFormat,
            speed: speed,
            temperature: temperature
        )
    }

    static func transcriptionPlan(from form: MultipartFormData) throws -> TranscriptionPlan {
        let modelID = normalizedTranscriptionModelID(form.field("model"))
        return TranscriptionPlan(
            modelID: modelID,
            language: normalizedOptional(form.field("language")),
            responseFormat: try transcriptionResponseFormat(form.field("response_format")),
            task: try transcriptionTask(form.field("task")),
            maxTokens: try transcriptionMaxTokens(form.field("max_tokens"))
        )
    }

    static func transcriptionResponse(
        from result: ASRResult,
        verbose: Bool
    ) -> OpenAIAudioTranscriptionResponse {
        OpenAIAudioTranscriptionResponse(
            text: result.text,
            language: result.language,
            duration: verbose ? result.duration : nil,
            segments: verbose ? transcriptionSegments(from: result) : nil
        )
    }

    static func transcriptionSubtitle(from result: ASRResult, format: String) -> String {
        let segments = transcriptionSegments(from: result) ?? [
            OpenAIAudioTranscriptionSegment(
                id: 0,
                start: 0,
                end: max(result.duration, 0.001),
                text: result.text
            ),
        ]
        switch format {
        case "srt":
            return segments.enumerated()
                .map { index, segment in
                    let start = subtitleTimestamp(segment.start, separator: ",")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ",")
                    return "\(index + 1)\n\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n") + "\n"
        default:
            let body = segments
                .map { segment in
                    let start = subtitleTimestamp(segment.start, separator: ".")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ".")
                    return "\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n")
            return body.isEmpty ? "WEBVTT\n" : "WEBVTT\n\n\(body)\n"
        }
    }

    static func chatRequest(
        from openaiRequest: OpenAIChatRequest,
        fallbackLoraPath: String?,
        contextSize: Int,
        capabilities: APIEngineCapabilities = .localText,
        servedModelID: String? = nil
    ) throws -> ChatRequest {
        guard !openaiRequest.messages.isEmpty else {
            throw APIRequestValidationError.invalidField("messages", "must contain at least one message")
        }

        try validateTopLevelOptions(openaiRequest, capabilities: capabilities)

        if let requestLora = openaiRequest.lora?.trimmingCharacters(in: .whitespacesAndNewlines),
           !requestLora.isEmpty {
            throw APIRequestValidationError.invalidField(
                "lora",
                "per-request LoRA paths are not supported; start the server with --lora instead"
            )
        }

        let maxTokens = try validateMaxTokens(
            maxTokens: openaiRequest.max_tokens,
            maxCompletionTokens: openaiRequest.max_completion_tokens,
            contextSize: contextSize,
            capabilities: capabilities
        )
        let temperature = try validateTemperature(openaiRequest.temperature)
        let topP = try validateTopP(openaiRequest.top_p)
        let tools = try toolDefinitions(from: openaiRequest, capabilities: capabilities)
        let requiresJSON = try requiresJSONResponseFormat(
            openaiRequest.response_format,
            capabilities: capabilities
        )

        let messages = try openaiRequest.messages.map { msg in
            try chatMessage(from: msg, capabilities: capabilities)
        }

        let lora: LoRA?
        if let loraPath = fallbackLoraPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !loraPath.isEmpty {
            lora = LoRA.local(path: loraPath, scale: 1.0)
        } else {
            lora = nil
        }

        // R1-style lanes degenerate without reasoning; their published top_k
        // applies only when the client did not set explicit sampling.
        let laneModelID = servedModelID ?? ""
        let recommendedSampling = Q35Resources.recommendedSampling(forModelId: laneModelID)
        let usesExplicitSampling = openaiRequest.temperature != nil || openaiRequest.top_p != nil

        return ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: usesExplicitSampling ? nil : recommendedSampling?.topK,
            showThinking: Q35Resources.thinkingDefault(forModelId: laneModelID),
            lora: lora,
            requiresJSON: requiresJSON,
            tools: tools,
            stopSequences: openaiRequest.stop?.values ?? [],
            maxContextTokens: contextSize
        )
    }

    static func includeUsageInStreaming(
        _ openaiRequest: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard openaiRequest.stream_options?.include_usage == true else {
            return false
        }
        guard capabilities.supportsUsageInStreaming else {
            throw APIRequestValidationError.invalidField(
                "stream_options.include_usage",
                "this engine cannot emit usage chunks while streaming"
            )
        }
        return true
    }

    private static func chatMessage(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> ChatMessage {
        let normalizedRole = msg.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let role: ChatMessage.Role
        switch normalizedRole {
        case "system":
            role = .system
        case "developer":
            guard capabilities.supportsDeveloperRole else {
                throw APIRequestValidationError.invalidField(
                    "messages.role",
                    "developer messages are not supported by this engine"
                )
            }
            role = .system
        case "user":
            role = .user
        case "assistant":
            role = .assistant
        case "tool":
            role = .tool
        default:
            throw APIRequestValidationError.invalidField(
                "messages.role",
                "unsupported role '\(msg.role)'"
            )
        }

        let imageURL = try firstImageURL(from: msg, capabilities: capabilities)
        let content = renderMessageContent(msg)
        return ChatMessage(
            role: role,
            content: content,
            imageUrl: imageURL
        )
    }

    private static func firstImageURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.imageURLs.isEmpty else { return nil }
        guard capabilities.supportsVisionContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "image content parts are not supported by this engine"
            )
        }
        guard msg.imageURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one image content part is currently supported"
            )
        }
        return msg.imageURLs.first
    }

    private static func renderMessageContent(_ msg: OpenAIChatMessage) -> String {
        guard msg.role.lowercased() == "assistant",
              let toolCalls = msg.tool_calls,
              !toolCalls.isEmpty else {
            return msg.content
        }
        let renderedCalls = toolCalls.compactMap { call -> String? in
            guard call.type == "function", let function = call.function else { return nil }
            return "<|tool_call>call:\(function.name)\(function.arguments)<tool_call|>"
        }
        guard !renderedCalls.isEmpty else {
            return msg.content
        }
        return ([msg.content] + renderedCalls)
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func validateTopLevelOptions(
        _ request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws {
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        if request.store == true {
            throw APIRequestValidationError.invalidField("store", "stored chat completions are not supported")
        }
        if let modalities = request.modalities,
           modalities.contains(where: { $0 != "text" }) {
            throw APIRequestValidationError.invalidField("modalities", "only text output is supported")
        }
        if request.audio != nil {
            throw APIRequestValidationError.invalidField("audio", "audio output is not supported by /v1/chat/completions")
        }
        if request.prediction != nil {
            throw APIRequestValidationError.invalidField("prediction", "predicted outputs are not supported")
        }
        if let stop = request.stop, !stop.values.isEmpty, !capabilities.supportsStopSequences {
            throw APIRequestValidationError.invalidField("stop", "stop sequences are not supported by this engine")
        }
        if request.seed != nil, !capabilities.supportsSeed {
            throw APIRequestValidationError.invalidField("seed", "deterministic seeds are not supported by this engine")
        }
        if let penalty = request.presence_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("presence_penalty", "presence penalties are not supported by this engine")
        }
        if let penalty = request.frequency_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("frequency_penalty", "frequency penalties are not supported by this engine")
        }
        if request.logprobs == true, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("logprobs", "token log probabilities are not supported by this engine")
        }
        if request.top_logprobs != nil, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("top_logprobs", "token log probabilities are not supported by this engine")
        }
        if request.reasoning_effort != nil, !capabilities.supportsReasoningEffort {
            throw APIRequestValidationError.invalidField("reasoning_effort", "reasoning effort is not supported by this engine")
        }
        if request.think != nil || request.thinking != nil {
            guard capabilities.supportsProviderThinkingControls else {
                throw APIRequestValidationError.invalidField(
                    "thinking",
                    "provider thinking controls are not supported by this engine"
                )
            }
        }
    }

    private static func toolDefinitions(
        from request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> [ToolDefinition]? {
        guard let tools = request.tools, !tools.isEmpty else {
            return nil
        }
        guard capabilities.supportsTools else {
            throw APIRequestValidationError.invalidField("tools", "tools are not supported by this engine")
        }

        let convertedTools = try tools.map { try toolDefinition(from: $0) }
        switch request.tool_choice {
        case nil, .mode("auto")?, .mode("required")?:
            return convertedTools
        case .mode("none")?:
            return nil
        case .function(let name)?:
            guard let selected = convertedTools.first(where: { $0.name == name }) else {
                throw APIRequestValidationError.invalidField(
                    "tool_choice",
                    "requested tool '\(name)' is not present in tools"
                )
            }
            return [selected]
        case .custom?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported object shape")
        case .mode(let value)?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported mode '\(value)'")
        }
    }

    private static func toolDefinition(from tool: OpenAIChatTool) throws -> ToolDefinition {
        guard tool.type == "function", let function = tool.function else {
            throw APIRequestValidationError.invalidField("tools", "only function tools are supported")
        }

        let schema = function.parameters?.objectValue ?? [:]
        let properties = schema["properties"]?.objectValue ?? [:]
        var converted: [String: ToolParameterProperty] = [:]
        for (name, rawProperty) in properties {
            guard let property = rawProperty.objectValue else { continue }
            let type = property["type"]?.stringValue ?? "string"
            let description = property["description"]?.stringValue ?? ""
            converted[name] = ToolParameterProperty(type: type, description: description)
        }
        let required = schema["required"]?.arrayValue?.compactMap(\.stringValue) ?? []

        return ToolDefinition(
            name: function.name,
            description: function.description ?? "",
            parameters: converted,
            required: required
        )
    }

    private static func requiresJSONResponseFormat(
        _ responseFormat: OpenAIResponseFormat?,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard let responseFormat else { return false }
        switch responseFormat.type {
        case "text":
            return false
        case "json_object":
            guard capabilities.supportsStructuredOutputs else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "JSON mode is not supported by this engine"
                )
            }
            return true
        case "json_schema":
            guard capabilities.supportsStrictMode else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "strict JSON schema outputs are not supported by this engine"
                )
            }
            return true
        default:
            throw APIRequestValidationError.invalidField(
                "response_format",
                "unsupported response format '\(responseFormat.type)'"
            )
        }
    }

    private static func validateMaxTokens(
        maxTokens: Int?,
        maxCompletionTokens: Int?,
        contextSize: Int,
        capabilities: APIEngineCapabilities
    ) throws -> Int {
        if maxCompletionTokens != nil, !capabilities.supportsMaxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "this engine does not support max_completion_tokens"
            )
        }
        if let maxTokens, let maxCompletionTokens, maxTokens != maxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "must match max_tokens when both are provided"
            )
        }
        return try validateMaxTokens(maxCompletionTokens ?? maxTokens, contextSize: contextSize)
    }

    static func acceptsJSONContentType(_ rawValue: String?) -> Bool {
        guard let mediaType = rawValue?
            .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
            .first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
            !mediaType.isEmpty
        else {
            return false
        }
        return mediaType == "application/json" || mediaType.hasSuffix("+json")
    }

    static func multipartBoundary(from rawValue: String?) -> String? {
        guard let pieces = rawValue?.split(separator: ";", omittingEmptySubsequences: true),
              let mediaType = pieces.first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              mediaType == "multipart/form-data" else {
            return nil
        }
        for piece in pieces.dropFirst() {
            let pair = piece.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2,
                  pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "boundary" else {
                continue
            }
            var boundary = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if boundary.hasPrefix("\""), boundary.hasSuffix("\""), boundary.count >= 2 {
                boundary.removeFirst()
                boundary.removeLast()
            }
            return boundary.isEmpty ? nil : String(boundary)
        }
        return nil
    }

    static func isStreamingStatusMessage(_ message: String) -> Bool {
        switch message {
        case "Generating...", "Generating response", "Retrying generation", "DS4 chat completion":
            return true
        default:
            return false
        }
    }

    private static func validateMaxTokens(_ rawValue: Int?, contextSize: Int) throws -> Int {
        let value = rawValue ?? defaultMaxTokens
        let upperBound = min(contextSize, Int(Int32.max))
        guard upperBound > 0 else {
            throw APIRequestValidationError.invalidField("context_size", "must be greater than zero")
        }
        guard (1...upperBound).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "max_tokens",
                "must be between 1 and \(upperBound)"
            )
        }
        return value
    }

    private static func validateTemperature(_ rawValue: Double?) throws -> Double {
        let value = rawValue ?? 1.0
        guard value.isFinite, (0...2).contains(value) else {
            throw APIRequestValidationError.invalidField("temperature", "must be between 0 and 2")
        }
        return value
    }

    private static func validateTopP(_ rawValue: Double?) throws -> Double {
        let value = rawValue ?? 0.95
        guard value.isFinite, (0...1).contains(value) else {
            throw APIRequestValidationError.invalidField("top_p", "must be between 0 and 1")
        }
        return value
    }

    private static func imageSize(from rawValue: String?) throws -> (width: Int, height: Int) {
        guard let rawValue = normalizedOptional(rawValue), rawValue.lowercased() != "auto" else {
            return (1024, 1024)
        }
        let parts = rawValue.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]),
              let height = Int(parts[1]),
              width > 0,
              height > 0 else {
            throw APIRequestValidationError.invalidField("size", "expected WIDTHxHEIGHT, for example 1024x1024")
        }
        return (width, height)
    }

    private static func imageResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "b64_json"
        guard value == "b64_json" || value == "url" else {
            throw APIRequestValidationError.invalidField("response_format", "expected b64_json or url")
        }
        return value
    }

    private static func speechResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "wav"
        guard ["wav", "mp3", "opus", "aac", "flac"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected wav, mp3, opus, aac, or flac"
            )
        }
        return value
    }

    static func speechContentType(for responseFormat: String) -> String {
        switch responseFormat {
        case "mp3":
            return "audio/mpeg"
        case "opus":
            return "audio/ogg"
        case "aac":
            return "audio/aac"
        case "flac":
            return "audio/flac"
        default:
            return "audio/wav"
        }
    }

    private static func speechSpeed(_ rawValue: Double?) throws -> Float {
        let value = rawValue ?? 1.0
        guard value.isFinite, (0.25...4.0).contains(value) else {
            throw APIRequestValidationError.invalidField("speed", "must be between 0.25 and 4.0")
        }
        return Float(value)
    }

    private static func transcriptionResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "json"
        guard ["json", "text", "verbose_json", "srt", "vtt"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected json, text, verbose_json, srt, or vtt"
            )
        }
        return value
    }

    private static func transcriptionTask(_ rawValue: String?) throws -> ASRTask {
        let value = normalizedOptional(rawValue)?.lowercased() ?? ASRTask.transcribe.rawValue
        guard let task = ASRTask(rawValue: value) else {
            throw APIRequestValidationError.invalidField("task", "expected transcribe or translate")
        }
        return task
    }

    private static func transcriptionMaxTokens(_ rawValue: String?) throws -> Int {
        guard let rawValue = normalizedOptional(rawValue) else {
            return 448
        }
        guard let value = Int(rawValue), (1...Int(Int32.max)).contains(value) else {
            throw APIRequestValidationError.invalidField("max_tokens", "must be a positive integer")
        }
        return value
    }

    private static func optionalUInt64Field(_ rawValue: String?) throws -> UInt64? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = UInt64(rawValue) else {
            throw APIRequestValidationError.invalidField("seed", "must be an unsigned integer")
        }
        return value
    }

    private static func optionalPositiveIntField(_ rawValue: String?, field: String) throws -> Int? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Int(rawValue), value > 0 else {
            throw APIRequestValidationError.invalidField(field, "must be greater than zero")
        }
        return value
    }

    private static func optionalPositiveDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, value >= 0 else {
            throw APIRequestValidationError.invalidField(field, "must be a positive number")
        }
        return value
    }

    private static func optionalUnitDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, (0...1).contains(value) else {
            throw APIRequestValidationError.invalidField(field, "must be between 0 and 1")
        }
        return value
    }

    private static func normalizedModelID(_ rawValue: String?, defaultID: String) -> String {
        normalizedOptional(rawValue) ?? defaultID
    }

    private static func normalizedImageModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultImageModelID)
        switch modelID.lowercased() {
        case "gpt-image-1", "dall-e-3", "dall-e-2":
            return defaultImageModelID
        default:
            return modelID
        }
    }

    private static func normalizedSpeechModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultSpeechModelID)
        switch modelID.lowercased() {
        case "tts-1", "tts-1-hd", "gpt-4o-mini-tts":
            return defaultSpeechModelID
        default:
            return modelID
        }
    }

    private static func normalizedTranscriptionModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultTranscriptionModelID)
        switch modelID.lowercased() {
        case "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe":
            return defaultTranscriptionModelID
        default:
            return modelID
        }
    }

    private static func normalizedOptional(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func isCompanionModelInstalled(
        _ spec: ManagedModelSpec,
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(spec.id)
        }
        return spec.managedRuntimeURL(fileManager: fileManager) != nil
    }

    private static func isQwenImageEditInstalled(
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(QwenImageEditRepository.modelId)
        }
        return QwenImageEditRepository.resolveInstalledModelRoot(fileManager: fileManager) != nil
    }

    private static func voiceDescription(for rawVoice: String?, instructions: String?) -> String {
        let instructionText = normalizedOptional(instructions)
        let voice = normalizedOptional(rawVoice)?.lowercased() ?? "nova"
        let base: String
        switch voice {
        case "alloy":
            base = "A balanced, natural voice with clear pronunciation"
        case "ash":
            base = "A calm, low voice with a steady delivery"
        case "ballad":
            base = "A warm, expressive voice with a storytelling cadence"
        case "coral":
            base = "A bright, friendly voice with gentle energy"
        case "echo":
            base = "A clear male voice with an even, conversational tone"
        case "fable":
            base = "A warm narrative voice with a measured pace"
        case "nova":
            base = "A calm female voice with clear pronunciation"
        case "onyx":
            base = "A deep, confident voice with crisp articulation"
        case "sage":
            base = "A thoughtful, composed voice with soft emphasis"
        case "shimmer":
            base = "A bright, gentle voice with smooth pronunciation"
        default:
            base = rawVoice?.trimmingCharacters(in: .whitespacesAndNewlines)
                ?? "A calm female voice with clear pronunciation"
        }
        if let instructionText {
            return "\(base). \(instructionText)"
        }
        return base
    }

    private static func transcriptionSegments(
        from result: ASRResult
    ) -> [OpenAIAudioTranscriptionSegment]? {
        if let sentences = result.sentenceAlignments, !sentences.isEmpty {
            return sentences.enumerated().map { index, sentence in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: sentence.startSeconds,
                    end: sentence.endSeconds,
                    text: sentence.text
                )
            }
        }
        if let tokens = result.tokenAlignments, !tokens.isEmpty {
            return tokens.enumerated().map { index, token in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: token.startSeconds,
                    end: token.endSeconds,
                    text: token.text
                )
            }
        }
        return nil
    }

    private static func subtitleTimestamp(_ seconds: Double, separator: String) -> String {
        let milliseconds = max(0, Int((seconds * 1_000).rounded()))
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds % 3_600_000) / 60_000
        let secs = (milliseconds % 60_000) / 1_000
        let millis = milliseconds % 1_000
        return String(format: "%02d:%02d:%02d%@%03d", hours, minutes, secs, separator, millis)
    }
}

enum APIRequestValidationError: LocalizedError, Equatable {
    case invalidField(String, String)

    var errorDescription: String? {
        switch self {
        case .invalidField(let field, let reason):
            return "Invalid '\(field)': \(reason)."
        }
    }
}

struct MultipartFormData: Equatable, Sendable {
    struct Part: Equatable, Sendable {
        let name: String
        let filename: String?
        let contentType: String?
        let body: Data
    }

    enum ParseError: LocalizedError, Equatable {
        case missingBoundary
        case malformedBody
        case missingName

        var errorDescription: String? {
            switch self {
            case .missingBoundary:
                return "Missing multipart boundary."
            case .malformedBody:
                return "Malformed multipart body."
            case .missingName:
                return "Multipart part is missing a form-data name."
            }
        }
    }

    let parts: [Part]

    static func parse(body: Data, boundary: String?) throws -> MultipartFormData {
        guard let boundary, !boundary.isEmpty else {
            throw ParseError.missingBoundary
        }
        let marker = Data("--\(boundary)".utf8)
        guard !marker.isEmpty,
              let firstMarker = body.range(of: marker, options: [], in: body.startIndex..<body.endIndex) else {
            throw ParseError.malformedBody
        }

        var parts: [Part] = []
        var markerRange = firstMarker
        while true {
            let afterMarker = markerRange.upperBound
            if body.hasBytes(Data("--".utf8), at: afterMarker) {
                break
            }
            let partStart = body.indexAfterLineBreak(at: afterMarker)
            guard let nextMarker = body.range(of: marker, options: [], in: partStart..<body.endIndex) else {
                throw ParseError.malformedBody
            }
            let partEnd = body.indexTrimmingLineBreak(before: nextMarker.lowerBound)
            if partStart < partEnd {
                parts.append(try parsePart(Data(body[partStart..<partEnd])))
            }
            markerRange = nextMarker
        }

        return MultipartFormData(parts: parts)
    }

    func field(_ name: String) -> String? {
        guard let part = parts.first(where: { $0.name == name && $0.filename == nil }) else {
            return nil
        }
        return String(data: part.body, encoding: .utf8)
    }

    func file(named name: String) -> Part? {
        parts.first { $0.name == name && $0.filename != nil }
    }

    func files(named name: String) -> [Part] {
        parts.filter { $0.name == name && $0.filename != nil }
    }

    private static func parsePart(_ data: Data) throws -> Part {
        let separator = Data("\r\n\r\n".utf8)
        let fallbackSeparator = Data("\n\n".utf8)
        let separatorRange = data.range(of: separator, options: [], in: data.startIndex..<data.endIndex)
            ?? data.range(of: fallbackSeparator, options: [], in: data.startIndex..<data.endIndex)
        guard let separatorRange,
              let headerText = String(data: data[data.startIndex..<separatorRange.lowerBound], encoding: .utf8) else {
            throw ParseError.malformedBody
        }
        let body = Data(data[separatorRange.upperBound..<data.endIndex])
        let headers = parseHeaders(headerText)
        guard let disposition = headers["content-disposition"] else {
            throw ParseError.missingName
        }
        let params = parseDispositionParameters(disposition)
        guard let name = params["name"], !name.isEmpty else {
            throw ParseError.missingName
        }
        return Part(
            name: name,
            filename: params["filename"],
            contentType: headers["content-type"],
            body: body
        )
    }

    private static func parseHeaders(_ text: String) -> [String: String] {
        var headers: [String: String] = [:]
        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[name] = value
        }
        return headers
    }

    private static func parseDispositionParameters(_ value: String) -> [String: String] {
        var params: [String: String] = [:]
        for part in value.split(separator: ";", omittingEmptySubsequences: false).dropFirst() {
            let pair = part.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            var rawValue = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if rawValue.hasPrefix("\""), rawValue.hasSuffix("\""), rawValue.count >= 2 {
                rawValue.removeFirst()
                rawValue.removeLast()
            }
            params[key] = rawValue
        }
        return params
    }
}

private extension Data {
    func hasBytes(_ bytes: Data, at index: Data.Index) -> Bool {
        guard index >= startIndex, index + bytes.count <= endIndex else {
            return false
        }
        return self[index..<(index + bytes.count)].elementsEqual(bytes)
    }

    func indexAfterLineBreak(at index: Data.Index) -> Data.Index {
        if hasBytes(Data("\r\n".utf8), at: index) {
            return index + 2
        }
        if hasBytes(Data("\n".utf8), at: index) {
            return index + 1
        }
        return index
    }

    func indexTrimmingLineBreak(before index: Data.Index) -> Data.Index {
        if index >= 2, self[(index - 2)..<index].elementsEqual(Data("\r\n".utf8)) {
            return index - 2
        }
        if index >= 1, self[(index - 1)..<index].elementsEqual(Data("\n".utf8)) {
            return index - 1
        }
        return index
    }
}

// MARK: - Server Implementation

actor CodeGenServer {
    private let apiKey: String?
    private let fallbackLoraPath: String?
    private let defaultModelID: String
    private let contextSize: Int
    private let requestLimiter: APIRateLimiter
    private let requestAdmission: RuntimeRequestAdmission
    private let pool: RuntimeModelPool
    private var embeddingModels: [String: Qwen3EmbeddingModel] = [:]

    init(
        defaultModelID: String,
        modelPath: String?,
        fallbackLoraPath: String?,
        apiKey: String?,
        rateLimitPerMinute: Int,
        maxActiveRequests: Int = 1,
        engine: APIEngine,
        contextSize: Int = 32768,
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        memoryPressurePolicy: RuntimeMemoryPressurePolicy = .default
    ) async throws {
        self.apiKey = apiKey
        self.contextSize = contextSize
        self.fallbackLoraPath = fallbackLoraPath
        self.defaultModelID = defaultModelID
        self.requestLimiter = APIRateLimiter(limitPerMinute: rateLimitPerMinute)
        let runtimePool = RuntimeModelPool(
            defaultModelID: defaultModelID,
            defaultEngine: engine.runtimeServingEngine,
            startupModelPath: modelPath,
            gemma4KVCacheQuantization: gemma4KVCacheQuantization,
            memoryPressurePolicy: memoryPressurePolicy
        )
        self.pool = runtimePool
        self.requestAdmission = RuntimeRequestAdmission(
            maxActiveRequests: maxActiveRequests,
            pressureProvider: {
                await runtimePool.currentMemoryPressure()
            }
        )
        try await pool.preloadDefault()
    }

    func run(host: String, port: Int) async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        print("Starting server at http://\(host):\(port)")
        print("OpenAI-compatible base URL: http://\(host):\(port)/v1")
        print("Chat endpoint: http://\(host):\(port)/v1/chat/completions")
        print("Embeddings endpoint: http://\(host):\(port)/v1/embeddings")
        print("Images endpoint: http://\(host):\(port)/v1/images/generations")
        print("Image edits endpoint: http://\(host):\(port)/v1/images/edits")
        print("Speech endpoint: http://\(host):\(port)/v1/audio/speech")
        print("Transcriptions endpoint: http://\(host):\(port)/v1/audio/transcriptions")
        print("Press Ctrl+C to stop.")

        try await app.runService()
    }

    nonisolated func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()

        // Health check
        router.get("/health") { _, _ in
            let data = try JSONEncoder().encode(APIServerContract.healthStatus())
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(bytes: data))
            )
        }

        // List models
        router.get("/v1/models") { [self] request, _ in
            return try await self.handleModels(request)
        }

        // Chat completions
        router.post("/v1/chat/completions") { [self] request, _ in
            return try await self.handleChatCompletions(request)
        }

        // Embeddings
        router.post("/v1/embeddings") { [self] request, _ in
            return try await self.handleEmbeddings(request)
        }

        router.post("/v1/images/generations") { [self] request, _ in
            return try await self.handleImageGenerations(request)
        }

        router.post("/v1/images/edits") { [self] request, _ in
            return try await self.handleImageEdits(request)
        }

        router.post("/v1/audio/speech") { [self] request, _ in
            return try await self.handleAudioSpeech(request)
        }

        router.post("/v1/audio/transcriptions") { [self] request, _ in
            return try await self.handleAudioTranscriptions(request)
        }

        router.get("/runtime/status") { [self] request, _ in
            return try await self.handleRuntimeStatus(request)
        }

        router.post("/runtime/models/:id/load") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeLoad(request, id: id)
        }

        router.post("/runtime/models/:id/unload") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeUnload(request, id: id)
        }

        router.get("/runtime/models/:id/settings") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeSettings(request, id: id)
        }

        router.patch("/runtime/models/:id/settings") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeSettingsPatch(request, id: id)
        }

        return router
    }

    private func handleModels(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        var models = try await pool.modelsResponse()
        for modelID in APIServerContract.companionModelIDs()
            where !models.data.contains(where: { $0.id == modelID }) {
            models.data.append(
                OpenAIModel(
                    id: modelID,
                    object: "model",
                    created: Int(Date().timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            )
        }
        models.data.sort { $0.id < $1.id }

        let data = try JSONEncoder().encode(models)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func handleChatCompletions(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        // Decode request body
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024) // 10MB limit
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        let openaiRequest: OpenAIChatRequest
        do {
            openaiRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.readableBytesView))
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request payload.", type: "invalid_request_error")
        }

        let admissionLease = try await requestAdmission.acquire()
        let plan: RuntimeChatPlan
        do {
            plan = try await pool.makeChatPlan(
                for: openaiRequest,
                fallbackLoraPath: fallbackLoraPath,
                serverContextSize: contextSize
            )
        } catch {
            await admissionLease.release()
            return runtimeErrorResponse(error)
        }

        if plan.engine == .textChatDeepseekV4Flash {
            do {
                return try await proxyDeepseekV4FlashChatCompletions(
                    body: body,
                    contentType: request.headers[.contentType],
                    lease: plan.lease,
                    admissionLease: admissionLease
                )
            } catch {
                await plan.lease.release()
                await admissionLease.release()
                return makeErrorResponse(status: .internalServerError, message: "Request failed.", type: "server_error")
            }
        }

        do {
            if openaiRequest.stream == true {
                return try await handleStreamingChat(
                    plan.request,
                    modelID: plan.modelID,
                    includeUsage: plan.includeUsage,
                    lease: plan.lease,
                    admissionLease: admissionLease
                )
            } else {
                return try await handleNonStreamingChat(
                    plan.request,
                    modelID: plan.modelID,
                    lease: plan.lease,
                    admissionLease: admissionLease
                )
            }
        } catch let error as APIRequestValidationError {
            await plan.lease.release()
            await admissionLease.release()
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        } catch {
            await plan.lease.release()
            await admissionLease.release()
            return makeErrorResponse(status: .internalServerError, message: "Request failed.", type: "server_error")
        }
    }

    private func handleEmbeddings(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        let openaiRequest: OpenAIEmbeddingRequest
        do {
            openaiRequest = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: Data(body.readableBytesView))
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request payload.", type: "invalid_request_error")
        }

        do {
            let texts = try APIServerContract.embeddingTexts(from: openaiRequest)
            let resolved = try await embeddingModel(for: openaiRequest.model)
            let result = try resolved.model.embed(texts: texts)
            let response = APIServerContract.embeddingResponse(
                modelId: resolved.modelID,
                embeddings: result.embeddings,
                tokenCounts: result.tokenCounts
            )
            return try jsonResponse(response)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleImageGenerations(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let openaiRequest = try JSONDecoder().decode(
                OpenAIImageGenerationRequest.self,
                from: Data(body.readableBytesView)
            )
            let plan = try APIServerContract.imageGenerationPlan(from: openaiRequest)
            let outputURL = try await generateImage(plan)
            let response = try APIServerContract.imageResponse(outputURL: outputURL, plan: plan)
            return try jsonResponse(response)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleImageEdits(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 100 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let form = try MultipartFormData.parse(body: Data(body.readableBytesView), boundary: boundary)
            let imageFiles = (form.files(named: "image") + form.files(named: "image[]"))
                .filter { !$0.body.isEmpty }
            guard !imageFiles.isEmpty else {
                throw APIRequestValidationError.invalidField("image", "image file is required")
            }
            let inputImageURLs = try imageFiles.map {
                try writeMultipartFile($0, directoryName: "mere-run-api-image-edits")
            }
            let maskImageURL = try form.file(named: "mask").flatMap { mask -> URL? in
                guard !mask.body.isEmpty else { return nil }
                return try writeMultipartFile(mask, directoryName: "mere-run-api-image-edits")
            }
            let plan = try APIServerContract.imageEditPlan(
                from: form,
                inputImageURLs: inputImageURLs,
                maskImageURL: maskImageURL
            )
            let outputURL = try await generateImage(plan)
            let response = try APIServerContract.imageResponse(outputURL: outputURL, plan: plan)
            return try jsonResponse(response)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleAudioSpeech(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let openaiRequest = try JSONDecoder().decode(
                OpenAIAudioSpeechRequest.self,
                from: Data(body.readableBytesView)
            )
            let plan = try APIServerContract.speechPlan(from: openaiRequest)
            let outputURL = try await synthesizeSpeech(plan)
            let responseURL = try speechResponseURL(outputURL, responseFormat: plan.responseFormat)
            let data = try Data(contentsOf: responseURL)
            return binaryResponse(data, contentType: APIServerContract.speechContentType(for: plan.responseFormat))
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleAudioTranscriptions(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 100 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let form = try MultipartFormData.parse(body: Data(body.readableBytesView), boundary: boundary)
            let plan = try APIServerContract.transcriptionPlan(from: form)
            guard let file = form.file(named: "file"), !file.body.isEmpty else {
                throw APIRequestValidationError.invalidField("file", "audio file is required")
            }
            let audioURL = try writeMultipartFile(file, directoryName: "mere-run-api-audio")
            let result = try await transcribeAudio(audioURL: audioURL, plan: plan)
            switch plan.responseFormat {
            case "text":
                return binaryResponse(Data(result.text.utf8), contentType: "text/plain; charset=utf-8")
            case "srt", "vtt":
                let subtitle = APIServerContract.transcriptionSubtitle(from: result, format: plan.responseFormat)
                let contentType = plan.responseFormat == "srt"
                    ? "application/x-subrip; charset=utf-8"
                    : "text/vtt; charset=utf-8"
                return binaryResponse(Data(subtitle.utf8), contentType: contentType)
            case "verbose_json":
                return try jsonResponse(
                    APIServerContract.transcriptionResponse(from: result, verbose: true)
                )
            default:
                return try jsonResponse(
                    APIServerContract.transcriptionResponse(from: result, verbose: false)
                )
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleNonStreamingChat(
        _ request: ChatRequest,
        modelID: String,
        lease: RuntimeModelLease,
        admissionLease: RuntimeRequestAdmissionLease
    ) async throws -> Response {
        defer {
            Task {
                await lease.release()
                await admissionLease.release()
            }
        }
        let result = try await lease.chat(request, progressHandler: nil)

        let response = OpenAIChatResponse(
            id: "chatcmpl-\(UUID().uuidString.prefix(8))",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelID,
            choices: [
                OpenAIChatChoice(
                    index: 0,
                    message: OpenAIChatMessage(
                        role: "assistant",
                        content: result.response,
                        reasoning_content: result.reasoningContent,
                        tool_calls: openAIToolCalls(from: result.toolCalls)
                    ),
                    finish_reason: openAIFinishReason(for: result)
                )
            ],
            usage: OpenAIUsage(
                prompt_tokens: 0,
                completion_tokens: result.tokensGenerated,
                total_tokens: result.tokensGenerated
            )
        )

        let data = try JSONEncoder().encode(response)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func embeddingModel(for requestedModel: String) async throws -> (modelID: String, model: Qwen3EmbeddingModel) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if let spec = ManagedModelCatalog.spec(for: normalized),
           spec.id != Qwen3EmbeddingCatalog.modelId {
            throw APIRequestValidationError.invalidField(
                "model",
                "use \(Qwen3EmbeddingCatalog.modelId) or a local Qwen3 embedding model path"
            )
        }

        if let model = embeddingModels[normalized] {
            let modelID = ManagedModelCatalog.spec(for: normalized)?.id ?? normalized
            return (modelID, model)
        }

        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: normalized,
            defaultModelID: Qwen3EmbeddingCatalog.modelId,
            progress: nil
        )
        let modelID = resolution.source == .explicitPath ? normalized : resolution.spec.id
        let model = try Qwen3EmbeddingModel(
            resources: Qwen3EmbeddingResources(rootURL: resolution.url)
        )
        embeddingModels[normalized] = model
        if modelID != normalized {
            embeddingModels[modelID] = model
        }
        return (modelID, model)
    }

    private func generateImage(_ plan: APIServerContract.ImageGenerationPlan) async throws -> URL {
        try MLXBundleSupport.ensureAvailable(quiet: true)
        if plan.inputImage != nil,
           QwenImageEditRepository.canonicalModelId(for: plan.modelID) != nil {
            return try await generateQwenImageEdit(plan)
        }
        let resolved = try resolveImageModel(plan.modelID)
        let effectiveSteps = plan.steps
            ?? ((resolved.manifest.family == .hidream || resolved.manifest.family == .krea || resolved.manifest.family == .ideogram)
                ? (resolved.manifest.defaults?.steps ?? 4)
                : 4)
        let effectiveCFG = plan.guidanceScale
            ?? ((resolved.manifest.family == .hidream || resolved.manifest.family == .krea || resolved.manifest.family == .ideogram)
                ? (resolved.manifest.defaults?.cfg ?? 1.0)
                : 1.0)
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-images", extension: "png")
        let request = GenerationRequest(
            prompt: plan.prompt,
            negativePrompt: plan.negativePrompt,
            referenceImages: plan.additionalInputImages,
            width: plan.width,
            height: plan.height,
            steps: effectiveSteps,
            guidanceScale: effectiveCFG,
            seed: plan.seed,
            outputURL: outputURL,
            model: resolved.rootURL.path,
            maxSequenceLength: 512,
            lora: nil,
            enhancePrompt: false,
            inputImage: plan.inputImage,
            strength: plan.strength ?? 0.75,
            keepOriginalAspect: false,
            useBetaSigmas: false,
            sigmaShift: resolved.manifest.defaults?.sigmaShift.map { Float($0) }
        )

        switch resolved.manifest.family {
        case .klein:
            _ = try await Flux2KleinGenerator().generate(request, progressHandler: nil)
        case .zimage:
            _ = try await ZImageTurboGenerator().generate(request, progressHandler: nil)
        case .hidream:
            let generator = HiDreamO1Generator()
            defer { generator.unload() }
            _ = try await generator.generate(request, progressHandler: nil)
        case .krea:
            let generator = Krea2Generator()
            defer { generator.unload() }
            _ = try await generator.generate(request, progressHandler: nil)
        case .ideogram:
            let generator = Ideogram4Generator()
            defer { generator.unload() }
            _ = try await generator.generate(request, progressHandler: nil)
        case .gemma, .liquid, .qwen, .sam, .falcon, .tts, .asr, .embed, .code, .ocr, .music, .sfx, .video, .psi, .privacy, .deepseek, nil:
            throw APIRequestValidationError.invalidField(
                "model",
                "model \(resolved.modelID) is not an image generation model"
            )
        }
        return outputURL
    }

    private func generateQwenImageEdit(_ plan: APIServerContract.ImageGenerationPlan) async throws -> URL {
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-images", extension: "png")
        let request = GenerationRequest(
            prompt: plan.prompt,
            negativePrompt: plan.negativePrompt,
            width: plan.width,
            height: plan.height,
            steps: plan.steps ?? 20,
            guidanceScale: plan.guidanceScale ?? 4.0,
            seed: plan.seed,
            outputURL: outputURL,
            model: plan.modelID,
            maxSequenceLength: 512,
            inputImage: plan.inputImage,
            strength: plan.strength ?? 0.75
        )
        let generator = QwenImageEditGenerator()
        _ = try await generator.generate(request, progressHandler: nil)
        return outputURL
    }

    private func synthesizeSpeech(_ plan: APIServerContract.SpeechPlan) async throws -> URL {
        try MLXBundleSupport.ensureAvailable(quiet: true)
        let selection = try resolveSpeechModel(plan.modelID)
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-speech", extension: "wav")
        let request = TTSRequest(
            text: plan.input,
            voiceDescription: plan.voiceDescription,
            voiceMode: .style,
            cloneReference: nil,
            language: "auto",
            speed: plan.speed,
            temperature: plan.temperature,
            outputURL: outputURL
        )
        let generator = Qwen3TTSGenerator(modelId: selection.modelID)
        _ = try await generator.generate(
            request,
            modelPath: selection.modelPath,
            progressHandler: nil
        )
        return outputURL
    }

    private func speechResponseURL(_ wavURL: URL, responseFormat: String) throws -> URL {
        guard responseFormat != "wav" else {
            return wavURL
        }
        let outputURL = try temporaryOutputURL(
            directoryName: "mere-run-api-speech",
            extension: responseFormat
        )
        try MediaAudioIO.transcode(wavURL, to: outputURL, format: responseFormat)
        return outputURL
    }

    private func transcribeAudio(
        audioURL: URL,
        plan: APIServerContract.TranscriptionPlan
    ) async throws -> ASRResult {
        try MLXBundleSupport.ensureAvailable(quiet: true)
        let selection = try resolveTranscriptionModel(plan.modelID)
        let request = ASRRequest(
            audioURL: audioURL,
            language: plan.language,
            task: plan.task,
            maxTokens: plan.maxTokens
        )
        let execution = try await CLIASRRouting.transcribe(
            request: request,
            preferredBackend: selection.backend,
            modelOverride: selection.modelOverride,
            progressHandler: nil
        )
        return execution.result
    }

    private func resolveImageModel(
        _ requestedModel: String
    ) throws -> (modelID: String, rootURL: URL, manifest: MereRunModelManifest) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let asPath = URL(fileURLWithPath: normalized).standardizedFileURL
        if FileManager.default.fileExists(atPath: asPath.path) {
            let manifest = try MereRunModelManifest.loadRequired(from: asPath)
            return (manifest.id, asPath, manifest)
        }
        guard let modelID = ModelResolver.ModelID(rawValue: normalized) else {
            throw APIRequestValidationError.invalidField(
                "model",
                "use a mere.run image model id or a local model path"
            )
        }
        let resolution = try ModelResolver().resolve(modelID)
        let manifest = try MereRunModelManifest.loadRequired(from: resolution.rootURL)
        return (modelID.rawValue, resolution.rootURL, manifest)
    }

    private func resolveSpeechModel(
        _ requestedModel: String
    ) throws -> (modelID: String, modelPath: String?) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let asPath = URL(fileURLWithPath: normalized).standardizedFileURL
        if FileManager.default.fileExists(atPath: asPath.path) {
            return (Qwen3TTSResources.defaultModelId, asPath.path)
        }
        if let spec = ManagedModelCatalog.spec(for: normalized),
           spec.category == .speechTTS {
            return (spec.id, nil)
        }
        throw APIRequestValidationError.invalidField(
            "model",
            "use a mere.run TTS model id or a local Qwen3-TTS model path"
        )
    }

    private func resolveTranscriptionModel(
        _ requestedModel: String
    ) throws -> (modelOverride: String?, backend: ASRBackend) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let asPath = URL(fileURLWithPath: normalized).standardizedFileURL
        if FileManager.default.fileExists(atPath: asPath.path) {
            return (asPath.path, .auto)
        }
        guard let spec = ManagedModelCatalog.spec(for: normalized),
              spec.category == .speechASR else {
            throw APIRequestValidationError.invalidField(
                "model",
                "use a mere.run ASR model id or a local ASR model path"
            )
        }
        let backend: ASRBackend = spec.id.contains("parakeet") ? .parakeet : .qwen
        return (spec.id, backend)
    }

    private nonisolated func temporaryOutputURL(
        directoryName: String,
        extension pathExtension: String
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("\(UUID().uuidString).\(pathExtension)")
    }

    private nonisolated func writeMultipartFile(
        _ file: MultipartFormData.Part,
        directoryName: String
    ) throws -> URL {
        let pathExtension = sanitizedPathExtension(from: file.filename) ?? "wav"
        let outputURL = try temporaryOutputURL(directoryName: directoryName, extension: pathExtension)
        try file.body.write(to: outputURL)
        return outputURL
    }

    private nonisolated func sanitizedPathExtension(from filename: String?) -> String? {
        guard let rawExtension = filename.flatMap({ URL(fileURLWithPath: $0).pathExtension.lowercased() }),
              !rawExtension.isEmpty,
              rawExtension.allSatisfy({ $0.isLetter || $0.isNumber }) else {
            return nil
        }
        return rawExtension
    }

    private func handleRuntimeStatus(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let admission = await requestAdmission.snapshot()
        let data = try JSONEncoder().encode(await pool.status(admission: admission))
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func handleRuntimeLoad(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        do {
            let snapshot = try await pool.loadModel(idOrAlias: id)
            return try jsonResponse(snapshot)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeUnload(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        do {
            let snapshot = try await pool.unloadModel(idOrAlias: id)
            return try jsonResponse(snapshot)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeSettings(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        do {
            let settings = try await pool.settings(idOrAlias: id)
            return try jsonResponse(settings)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeSettingsPatch(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }
        do {
            let settings = try JSONDecoder().decode(RuntimeModelSettings.self, from: Data(body.readableBytesView))
            let updated = try await pool.updateSettings(idOrAlias: id, settings: settings)
            return try jsonResponse(updated)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func proxyDeepseekV4FlashChatCompletions(
        body: ByteBuffer,
        contentType: String?,
        lease: RuntimeModelLease,
        admissionLease: RuntimeRequestAdmissionLease
    ) async throws -> Response {
        let upstreamURL = try await lease.deepseekChatCompletionsURL(progressHandler: nil)
        let data = Data(body.readableBytesView)

        if !DeepseekV4FlashClient.requestWantsStreamingResponse(data) {
            let upstreamResponse: DeepseekV4FlashClientResponse
            do {
                upstreamResponse = try await DeepseekV4FlashClient.normalizedChatCompletionData(
                    url: upstreamURL,
                    requestBody: data,
                    contentType: contentType
                )
            } catch {
                await lease.release()
                await admissionLease.release()
                throw error
            }
            await lease.release()
            await admissionLease.release()
            var headers: HTTPFields = [:]
            headers[.contentType] = upstreamResponse.contentType
            return Response(
                status: .init(code: upstreamResponse.statusCode),
                headers: headers,
                body: .init(byteBuffer: ByteBuffer(bytes: upstreamResponse.body))
            )
        }

        let upstreamRequest = DeepseekV4FlashClient.makeChatCompletionsRequest(
            url: upstreamURL,
            requestBody: data,
            contentType: contentType
        )
#if os(Linux)
        let (upstreamData, response) = try await URLSession.shared.data(for: upstreamRequest)
        let http = response as? HTTPURLResponse
        var headers: HTTPFields = [:]
        headers[.contentType] = http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json"
        if DeepseekV4FlashClient.isEventStreamContentType(headers[.contentType]) {
            headers[.init("Cache-Control")!] = "no-cache"
            headers[.connection] = "keep-alive"
            let stream = AsyncStream<ByteBuffer> { continuation in
                if !upstreamData.isEmpty {
                    continuation.yield(ByteBuffer(bytes: upstreamData))
                }
                Task {
                    await lease.release()
                    await admissionLease.release()
                }
                continuation.finish()
            }
            return Response(
                status: .init(code: http?.statusCode ?? 502),
                headers: headers,
                body: .init(asyncSequence: stream)
            )
        }
        let repairedData = DeepseekV4FlashClient.normalizedChatCompletionBody(
            upstreamData,
            contentType: headers[.contentType]
        )
        await lease.release()
        await admissionLease.release()
        return Response(
            status: .init(code: http?.statusCode ?? 502),
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: repairedData))
        )
#else
        let (upstreamBytes, response) = try await URLSession.shared.bytes(for: upstreamRequest)
        let http = response as? HTTPURLResponse
        var headers: HTTPFields = [:]
        headers[.contentType] = http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json"
        if DeepseekV4FlashClient.isEventStreamContentType(headers[.contentType]) {
            headers[.init("Cache-Control")!] = "no-cache"
            headers[.connection] = "keep-alive"
            let stream = AsyncStream<ByteBuffer> { continuation in
                Task {
                    var buffer = Data()
                    buffer.reserveCapacity(4_096)
                    do {
                        for try await byte in upstreamBytes {
                            buffer.append(byte)
                            if byte == 10 || buffer.count >= 4_096 {
                                continuation.yield(ByteBuffer(bytes: buffer))
                                buffer.removeAll(keepingCapacity: true)
                            }
                        }
                        if !buffer.isEmpty {
                            continuation.yield(ByteBuffer(bytes: buffer))
                        }
                        await lease.release()
                        await admissionLease.release()
                        continuation.finish()
                    } catch {
                        await lease.release()
                        await admissionLease.release()
                        continuation.finish()
                    }
                }
            }
            return Response(
                status: .init(code: http?.statusCode ?? 502),
                headers: headers,
                body: .init(asyncSequence: stream)
            )
        }

        var upstreamData = Data()
        do {
            for try await byte in upstreamBytes {
                upstreamData.append(byte)
            }
        } catch {
            await lease.release()
            await admissionLease.release()
            throw error
        }
        await lease.release()
        await admissionLease.release()
        let repairedData = DeepseekV4FlashClient.normalizedChatCompletionBody(
            upstreamData,
            contentType: headers[.contentType]
        )
        return Response(
            status: .init(code: http?.statusCode ?? 502),
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: repairedData))
        )
#endif
    }

    private func handleStreamingChat(
        _ request: ChatRequest,
        modelID: String,
        includeUsage: Bool,
        lease: RuntimeModelLease,
        admissionLease: RuntimeRequestAdmissionLease
    ) async throws -> Response {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let encoder = JSONEncoder()

        // Create async stream for SSE
        let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()

        // Start generation in a detached task
        Task {
            do {
                let streamedContent = StreamingContentTracker()
                let shouldBufferForToolCalls = request.tools?.isEmpty == false
                let result = try await lease.chat(request) { progress in
                    guard !shouldBufferForToolCalls else { return }
                    if progress.stage == .generating,
                       let token = progress.message,
                       !token.isEmpty,
                       !APIServerContract.isStreamingStatusMessage(token) {
                        streamedContent.markStreamed()
                        let chunk = OpenAIChatResponse(
                            id: id,
                            object: "chat.completion.chunk",
                            created: Int(Date().timeIntervalSince1970),
                            model: modelID,
                            choices: [
                                OpenAIChatChoice(
                                    index: 0,
                                    delta: OpenAIChatDelta(content: token),
                                    finish_reason: nil
                                )
                            ]
                        )
                        if let data = try? encoder.encode(chunk),
                           let json = String(data: data, encoding: .utf8) {
                            let line = "data: \(json)\n\n"
                            continuation.yield(ByteBuffer(string: line))
                        }
                    }
                }
                if let toolCalls = openAIToolCalls(from: result.toolCalls), !toolCalls.isEmpty {
                    let chunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [
                            OpenAIChatChoice(
                                index: 0,
                                delta: OpenAIChatDelta(
                                    role: "assistant",
                                    tool_calls: toolCalls.enumerated().map(OpenAIChatToolCallDelta.init(indexAndToolCall:))
                                ),
                                finish_reason: nil
                            )
                        ]
                    )
                    if let data = try? encoder.encode(chunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                } else if !streamedContent.didStream, !result.response.isEmpty {
                    let chunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [
                            OpenAIChatChoice(
                                index: 0,
                                delta: OpenAIChatDelta(
                                    content: result.response,
                                    reasoning_content: result.reasoningContent
                                ),
                                finish_reason: nil
                            )
                        ]
                    )
                    if let data = try? encoder.encode(chunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }

                // Final chunk with finish_reason
                let finalChunk = OpenAIChatResponse(
                    id: id,
                    object: "chat.completion.chunk",
                    created: Int(Date().timeIntervalSince1970),
                    model: modelID,
                    choices: [
                        OpenAIChatChoice(
                            index: 0,
                            delta: OpenAIChatDelta(),
                            finish_reason: openAIFinishReason(for: result)
                        )
                    ]
                )
                if let data = try? encoder.encode(finalChunk),
                   let json = String(data: data, encoding: .utf8) {
                    continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                }

                if includeUsage {
                    let usageChunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [],
                        usage: OpenAIUsage(
                            prompt_tokens: 0,
                            completion_tokens: result.tokensGenerated,
                            total_tokens: result.tokensGenerated
                        )
                    )
                    if let data = try? encoder.encode(usageChunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }

                continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
                await lease.release()
                await admissionLease.release()
                continuation.finish()
            } catch {
                // Send error in SSE format
                let errorResponse = OpenAIErrorResponse(
                    error: OpenAIError(message: "Request failed.", type: "server_error")
                )
                if let data = try? encoder.encode(errorResponse),
                   let json = String(data: data, encoding: .utf8) {
                    continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                }
                await lease.release()
                await admissionLease.release()
                continuation.finish()
            }
        }

        return Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .init("Cache-Control")!: "no-cache",
                .connection: "keep-alive"
            ],
            body: .init(asyncSequence: stream)
        )
    }

    private nonisolated func openAIToolCalls(from toolCalls: [ToolCall]?) -> [OpenAIChatToolCall]? {
        guard let toolCalls, !toolCalls.isEmpty else { return nil }
        return toolCalls.enumerated().map { index, call in
            OpenAIChatToolCall(
                id: "call_\(index)_\(UUID().uuidString.prefix(8))",
                function: OpenAIChatToolCallFunction(
                    name: call.name,
                    arguments: jsonString(from: call.arguments)
                )
            )
        }
    }

    private nonisolated func openAIFinishReason(for result: ChatResponse) -> String {
        if result.toolCalls?.isEmpty == false {
            return "tool_calls"
        }
        return result.finishReason == .length ? "length" : "stop"
    }

    private nonisolated func jsonString(from arguments: [String: String]) -> String {
        let data = (try? JSONEncoder().encode(arguments)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private func unauthorizedResponseIfNeeded(for request: Request) -> Response? {
        guard let apiKey, !apiKey.isEmpty else { return nil }
        guard request.headers[.authorization] == "Bearer \(apiKey)" else {
            return makeErrorResponse(
                status: .unauthorized,
                message: "Unauthorized.",
                type: "authentication_error",
                extraHeaders: [.init("WWW-Authenticate")!: "Bearer"]
            )
        }
        return nil
    }

    private nonisolated func jsonResponse<T: Encodable>(
        _ payload: T,
        status: HTTPResponse.Status = .ok
    ) throws -> Response {
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private nonisolated func binaryResponse(_ data: Data, contentType: String) -> Response {
        Response(
            status: .ok,
            headers: [.contentType: contentType],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private nonisolated func runtimeErrorResponse(_ error: Error) -> Response {
        switch error {
        case let error as RuntimeModelPoolError:
            switch error {
            case .unknownModel, .unsupportedModel, .modelNotInstalled, .incompatibleEngine, .invalidSettings:
                return makeErrorResponse(
                    status: .badRequest,
                    message: error.localizedDescription,
                    type: "invalid_request_error"
                )
            case .unloadConflict:
                return makeErrorResponse(
                    status: .conflict,
                    message: error.localizedDescription,
                    type: "conflict_error"
                )
            case .rawProxyUnavailable:
                return makeErrorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    type: "server_error"
                )
            }
        case let error as APIRequestValidationError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as MultipartFormData.ParseError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as ManagedModelResolver.ResolverError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as Qwen3EmbeddingModel.EmbeddingError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        default:
            return makeErrorResponse(
                status: .internalServerError,
                message: "Request failed.",
                type: "server_error"
            )
        }
    }

    private nonisolated func makeErrorResponse(
        status: HTTPResponse.Status,
        message: String,
        type: String,
        extraHeaders: HTTPFields = [:]
    ) -> Response {
        let payload = OpenAIErrorResponse(error: OpenAIError(message: message, type: type))
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{\"error\":{\"message\":\"\(message)\",\"type\":\"\(type)\"}}".utf8)
        var headers: HTTPFields = [.contentType: "application/json"]
        for field in extraHeaders {
            headers.append(field)
        }
        return Response(
            status: status,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}

actor APIRateLimiter {
    private let limitPerMinute: Int
    private var requestTimes: [Date] = []

    init(limitPerMinute: Int) {
        self.limitPerMinute = limitPerMinute
    }

    func allowRequest(now: Date = Date()) -> Bool {
        let cutoff = now.addingTimeInterval(-60)
        requestTimes.removeAll { $0 < cutoff }
        guard requestTimes.count < limitPerMinute else {
            return false
        }
        requestTimes.append(now)
        return true
    }
}

final class StreamingContentTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var streamed = false

    var didStream: Bool {
        lock.lock()
        defer { lock.unlock() }
        return streamed
    }

    func markStreamed() {
        lock.lock()
        streamed = true
        lock.unlock()
    }
}
