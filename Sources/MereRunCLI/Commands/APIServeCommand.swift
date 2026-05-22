import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Hummingbird
import NIOCore
import MereRunCore

struct APIServe: AsyncParsableCommand {
    private static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start an OpenAI-compatible API server for local text-code or text-chat models.",
        discussion: """
        Runs an HTTP server that exposes an OpenAI-compatible API for the selected engine.
        Compatible with any OpenAI client (VS Code extensions, Continue, Cursor, etc.).

        Endpoints:
          GET  /health              - Health check
          GET  /v1/models           - List available models
          POST /v1/chat/completions - Chat completions (streaming supported)

        Example:
          # Start with the default local code model
          mere.run api serve

          # Start with a specific GGUF model
          mere.run api serve -m ~/models/Qwen3-Coder-Next-Q4_K_M.gguf

          # Start a Gemma 4 text-chat server
          mere.run api serve --engine text-chat-gemma4

          # Start a Q35 text-chat server with an explicit nano model root
          mere.run api serve --engine text-chat-q35 -m ~/Models/text-chat-q35-nano

          # Start the DeepSeek V4 Flash OpenAI-compatible server
          mere.run api serve --engine text-chat-deepseek-v4-flash

          # Custom host/port (non-loopback binds require an API key)
          export MERERUN_API_KEY=change-me
          mere.run api serve --host 0.0.0.0 --port 11434 --api-key "$MERERUN_API_KEY"

          # Check server health and the served model from another terminal
          mere.run status --port 11434

          # Test with curl (request.json contains an OpenAI chat payload)
          curl http://localhost:8080/v1/chat/completions \\
            -H "Content-Type: application/json" \\
            --data @request.json
        """
    )

    @Option(name: [.short, .long], help: "Port to listen on.")
    var port: Int = 8080

    @Option(name: [.long], help: "Host to bind to.")
    var host: String = "127.0.0.1"

    @Option(name: [.customShort("m"), .long, .customLong("model-path")], help: "Model path. For --engine text-code, pass a GGUF file. For --engine text-chat-klein, pass a Klein-root text chat model. For --engine text-chat-gemma4, pass a Gemma 4 model root or repo ID. For --engine text-chat-q35, pass a Q35 text chat model root. For --engine text-chat-deepseek-v4-flash, pass a DS4 GGUF file or managed model root.")
    var model: String?

    @Option(name: [.long], help: "Serving engine: text-code (default), text-chat-klein, text-chat-gemma4, text-chat-q35, or text-chat-deepseek-v4-flash.")
    var engine: APIEngine = .textCode

    @Option(name: [.long], help: "Default LoRA adapter path for all requests.")
    var lora: String?

    @Option(name: [.long], help: "Bearer token required by /v1/models and /v1/chat/completions. Also read from MERERUN_API_KEY.")
    var apiKey: String?

    @Option(name: [.long], help: "Global request limit for /v1/chat/completions per rolling minute.")
    var rateLimitPerMinute: Int = 60

    @Option(name: [.long], help: "Context size (default: 32768).")
    var contextSize: Int = 32768

    @Option(name: [.long], help: "Quantize the Gemma4 KV cache to this many bits. Supports integer widths for uniform and integer/.5 widths for turboquant.")
    var kvBits: Double?

    @Option(name: [.long], help: "Gemma4 KV cache quantization backend: uniform or turboquant.")
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
        let server = try await CodeGenServer(
            modelPath: resolvedModelPath,
            fallbackLoraPath: lora,
            apiKey: resolvedAPIKey,
            rateLimitPerMinute: rateLimitPerMinute,
            engine: engine,
            contextSize: contextSize,
            gemma4KVCacheQuantization: gemma4KVCacheQuantization
        )
        try await server.run(host: host, port: port)
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
        case .textChatQ35:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.q35) {
                return resolved.rootURL.path
            }
            if let resolved = ModelResolver().resolveIfPresent(.q35Nano) {
                return resolved.rootURL.path
            }
            // Allow Q35Generator to auto-download from Hugging Face when model path is omitted.
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
            throw ValidationError("Unsupported --kv-quant-scheme '\(raw)'. Expected 'uniform' or 'turboquant'.")
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
        guard (1...Int(Int32.max)).contains(contextSize) else {
            throw ValidationError("--context-size must be between 1 and \(Int(Int32.max)).")
        }
        guard Self.isLoopbackHost(host) || apiKey != nil else {
            throw ValidationError("Binding to non-loopback hosts requires --api-key or MERERUN_API_KEY.")
        }
    }

    private static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "127.0.0.1" || normalized == "localhost" || normalized == "::1"
    }
}

enum APIEngine: String, ExpressibleByArgument {
    case textCode = "text-code"
    case textChatKlein = "text-chat-klein"
    case textChatGemma4 = "text-chat-gemma4"
    case textChatQ35 = "text-chat-q35"
    case textChatDeepseekV4Flash = "text-chat-deepseek-v4-flash"

    var openAICompatibility: APIEngineCapabilities {
        switch self {
        case .textCode:
            return .localText
        case .textChatKlein:
            return .localTextWithStructuredJSON
        case .textChatGemma4:
            return .localTextWithTools
        case .textChatQ35:
            return .localTextWithToolsAndVision
        case .textChatDeepseekV4Flash:
            return .rawProxy
        }
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

    static let localTextWithStructuredJSON = APIEngineCapabilities(
        supportsStructuredOutputs: true
    )

    static let localTextWithTools = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true
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

    static func healthStatus() -> APIHealthStatus {
        APIHealthStatus(status: "ok")
    }

    static func modelsResponse(modelId: String, createdAt: Date = Date()) -> OpenAIModelsResponse {
        OpenAIModelsResponse(
            object: "list",
            data: [
                OpenAIModel(
                    id: modelId,
                    object: "model",
                    created: Int(createdAt.timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            ]
        )
    }

    static func chatRequest(
        from openaiRequest: OpenAIChatRequest,
        fallbackLoraPath: String?,
        contextSize: Int,
        capabilities: APIEngineCapabilities = .localText
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

        return ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            lora: lora,
            requiresJSON: requiresJSON,
            tools: tools
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

        switch request.tool_choice {
        case nil, .mode("auto")?, .mode("required")?:
            break
        case .mode("none")?:
            return nil
        case .function?, .custom?:
            throw APIRequestValidationError.invalidField(
                "tool_choice",
                "specific tool forcing is not supported by this engine"
            )
        case .mode(let value)?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported mode '\(value)'")
        }

        return try tools.map { try toolDefinition(from: $0) }
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

// MARK: - Server Implementation

actor CodeGenServer {
    private let apiKey: String?
    private let engine: APIEngine
    private let llamaGenerator: CodeGenGenerator?
    private let mlxGenerator: Flux2KleinGenerator?
    private let gemma4Generator: Gemma4Generator?
    private let q35Generator: Q35Generator?
    private let deepseekV4FlashGenerator: DeepseekV4FlashGenerator?
    private let modelPath: String?
    private let fallbackLoraPath: String?
    private let modelId: String
    private let contextSize: Int
    private let useStandaloneModel: Bool
    private let requestLimiter: APIRateLimiter

    init(
        modelPath: String?,
        fallbackLoraPath: String?,
        apiKey: String?,
        rateLimitPerMinute: Int,
        engine: APIEngine,
        contextSize: Int = 32768,
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization()
    ) async throws {
        self.apiKey = apiKey
        self.contextSize = contextSize
        self.engine = engine
        self.fallbackLoraPath = fallbackLoraPath
        self.modelPath = modelPath
        self.requestLimiter = APIRateLimiter(limitPerMinute: rateLimitPerMinute)
        self.modelId = modelPath.map { URL(fileURLWithPath: $0).lastPathComponent }
            ?? {
                switch engine {
                case .textChatKlein:
                    return ModelResolver.ModelID.mebot.rawValue
                case .textChatGemma4:
                    return ModelResolver.ModelID.gemma4.rawValue
                case .textChatQ35:
                    return ModelResolver.ModelID.q35.rawValue
                case .textCode:
                    return CodeGenResources.defaultModelId
                case .textChatDeepseekV4Flash:
                    return DeepseekV4FlashResources.defaultModelId
                }
            }()

        switch engine {
        case .textCode:
            let generator = CodeGenGenerator(modelId: modelPath ?? CodeGenResources.defaultModelId)
            self.llamaGenerator = generator
            self.mlxGenerator = nil
            self.gemma4Generator = nil
            self.q35Generator = nil
            self.deepseekV4FlashGenerator = nil
            self.useStandaloneModel = false

            // Pre-load model
            try await generator.prepare(modelPath: modelPath) { progress in
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
        case .textChatKlein:
            self.llamaGenerator = nil
            self.mlxGenerator = Flux2KleinGenerator()
            self.gemma4Generator = nil
            self.q35Generator = nil
            self.deepseekV4FlashGenerator = nil
            // Check if the resolved path is a standalone MeBot Instruct model
            self.useStandaloneModel = MeBotModelCatalog.resolveModelPath() != nil
                && modelPath == MeBotModelCatalog.resolveModelPath()
        case .textChatGemma4:
            let generator = Gemma4Generator(
                modelId: Gemma4Resources.defaultModelId,
                kvCacheQuantization: gemma4KVCacheQuantization
            )
            self.llamaGenerator = nil
            self.mlxGenerator = nil
            self.gemma4Generator = generator
            self.q35Generator = nil
            self.deepseekV4FlashGenerator = nil
            self.useStandaloneModel = false

            try await generator.prepare(modelPath: modelPath) { progress in
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
        case .textChatQ35:
            let generator = Q35Generator(modelId: Q35Resources.defaultModelId)
            self.llamaGenerator = nil
            self.mlxGenerator = nil
            self.gemma4Generator = nil
            self.q35Generator = generator
            self.deepseekV4FlashGenerator = nil
            self.useStandaloneModel = false

            try await generator.prepare(modelPath: modelPath) { progress in
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
        case .textChatDeepseekV4Flash:
            let generator = DeepseekV4FlashGenerator()
            self.llamaGenerator = nil
            self.mlxGenerator = nil
            self.gemma4Generator = nil
            self.q35Generator = nil
            self.deepseekV4FlashGenerator = generator
            self.useStandaloneModel = false

            try await generator.prepare(modelPath: modelPath) { progress in
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
        }
    }

    func run(host: String, port: Int) async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        print("Starting server at http://\(host):\(port)")
        print("OpenAI-compatible endpoint: http://\(host):\(port)/v1/chat/completions")
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

        return router
    }

    private func handleModels(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let models = APIServerContract.modelsResponse(modelId: modelId)

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

        if engine == .textChatDeepseekV4Flash {
            do {
                return try await proxyDeepseekV4FlashChatCompletions(body: body, contentType: request.headers[.contentType])
            } catch {
                return makeErrorResponse(status: .internalServerError, message: "Request failed.", type: "server_error")
            }
        }

        let openaiRequest: OpenAIChatRequest
        do {
            openaiRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.readableBytesView))
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request payload.", type: "invalid_request_error")
        }

        let chatRequest: ChatRequest
        do {
            chatRequest = try APIServerContract.chatRequest(
                from: openaiRequest,
                fallbackLoraPath: fallbackLoraPath,
                contextSize: contextSize,
                capabilities: engine.openAICompatibility
            )
        } catch {
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        }

        do {
            if openaiRequest.stream == true {
                let includeUsage = try APIServerContract.includeUsageInStreaming(
                    openaiRequest,
                    capabilities: engine.openAICompatibility
                )
                return try await handleStreamingChat(chatRequest, includeUsage: includeUsage)
            } else {
                return try await handleNonStreamingChat(chatRequest)
            }
        } catch let error as APIRequestValidationError {
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        } catch {
            return makeErrorResponse(status: .internalServerError, message: "Request failed.", type: "server_error")
        }
    }

    private func handleNonStreamingChat(_ request: ChatRequest) async throws -> Response {
        let result = try await generateChat(request, progressHandler: nil)

        let response = OpenAIChatResponse(
            id: "chatcmpl-\(UUID().uuidString.prefix(8))",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelId,
            choices: [
                OpenAIChatChoice(
                    index: 0,
                    message: OpenAIChatMessage(
                        role: "assistant",
                        content: result.response,
                        tool_calls: openAIToolCalls(from: result.toolCalls)
                    ),
                    finish_reason: result.toolCalls?.isEmpty == false ? "tool_calls" : "stop"
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

    private func proxyDeepseekV4FlashChatCompletions(body: ByteBuffer, contentType: String?) async throws -> Response {
        guard let generator = deepseekV4FlashGenerator else {
            throw DeepseekV4FlashError.serverFailedToStart("generator not initialized")
        }
        let upstreamURL = try await generator.chatCompletionsURL(modelPath: modelPath, progressHandler: nil)
        let data = Data(body.readableBytesView)

        if !DeepseekV4FlashClient.requestWantsStreamingResponse(data) {
            let upstreamResponse = try await DeepseekV4FlashClient.normalizedChatCompletionData(
                url: upstreamURL,
                requestBody: data,
                contentType: contentType
            )
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
                        continuation.finish()
                    } catch {
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
        for try await byte in upstreamBytes {
            upstreamData.append(byte)
        }
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

    private func handleStreamingChat(_ request: ChatRequest, includeUsage: Bool) async throws -> Response {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let encoder = JSONEncoder()

        // Create async stream for SSE
        let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()

        // Start generation in a detached task
        Task { [self, modelId] in
            do {
                let streamedContent = StreamingContentTracker()
                let shouldBufferForToolCalls = request.tools?.isEmpty == false
                let result = try await self.generateChat(request) { progress in
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
                            model: modelId,
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
                        model: modelId,
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
                        model: modelId,
                        choices: [
                            OpenAIChatChoice(
                                index: 0,
                                delta: OpenAIChatDelta(content: result.response),
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
                    model: modelId,
                    choices: [
                        OpenAIChatChoice(
                            index: 0,
                            delta: OpenAIChatDelta(),
                            finish_reason: result.toolCalls?.isEmpty == false ? "tool_calls" : "stop"
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
                        model: modelId,
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

    private nonisolated func jsonString(from arguments: [String: String]) -> String {
        let data = (try? JSONEncoder().encode(arguments)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private func generateChat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        switch engine {
        case .textCode:
            guard let generator = llamaGenerator else {
                throw CodeGenError.modelNotLoaded
            }
            return try await generator.chat(request, progressHandler: progressHandler)
        case .textChatKlein:
            guard let generator = mlxGenerator else {
                throw Flux2Error.modelsNotLoaded
            }
            guard let modelPath else {
                throw Flux2Error.modelNotFound(ModelResolver.ModelID.mebot.rawValue)
            }
            if useStandaloneModel {
                return try await generator.chatStandalone(request, modelPath: modelPath, progressHandler: progressHandler)
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatGemma4:
            guard let generator = gemma4Generator else {
                throw Gemma4Error.modelNotLoaded
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatQ35:
            guard let generator = q35Generator else {
                throw Q35Error.modelNotLoaded
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDeepseekV4Flash:
            guard let generator = deepseekV4FlashGenerator else {
                throw DeepseekV4FlashError.serverFailedToStart("generator not initialized")
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        }
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

    private func makeErrorResponse(
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
