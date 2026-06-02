import Foundation

public enum LoRA: Sendable, Hashable {
    case local(path: String, scale: Double)
    case remote(reference: String, scale: Double)
}

public struct GenerationRequest: Sendable, Hashable {
    public var prompt: String
    public var negativePrompt: String?
    /// Reference images for model families that support edit or personalization modes.
    public var referenceImages: [URL]
    /// Reference strength for FLUX.2 Klein editing (0.0 = preserve, 1.0 = max change).
    public var referenceStrength: Double
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidanceScale: Double
    public var seed: UInt64?
    public var outputURL: URL
    public var model: String?
    public var maxSequenceLength: Int
    public var lora: LoRA?
    public var enhancePrompt: Bool
    public var inputImage: URL?
    public var strength: Double
    /// Preserve the single reference image aspect ratio for model families that support it.
    public var keepOriginalAspect: Bool
    public var useBetaSigmas: Bool
    public var sigmaShift: Float?

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        referenceImages: [URL] = [],
        referenceStrength: Double = 0.0,
        width: Int,
        height: Int,
        steps: Int,
        guidanceScale: Double = 0,
        seed: UInt64? = nil,
        outputURL: URL,
        model: String? = nil,
        maxSequenceLength: Int = 512,
        lora: LoRA? = nil,
        enhancePrompt: Bool = false,
        inputImage: URL? = nil,
        strength: Double = 0.75,
        keepOriginalAspect: Bool = false,
        useBetaSigmas: Bool = false,
        sigmaShift: Float? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.referenceImages = referenceImages
        self.referenceStrength = referenceStrength
        self.width = width
        self.height = height
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.outputURL = outputURL
        self.model = model
        self.maxSequenceLength = maxSequenceLength
        self.lora = lora
        self.enhancePrompt = enhancePrompt
        self.inputImage = inputImage
        self.strength = strength
        self.keepOriginalAspect = keepOriginalAspect
        self.useBetaSigmas = useBetaSigmas
        self.sigmaShift = sigmaShift
    }
}

public enum GenerationStage: String, Sendable, Hashable {
    case loadingModel
    case loadingEncoder
    case encodingText
    case encodingReferenceImages
    case loadingTransformer
    case loadingLoRA
    case denoising
    case loadingVAE
    case decoding
    case saving
}

public struct GenerationProgress: Sendable, Hashable {
    public let stage: GenerationStage
    public let stepIndex: Int
    public let totalSteps: Int

    public init(stage: GenerationStage, stepIndex: Int, totalSteps: Int) {
        self.stage = stage
        self.stepIndex = stepIndex
        self.totalSteps = totalSteps
    }

    public var fractionCompleted: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(stepIndex) / Double(totalSteps)
    }
}

public struct GenerationResult: Sendable, Hashable {
    public let outputURL: URL
    public let seed: UInt64

    public init(outputURL: URL, seed: UInt64) {
        self.outputURL = outputURL
        self.seed = seed
    }
}

public protocol ImageGenerator {
    func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult
}

public extension ImageGenerator {
    func generate(_ request: GenerationRequest) async throws -> GenerationResult {
        try await generate(request, progressHandler: nil)
    }
}

// MARK: - Chat Types

public struct ChatMessage: Sendable, Hashable, Codable {
    public enum Role: String, Sendable, Hashable, Codable {
        case system
        case user
        case assistant
        case tool
    }

    public var role: Role
    public var content: String
    public var imageUrl: String?

    public init(role: Role, content: String, imageUrl: String? = nil) {
        self.role = role
        self.content = content
        self.imageUrl = imageUrl
    }
}

// MARK: - Tool Types

public struct ToolParameterProperty: Sendable, Hashable, Codable {
    public let type: String
    public let description: String

    public init(type: String, description: String) {
        self.type = type
        self.description = description
    }
}

public struct ToolDefinition: Sendable, Hashable, Codable {
    public let name: String
    public let description: String
    public let parameters: [String: ToolParameterProperty]
    public let required: [String]

    public init(name: String, description: String, parameters: [String: ToolParameterProperty], required: [String]? = nil) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required ?? Array(parameters.keys)
    }

    /// Convert to the ToolSpec format expected by swift-transformers applyChatTemplate.
    public func toToolSpec() -> [String: any Sendable] {
        var properties: [String: any Sendable] = [:]
        for (key, prop) in parameters {
            properties[key] = ["type": prop.type, "description": prop.description] as [String: String]
        }
        return [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": required,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable]
    }

    public func promptSchemaJSONString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(ToolPromptSchema(definition: self))
        return String(decoding: data, as: UTF8.self)
    }
}

private struct ToolPromptSchema: Encodable {
    let type = "function"
    let function: ToolFunctionSchema

    init(definition: ToolDefinition) {
        self.function = ToolFunctionSchema(definition: definition)
    }
}

private struct ToolFunctionSchema: Encodable {
    let name: String
    let description: String
    let parameters: ToolParametersSchema

    init(definition: ToolDefinition) {
        self.name = definition.name
        self.description = definition.description
        self.parameters = ToolParametersSchema(
            properties: definition.parameters,
            required: definition.required
        )
    }
}

private struct ToolParametersSchema: Encodable {
    let type = "object"
    let properties: [String: ToolParameterProperty]
    let required: [String]
}

public struct ToolCall: Sendable, Hashable {
    public let name: String
    public let arguments: [String: String]

    public init(name: String, arguments: [String: String]) {
        self.name = name
        self.arguments = arguments
    }
}

public struct ChatRequest: Sendable, Hashable {
    public var messages: [ChatMessage]
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double
    public var showThinking: Bool
    public var lora: LoRA?
    public var requiresJSON: Bool
    public var tools: [ToolDefinition]?
    public var stopOnEOS: Bool
    public var kvCacheMode: RuntimeKVCacheMode?
    public var maxContextTokens: Int?

    public init(
        messages: [ChatMessage],
        maxTokens: Int = 512,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        showThinking: Bool = true,
        lora: LoRA? = nil,
        requiresJSON: Bool = false,
        tools: [ToolDefinition]? = nil,
        stopOnEOS: Bool = true,
        kvCacheMode: RuntimeKVCacheMode? = nil,
        maxContextTokens: Int? = nil
    ) {
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.showThinking = showThinking
        self.lora = lora
        self.requiresJSON = requiresJSON
        self.tools = tools
        self.stopOnEOS = stopOnEOS
        self.kvCacheMode = kvCacheMode
        self.maxContextTokens = maxContextTokens
    }
}

public struct ChatTiming: Sendable, Hashable {
    public var loadSeconds: Double
    public var prefillSeconds: Double
    public var cacheConversionSeconds: Double?
    public var decodeSeconds: Double
    public var firstTokenSeconds: Double?
    public var kvCacheMode: RuntimeKVCacheMode?
    public var prefillKVCache: String?
    public var decodeKVCache: String?

    public init(
        loadSeconds: Double = 0,
        prefillSeconds: Double = 0,
        cacheConversionSeconds: Double? = nil,
        decodeSeconds: Double = 0,
        firstTokenSeconds: Double? = nil,
        kvCacheMode: RuntimeKVCacheMode? = nil,
        prefillKVCache: String? = nil,
        decodeKVCache: String? = nil
    ) {
        self.loadSeconds = loadSeconds
        self.prefillSeconds = prefillSeconds
        self.cacheConversionSeconds = cacheConversionSeconds
        self.decodeSeconds = decodeSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.kvCacheMode = kvCacheMode
        self.prefillKVCache = prefillKVCache
        self.decodeKVCache = decodeKVCache
    }
}

public struct ChatResponse: Sendable, Hashable {
    public var response: String
    public var tokensGenerated: Int
    public var timing: ChatTiming?
    public var toolCalls: [ToolCall]?
    public var promptTokens: Int?

    public init(
        response: String,
        tokensGenerated: Int,
        timing: ChatTiming? = nil,
        toolCalls: [ToolCall]? = nil,
        promptTokens: Int? = nil
    ) {
        self.response = response
        self.tokensGenerated = tokensGenerated
        self.timing = timing
        self.toolCalls = toolCalls
        self.promptTokens = promptTokens
    }
}

public enum ChatStage: String, Sendable, Hashable {
    case loadingModel
    case encoding
    case generating
}

public struct ChatProgress: Sendable, Hashable {
    public let stage: ChatStage
    public let message: String?

    public init(stage: ChatStage, message: String? = nil) {
        self.stage = stage
        self.message = message
    }
}

public protocol ChatGenerator {
    func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse
}
