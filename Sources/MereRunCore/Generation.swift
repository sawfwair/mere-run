import Foundation

public enum LoRA: Sendable, Hashable {
    case local(path: String, scale: Double)
    case remote(reference: String, scale: Double)
}

public struct Krea2ConditioningRebalance: Sendable, Hashable {
    public var multiplier: Float
    public var layerWeights: [Float]

    public init(multiplier: Float = 1, layerWeights: [Float] = []) {
        self.multiplier = multiplier
        self.layerWeights = layerWeights
    }
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
    public var kreaConditioningRebalance: Krea2ConditioningRebalance?
    public var kreaBaseQuantizationBits: Int?

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
        sigmaShift: Float? = nil,
        kreaConditioningRebalance: Krea2ConditioningRebalance? = nil,
        kreaBaseQuantizationBits: Int? = nil
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
        self.kreaConditioningRebalance = kreaConditioningRebalance
        self.kreaBaseQuantizationBits = kreaBaseQuantizationBits
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

public struct ChatMessageToolCall: Sendable, Hashable, Codable {
    public var id: String?
    public var name: String
    public var arguments: [String: OpenAIJSONValue]

    public init(id: String? = nil, name: String, arguments: [String: OpenAIJSONValue]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

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
    public var reasoningContent: String?
    public var name: String?
    public var toolCallID: String?
    public var toolCalls: [ChatMessageToolCall]?

    public init(
        role: Role,
        content: String,
        imageUrl: String? = nil,
        reasoningContent: String? = nil,
        name: String? = nil,
        toolCallID: String? = nil,
        toolCalls: [ChatMessageToolCall]? = nil
    ) {
        self.role = role
        self.content = content
        self.imageUrl = imageUrl
        self.reasoningContent = reasoningContent
        self.name = name
        self.toolCallID = toolCallID
        self.toolCalls = toolCalls
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
    /// Top-k sampling cutoff; nil leaves the engine's default (no cutoff).
    public var topK: Int?
    /// Min-p sampling cutoff relative to the most likely token; zero disables it.
    public var minP: Double
    /// Model-specific reasoning budget. Inkling-Small accepts values from 0 through 0.99.
    public var reasoningEffort: Double?
    public var showThinking: Bool
    public var lora: LoRA?
    public var requiresJSON: Bool
    public var tools: [ToolDefinition]?
    public var stopOnEOS: Bool
    public var stopSequences: [String]
    /// Prevent generation of an n-gram already present in the full prompt and decode history.
    public var noRepeatNgramSize: Int?
    public var kvCacheMode: RuntimeKVCacheMode?
    public var maxContextTokens: Int?

    public init(
        messages: [ChatMessage],
        maxTokens: Int = 512,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        topK: Int? = nil,
        minP: Double = 0,
        reasoningEffort: Double? = nil,
        showThinking: Bool = true,
        lora: LoRA? = nil,
        requiresJSON: Bool = false,
        tools: [ToolDefinition]? = nil,
        stopOnEOS: Bool = true,
        stopSequences: [String] = [],
        noRepeatNgramSize: Int? = nil,
        kvCacheMode: RuntimeKVCacheMode? = nil,
        maxContextTokens: Int? = nil
    ) {
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.reasoningEffort = reasoningEffort
        self.showThinking = showThinking
        self.lora = lora
        self.requiresJSON = requiresJSON
        self.tools = tools
        self.stopOnEOS = stopOnEOS
        self.stopSequences = stopSequences
        self.noRepeatNgramSize = noRepeatNgramSize
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
    public var prefillTokensPerSecond: Double?
    public var decodeTokensPerSecond: Double?

    public init(
        loadSeconds: Double = 0,
        prefillSeconds: Double = 0,
        cacheConversionSeconds: Double? = nil,
        decodeSeconds: Double = 0,
        firstTokenSeconds: Double? = nil,
        kvCacheMode: RuntimeKVCacheMode? = nil,
        prefillKVCache: String? = nil,
        decodeKVCache: String? = nil,
        prefillTokensPerSecond: Double? = nil,
        decodeTokensPerSecond: Double? = nil
    ) {
        self.loadSeconds = loadSeconds
        self.prefillSeconds = prefillSeconds
        self.cacheConversionSeconds = cacheConversionSeconds
        self.decodeSeconds = decodeSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.kvCacheMode = kvCacheMode
        self.prefillKVCache = prefillKVCache
        self.decodeKVCache = decodeKVCache
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
    }
}

public struct ChatResponse: Sendable, Hashable {
    public var response: String
    public var tokensGenerated: Int
    public var timing: ChatTiming?
    public var toolCalls: [ToolCall]?
    public var promptTokens: Int?
    public var finishReason: ChatFinishReason?
    public var reasoningContent: String?
    public var hasIncompleteReasoning: Bool
    public var reasoningBlockCount: Int
    public var hasReopenedReasoning: Bool

    public init(
        response: String,
        tokensGenerated: Int,
        timing: ChatTiming? = nil,
        toolCalls: [ToolCall]? = nil,
        promptTokens: Int? = nil,
        finishReason: ChatFinishReason? = nil,
        reasoningContent: String? = nil,
        hasIncompleteReasoning: Bool = false,
        reasoningBlockCount: Int = 0,
        hasReopenedReasoning: Bool = false
    ) {
        self.response = response
        self.tokensGenerated = tokensGenerated
        self.timing = timing
        self.toolCalls = toolCalls
        self.promptTokens = promptTokens
        self.finishReason = finishReason
        let trimmedReasoning = reasoningContent?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.reasoningContent = trimmedReasoning?.nilIfEmpty
        self.hasIncompleteReasoning = hasIncompleteReasoning
        self.reasoningBlockCount = reasoningBlockCount
        self.hasReopenedReasoning = hasReopenedReasoning
    }

    public init(
        generatedText: String,
        tokensGenerated: Int,
        showThinking: Bool,
        timing: ChatTiming? = nil,
        toolCalls: [ToolCall]? = nil,
        promptTokens: Int? = nil,
        finishReason: ChatFinishReason? = nil
    ) {
        let split = ChatReasoningMarkup.splitThinkBlocks(in: generatedText)
        let visibleResponse = showThinking
            ? generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            : split.visibleContent
        self.init(
            response: visibleResponse,
            tokensGenerated: tokensGenerated,
            timing: timing,
            toolCalls: toolCalls,
            promptTokens: promptTokens,
            finishReason: finishReason,
            reasoningContent: split.reasoningContent,
            hasIncompleteReasoning: split.hasIncompleteReasoning,
            reasoningBlockCount: split.reasoningBlockCount,
            hasReopenedReasoning: split.hasReopenedReasoning
        )
    }
}

public enum ChatFinishReason: String, Sendable, Hashable {
    case stop
    case stopSequence = "stop_sequence"
    case length
}

public enum TextGenerationStopSequences {
    public static let defaultRenderedChatStops = [
        "<|END_OF_TURN_TOKEN|>",
        "<|END_OFTURN_TOKEN|>",
        "<|CHANNEL_END|>",
        "<|im_end|>",
        "</s>",
        "<|endoftext|>",
    ]

    public static func merged(_ sequences: [String]) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        result.reserveCapacity(sequences.count)
        for sequence in sequences {
            guard !sequence.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  seen.insert(sequence).inserted else {
                continue
            }
            result.append(sequence)
        }
        return result
    }

    public static func firstMatch(
        in text: String,
        sequences: [String]
    ) -> (sequence: String, range: Range<String.Index>)? {
        var first: (sequence: String, range: Range<String.Index>)?
        for sequence in merged(sequences) {
            guard let range = text.range(of: sequence) else {
                continue
            }
            if let current = first {
                if range.lowerBound < current.range.lowerBound
                    || (range.lowerBound == current.range.lowerBound && range.upperBound > current.range.upperBound) {
                    first = (sequence, range)
                }
            } else {
                first = (sequence, range)
            }
        }
        return first
    }

    public static func trimming(
        _ text: String,
        sequences: [String]
    ) -> (text: String, matchedSequence: String?) {
        guard let match = firstMatch(in: text, sequences: sequences) else {
            return (text.trimmingCharacters(in: .whitespacesAndNewlines), nil)
        }
        let trimmed = String(text[..<match.range.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (trimmed, match.sequence)
    }
}

public struct ChatReasoningSplit: Sendable, Hashable {
    public var visibleContent: String
    public var reasoningContent: String?
    public var hasIncompleteReasoning: Bool
    public var reasoningBlockCount: Int
    public var hasReopenedReasoning: Bool

    public init(
        visibleContent: String,
        reasoningContent: String? = nil,
        hasIncompleteReasoning: Bool = false,
        reasoningBlockCount: Int = 0,
        hasReopenedReasoning: Bool = false
    ) {
        self.visibleContent = visibleContent
        let trimmedReasoning = reasoningContent?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.reasoningContent = trimmedReasoning?.nilIfEmpty
        self.hasIncompleteReasoning = hasIncompleteReasoning
        self.reasoningBlockCount = reasoningBlockCount
        self.hasReopenedReasoning = hasReopenedReasoning
    }
}

public enum ChatReasoningMarkup {
    private static let openThinkTag = "<think>"
    private static let closeThinkTag = "</think>"

    public static func splitThinkBlocks(in text: String) -> ChatReasoningSplit {
        var visible = ""
        var reasoningParts: [String] = []
        var index = text.startIndex
        var hasIncompleteReasoning = false
        var reasoningBlockCount = 0

        func appendReasoning(_ slice: Substring) {
            let part = String(slice).trimmingCharacters(in: .whitespacesAndNewlines)
            if !part.isEmpty {
                reasoningParts.append(part)
            }
        }

        while index < text.endIndex {
            let remaining = index..<text.endIndex
            let openRange = text.range(of: openThinkTag, options: .caseInsensitive, range: remaining)
            let closeRange = text.range(of: closeThinkTag, options: .caseInsensitive, range: remaining)

            if let closeRange {
                let closeBeforeOpen = openRange.map { closeRange.lowerBound < $0.lowerBound } ?? true
                if closeBeforeOpen {
                    let prefix = text[index..<closeRange.lowerBound]
                    let prefixIsOnlyReasoning = visible
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                        .isEmpty && reasoningParts.isEmpty
                    if prefixIsOnlyReasoning {
                        reasoningBlockCount += 1
                        appendReasoning(prefix)
                    } else {
                        visible += prefix
                    }
                    index = closeRange.upperBound
                    continue
                }
            }

            guard let openRange else {
                visible += text[index...]
                break
            }

            visible += text[index..<openRange.lowerBound]
            reasoningBlockCount += 1
            let reasoningStart = openRange.upperBound
            if let closeRange = text.range(
                of: closeThinkTag,
                options: .caseInsensitive,
                range: reasoningStart..<text.endIndex
            ) {
                appendReasoning(text[reasoningStart..<closeRange.lowerBound])
                index = closeRange.upperBound
            } else {
                appendReasoning(text[reasoningStart..<text.endIndex])
                hasIncompleteReasoning = true
                break
            }
        }

        return ChatReasoningSplit(
            visibleContent: visible.trimmingCharacters(in: .whitespacesAndNewlines),
            reasoningContent: reasoningParts.joined(separator: "\n\n"),
            hasIncompleteReasoning: hasIncompleteReasoning,
            reasoningBlockCount: reasoningBlockCount,
            hasReopenedReasoning: reasoningBlockCount > 1
        )
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

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
