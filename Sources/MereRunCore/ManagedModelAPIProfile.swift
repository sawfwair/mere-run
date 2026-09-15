import Foundation

public enum ManagedModelAPITask: String, Hashable, Sendable {
    case chatCompletions = "chat.completions"
    case imageGenerations = "images.generations"
    case imageEdits = "images.edits"
    case audioSpeech = "audio.speech"
    case audioTranscriptions = "audio.transcriptions"
    case embeddings
    case visionGeometry = "vision.geometry"
    case visionDepth = "vision.depth"
    case visionImageTo3D = "vision.image_to_3d"
}

public enum ManagedModelAPIModality: String, Hashable, Sendable {
    case text
    case image
    case audio
    case video
    case embedding
    case geometry
    case threeD = "3d"
}

public enum ManagedModelThinkingLevel: String, CaseIterable, Hashable, Sendable {
    case off
    case minimal
    case low
    case medium
    case high
    case xhigh
    case max
}

public enum ManagedModelMaxTokensField: String, Hashable, Sendable {
    case maxTokens = "max_tokens"
    case maxCompletionTokens = "max_completion_tokens"
}

public enum ManagedModelThinkingFormat: String, Hashable, Sendable {
    case deepseek
}

public struct ManagedModelOpenAICompatibilityProfile: Hashable, Sendable {
    public let supportsStore: Bool
    public let supportsDeveloperRole: Bool
    public let supportsReasoningEffort: Bool
    public let supportsUsageInStreaming: Bool
    public let supportsFinishReason: Bool
    public let maxTokensField: ManagedModelMaxTokensField
    public let supportsStrictMode: Bool
    public let thinkingFormat: ManagedModelThinkingFormat?
    public let requiresReasoningContentOnAssistantMessages: Bool

    public init(
        supportsStore: Bool = false,
        supportsDeveloperRole: Bool = true,
        supportsReasoningEffort: Bool = false,
        supportsUsageInStreaming: Bool = true,
        supportsFinishReason: Bool = true,
        maxTokensField: ManagedModelMaxTokensField = .maxCompletionTokens,
        supportsStrictMode: Bool = false,
        thinkingFormat: ManagedModelThinkingFormat? = nil,
        requiresReasoningContentOnAssistantMessages: Bool = false
    ) {
        self.supportsStore = supportsStore
        self.supportsDeveloperRole = supportsDeveloperRole
        self.supportsReasoningEffort = supportsReasoningEffort
        self.supportsUsageInStreaming = supportsUsageInStreaming
        self.supportsFinishReason = supportsFinishReason
        self.maxTokensField = maxTokensField
        self.supportsStrictMode = supportsStrictMode
        self.thinkingFormat = thinkingFormat
        self.requiresReasoningContentOnAssistantMessages = requiresReasoningContentOnAssistantMessages
    }
}

/// The capabilities mere.run promises when it serves a managed model.
///
/// This is deliberately catalog metadata rather than a client-specific model
/// definition. API discovery, request validation, and harness integrations all
/// project from the same profile, then apply runtime settings such as context
/// and output-token overrides.
public struct ManagedModelAPIProfile: Hashable, Sendable {
    public let task: ManagedModelAPITask
    public let servingEngine: RuntimeServingEngine?
    public let inputModalities: [ManagedModelAPIModality]
    public let outputModalities: [ManagedModelAPIModality]
    public let contextWindow: Int?
    public let maximumOutputTokens: Int?
    public let thinkingLevels: [ManagedModelThinkingLevel]
    public let thinkingLevelMap: [ManagedModelThinkingLevel: ManagedModelThinkingLevel]
    public let reasoningEffortStrengths: [ManagedModelThinkingLevel: Double]
    public let toolCall: Bool
    public let structuredOutput: Bool
    public let compatibility: ManagedModelOpenAICompatibilityProfile
    public let supportsRawProxy: Bool
    public let supportsToolChoice: Bool
    public let supportsStopSequences: Bool
    public let supportsSeed: Bool
    public let supportsPenalties: Bool
    public let supportsLogprobs: Bool
    public let supportsProviderThinkingControls: Bool

    public var reasoning: Bool {
        !thinkingLevels.isEmpty
    }

    public init(
        task: ManagedModelAPITask,
        servingEngine: RuntimeServingEngine? = nil,
        inputModalities: [ManagedModelAPIModality],
        outputModalities: [ManagedModelAPIModality],
        contextWindow: Int? = nil,
        maximumOutputTokens: Int? = nil,
        thinkingLevels: [ManagedModelThinkingLevel] = [],
        thinkingLevelMap: [ManagedModelThinkingLevel: ManagedModelThinkingLevel] = [:],
        reasoningEffortStrengths: [ManagedModelThinkingLevel: Double] = [:],
        toolCall: Bool = false,
        structuredOutput: Bool = false,
        compatibility: ManagedModelOpenAICompatibilityProfile = .init(),
        supportsRawProxy: Bool = false,
        supportsToolChoice: Bool = false,
        supportsStopSequences: Bool = false,
        supportsSeed: Bool = false,
        supportsPenalties: Bool = false,
        supportsLogprobs: Bool = false,
        supportsProviderThinkingControls: Bool = false
    ) {
        self.task = task
        self.servingEngine = servingEngine
        self.inputModalities = inputModalities
        self.outputModalities = outputModalities
        self.contextWindow = contextWindow
        self.maximumOutputTokens = maximumOutputTokens
        self.thinkingLevels = thinkingLevels
        self.thinkingLevelMap = thinkingLevelMap
        self.reasoningEffortStrengths = reasoningEffortStrengths
        self.toolCall = toolCall
        self.structuredOutput = structuredOutput
        self.compatibility = compatibility
        self.supportsRawProxy = supportsRawProxy
        self.supportsToolChoice = supportsToolChoice
        self.supportsStopSequences = supportsStopSequences
        self.supportsSeed = supportsSeed
        self.supportsPenalties = supportsPenalties
        self.supportsLogprobs = supportsLogprobs
        self.supportsProviderThinkingControls = supportsProviderThinkingControls
    }
}

public extension ManagedModelAPIProfile {
    static func textCode(
        contextWindow: Int = 32_768,
        maximumOutputTokens: Int = 4_096
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textCode,
            contextWindow: contextWindow,
            maximumOutputTokens: maximumOutputTokens,
            supportsStopSequences: true
        )
    }

    static func klein(contextWindow: Int = 32_768) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatKlein,
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            structuredOutput: true
        )
    }

    static func gemma4(
        inputModalities: [ManagedModelAPIModality] = [.text],
        contextWindow: Int = Gemma4Resources.defaultContextLength
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatGemma4,
            inputModalities: inputModalities,
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            toolCall: true,
            structuredOutput: true
        )
    }

    static func diffusionGemma() -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatDiffusionGemma,
            contextWindow: DiffusionGemmaResources.defaultContextLength,
            maximumOutputTokens: DiffusionGemmaResources.maximumCanvasLength,
            toolCall: true,
            supportsStopSequences: true,
            supportsSeed: true
        )
    }

    static func laguna(contextWindow: Int = LagunaResources.defaultContextLength) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatLaguna,
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            thinkingLevels: [.high],
            toolCall: true,
            supportsStopSequences: true,
            supportsLogprobs: true
        )
    }

    static func q36(
        contextWindow: Int,
        fixedReasoning: Bool = false
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatQ36,
            inputModalities: [.text, .image],
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            thinkingLevels: fixedReasoning ? [.high] : [],
            toolCall: true,
            structuredOutput: true,
            supportsPenalties: true,
            supportsLogprobs: true
        )
    }

    static func q38(contextWindow: Int) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatQ36,
            inputModalities: [.text, .image],
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            thinkingLevels: [.low, .medium, .xhigh],
            thinkingLevelMap: [.minimal: .low, .high: .xhigh, .max: .xhigh],
            reasoningEffortStrengths: [
                .minimal: 0.2,
                .low: 0.2,
                .medium: 0.5,
                .high: 1,
                .xhigh: 1,
                .max: 1,
            ],
            toolCall: true,
            structuredOutput: true,
            compatibility: ManagedModelOpenAICompatibilityProfile(
                supportsReasoningEffort: true
            ),
            supportsPenalties: true,
            supportsLogprobs: true
        )
    }

    static func lfm2(
        inputModalities: [ManagedModelAPIModality] = [.text],
        contextWindow: Int = LFM2Resources.defaultContextLength
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatLFM2,
            inputModalities: inputModalities,
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            toolCall: true
        )
    }

    static func deepseekV4Flash(
        contextWindow: Int = DeepseekV4FlashResources.defaultContextLength
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatDeepseekV4Flash,
            inputModalities: [.text, .image],
            contextWindow: contextWindow,
            maximumOutputTokens: contextWindow,
            thinkingLevels: [.off, .minimal, .low, .medium, .high, .xhigh],
            thinkingLevelMap: [.minimal: .low],
            toolCall: true,
            compatibility: ManagedModelOpenAICompatibilityProfile(
                supportsDeveloperRole: false,
                supportsReasoningEffort: true,
                maxTokensField: .maxTokens,
                thinkingFormat: .deepseek,
                requiresReasoningContentOnAssistantMessages: true
            ),
            supportsRawProxy: true,
            supportsStopSequences: true,
            supportsSeed: true,
            supportsPenalties: true,
            supportsLogprobs: true,
            supportsProviderThinkingControls: true
        )
    }

    static func museGlimmer(
        contextWindow: Int = MuseGlimmerResources.defaultContextLength
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatMuseGlimmer,
            inputModalities: [.text, .image],
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            thinkingLevels: [.minimal, .low, .medium, .high, .xhigh, .max],
            thinkingLevelMap: [.minimal: .low, .max: .xhigh],
            reasoningEffortStrengths: [
                .minimal: 0.1,
                .low: 0.25,
                .medium: 0.5,
                .high: 0.8,
                .xhigh: 1,
                .max: 1,
            ],
            toolCall: true,
            compatibility: ManagedModelOpenAICompatibilityProfile(
                supportsReasoningEffort: true
            )
        )
    }

    static func nemotronH(
        contextWindow: Int = NemotronHResources.defaultContextLength
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatNemotronH,
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
            toolCall: true,
            supportsStopSequences: true,
            supportsLogprobs: true
        )
    }

    static func nemotronOmni(
        contextWindow: Int = NemotronOmniResources.maximumContextLength
    ) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatNemotronOmni,
            inputModalities: [.text, .image, .audio, .video],
            contextWindow: contextWindow,
            maximumOutputTokens: NemotronOmniResources.maximumOutputTokens,
            thinkingLevels: [.off, .high],
            thinkingLevelMap: [.high: .high],
            toolCall: true,
            supportsStopSequences: true,
            supportsProviderThinkingControls: true
        )
    }

    static func runtimeFallback(for engine: RuntimeServingEngine) -> ManagedModelAPIProfile {
        switch engine {
        case .textCode:
            return .textCode()
        case .textChatKlein:
            return .klein()
        case .textChatGemma4:
            return .gemma4()
        case .textChatDiffusionGemma:
            return .diffusionGemma()
        case .textChatLaguna:
            return .laguna()
        case .textChatQ36, .textChatQ35:
            return .q36(contextWindow: Q35Resources.defaultContextLength)
        case .textChatLFM2:
            return .lfm2()
        case .textChatDeepseekV4Flash:
            return .deepseekV4Flash()
        case .textChatMuseGlimmer:
            return .museGlimmer()
        case .textChatNemotronH:
            return .nemotronH()
        case .textChatNemotronOmni:
            return .nemotronOmni()
        }
    }

    static func companion(
        modelID: String,
        category: ManagedModelCategory?
    ) -> ManagedModelAPIProfile? {
        if QwenImageEditRepository.canonicalModelId(for: modelID) != nil {
            return ManagedModelAPIProfile(
                task: .imageEdits,
                inputModalities: [.text, .image],
                outputModalities: [.image]
            )
        }
        switch category {
        case .image:
            return ManagedModelAPIProfile(
                task: .imageGenerations,
                inputModalities: [.text],
                outputModalities: [.image]
            )
        case .image3D:
            return ManagedModelAPIProfile(
                task: .visionImageTo3D,
                inputModalities: [.image],
                outputModalities: [.threeD]
            )
        case .speechTTS:
            return ManagedModelAPIProfile(
                task: .audioSpeech,
                inputModalities: [.text],
                outputModalities: [.audio]
            )
        case .speechASR:
            return ManagedModelAPIProfile(
                task: .audioTranscriptions,
                inputModalities: [.audio],
                outputModalities: [.text]
            )
        case .textEmbed:
            return ManagedModelAPIProfile(
                task: .embeddings,
                inputModalities: [.text],
                outputModalities: [.embedding]
            )
        case .visionGeometry:
            return ManagedModelAPIProfile(
                task: .visionGeometry,
                inputModalities: [.image],
                outputModalities: [.geometry]
            )
        case .visionDepth:
            return ManagedModelAPIProfile(
                task: .visionDepth,
                inputModalities: [.video],
                outputModalities: [.video]
            )
        default:
            return nil
        }
    }

    private static func chat(
        servingEngine: RuntimeServingEngine,
        inputModalities: [ManagedModelAPIModality] = [.text],
        contextWindow: Int,
        maximumOutputTokens: Int,
        thinkingLevels: [ManagedModelThinkingLevel] = [],
        thinkingLevelMap: [ManagedModelThinkingLevel: ManagedModelThinkingLevel] = [:],
        reasoningEffortStrengths: [ManagedModelThinkingLevel: Double] = [:],
        toolCall: Bool = false,
        structuredOutput: Bool = false,
        compatibility: ManagedModelOpenAICompatibilityProfile = .init(),
        supportsRawProxy: Bool = false,
        supportsStopSequences: Bool = false,
        supportsSeed: Bool = false,
        supportsPenalties: Bool = false,
        supportsLogprobs: Bool = false,
        supportsProviderThinkingControls: Bool = false
    ) -> ManagedModelAPIProfile {
        ManagedModelAPIProfile(
            task: .chatCompletions,
            servingEngine: servingEngine,
            inputModalities: inputModalities,
            outputModalities: [.text],
            contextWindow: contextWindow,
            maximumOutputTokens: maximumOutputTokens,
            thinkingLevels: thinkingLevels,
            thinkingLevelMap: thinkingLevelMap,
            reasoningEffortStrengths: reasoningEffortStrengths,
            toolCall: toolCall,
            structuredOutput: structuredOutput,
            compatibility: compatibility,
            supportsRawProxy: supportsRawProxy,
            supportsToolChoice: toolCall,
            supportsStopSequences: supportsStopSequences,
            supportsSeed: supportsSeed,
            supportsPenalties: supportsPenalties,
            supportsLogprobs: supportsLogprobs,
            supportsProviderThinkingControls: supportsProviderThinkingControls
        )
    }
}
