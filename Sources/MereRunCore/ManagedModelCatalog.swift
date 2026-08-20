import Foundation

public enum ManagedModelCategory: String, CaseIterable, Hashable, Sendable {
    case image = "image"
    case textChat = "text-chat"
    case textCode = "text-code"
    case textEmbed = "text-embed"
    case textAnonymize = "text-anonymize"
    case speechTTS = "speech-tts"
    case speechASR = "speech-asr"
    case speechDiarization = "speech-diarization"
    case visionOCR = "vision-ocr"
    case visionChat = "vision-chat"
    case omniChat = "omni-chat"
    case visionSegment = "vision-segment"
    case visionGround = "vision-ground"
    case visionFlood = "vision-flood"
    case visionFire = "vision-fire"
    case visionEmbed = "vision-embed"
    case visionFace = "vision-face"
    case visionGeometry = "vision-geometry"
    case visionDepth = "vision-depth"
    case image3D = "image-3d"
    case audio = "audio"
    case music = "music"
    case sfx = "sfx"
    case video = "video"
}

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

    static func laguna(contextWindow: Int = LagunaResources.defaultContextLength) -> ManagedModelAPIProfile {
        chat(
            servingEngine: .textChatLaguna,
            contextWindow: contextWindow,
            maximumOutputTokens: 4_096,
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
        if modelID == QwenImageEditRepository.modelId {
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

public enum ManagedModelInstallShape: Hashable, Sendable {
    case directoryRoot
    case singleFile(relativePath: String)
    case structuredRoot
}

public enum ManagedModelValidationKind: String, Hashable, Sendable {
    case flux2Klein
    case bonsaiImage
    case zimageTurbo
    case hidreamO1
    case krea2
    case ideogram4SDNQ
    case gemma4
    case gemma4Unified
    case gemma4MTPAssistant
    case laguna
    case lagunaDFlash
    case q35
    case lfm2
    case inkling
    case museGlimmer
    case museGlimmerAssistant
    case nemotronH
    case nemotronHDSpark
    case nemotronOmni
    case qwen3TTS
    case qwen3ASR
    case parakeet
    case sortformer
    case qwen3Embedding
    case privacyFilter
    case codegenGGUF
    case deepseekV4FlashIMatrixGGUF
    case lightOnOCR
    case sam31
    case falconPerception
    case terramindFlood
    case terramindFire
    case tessera
    case olmoEarth
    case insightFaceBuffaloL
    case moge2
    case videoDepthAnything
    case depthAnything3
    case tripoSR
    case instantMesh
    case trellis2
    case aceStep
    case aceStepLM
    case miniMaxMusic3
    case magentaRT2
    case muScriptor
    case roFormer
    case apBWE
    case univerSR
    case woosh
    case wooshClap
    case wooshSynchformer
    case mmaudio
    case ltxVideo
    case ltxVideo23MLX
    case ltxVideo23FullMLX
    case ltxVideo23A2VMLX
    case ltxVideo25
    case wan22TI2VMLX
    case miniMaxH3MLX
    case cosmos3EdgeMLX
    case scail2MLX
    case dreamXCausalMLX
    case hfTextChat
}

public enum ManagedModelNormalizationKind: String, Hashable, Sendable {
    case none
    case qwen3ASRNested
    case parakeetNested
    case musicACEStep
    case musicACEStepLM
}

public enum ManagedModelAliasKind: String, Hashable, Sendable {
    case none
    case codegenGGUF
}

public struct ManagedModelUsageTerm: Codable, Hashable, Sendable {
    public let component: String
    public let license: String
    public let summary: String
    public let sourceRepoId: String
    public let sourceRevision: String
    public let licenseURL: String

    public init(
        component: String,
        license: String,
        summary: String,
        sourceRepoId: String,
        sourceRevision: String,
        licenseURL: String
    ) {
        self.component = component
        self.license = license
        self.summary = summary
        self.sourceRepoId = sourceRepoId
        self.sourceRevision = sourceRevision
        self.licenseURL = licenseURL
    }

    private enum CodingKeys: String, CodingKey {
        case component
        case license
        case summary
        case sourceRepoId = "source_repo_id"
        case sourceRevision = "source_revision"
        case licenseURL = "license_url"
    }
}

public struct ManagedModelUsageRestriction: Hashable, Sendable {
    public let summary: String
    public let terms: [ManagedModelUsageTerm]

    public var licenseURL: String {
        terms.first?.licenseURL ?? ""
    }

    public init(summary: String, terms: [ManagedModelUsageTerm]) {
        self.summary = summary
        self.terms = terms
    }
}

public struct ManagedModelSpec: Hashable, Sendable {
    public let id: String
    public let category: ManagedModelCategory
    public let installShape: ManagedModelInstallShape
    public let hubFallback: HubFallbackConfig?
    public let mountedHubFallbacks: [MountedHubFallbackConfig]
    public let upstreamRepoId: String?
    public let upstreamRevision: String?
    public let usageRestriction: ManagedModelUsageRestriction?
    public let validationKind: ManagedModelValidationKind
    public let normalizationKind: ManagedModelNormalizationKind
    public let aliasKind: ManagedModelAliasKind
    public let runtimeAutoDownloadAllowed: Bool
    public let resolutionFallbackIDs: [String]
    public let estimatedDownloadBytes: Int64?
    public let defaultCLICommands: [String]
    public let companionModelIDs: [String]
    public let apiProfile: ManagedModelAPIProfile?

    public init(
        id: String,
        category: ManagedModelCategory,
        installShape: ManagedModelInstallShape,
        hubFallback: HubFallbackConfig? = nil,
        mountedHubFallbacks: [MountedHubFallbackConfig] = [],
        upstreamRepoId: String? = nil,
        upstreamRevision: String? = nil,
        usageRestriction: ManagedModelUsageRestriction? = nil,
        validationKind: ManagedModelValidationKind,
        normalizationKind: ManagedModelNormalizationKind = .none,
        aliasKind: ManagedModelAliasKind = .none,
        runtimeAutoDownloadAllowed: Bool = true,
        resolutionFallbackIDs: [String] = [],
        estimatedDownloadBytes: Int64? = nil,
        defaultCLICommands: [String] = [],
        companionModelIDs: [String] = [],
        apiProfile: ManagedModelAPIProfile? = nil
    ) {
        self.id = id
        self.category = category
        self.installShape = installShape
        self.hubFallback = hubFallback
        self.mountedHubFallbacks = mountedHubFallbacks
        self.upstreamRepoId = upstreamRepoId
        self.upstreamRevision = upstreamRevision
        self.usageRestriction = usageRestriction
        self.validationKind = validationKind
        self.normalizationKind = normalizationKind
        self.aliasKind = aliasKind
        self.runtimeAutoDownloadAllowed = runtimeAutoDownloadAllowed
        self.resolutionFallbackIDs = resolutionFallbackIDs
        self.estimatedDownloadBytes = estimatedDownloadBytes
        self.defaultCLICommands = defaultCLICommands
        self.companionModelIDs = companionModelIDs
        self.apiProfile = apiProfile ?? ManagedModelAPIProfile.companion(
            modelID: id,
            category: category
        )
    }
}

public enum ManagedModelCatalog {
    public static func apiProfile(for modelID: String) -> ManagedModelAPIProfile? {
        spec(for: modelID)?.apiProfile
            ?? ManagedModelAPIProfile.companion(modelID: modelID, category: nil)
    }

    private static let diffusersImageSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "model_index.json",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
        "scheduler/*",
    ]
    private static let kleinBase9BTransformerSnapshotPatterns = [
        "model_index.json",
        "transformer/*",
    ]
    private static let kleinNanoSnapshotPatterns = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    private static let zImageNanoSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "text_encoder/0.safetensors",
        "text_encoder/1.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/0.safetensors",
        "transformer/1.safetensors",
        "transformer/model.safetensors.index.json",
        "vae/0.safetensors",
        "vae/model.safetensors.index.json",
    ]

    private static let zImageNanoUpstreamRepoId = "filipstrand/Z-Image-Turbo-mflux-4bit"
    private static let zImageNanoUpstreamRevision = "b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b"
    private static let klein9BUpstreamRevision = "b0f0826a36667ec7c58253e50557ba76f8c0255e"
    private static let kleinBase9BUpstreamRevision = "32773329fbe7e81a90ef971740e8ba4b0364ecf3"
    private static let sam31MLXRevision = "a992e302ea9b0f03f41dfd93414a4fd0e818f65b"
    private static let sam31TokenizerRevision = "694239a1479aab8fd1317c87c433c58acd7c6eab"
    private static let wooshWeightsRevision = "f7b524db359f95b2b0bdc4afce12120b72e68bff"
    private static let ltx2DistilledRevision = "c38acc2729229140f083c3a834041e8735ee5260"
    private static let ltx23MLXRevision = "baa5f235ea04fd9c95899d751295c4fd825ee4e2"
    private static let kleinNanoUpstreamRepoId = "stereovoid/flux2-klein-4b-4bit"
    private static let bonsaiBinaryUpstreamRepoId = "prism-ml/bonsai-image-binary-4B-mlx-1bit"
    private static let bonsaiTernaryUpstreamRepoId = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
    private static let magentaRT2UpstreamRepoId = "google/magenta-realtime-2"
    private static let magentaRT2UpstreamRevision = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"
    private static let sortformerUpstreamRevision = "e23e6404bd9859e93edbf94a740eb1c7fc58f12e"
    private static let aceStepSharedRevision = "19671f406d603126926c1b7e2adc169acbcade22"
    private static let aceStepXLBaseRevision = "220c1166efbdd9583eafcb12eb160594bbfcb241"
    private static let aceStepXLSFTRevision = "d06de46b4622f781cf07f4a013a67d591ca52819"
    private static let aceStepXLTurboRevision = "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee"
    private static let aceStepLM4BRevision = "0a3ec94b557aea7d508da38b31cfe7341f6ff737"
    private static let magentaRT2ResourcePatterns = [
        "resources/musiccoca/audio_preprocessor.tflite",
        "resources/musiccoca/mapper.tflite",
        "resources/musiccoca/music_encoder.tflite",
        "resources/musiccoca/pretrained_vector_quantizer.tflite",
        "resources/musiccoca/spm.model",
        "resources/musiccoca/text_encoder.tflite",
        "resources/spectrostream/decoder.safetensors",
        "resources/spectrostream/encoder.safetensors",
        "resources/spectrostream/quantizer.safetensors",
        "resources/spectrostream/spectrostream_encoder.mlxfn",
    ]
    private static let wooshDFlowSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-DFlow/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerA/*",
    ]
    private static let wooshFlowSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-Flow/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerA/*",
    ]
    private static let wooshCLAPSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-CLAP/*",
    ]
    private static let wooshSynchformerSnapshotPatterns = [
        WooshResources.synchformerFilename,
    ]
    private static let wooshVFlow8sSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-VFlow-8s/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerV/*",
    ]
    private static let wooshDVFlow8sSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-DVFlow-8s/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerV/*",
    ]
    private static let wooshRobertaTokenizerPatterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    private static let mmaudioSnapshotPatterns = [
        "README.md",
        MMAudioResources.networkFilename,
        MMAudioResources.clipFilename,
        MMAudioResources.synchformerFilename,
        MMAudioResources.vaeFilename,
    ]
    private static let mmaudioCLIPTokenizerPatterns = [
        "LICENSE",
        "open_clip_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    private static let mmaudioBigVGANPatterns = [
        "LICENSE",
        "config.json",
        MMAudioResources.bigVGANPyTorchFilename,
    ]
    private static let ltx23MLXUpstreamRepoId = "dgrauet/ltx-2.3-mlx"
    private static let ltx23MLXSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "config.json",
        "embedded_config.json",
        "split_model.json",
        "connector.safetensors",
        "transformer-distilled.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2_v1_1_config.json",
        "spatial_upscaler_x1_5_v1_0.safetensors",
        "spatial_upscaler_x1_5_v1_0_config.json",
        "temporal_upscaler_x2_v1_0.safetensors",
        "temporal_upscaler_x2_v1_0_config.json",
    ]
    private static let ltx23A2VMLXSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "config.json",
        "embedded_config.json",
        "split_model.json",
        "connector.safetensors",
        "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2_v1_1_config.json",
    ]
    private static let ltx23FullMLXSnapshotPatterns = ltx23A2VMLXSnapshotPatterns + [
        "vocoder.safetensors",
    ]
    private static let ltxGemma3TextEncoderRepoId = "mlx-community/gemma-3-12b-it-4bit"
    private static let ltxGemma3TextEncoderRevision = "14d891e009084901c434304fe93a86fd9013e84c"
    private static let ltxGemma3TextEncoderSnapshotPatterns = [
        "README.md",
        "config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]

    private static let miniMaxH3UsageRestriction = usageRestriction(
        summary: "MiniMax-H3 weights may not be used, distributed, or displayed in the United States, European Union, United Kingdom, or Republic of Korea; downstream distribution also requires the Community License agreement, notice, and safeguards.",
        license: "MiniMax-H3 Community License",
        sourceRepoId: MiniMaxH3Resources.sourceRepository,
        sourceRevision: MiniMaxH3Resources.sourceRevision,
        licenseURL: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/ec19cc6daf5d8add9417c18e86b6b58cc6c55027/LICENSE"
    )

    private static func usageRestriction(
        summary: String,
        component: String = "model",
        license: String,
        termSummary: String? = nil,
        sourceRepoId: String,
        sourceRevision: String,
        licenseURL: String
    ) -> ManagedModelUsageRestriction {
        ManagedModelUsageRestriction(
            summary: summary,
            terms: [
                ManagedModelUsageTerm(
                    component: component,
                    license: license,
                    summary: termSummary ?? summary,
                    sourceRepoId: sourceRepoId,
                    sourceRevision: sourceRevision,
                    licenseURL: licenseURL
                ),
            ]
        )
    }

    private static func flux9BUsageRestriction(
        sourceRepoId: String,
        sourceRevision: String,
        component: String = "model",
        additionalTerms: [ManagedModelUsageTerm] = []
    ) -> ManagedModelUsageRestriction {
        let summary = "FLUX.2 Klein 9B weights are licensed only for non-commercial, non-production use."
        let term = ManagedModelUsageTerm(
            component: component,
            license: "FLUX Non-Commercial License v2.1",
            summary: summary,
            sourceRepoId: sourceRepoId,
            sourceRevision: sourceRevision,
            licenseURL: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md"
        )
        return ManagedModelUsageRestriction(
            summary: summary,
            terms: [term] + additionalTerms
        )
    }

    private static func krea2UsageRestriction(
        sourceRepoId: String,
        sourceRevision: String
    ) -> ManagedModelUsageRestriction {
        usageRestriction(
            summary: "Krea 2 uses a custom community license; commercial use is limited to entities below USD 1M in trailing annual revenue and remains subject to its use and distribution conditions.",
            license: "Krea 2 Community License Agreement",
            sourceRepoId: sourceRepoId,
            sourceRevision: sourceRevision,
            licenseURL: "https://huggingface.co/\(sourceRepoId)/blob/\(sourceRevision)/LICENSE.pdf"
        )
    }

    private static func muScriptorUsageRestriction(
        sourceRepoId: String,
        sourceRevision: String
    ) -> ManagedModelUsageRestriction {
        usageRestriction(
            summary: "MuScriptor weights are licensed CC BY-NC 4.0 for non-commercial use.",
            license: "CC BY-NC 4.0",
            sourceRepoId: sourceRepoId,
            sourceRevision: sourceRevision,
            licenseURL: "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en"
        )
    }

    private static let wooshUsageRestriction = usageRestriction(
        summary: "Woosh model weights are licensed CC BY-NC 4.0 for non-commercial use.",
        license: "CC BY-NC 4.0",
        sourceRepoId: WooshResources.huggingFaceMirrorRepoId,
        sourceRevision: wooshWeightsRevision,
        licenseURL: "https://github.com/SonyResearch/Woosh/blob/v1.0.0/LICENSE"
    )

    private static let wooshSynchformerUsageRestriction = usageRestriction(
        summary: "The MMAudio Synchformer checkpoint used by Woosh is licensed CC BY-NC 4.0 for non-commercial use.",
        component: "synchformer",
        license: "CC BY-NC 4.0",
        sourceRepoId: WooshResources.synchformerRepoId,
        sourceRevision: MMAudioResources.convertedWeightsRevision,
        licenseURL: "https://github.com/hkchengrex/MMAudio#pre-trained-weights"
    )

    private static let mmaudioUsageRestriction = ManagedModelUsageRestriction(
        summary: "MMAudio combines non-commercial checkpoint terms with an Apple research-only visual encoder; each component's terms apply independently.",
        terms: [
            ManagedModelUsageTerm(
                component: "MMAudio checkpoints",
                license: "CC BY-NC 4.0",
                summary: "Published MMAudio checkpoints are limited to non-commercial use.",
                sourceRepoId: MMAudioResources.convertedWeightsRepoID,
                sourceRevision: MMAudioResources.convertedWeightsRevision,
                licenseURL: "https://github.com/hkchengrex/MMAudio#pre-trained-weights"
            ),
            ManagedModelUsageTerm(
                component: "Apple DFN5B CLIP visual encoder",
                license: "Apple Machine Learning Research Model License Agreement",
                summary: "Apple licenses this component exclusively for non-commercial scientific research and academic development.",
                sourceRepoId: MMAudioResources.clipRepoID,
                sourceRevision: MMAudioResources.clipRevision,
                licenseURL: "https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/blob/main/LICENSE"
            ),
        ]
    )

    private static func ltxUsageRestriction(
        sourceRepoId: String,
        sourceRevision: String,
        additionalTerms: [ManagedModelUsageTerm] = []
    ) -> ManagedModelUsageRestriction {
        let summary = "LTX-2 uses a custom community license; entities with at least USD 10M annual revenue need a paid commercial license, and acceptable-use conditions apply."
        let ltxTerm = ManagedModelUsageTerm(
            component: "model",
            license: "LTX-2 Community License Agreement",
            summary: summary,
            sourceRepoId: sourceRepoId,
            sourceRevision: sourceRevision,
            licenseURL: "https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md"
        )
        return ManagedModelUsageRestriction(
            summary: summary,
            terms: [ltxTerm] + additionalTerms
        )
    }

    private static let ltxGemmaTextEncoderUsageTerm = ManagedModelUsageTerm(
        component: "Gemma 3 text encoder",
        license: "Gemma Terms of Use",
        summary: "The LTX 2.3 text-encoder companion is distributed under the Gemma Terms of Use and Gemma Prohibited Use Policy.",
        sourceRepoId: ltxGemma3TextEncoderRepoId,
        sourceRevision: ltxGemma3TextEncoderRevision,
        licenseURL: "https://ai.google.dev/gemma/terms"
    )

    private static let ltx25GemmaTextEncoderUsageTerm = ManagedModelUsageTerm(
        component: "Gemma 4 text encoder",
        license: "Gemma Terms of Use",
        summary: "The packed LTX 2.5 text encoder includes Gemma 4 weights distributed under the Gemma Terms of Use and Gemma Prohibited Use Policy.",
        sourceRepoId: LTX25Resources.sourceRepository,
        sourceRevision: LTX25Resources.sourceRevision,
        licenseURL: "https://ai.google.dev/gemma/terms"
    )

    private static let geoExpansionSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionFireTerraMindBase.rawValue,
            category: .visionFire,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: TerraMindFireResources.sourceRepository,
                revision: TerraMindFireResources.sourceRevision,
                patterns: [
                    "README.md",
                    "LICENSE*",
                    "NOTICE*",
                    TerraMindFireResources.sourceCheckpointFilename,
                    TerraMindFireResources.sourceConfigurationFilename,
                ]
            ),
            upstreamRepoId: TerraMindFireResources.sourceRepository,
            upstreamRevision: TerraMindFireResources.sourceRevision,
            validationKind: .terramindFire,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 673_193_610,
            defaultCLICommands: ["geo fire"]
        ),
    ] + TESSERAResources.allSpecs.map { source in
        ManagedModelSpec(
            id: source.modelID,
            category: .visionEmbed,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: source.sourceRepository,
                revision: source.sourceRevision,
                patterns: ["README.md", source.sourceCheckpointFilename]
            ),
            upstreamRepoId: source.sourceRepository,
            upstreamRevision: source.sourceRevision,
            validationKind: .tessera,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: source.sourceCheckpointByteCount,
            defaultCLICommands: ["geo tessera"]
        )
    } + OlmoEarthResources.allSpecs.map { source in
        ManagedModelSpec(
            id: source.modelID,
            category: .visionEmbed,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: source.sourceRepository,
                revision: source.sourceRevision,
                patterns: [
                    "README.md",
                    "LICENSE*",
                    OlmoEarthResources.sourceWeightsFilename,
                    OlmoEarthResources.sourceConfigurationFilename,
                ]
            ),
            upstreamRepoId: source.sourceRepository,
            upstreamRevision: source.sourceRevision,
            usageRestriction: usageRestriction(
                summary: "OlmoEarth permits broad use but prohibits military and defense applications, intelligence gathering, human surveillance and policing, and extractive activities such as drilling, mining, and deforestation.",
                license: "OlmoEarth Artifact License",
                sourceRepoId: source.sourceRepository,
                sourceRevision: source.sourceRevision,
                licenseURL: "https://huggingface.co/\(source.sourceRepository)/blob/\(source.sourceRevision)/LICENSE.txt"
            ),
            validationKind: .olmoEarth,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: source.sourceWeightsByteCount,
            defaultCLICommands: ["geo olmoearth"]
        )
    }

    public static let allSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: "image-klein-nano",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: kleinNanoUpstreamRepoId,
                patterns: kleinNanoSnapshotPatterns
            ),
            upstreamRepoId: kleinNanoUpstreamRepoId,
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 4_627_979_498,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-klein-max",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "black-forest-labs/FLUX.2-klein-4B",
                patterns: diffusersImageSnapshotPatterns
            ),
            upstreamRepoId: "black-forest-labs/FLUX.2-klein-4B",
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 15_980_131_745,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            // FLUX.2 [klein] 9B — 9B flow + 8B Qwen3 text embedder, step-distilled to 4 steps,
            // native single/multi-reference editing. bf16 diffusers format (same loader path as
            // klein-max). Pulled from the mlx-community mirror, which is UNGATED (gated:false) —
            // no HF token required — unlike black-forest-labs/FLUX.2-klein-9B which is gated.
            // diffusersImageSnapshotPatterns skips the redundant single-file root checkpoint.
            id: "image-klein-9b",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/FLUX.2-klein-9B",
                revision: klein9BUpstreamRevision,
                patterns: diffusersImageSnapshotPatterns
            ),
            upstreamRepoId: "mlx-community/FLUX.2-klein-9B",
            upstreamRevision: klein9BUpstreamRevision,
            usageRestriction: flux9BUsageRestriction(
                sourceRepoId: "mlx-community/FLUX.2-klein-9B",
                sourceRevision: klein9BUpstreamRevision
            ),
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 34_722_771_551,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-klein-base",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "black-forest-labs/FLUX.2-klein-base-4B",
                patterns: diffusersImageSnapshotPatterns
            ),
            upstreamRepoId: "black-forest-labs/FLUX.2-klein-base-4B",
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 15_980_131_711,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            // FLUX.2 [klein] Base 9B — undistilled bf16 9B transformer for LoRA/fine-tuning.
            // BFL publishes the base transformer separately from reusable 9B text/VAE
            // components in common training recipes, so mount the shared components from the
            // ungated mlx-community mirror while keeping the gated Base 9B transformer source
            // explicit.
            id: "image-klein-base-9b",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "black-forest-labs/FLUX.2-klein-base-9B",
                revision: kleinBase9BUpstreamRevision,
                patterns: kleinBase9BTransformerSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "text_encoder",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["text_encoder/*"]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["tokenizer/*"]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "vae",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["vae/*"]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "scheduler",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["scheduler/*"]
                    )
                ),
            ],
            upstreamRepoId: "black-forest-labs/FLUX.2-klein-base-9B",
            upstreamRevision: kleinBase9BUpstreamRevision,
            usageRestriction: flux9BUsageRestriction(
                sourceRepoId: "black-forest-labs/FLUX.2-klein-base-9B",
                sourceRevision: kleinBase9BUpstreamRevision,
                component: "Base 9B transformer",
                additionalTerms: [
                    ManagedModelUsageTerm(
                        component: "shared 9B text encoder, tokenizer, VAE, and scheduler",
                        license: "FLUX Non-Commercial License v2.1",
                        summary: "The shared FLUX.2 Klein 9B components are licensed only for non-commercial, non-production use.",
                        sourceRepoId: "mlx-community/FLUX.2-klein-9B",
                        sourceRevision: klein9BUpstreamRevision,
                        licenseURL: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md"
                    ),
                ]
            ),
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 48 * 1_073_741_824,
            defaultCLICommands: ["image generate", "image train-lora"]
        ),
        ManagedModelSpec(
            id: "image-klein-shared",
            category: .image,
            installShape: .directoryRoot,
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false
        ),
        ManagedModelSpec(
            id: "image-bonsai-binary",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: bonsaiBinaryUpstreamRepoId,
                revision: "main",
                patterns: [
                    "manifest.json",
                    "scheduler/scheduler_config.json",
                    "tokenizer/*",
                    "text_encoder-mlx-4bit/*",
                    "transformer-packed-mflux/*",
                    "vae/*",
                ]
            ),
            upstreamRepoId: bonsaiBinaryUpstreamRepoId,
            upstreamRevision: "main",
            validationKind: .bonsaiImage,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 3_428_210_775,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-bonsai-ternary",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: bonsaiTernaryUpstreamRepoId,
                revision: "main",
                patterns: [
                    "manifest.json",
                    "scheduler/scheduler_config.json",
                    "tokenizer/*",
                    "text_encoder-mlx-4bit/*",
                    "transformer-packed-mflux/*",
                    "vae/*",
                ]
            ),
            upstreamRepoId: bonsaiTernaryUpstreamRepoId,
            upstreamRevision: "main",
            validationKind: .bonsaiImage,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 3_888_274_558,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-zimage-nano",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: zImageNanoUpstreamRepoId,
                revision: zImageNanoUpstreamRevision,
                patterns: zImageNanoSnapshotPatterns
            ),
            upstreamRepoId: zImageNanoUpstreamRepoId,
            upstreamRevision: zImageNanoUpstreamRevision,
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 20_538_488_559,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-zimage-max",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Tongyi-MAI/Z-Image-Turbo",
                revision: "main",
                patterns: diffusersImageSnapshotPatterns
            ),
            upstreamRepoId: "Tongyi-MAI/Z-Image-Turbo",
            upstreamRevision: "main",
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 32_848_305_533,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-zimage-base",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Tongyi-MAI/Z-Image",
                revision: "main",
                patterns: diffusersImageSnapshotPatterns
            ),
            upstreamRepoId: "Tongyi-MAI/Z-Image",
            upstreamRevision: "main",
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5_907_438_792,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-hidream-o1",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "HiDream-ai/HiDream-O1-Image",
                revision: "main",
                patterns: [
                    "config.json",
                    "configuration.json",
                    "generation_config.json",
                    "chat_template.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "preprocessor_config.json",
                    "video_preprocessor_config.json",
                    "model.safetensors.index.json",
                    "model-*.safetensors",
                ]
            ),
            upstreamRepoId: "HiDream-ai/HiDream-O1-Image",
            upstreamRevision: "main",
            validationKind: .hidreamO1,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 35_231_213_079,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-hidream-o1-dev",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "HiDream-ai/HiDream-O1-Image-Dev",
                revision: "main",
                patterns: [
                    "config.json",
                    "configuration.json",
                    "generation_config.json",
                    "chat_template.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "preprocessor_config.json",
                    "video_preprocessor_config.json",
                    "model.safetensors.index.json",
                    "model-*.safetensors",
                ]
            ),
            upstreamRepoId: "HiDream-ai/HiDream-O1-Image-Dev",
            upstreamRevision: "main",
            validationKind: .hidreamO1,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 35_231_213_079,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: Krea2RawResources.modelId,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Krea2RawResources.upstreamRepoId,
                revision: Krea2RawResources.upstreamRevision,
                patterns: Krea2RawResources.snapshotPatterns
            ),
            upstreamRepoId: Krea2RawResources.upstreamRepoId,
            upstreamRevision: Krea2RawResources.upstreamRevision,
            usageRestriction: krea2UsageRestriction(
                sourceRepoId: Krea2RawResources.upstreamRepoId,
                sourceRevision: Krea2RawResources.upstreamRevision
            ),
            validationKind: .krea2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Krea2RawResources.estimatedDownloadBytes,
            defaultCLICommands: ["image train-lora"]
        ),
        ManagedModelSpec(
            id: Krea2Resources.modelId,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Krea2Resources.upstreamRepoId,
                revision: Krea2Resources.upstreamRevision,
                patterns: Krea2Resources.snapshotPatterns
            ),
            upstreamRepoId: Krea2Resources.upstreamRepoId,
            upstreamRevision: Krea2Resources.upstreamRevision,
            usageRestriction: krea2UsageRestriction(
                sourceRepoId: Krea2Resources.upstreamRepoId,
                sourceRevision: Krea2Resources.upstreamRevision
            ),
            validationKind: .krea2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Krea2Resources.estimatedDownloadBytes,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: Ideogram4Resources.modelId,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Ideogram4Resources.upstreamRepoId,
                revision: Ideogram4Resources.upstreamRevision,
                patterns: Ideogram4Resources.snapshotPatterns
            ),
            upstreamRepoId: Ideogram4Resources.upstreamRepoId,
            upstreamRevision: Ideogram4Resources.upstreamRevision,
            usageRestriction: usageRestriction(
                summary: "Ideogram 4 weights are licensed only for non-commercial purposes.",
                license: "Ideogram Non-Commercial Model Agreement",
                sourceRepoId: Ideogram4Resources.upstreamRepoId,
                sourceRevision: Ideogram4Resources.upstreamRevision,
                licenseURL: "https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md"
            ),
            validationKind: .ideogram4SDNQ,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Ideogram4Resources.estimatedDownloadBytes,
            defaultCLICommands: []
        ),
        ManagedModelSpec(
            id: "text-chat-mebot",
            category: .textChat,
            installShape: .directoryRoot,
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["api serve"],
            apiProfile: .klein()
        ),
        ManagedModelSpec(
            id: "text-chat-psi-agent",
            category: .textChat,
            installShape: .directoryRoot,
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false
        ),
        ManagedModelSpec(
            id: "text-chat-gemma4",
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.defaultUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.defaultUpstreamModelId,
            validationKind: .gemma4,
            resolutionFallbackIDs: ["text-chat-gemma4-max", "text-chat-gemma4-nano"],
            estimatedDownloadBytes: 62_578_654_199,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: Gemma4Resources.turboModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.turboUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.turboUpstreamModelId,
            validationKind: .gemma4,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 31 * 1_073_741_824,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: Gemma4Resources.twelveBModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.twelveBUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.twelveBUpstreamModelId,
            validationKind: .gemma4,
            estimatedDownloadBytes: 25 * 1_073_741_824,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            companionModelIDs: [Gemma4MTPResources.modelId],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: Gemma4Resources.twelveB4BitModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.twelveB4BitUpstreamModelId,
                revision: Gemma4Resources.twelveB4BitUpstreamRevision,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.twelveB4BitUpstreamModelId,
            upstreamRevision: Gemma4Resources.twelveB4BitUpstreamRevision,
            validationKind: .gemma4,
            estimatedDownloadBytes: 6_773_374_762,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            companionModelIDs: [Gemma4MTPResources.modelId],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: Gemma4Resources.visionTwelveBModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.twelveBUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.twelveBUpstreamModelId,
            validationKind: .gemma4Unified,
            estimatedDownloadBytes: 25 * 1_073_741_824,
            defaultCLICommands: ["api serve"],
            companionModelIDs: [Gemma4MTPResources.modelId],
            apiProfile: .gemma4(inputModalities: [.text, .image])
        ),
        ManagedModelSpec(
            id: "text-chat-gemma4-nano",
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.nanoUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.nanoUpstreamModelId,
            validationKind: .gemma4,
            estimatedDownloadBytes: 16_024_791_983,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: "text-chat-gemma4-max",
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.maxUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.maxUpstreamModelId,
            validationKind: .gemma4,
            estimatedDownloadBytes: 62_578_654_199,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: LagunaResources.modelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LagunaResources.upstreamModelID,
                revision: LagunaResources.upstreamRevision,
                patterns: LagunaResources.snapshotPatterns
            ),
            upstreamRepoId: LagunaResources.upstreamModelID,
            upstreamRevision: LagunaResources.upstreamRevision,
            validationKind: .laguna,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LagunaResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            companionModelIDs: [LagunaResources.dflashModelID],
            apiProfile: .laguna()
        ),
        ManagedModelSpec(
            id: LagunaResources.xsModelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LagunaResources.xsUpstreamModelID,
                revision: LagunaResources.xsUpstreamRevision,
                patterns: LagunaResources.snapshotPatterns
            ),
            upstreamRepoId: LagunaResources.xsUpstreamModelID,
            upstreamRevision: LagunaResources.xsUpstreamRevision,
            validationKind: .laguna,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LagunaResources.xsEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "text train-lora",
                "api serve",
                "model benchmark chat",
            ],
            apiProfile: .laguna()
        ),
        ManagedModelSpec(
            id: InklingResources.modelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: InklingResources.hubFallbackConfig,
            upstreamRepoId: InklingResources.artifactRepoID,
            upstreamRevision: InklingResources.artifactRevision,
            validationKind: .inkling,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: InklingResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "text train-lora"]
        ),
        ManagedModelSpec(
            id: MuseGlimmerResources.modelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: MuseGlimmerResources.artifactRepoId,
                revision: MuseGlimmerResources.artifactRevision,
                patterns: MuseGlimmerResources.snapshotPatterns
            ),
            upstreamRepoId: MuseGlimmerResources.artifactRepoId,
            upstreamRevision: MuseGlimmerResources.artifactRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Apache-2.0 model subject to Meta's bundled usage policy; upstream states it is not intended for download or use by people under 18.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Muse Glimmer 30B",
                        license: "Apache-2.0 with upstream usage policy",
                        summary: "Review LICENSE and USAGE_POLICY.md before installing or deploying.",
                        sourceRepoId: MuseGlimmerResources.upstreamRepoId,
                        sourceRevision: MuseGlimmerResources.upstreamRevision,
                        licenseURL: "https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/\(MuseGlimmerResources.upstreamRevision)/USAGE_POLICY.md"
                    ),
                ]
            ),
            validationKind: .museGlimmer,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: MuseGlimmerResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            companionModelIDs: [MuseGlimmerResources.dflash2ModelId],
            apiProfile: .museGlimmer()
        ),
        ManagedModelSpec(
            id: NemotronHResources.modelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: NemotronHResources.artifactRepoID,
                revision: NemotronHResources.artifactRevision,
                patterns: NemotronHResources.snapshotPatterns
            ),
            upstreamRepoId: NemotronHResources.artifactRepoID,
            upstreamRevision: NemotronHResources.artifactRevision,
            validationKind: .nemotronH,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: NemotronHResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            companionModelIDs: [NemotronHResources.dsparkModelID],
            apiProfile: .nemotronH()
        ),
        ManagedModelSpec(
            id: NemotronOmniResources.modelID,
            category: .omniChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: NemotronOmniResources.upstreamRepoID,
                revision: NemotronOmniResources.upstreamRevision,
                patterns: NemotronOmniResources.snapshotPatterns
            ),
            upstreamRepoId: NemotronOmniResources.upstreamRepoID,
            upstreamRevision: NemotronOmniResources.upstreamRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Use is governed by the NVIDIA Open Model Agreement; review and acknowledge the governing terms before download.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Nemotron 3 Nano Omni 30B-A3B Reasoning BF16",
                        license: "NVIDIA Open Model Agreement",
                        summary: "Review the NVIDIA Open Model Agreement before installing or deploying.",
                        sourceRepoId: NemotronOmniResources.upstreamRepoID,
                        sourceRevision: NemotronOmniResources.upstreamRevision,
                        licenseURL: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/"
                    ),
                ]
            ),
            validationKind: .nemotronOmni,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: NemotronOmniResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .nemotronOmni()
        ),
        ManagedModelSpec(
            id: Q35Resources.q36NanoModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.q36NanoModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.q36NanoUpstreamRepoId,
            upstreamRevision: Q35Resources.q36NanoUpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: 24 * 1_073_741_824,
            defaultCLICommands: ["chat", "api serve"],
            apiProfile: .q36(contextWindow: Q35Resources.defaultContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38TwentySevenBModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.q38TwentySevenBModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.q38TwentySevenBUpstreamRepoId,
            upstreamRevision: Q35Resources.q38TwentySevenBUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38TwentySevenBEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
                "model benchmark vlm",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38TwentySevenB4BitModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(
                for: Q35Resources.q38TwentySevenB4BitModelId
            )?.hubFallbackConfig,
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: Q35Resources.q38MTPComponentPath,
                    hubFallback: HubFallbackConfig(
                        repoId: Q35Resources.q38MTP4BitUpstreamRepoId,
                        revision: Q35Resources.q38MTP4BitUpstreamRevision,
                        patterns: Q35Resources.q38MTPComponentSnapshotPatterns
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: Q35Resources.q38LicenseComponentPath,
                    hubFallback: HubFallbackConfig(
                        repoId: Q35Resources.q38TwentySevenBUpstreamRepoId,
                        revision: Q35Resources.q38TwentySevenBUpstreamRevision,
                        patterns: Q35Resources.q38LicenseComponentSnapshotPatterns
                    )
                ),
            ],
            upstreamRepoId: Q35Resources.q38TwentySevenB4BitUpstreamRepoId,
            upstreamRevision: Q35Resources.q38TwentySevenB4BitUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38TwentySevenB4BitEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
                "model benchmark vlm",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai27B1BitModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai27B1BitModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.bonsai27B1BitUpstreamRepoId,
            upstreamRevision: Q35Resources.bonsai27B1BitUpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.bonsai27B1BitEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(
                contextWindow: Q35Resources.bonsai27B1BitContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai27B2BitModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai27B2BitModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.bonsai27B2BitUpstreamRepoId,
            upstreamRevision: Q35Resources.bonsai27B2BitUpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.bonsai27B2BitEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(
                contextWindow: Q35Resources.bonsai27B2BitContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith9BModelId,
            category: .textCode,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith9BModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith9BUpstreamRepoId,
            upstreamRevision: Q35Resources.ornith9BUpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.ornith9BEstimatedDownloadBytes,
            defaultCLICommands: ["chat", "api serve", "agent start"],
            apiProfile: .q36(
                contextWindow: Q35Resources.defaultContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith35BMLXModelId,
            category: .textCode,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith35BMLXModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith35BMLXUpstreamRepoId,
            upstreamRevision: Q35Resources.ornith35BMLXUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.ornith35BMLXEstimatedDownloadBytes,
            defaultCLICommands: ["chat", "api serve", "agent start", "model benchmark code"],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: AgentModelResources.qwen35NineBModelId,
            category: .textCode,
            installShape: .singleFile(relativePath: AgentModelResources.qwen35NineBRelativePath),
            hubFallback: AgentModelResources.qwen35NineBHubFallbackConfig,
            upstreamRepoId: AgentModelResources.qwen35NineBRepoId,
            upstreamRevision: AgentModelResources.qwen35NineBRevision,
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: 5_680_522_464,
            defaultCLICommands: ["api serve", "text code"],
            apiProfile: .textCode()
        ),
        ManagedModelSpec(
            id: NorthMiniCodeResources.modelId,
            category: .textCode,
            installShape: .singleFile(relativePath: NorthMiniCodeResources.managedRelativePath),
            hubFallback: NorthMiniCodeResources.hubFallbackConfig,
            upstreamRepoId: NorthMiniCodeResources.upstreamRepoId,
            upstreamRevision: NorthMiniCodeResources.upstreamRevision,
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: NorthMiniCodeResources.estimatedDownloadBytes,
            defaultCLICommands: ["text code", "api serve", "agent start"],
            apiProfile: .textCode(
                contextWindow: NorthMiniCodeResources.runtimeContextLength,
                maximumOutputTokens: NorthMiniCodeResources.maxOutputTokens
            )
        ),
        ManagedModelSpec(
            id: Ornith35BCodeResources.modelId,
            category: .textCode,
            installShape: .singleFile(relativePath: Ornith35BCodeResources.managedRelativePath),
            hubFallback: Ornith35BCodeResources.hubFallbackConfig,
            upstreamRepoId: Ornith35BCodeResources.upstreamRepoId,
            upstreamRevision: Ornith35BCodeResources.upstreamRevision,
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: Ornith35BCodeResources.estimatedDownloadBytes,
            defaultCLICommands: ["text code", "api serve", "agent start"],
            apiProfile: .textCode(
                contextWindow: Ornith35BCodeResources.runtimeContextLength,
                maximumOutputTokens: Ornith35BCodeResources.maxOutputTokens
            )
        ),
        ManagedModelSpec(
            // GGUF Qwen3.6-35B-A3B: the CUDA default chat model. Routes through
            // llama.cpp (.codegenGGUF) for the GB10-optimized quantized-MoE
            // kernels (~68 tok/s on GB10 vs ~13 for the MLX path). Same model
            // family as the Apple-Silicon default (text-chat-q36-nano, MLX).
            id: "text-chat-q36-nano-gguf",
            category: .textChat,
            installShape: .singleFile(relativePath: "text-chat-q36-nano-gguf.gguf"),
            hubFallback: HubFallbackConfig(
                repoId: "unsloth/Qwen3.6-35B-A3B-GGUF",
                revision: "main",
                patterns: ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"],
                filePath: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
            ),
            upstreamRepoId: "unsloth/Qwen3.6-35B-A3B-GGUF",
            upstreamRevision: "main",
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: 22 * 1_073_741_824,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .textCode()
        ),
        ManagedModelSpec(
            id: DeepseekV4FlashResources.defaultModelId,
            category: .textChat,
            installShape: .singleFile(relativePath: DeepseekV4FlashResources.managedRelativePath),
            hubFallback: DeepseekV4FlashResources.hubFallbackConfig,
            upstreamRepoId: DeepseekV4FlashResources.defaultRepoId,
            upstreamRevision: DeepseekV4FlashResources.defaultRevision,
            validationKind: .deepseekV4FlashIMatrixGGUF,
            estimatedDownloadBytes: DeepseekV4FlashResources.defaultGGUFByteCount,
            defaultCLICommands: ["api serve", "agent"],
            apiProfile: .deepseekV4Flash()
        ),
        ManagedModelSpec(
            id: LFM2Resources.defaultModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.upstreamRepoId,
                revision: LFM2Resources.upstreamRevision,
                patterns: LFM2Resources.snapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.upstreamRepoId,
            upstreamRevision: LFM2Resources.upstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.upstreamRepoId,
                sourceRevision: LFM2Resources.upstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-8bit/blob/\(LFM2Resources.upstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 10 * 1_073_741_824,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.denseModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.denseUpstreamRepoId,
                revision: LFM2Resources.denseUpstreamRevision,
                patterns: LFM2Resources.denseSnapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.denseUpstreamRepoId,
            upstreamRevision: LFM2Resources.denseUpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.denseUpstreamRepoId,
                sourceRevision: LFM2Resources.denseUpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-2.6B-MLX/blob/\(LFM2Resources.denseUpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_601_108_788,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.visionModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.visionUpstreamRepoId,
                revision: LFM2Resources.visionUpstreamRevision,
                patterns: LFM2Resources.visionSnapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.visionUpstreamRepoId,
            upstreamRevision: LFM2Resources.visionUpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.visionUpstreamRepoId,
                sourceRevision: LFM2Resources.visionUpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-VL-3B-MLX-8bit/blob/\(LFM2Resources.visionUpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.visionEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2(inputModalities: [.text, .image])
        ),
        ManagedModelSpec(
            id: "speech-tts-qwen3-nano",
            category: .speechTTS,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                revision: "main",
                patterns: [
                    "LICENSE*",
                    "README.md",
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model.safetensors",
                    "speech_tokenizer/*",
                    "tokenizer_config.json",
                    "vocab.json",
                ]
            ),
            upstreamRepoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            upstreamRevision: "main",
            validationKind: .qwen3TTS,
            estimatedDownloadBytes: 4_520_158_972,
            defaultCLICommands: ["speech synthesize"]
        ),
        ManagedModelSpec(
            id: "speech-tts-qwen3-customvoice",
            category: .speechTTS,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                revision: "main",
                patterns: [
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model.safetensors",
                    "speech_tokenizer/*",
                    "tokenizer_config.json",
                    "vocab.json",
                ]
            ),
            upstreamRepoId: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            upstreamRevision: "main",
            validationKind: .qwen3TTS,
            estimatedDownloadBytes: 4_520_159_459,
            defaultCLICommands: ["speech synthesize"]
        ),
        ManagedModelSpec(
            id: "speech-asr-qwen3",
            category: .speechASR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/Qwen3-ASR-1.7B-8bit",
                revision: "main",
                patterns: [
                    "config.json",
                    "generation_config.json",
                    "preprocessor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "added_tokens.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "mlx-community/Qwen3-ASR-1.7B-8bit",
            upstreamRevision: "main",
            validationKind: .qwen3ASR,
            normalizationKind: .qwen3ASRNested,
            estimatedDownloadBytes: 2_467_855_342,
            defaultCLICommands: ["speech transcribe"]
        ),
        ManagedModelSpec(
            id: "speech-asr-parakeet",
            category: .speechASR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/parakeet-tdt-0.6b-v3",
                revision: "main",
                patterns: [
                    "config.json",
                    "tokenizer.model",
                    "tokenizer.vocab",
                    "vocab.txt",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "mlx-community/parakeet-tdt-0.6b-v3",
            upstreamRevision: "main",
            validationKind: .parakeet,
            normalizationKind: .parakeetNested,
            estimatedDownloadBytes: 2 * 1_073_741_824,
            defaultCLICommands: ["speech transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.sortformerDiarization.rawValue,
            category: .speechDiarization,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
                revision: sortformerUpstreamRevision,
                patterns: [
                    "README.md",
                    "config.json",
                    "model.safetensors",
                ]
            ),
            upstreamRepoId: "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
            upstreamRevision: sortformerUpstreamRevision,
            validationKind: .sortformer,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 236_108_132,
            defaultCLICommands: ["speech diarize"]
        ),
        ManagedModelSpec(
            id: "text-code-qwen3",
            category: .textCode,
            installShape: .singleFile(relativePath: CodeGenResources.managedRelativePath),
            hubFallback: CodeGenResources.hubFallbackConfig,
            upstreamRepoId: CodeGenResources.defaultRepoId,
            upstreamRevision: CodeGenResources.defaultRevision,
            validationKind: .codegenGGUF,
            aliasKind: .codegenGGUF,
            estimatedDownloadBytes: 48_410_992_032,
            defaultCLICommands: ["text code"],
            apiProfile: .textCode()
        ),
        ManagedModelSpec(
            id: "text-embed-qwen3-0.6b",
            category: .textEmbed,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Qwen/Qwen3-Embedding-0.6B",
                revision: "main",
                patterns: [
                    "config.json",
                    "config_sentence_transformers.json",
                    "generation_config.json",
                    "modules.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "merges.txt",
                    "vocab.json",
                    "1_Pooling/*",
                ]
            ),
            upstreamRepoId: "Qwen/Qwen3-Embedding-0.6B",
            upstreamRevision: "main",
            validationKind: .qwen3Embedding,
            estimatedDownloadBytes: 2 * 1_073_741_824,
            defaultCLICommands: ["text embed"]
        ),
        ManagedModelSpec(
            id: OpenAIPrivacyFilterCatalog.modelId,
            category: .textAnonymize,
            installShape: .directoryRoot,
            hubFallback: OpenAIPrivacyFilterCatalog.hubFallbackConfig,
            upstreamRepoId: OpenAIPrivacyFilterCatalog.defaultRepoId,
            upstreamRevision: OpenAIPrivacyFilterCatalog.defaultRevision,
            validationKind: .privacyFilter,
            estimatedDownloadBytes: 2_826_861_317,
            defaultCLICommands: ["text anonymize"]
        ),
        ManagedModelSpec(
            id: Q35Resources.infinityParser2ProModelId,
            category: .visionOCR,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.infinityParser2ProModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.infinityParser2ProUpstreamRepoId,
            upstreamRevision: Q35Resources.infinityParser2ProUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 131 * 1_073_741_824,
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: Q35Resources.infinityParser2ProInt8ModelId,
            category: .visionOCR,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.infinityParser2ProInt8ModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.infinityParser2ProInt8UpstreamRepoId,
            upstreamRevision: Q35Resources.infinityParser2ProInt8UpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 38 * 1_073_741_824,
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: "vision-ocr-lighton",
            category: .visionOCR,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "lightonai/LightOnOCR-2-1B",
                revision: "main",
                patterns: [
                    "added_tokens.json",
                    "chat_template.jinja",
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "processor_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ]
            ),
            upstreamRepoId: "lightonai/LightOnOCR-2-1B",
            upstreamRevision: "main",
            validationKind: .lightOnOCR,
            estimatedDownloadBytes: 2_022_801_518,
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: "vision-segment-sam31",
            category: .visionSegment,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/sam3.1-bf16",
                revision: sam31MLXRevision,
                patterns: [
                    "LICENSE*",
                    "README.md",
                    "config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: "AEmotionStudio/sam3.1",
                        revision: sam31TokenizerRevision,
                        patterns: [
                            "LICENSE*",
                            "README.md",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "vocab.json",
                            "merges.txt",
                            "special_tokens_map.json",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "mlx-community/sam3.1-bf16",
            upstreamRevision: sam31MLXRevision,
            usageRestriction: usageRestriction(
                summary: "SAM 3.1 uses Meta's custom SAM License, including trade-control, prohibited-use, redistribution, and research-attribution conditions.",
                license: "SAM License",
                sourceRepoId: "mlx-community/sam3.1-bf16",
                sourceRevision: sam31MLXRevision,
                licenseURL: "https://huggingface.co/facebook/sam3.1/blob/main/LICENSE"
            ),
            validationKind: .sam31,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 3_498_072_777,
            defaultCLICommands: ["vision segment"]
        ),
        ManagedModelSpec(
            id: "vision-ground-falcon-perception",
            category: .visionGround,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "tiiuae/Falcon-Perception",
                revision: "main",
                patterns: [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "generation_config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "tiiuae/Falcon-Perception",
            upstreamRevision: "main",
            validationKind: .falconPerception,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 2_534_591_776,
            defaultCLICommands: ["vision ground"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionFloodTerraMindBase.rawValue,
            category: .visionFlood,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: TerraMindFloodResources.sourceRepository,
                revision: TerraMindFloodResources.sourceRevision,
                patterns: [
                    "README.md",
                    "LICENSE*",
                    "NOTICE*",
                    TerraMindFloodResources.sourceCheckpointFilename,
                    TerraMindFloodResources.sourceConfigurationFilename,
                ]
            ),
            upstreamRepoId: TerraMindFloodResources.sourceRepository,
            upstreamRevision: TerraMindFloodResources.sourceRevision,
            validationKind: .terramindFlood,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 673_199_088,
            defaultCLICommands: ["geo flood"]
        ),
        ManagedModelSpec(
            id: FaceAnalysisResources.modelID,
            category: .visionFace,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "deepghs/insightface",
                revision: "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb",
                patterns: [
                    "LICENSE*",
                    "README.md",
                    FaceAnalysisResources.detectorRelativePath,
                    FaceAnalysisResources.recognizerRelativePath,
                ]
            ),
            upstreamRepoId: "deepghs/insightface",
            upstreamRevision: "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb",
            usageRestriction: usageRestriction(
                summary: "InsightFace Buffalo-L pretrained weights are limited to non-commercial research use.",
                license: "InsightFace pretrained model non-commercial research terms",
                sourceRepoId: "deepghs/insightface",
                sourceRevision: "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb",
                licenseURL: "https://github.com/deepinsight/insightface#license"
            ),
            validationKind: .insightFaceBuffaloL,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: FaceAnalysisResources.detectorByteCount
                + FaceAnalysisResources.recognizerByteCount,
            defaultCLICommands: [
                "vision face detect",
                "vision face embed",
                "vision face compare",
                "vision face batch",
            ]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
            category: .visionGeometry,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Ruicheng/moge-2-vits-normal-onnx",
                revision: "e50ffda41565591092adea54c6ac83d6212e1e23",
                patterns: ["model.onnx", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "Ruicheng/moge-2-vits-normal-onnx",
            upstreamRevision: "e50ffda41565591092adea54c6ac83d6212e1e23",
            validationKind: .moge2,
            estimatedDownloadBytes: 140_852_051,
            defaultCLICommands: ["vision geometry"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionDepthVDASmall.rawValue,
            category: .visionDepth,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "depth-anything/Video-Depth-Anything-Small",
                revision: "256875362cff76724b920335dfb4b29dd611f66e",
                patterns: ["video_depth_anything_vits.pth", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "depth-anything/Video-Depth-Anything-Small",
            upstreamRevision: "256875362cff76724b920335dfb4b29dd611f66e",
            validationKind: .videoDepthAnything,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 116_440_756,
            defaultCLICommands: ["vision depth-video"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
            category: .visionDepth,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "depth-anything/Metric-Video-Depth-Anything-Small",
                revision: "273d090f2ce17df50c2872d82c8322c45da5b4dd",
                patterns: ["metric_video_depth_anything_vits.pth", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "depth-anything/Metric-Video-Depth-Anything-Small",
            upstreamRevision: "273d090f2ce17df50c2872d82c8322c45da5b4dd",
            validationKind: .videoDepthAnything,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 116_444_063,
            defaultCLICommands: ["vision depth-video"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionGeometryDA3Small.rawValue,
            category: .visionGeometry,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "depth-anything/DA3-SMALL",
                revision: "e08cab65ca0ec38e7826075418411ab90cab4da3",
                patterns: ["config.json", "model.safetensors", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "depth-anything/DA3-SMALL",
            upstreamRevision: "e08cab65ca0ec38e7826075418411ab90cab4da3",
            validationKind: .depthAnything3,
            estimatedDownloadBytes: 137_248_940,
            defaultCLICommands: ["vision geometry-multiview"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.image3DTripoSR.rawValue,
            category: .image3D,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "stabilityai/TripoSR",
                revision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
                patterns: ["config.yaml", "model.ckpt", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "stabilityai/TripoSR",
            upstreamRevision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
            validationKind: .tripoSR,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_677_247_729,
            defaultCLICommands: ["image reconstruct-3d", "vision image-to-3d"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.image3DInstantMeshBase.rawValue,
            category: .image3D,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "TencentARC/InstantMesh",
                revision: "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
                patterns: ["instant_mesh_base.ckpt", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "TencentARC/InstantMesh",
            upstreamRevision: "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
            validationKind: .instantMesh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_253_574_354,
            defaultCLICommands: [
                "image reconstruct-3d-multiview",
                "vision image-to-3d-multiview",
            ]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.image3DTrellis2.rawValue,
            category: .image3D,
            installShape: .structuredRoot,
            hubFallback: Trellis2Resources.primaryHubFallback,
            mountedHubFallbacks: Trellis2Resources.mountedHubFallbacks,
            upstreamRepoId: Trellis2Resources.repository,
            upstreamRevision: Trellis2Resources.revision,
            usageRestriction: usageRestriction(
                summary: "TRELLIS.2 downloads a manually gated DINOv3 component governed by Meta's custom DINOv3 License.",
                component: "DINOv3 image encoder",
                license: "DINOv3 License",
                sourceRepoId: Trellis2Resources.dinoV3Repository,
                sourceRevision: Trellis2Resources.dinoV3Revision,
                licenseURL: "https://ai.meta.com/resources/models-and-libraries/dinov3-license/"
            ),
            validationKind: .trellis2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 11_010_775_158,
            defaultCLICommands: [
                "image reconstruct-3d-trellis2",
                "vision image-to-3d-trellis2",
            ]
        ),
        ManagedModelSpec(
            id: "music-acestep",
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "config.json",
                    "acestep-v15-turbo/*",
                    "acestep-5Hz-lm-1.7B/*",
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            upstreamRepoId: "ACE-Step/Ace-Step1.5",
            upstreamRevision: aceStepSharedRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 10_092_095_357,
            defaultCLICommands: ["music generate", "music analyze"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLBase.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-base",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-base",
                        revision: aceStepXLBaseRevision,
                        patterns: [
                            "apg_guidance.py",
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_base.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-base",
            upstreamRevision: aceStepXLBaseRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 23 * 1_073_741_824,
            defaultCLICommands: ["music generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLSFT.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-sft",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-sft",
                        revision: aceStepXLSFTRevision,
                        patterns: [
                            "apg_guidance.py",
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_base.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-sft",
            upstreamRevision: aceStepXLSFTRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 23 * 1_073_741_824,
            defaultCLICommands: ["music generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLTurbo.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-turbo",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-turbo",
                        revision: aceStepXLTurboRevision,
                        patterns: [
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_turbo.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-turbo",
            upstreamRevision: aceStepXLTurboRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 23 * 1_073_741_824,
            defaultCLICommands: ["music generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-turbo",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-turbo",
                        revision: aceStepXLTurboRevision,
                        patterns: [
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_turbo.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "acestep-5Hz-lm-4B",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-5Hz-lm-4B",
                        revision: aceStepLM4BRevision,
                        patterns: [
                            "*.json",
                            "*.safetensors",
                            "*.jinja",
                            "merges.txt",
                            "vocab.json",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-turbo + ACE-Step/acestep-5Hz-lm-4B",
            upstreamRevision: "\(aceStepXLTurboRevision)+\(aceStepLM4BRevision)",
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 32 * 1_073_741_824,
            defaultCLICommands: ["music generate", "music analyze"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepLM17B.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "acestep-5Hz-lm-1.7B/*",
                ]
            ),
            upstreamRepoId: "ACE-Step/Ace-Step1.5",
            upstreamRevision: aceStepSharedRevision,
            validationKind: .aceStepLM,
            normalizationKind: .musicACEStepLM,
            estimatedDownloadBytes: 4 * 1_073_741_824,
            defaultCLICommands: ["music generate", "music analyze", "music serve"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepLM4B.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/acestep-5Hz-lm-4B",
                revision: aceStepLM4BRevision,
                patterns: [
                    "*.json",
                    "*.safetensors",
                    "*.jinja",
                    "merges.txt",
                    "vocab.json",
                ]
            ),
            upstreamRepoId: "ACE-Step/acestep-5Hz-lm-4B",
            upstreamRevision: aceStepLM4BRevision,
            validationKind: .aceStepLM,
            normalizationKind: .musicACEStepLM,
            estimatedDownloadBytes: 9 * 1_073_741_824,
            defaultCLICommands: ["music generate", "music analyze", "music serve"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.miniMaxMusic3.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MiniMaxMusic3Resources.repository,
                revision: MiniMaxMusic3Resources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "config.json",
                    "modular_model_index.json",
                    "condition_encoder/*",
                    "language_model/*",
                    "rvq_depth_decoder/*",
                    "scheduler/*",
                    "tokenizer/*",
                    "transformer/*",
                    "vocoder/*",
                ]
            ),
            upstreamRepoId: MiniMaxMusic3Resources.repository,
            upstreamRevision: MiniMaxMusic3Resources.revision,
            usageRestriction: usageRestriction(
                summary: "MiniMax Music 3 uses the custom MiniMax-Music3 Community License, including product attribution, revenue-threshold authorization, and hosted-generation safeguards.",
                component: "MiniMax Music 3 weights",
                license: "MiniMax-Music3 Community License",
                sourceRepoId: MiniMaxMusic3Resources.repository,
                sourceRevision: MiniMaxMusic3Resources.revision,
                licenseURL: "https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/\(MiniMaxMusic3Resources.revision)/LICENSE"
            ),
            validationKind: .miniMaxMusic3,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: MiniMaxMusic3Resources.estimatedDownloadBytes,
            defaultCLICommands: ["music generate", "music serve"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.magentaRT2Small.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: magentaRT2UpstreamRepoId,
                revision: magentaRT2UpstreamRevision,
                patterns: [
                    "models/mrt2_small/mrt2_small.mlxfn",
                    "models/mrt2_small/mrt2_small_state.safetensors",
                ] + magentaRT2ResourcePatterns
            ),
            upstreamRepoId: magentaRT2UpstreamRepoId,
            upstreamRevision: magentaRT2UpstreamRevision,
            validationKind: .magentaRT2,
            estimatedDownloadBytes: 1_840_072_891,
            defaultCLICommands: ["music generate", "music realtime"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.magentaRT2Base.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: magentaRT2UpstreamRepoId,
                revision: magentaRT2UpstreamRevision,
                patterns: [
                    "models/mrt2_base/mrt2_base.mlxfn",
                    "models/mrt2_base/mrt2_base_state.safetensors",
                ] + magentaRT2ResourcePatterns
            ),
            upstreamRepoId: magentaRT2UpstreamRepoId,
            upstreamRevision: magentaRT2UpstreamRevision,
            validationKind: .magentaRT2,
            estimatedDownloadBytes: 4_164_096_058,
            defaultCLICommands: ["music generate", "music realtime"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.muScriptorSmall.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "MuScriptor/muscriptor-small",
                revision: "8c127f603b807520fa465c838e9bfee8a91ada4e",
                patterns: ["LICENSE*", "README.md", "config.json", "model.safetensors"]
            ),
            upstreamRepoId: "MuScriptor/muscriptor-small",
            upstreamRevision: "8c127f603b807520fa465c838e9bfee8a91ada4e",
            usageRestriction: muScriptorUsageRestriction(
                sourceRepoId: "MuScriptor/muscriptor-small",
                sourceRevision: "8c127f603b807520fa465c838e9bfee8a91ada4e"
            ),
            validationKind: .muScriptor,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 412_000_000,
            defaultCLICommands: ["music transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.muScriptorMedium.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "MuScriptor/muscriptor-medium",
                revision: "f32236969308476e01fd3aae67357de5feb05a2d",
                patterns: ["LICENSE*", "README.md", "config.json", "model.safetensors"]
            ),
            upstreamRepoId: "MuScriptor/muscriptor-medium",
            upstreamRevision: "f32236969308476e01fd3aae67357de5feb05a2d",
            usageRestriction: muScriptorUsageRestriction(
                sourceRepoId: "MuScriptor/muscriptor-medium",
                sourceRevision: "f32236969308476e01fd3aae67357de5feb05a2d"
            ),
            validationKind: .muScriptor,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_230_000_000,
            defaultCLICommands: ["music transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.muScriptorLarge.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "MuScriptor/muscriptor-large",
                revision: "8809fdfbed2affa7ade94a7059e746e3880720e7",
                patterns: ["LICENSE*", "README.md", "config.json", "model.safetensors"]
            ),
            upstreamRepoId: "MuScriptor/muscriptor-large",
            upstreamRevision: "8809fdfbed2affa7ade94a7059e746e3880720e7",
            usageRestriction: muScriptorUsageRestriction(
                sourceRepoId: "MuScriptor/muscriptor-large",
                sourceRevision: "8809fdfbed2affa7ade94a7059e746e3880720e7"
            ),
            validationKind: .muScriptor,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5_620_000_000,
            defaultCLICommands: ["music transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.roFormerViperX1297.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "bs_roformer/vocals_viperx/config.yaml",
                    "bs_roformer/vocals_viperx/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 639_114_645,
            defaultCLICommands: ["music separate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.roFormerFourStem.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "bs_roformer/multistem/config.yaml",
                    "bs_roformer/multistem/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 526_973_074,
            defaultCLICommands: ["music separate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.melRoFormerDereverb.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "mel_band_roformer/dereverb/config.yaml",
                    "mel_band_roformer/dereverb/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 912_891_178,
            defaultCLICommands: ["music separate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.melRoFormerDenoise.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "mel_band_roformer/denoise/config.yaml",
                    "mel_band_roformer/denoise/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 912_890_953,
            defaultCLICommands: ["music separate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.apBWE16kTo48k.rawValue,
            category: .audio,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: APBWEResources.artifactRepository,
                revision: APBWEResources.artifactRevision,
                patterns: APBWEResources.pins.map(\.filename)
            ),
            upstreamRepoId: APBWEResources.sourceRepository,
            upstreamRevision: APBWEResources.sourceRevision,
            validationKind: .apBWE,
            estimatedDownloadBytes: 119_099_717,
            defaultCLICommands: ["audio enhance"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.univerSRAudio.rawValue,
            category: .audio,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: UniverSRResources.artifactRepository,
                revision: UniverSRResources.artifactRevision,
                patterns: UniverSRResources.pins.map(\.filename)
            ),
            upstreamRepoId: UniverSRResources.sourceRepository,
            upstreamRevision: UniverSRResources.sourceRevision,
            validationKind: .univerSR,
            estimatedDownloadBytes: 229_074_334,
            defaultCLICommands: ["audio enhance"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshDFlow.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshDFlowSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerA/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5 * 1_073_741_824,
            defaultCLICommands: ["sfx generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshFlow.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshFlowSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerA/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5 * 1_073_741_824,
            defaultCLICommands: ["sfx generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshClap.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshCLAPSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/Woosh-CLAP/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .wooshClap,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 2 * 1_073_741_824,
            defaultCLICommands: ["sfx clap"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshSynchformer.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.synchformerRepoId,
                revision: MMAudioResources.convertedWeightsRevision,
                patterns: wooshSynchformerSnapshotPatterns
            ),
            upstreamRepoId: WooshResources.synchformerRepoId,
            upstreamRevision: MMAudioResources.convertedWeightsRevision,
            usageRestriction: wooshSynchformerUsageRestriction,
            validationKind: .wooshSynchformer,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 475 * 1_048_576,
            defaultCLICommands: ["sfx video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshVFlow8s.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshVFlow8sSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerV/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 6 * 1_073_741_824,
            defaultCLICommands: ["sfx video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshDVFlow8s.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshDVFlow8sSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerV/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 6 * 1_073_741_824,
            defaultCLICommands: ["sfx video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.mmaudioLarge44kV2.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MMAudioResources.convertedWeightsRepoID,
                revision: MMAudioResources.convertedWeightsRevision,
                patterns: mmaudioSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "clip",
                    hubFallback: HubFallbackConfig(
                        repoId: MMAudioResources.clipRepoID,
                        revision: MMAudioResources.clipRevision,
                        patterns: mmaudioCLIPTokenizerPatterns
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "bigvgan",
                    hubFallback: HubFallbackConfig(
                        repoId: MMAudioResources.bigVGANRepoID,
                        revision: MMAudioResources.bigVGANRevision,
                        patterns: mmaudioBigVGANPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(MMAudioResources.upstreamRepoID)@\(MMAudioResources.upstreamRevision)",
            upstreamRevision: MMAudioResources.upstreamRevision,
            usageRestriction: mmaudioUsageRestriction,
            validationKind: .mmaudio,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5_700_000_000,
            defaultCLICommands: ["sfx generate", "sfx video generate"]
        ),
        ManagedModelSpec(
            id: "video-ltx-av",
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/LTX-2-distilled-bf16",
                revision: ltx2DistilledRevision,
                patterns: [
                    "LICENSE*",
                    "README.md",
                    "ltx-2-19b-distilled.safetensors",
                    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
                    "text_encoder/*",
                    "tokenizer/*",
                ]
            ),
            upstreamRepoId: "mlx-community/LTX-2-distilled-bf16",
            upstreamRevision: ltx2DistilledRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: "mlx-community/LTX-2-distilled-bf16",
                sourceRevision: ltx2DistilledRevision
            ),
            validationKind: .ltxVideo,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 93_069_609_104,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.ltxVideo23AVMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: ltx23MLXUpstreamRepoId,
                revision: ltx23MLXRevision,
                patterns: ltx23MLXSnapshotPatterns
            ),
            upstreamRepoId: ltx23MLXUpstreamRepoId,
            upstreamRevision: ltx23MLXRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: ltx23MLXUpstreamRepoId,
                sourceRevision: ltx23MLXRevision,
                additionalTerms: [ltxGemmaTextEncoderUsageTerm]
            ),
            validationKind: .ltxVideo23MLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 120 * 1_073_741_824,
            defaultCLICommands: ["video generate"],
            companionModelIDs: [ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.ltxVideo23FullMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: ltx23MLXUpstreamRepoId,
                revision: ltx23MLXRevision,
                patterns: ltx23FullMLXSnapshotPatterns
            ),
            upstreamRepoId: ltx23MLXUpstreamRepoId,
            upstreamRevision: ltx23MLXRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: ltx23MLXUpstreamRepoId,
                sourceRevision: ltx23MLXRevision,
                additionalTerms: [ltxGemmaTextEncoderUsageTerm]
            ),
            validationKind: .ltxVideo23FullMLX,
            runtimeAutoDownloadAllowed: false,
            resolutionFallbackIDs: [ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue],
            estimatedDownloadBytes: 56_000_000_000,
            defaultCLICommands: ["video generate"],
            companionModelIDs: [ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: ltx23MLXUpstreamRepoId,
                revision: ltx23MLXRevision,
                patterns: ltx23A2VMLXSnapshotPatterns
            ),
            upstreamRepoId: ltx23MLXUpstreamRepoId,
            upstreamRevision: ltx23MLXRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: ltx23MLXUpstreamRepoId,
                sourceRevision: ltx23MLXRevision,
                additionalTerms: [ltxGemmaTextEncoderUsageTerm]
            ),
            validationKind: .ltxVideo23A2VMLX,
            runtimeAutoDownloadAllowed: false,
            resolutionFallbackIDs: [ModelResolver.ModelID.ltxVideo23FullMLX.rawValue],
            estimatedDownloadBytes: 54_000_000_000,
            defaultCLICommands: ["video generate"],
            companionModelIDs: [ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: LTX25Resources.sourceRepository,
                revision: LTX25Resources.sourceRevision,
                patterns: LTX25Resources.snapshotPatterns
            ),
            upstreamRepoId: LTX25Resources.sourceRepository,
            upstreamRevision: LTX25Resources.sourceRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: LTX25Resources.sourceRepository,
                sourceRevision: LTX25Resources.sourceRevision,
                additionalTerms: [ltx25GemmaTextEncoderUsageTerm]
            ),
            validationKind: .ltxVideo25,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LTX25Resources.estimatedDownloadBytes,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.ltxVideo25FullBF16.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: LTX25Resources.sourceRepository,
                revision: LTX25Resources.sourceRevision,
                patterns: LTX25Resources.fullSnapshotPatterns
            ),
            upstreamRepoId: LTX25Resources.sourceRepository,
            upstreamRevision: LTX25Resources.sourceRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: LTX25Resources.sourceRepository,
                sourceRevision: LTX25Resources.sourceRevision,
                additionalTerms: [ltx25GemmaTextEncoderUsageTerm]
            ),
            validationKind: .ltxVideo25,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LTX25Resources.fullEstimatedDownloadBytes,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wan22TI2V5BMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: Wan2Resources.managedRepoID,
                revision: Wan2Resources.managedRevision,
                patterns: Wan2Resources.snapshotPatterns
            ),
            upstreamRepoId: Wan2Resources.managedRepoID,
            upstreamRevision: Wan2Resources.managedRevision,
            validationKind: .wan22TI2VMLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 24_200_000_000,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.miniMaxH3FL2VAMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MiniMaxH3Resources.artifactRepository,
                revision: MiniMaxH3Resources.artifactRevision,
                patterns: MiniMaxH3Resources.compactArtifactFiles
            ),
            upstreamRepoId: MiniMaxH3Resources.artifactRepository,
            upstreamRevision: MiniMaxH3Resources.artifactRevision,
            usageRestriction: miniMaxH3UsageRestriction,
            validationKind: .miniMaxH3MLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 46_250_104_566,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MiniMaxH3Resources.compactBF16ArtifactRepository,
                revision: MiniMaxH3Resources.compactBF16ArtifactRevision,
                patterns: MiniMaxH3Resources.compactBF16AndQ8ArtifactFiles
            ),
            upstreamRepoId: MiniMaxH3Resources.compactBF16ArtifactRepository,
            upstreamRevision: MiniMaxH3Resources.compactBF16ArtifactRevision,
            usageRestriction: miniMaxH3UsageRestriction,
            validationKind: .miniMaxH3MLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 76_861_026_073,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MiniMaxH3Resources.q8ArtifactRepository,
                revision: MiniMaxH3Resources.q8ArtifactRevision,
                patterns: MiniMaxH3Resources.compactBF16AndQ8ArtifactFiles
            ),
            upstreamRepoId: MiniMaxH3Resources.q8ArtifactRepository,
            upstreamRevision: MiniMaxH3Resources.q8ArtifactRevision,
            usageRestriction: miniMaxH3UsageRestriction,
            validationKind: .miniMaxH3MLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 58_075_175_639,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MiniMaxH3Resources.ref2vaArtifactRepository,
                revision: MiniMaxH3Resources.ref2vaArtifactRevision,
                patterns: MiniMaxH3Resources.ref2vaArtifactFiles
            ),
            upstreamRepoId: MiniMaxH3Resources.ref2vaArtifactRepository,
            upstreamRevision: MiniMaxH3Resources.ref2vaArtifactRevision,
            usageRestriction: miniMaxH3UsageRestriction,
            validationKind: .miniMaxH3MLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 70_941_103_245,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.cosmos3EdgeMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: Cosmos3Resources.officialRepoID,
                revision: Cosmos3Resources.officialRevision,
                patterns: Cosmos3Resources.snapshotPatterns
            ),
            upstreamRepoId: Cosmos3Resources.officialRepoID,
            upstreamRevision: Cosmos3Resources.officialRevision,
            validationKind: .cosmos3EdgeMLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 9_200_000_000,
            defaultCLICommands: [
                "video cosmos3",
                "video cosmos3 --mode reasoner",
                "world serve --backend cosmos3",
            ]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.scail2Video14BMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: SCAIL2Resources.managedRepoID,
                revision: SCAIL2Resources.managedRevision,
                patterns: SCAIL2Resources.snapshotPatterns
            ),
            upstreamRepoId: SCAIL2Resources.upstreamRepoID,
            upstreamRevision: SCAIL2Resources.upstreamRevision,
            validationKind: .scail2MLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 46_648_000_000,
            defaultCLICommands: ["video animate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.dreamXWorld5BARMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            upstreamRepoId: Wan2DreamXCausalResources.upstreamRepoID,
            upstreamRevision: Wan2DreamXCausalResources.upstreamRevision,
            validationKind: .dreamXCausalMLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 10_566_339_320,
            defaultCLICommands: ["world serve"],
            companionModelIDs: [ModelResolver.ModelID.wan22TI2V5BMLX.rawValue]
        ),
    ] + geoExpansionSpecs

    public static var allModelIDs: [String] {
        allSpecs.map(\.id)
    }

    public static func spec(for id: String) -> ManagedModelSpec? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allKnownSpecs.first {
            $0.id == normalized || $0.upstreamRepoId?.lowercased() == normalized
        }
    }

    public static func missingHubSourceMessage(for modelId: String) -> String {
        "Model \(modelId) does not have a Hugging Face Hub source in this public build. Install it from a local path or choose a model listed by `mere.run model capabilities --recommended`."
    }
}

private extension ManagedModelCatalog {
    static var allKnownSpecs: [ManagedModelSpec] {
        allSpecs + companionSpecs
    }

    static var companionSpecs: [ManagedModelSpec] {
        [
            ManagedModelSpec(
                id: Gemma4MTPResources.modelId,
                category: .textChat,
                installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(
                    repoId: Gemma4MTPResources.upstreamModelId,
                    patterns: Gemma4MTPResources.snapshotPatterns
                ),
                upstreamRepoId: Gemma4MTPResources.upstreamModelId,
                validationKind: .gemma4MTPAssistant,
                runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: 4 * 1_073_741_824
            ),
            ManagedModelSpec(
                id: LagunaResources.dflashModelID,
                category: .textChat,
                installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(
                    repoId: LagunaResources.dflashUpstreamModelID,
                    revision: LagunaResources.dflashUpstreamRevision,
                    patterns: LagunaResources.dflashSnapshotPatterns
                ),
                upstreamRepoId: LagunaResources.dflashUpstreamModelID,
                upstreamRevision: LagunaResources.dflashUpstreamRevision,
                validationKind: .lagunaDFlash,
                runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: LagunaResources.dflashEstimatedDownloadBytes
            ),
            ManagedModelSpec(
                id: MuseGlimmerResources.dflash2ModelId,
                category: .textChat,
                installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(
                    repoId: MuseGlimmerResources.dflash2UpstreamRepoId,
                    revision: MuseGlimmerResources.dflash2UpstreamRevision,
                    patterns: MuseGlimmerResources.dflash2SnapshotPatterns
                ),
                upstreamRepoId: MuseGlimmerResources.dflash2UpstreamRepoId,
                upstreamRevision: MuseGlimmerResources.dflash2UpstreamRevision,
                validationKind: .museGlimmerAssistant,
                runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: MuseGlimmerResources.dflash2EstimatedDownloadBytes
            ),
            ManagedModelSpec(
                id: MuseGlimmerResources.assistantModelId,
                category: .textChat,
                installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(
                    repoId: MuseGlimmerResources.assistantUpstreamRepoId,
                    revision: MuseGlimmerResources.assistantUpstreamRevision,
                    patterns: MuseGlimmerResources.assistantSnapshotPatterns
                ),
                upstreamRepoId: MuseGlimmerResources.assistantUpstreamRepoId,
                upstreamRevision: MuseGlimmerResources.assistantUpstreamRevision,
                usageRestriction: ManagedModelUsageRestriction(
                    summary: "Apache-2.0 model subject to Meta's bundled usage policy; upstream states it is not intended for download or use by people under 18.",
                    terms: [
                        ManagedModelUsageTerm(
                            component: "Muse Glimmer 30B DFlash assistant",
                            license: "Apache-2.0 with upstream usage policy",
                            summary: "Review LICENSE and USAGE_POLICY.md before installing or deploying.",
                            sourceRepoId: MuseGlimmerResources.assistantUpstreamRepoId,
                            sourceRevision: MuseGlimmerResources.assistantUpstreamRevision,
                            licenseURL: "https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant/blob/\(MuseGlimmerResources.assistantUpstreamRevision)/USAGE_POLICY.md"
                        ),
                    ]
                ),
                validationKind: .museGlimmerAssistant,
                runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: MuseGlimmerResources.assistantEstimatedDownloadBytes
            ),
            ManagedModelSpec(
                id: NemotronHResources.dsparkModelID,
                category: .textChat,
                installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(
                    repoId: NemotronHResources.dsparkArtifactRepoID,
                    revision: NemotronHResources.dsparkArtifactRevision,
                    patterns: NemotronHResources.dsparkSnapshotPatterns
                ),
                upstreamRepoId: NemotronHResources.dsparkArtifactRepoID,
                upstreamRevision: NemotronHResources.dsparkArtifactRevision,
                validationKind: .nemotronHDSpark,
                runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: NemotronHResources.dsparkEstimatedDownloadBytes
            ),
            ManagedModelSpec(
                id: ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue,
                category: .textChat,
                installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(
                    repoId: ltxGemma3TextEncoderRepoId,
                    revision: ltxGemma3TextEncoderRevision,
                    patterns: ltxGemma3TextEncoderSnapshotPatterns
                ),
                upstreamRepoId: ltxGemma3TextEncoderRepoId,
                upstreamRevision: ltxGemma3TextEncoderRevision,
                validationKind: .hfTextChat,
                runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: 8 * 1_073_741_824
            ),
        ]
    }
}

public extension ManagedModelSpec {
    var modelID: ModelResolver.ModelID? {
        ModelResolver.ModelID(rawValue: id)
    }

    var canBePulledWithoutConfiguration: Bool {
        hubFallback != nil || !mountedHubFallbacks.isEmpty
    }

    var usesPinnedGeometryArtifacts: Bool {
        switch validationKind {
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR, .instantMesh, .trellis2:
            true
        default:
            false
        }
    }

    var requiresManagedConversion: Bool {
        switch validationKind {
        case .instantMesh, .terramindFlood, .terramindFire, .tessera, .olmoEarth:
            true
        default:
            false
        }
    }

    func managedConversionGuidance(at rootURL: URL) -> String? {
        guard requiresManagedConversion else { return nil }
        if validationKind == .terramindFlood {
            let source = rootURL.appendingPathComponent(TerraMindFloodResources.sourceCheckpointFilename).path
            let configuration = rootURL.appendingPathComponent(
                TerraMindFloodResources.sourceConfigurationFilename
            ).path
            return "Pinned TerraMind Flood source downloaded at \(source). Deterministic conversion is required "
                + "before native MLX inference; run scripts/convert-terramind-flood-mlx.py "
                + "--checkpoint \"\(source)\" --configuration \"\(configuration)\" "
                + "--output \"\(rootURL.path)\" --dtype float32. FP16 is intentionally unsupported by parity evidence."
        }
        if validationKind == .terramindFire {
            let source = rootURL.appendingPathComponent(TerraMindFireResources.sourceCheckpointFilename).path
            let configuration = rootURL.appendingPathComponent(
                TerraMindFireResources.sourceConfigurationFilename
            ).path
            return "Pinned TerraMind Fire source downloaded at \(source). Deterministic conversion is required "
                + "before native MLX inference; run scripts/convert-terramind-fire-mlx.py "
                + "--checkpoint \"\(source)\" --configuration \"\(configuration)\" "
                + "--output \"\(rootURL.path)\" --dtype float32."
        }
        if validationKind == .tessera, let source = TESSERAResources.spec(for: id) {
            let checkpoint = rootURL.appendingPathComponent(source.sourceCheckpointFilename).path
            return "Pinned TESSERA v2 \(source.variant.rawValue) source downloaded at \(checkpoint). "
                + "Deterministic conversion is required before native MLX inference; run "
                + "scripts/convert-tessera-v2-mlx.py --variant \(source.variant.rawValue) "
                + "--checkpoint \"\(checkpoint)\" --output \"\(rootURL.path)\" --dtype float32."
        }
        if validationKind == .olmoEarth, let source = OlmoEarthResources.spec(for: id) {
            let weights = rootURL.appendingPathComponent(OlmoEarthResources.sourceWeightsFilename).path
            let configuration = rootURL.appendingPathComponent(
                OlmoEarthResources.sourceConfigurationFilename
            ).path
            return "Pinned OlmoEarth v1.2 \(source.variant.rawValue) source downloaded at \(weights). "
                + "Deterministic conversion is required before native MLX inference; run "
                + "scripts/convert-olmoearth-v12-mlx.py --variant \(source.variant.rawValue) "
                + "--weights \"\(weights)\" --configuration \"\(configuration)\" "
                + "--output \"\(rootURL.path)\" --dtype float32."
        }
        let source = rootURL.appendingPathComponent("instant_mesh_base.ckpt").path
        let output = rootURL.appendingPathComponent(
            InstantMeshResources.managedConvertedDirectoryName,
            isDirectory: true
        ).path
        let license = rootURL.appendingPathComponent("LICENSE").path
        return "Pinned InstantMesh source downloaded at \(source). Conversion is required before runtime; "
            + "run scripts/model-conversion/convert_instantmesh_base.py "
            + "--source \"\(source)\" --output \"\(output)\" --license-file \"\(license)\"."
    }

    func hasAnyManagedDownloadSource() -> Bool {
        hubFallback != nil || !mountedHubFallbacks.isEmpty
    }

    func normalizedRootURL(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let base = rootURL.resolvingSymlinksInPath()
        if validationKind == .lfm2 {
            return LFM2Resources.normalizedRootURL(base, fileManager: fileManager)
        }
        switch normalizationKind {
        case .none, .musicACEStep:
            return base
        case .musicACEStepLM:
            if ACEStep5HzLMResources(rootURL: base).validate(fileManager: fileManager).isEmpty {
                return base
            }
            let candidates = [
                "acestep-5Hz-lm-1.7B",
                "acestep-5hz-lm-1.7b",
                "acestep-5Hz-lm-4B",
                "acestep-5hz-lm-4b",
            ]
            return candidates
                .map { base.appendingPathComponent($0, isDirectory: true) }
                .first {
                    ACEStep5HzLMResources(rootURL: $0).validate(fileManager: fileManager).isEmpty
                } ?? base
        case .qwen3ASRNested:
            let direct = missingPathsWithoutNormalization(in: base, fileManager: fileManager)
            if direct.isEmpty {
                return base
            }
            let nested = base.appendingPathComponent(id, isDirectory: true)
            return missingPathsWithoutNormalization(in: nested, fileManager: fileManager).isEmpty ? nested : base
        case .parakeetNested:
            let direct = missingPathsWithoutNormalization(in: base, fileManager: fileManager)
            if direct.isEmpty {
                return base
            }
            let nested = base.appendingPathComponent(id, isDirectory: true)
            return missingPathsWithoutNormalization(in: nested, fileManager: fileManager).isEmpty ? nested : base
        }
    }

    func missingPaths(in rootURL: URL, fileManager: FileManager = .default) -> [URL] {
        missingPathsWithoutNormalization(
            in: normalizedRootURL(rootURL, fileManager: fileManager),
            fileManager: fileManager
        )
    }

    private func missingPathsWithoutNormalization(in rootURL: URL, fileManager: FileManager = .default) -> [URL] {
        switch validationKind {
        case .flux2Klein:
            return Self.missingDiffusersImagePaths(in: rootURL, fileManager: fileManager)
        case .bonsaiImage:
            return Self.missingBonsaiImagePaths(in: rootURL, fileManager: fileManager)
        case .zimageTurbo:
            return ZImageTurboResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .hidreamO1:
            return HiDreamO1Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .krea2:
            return Krea2Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .ideogram4SDNQ:
            return Ideogram4Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .gemma4:
            return Gemma4Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .gemma4Unified:
            return Gemma4Resources(rootURL: rootURL).validate(
                fileManager: fileManager,
                requireUnifiedProcessor: true
            )
        case .gemma4MTPAssistant:
            return Gemma4MTPResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .laguna:
            return LagunaResources.missingTargetFiles(rootURL: rootURL, fileManager: fileManager)
        case .lagunaDFlash:
            return LagunaResources.missingDFlashFiles(rootURL: rootURL, fileManager: fileManager)
        case .q35:
            let resources = Q35Resources(rootURL: rootURL)
            var missing = resources.validate(fileManager: fileManager)
            if id == Q35Resources.q38TwentySevenB4BitModelId {
                missing.append(contentsOf: resources.validateQ38MTPComponent(fileManager: fileManager))
            }
            return missing
        case .lfm2:
            return LFM2Resources(rootURL: rootURL).validate(
                fileManager: fileManager,
                requireVisionProcessor: id == LFM2Resources.visionModelId
            )
        case .inkling:
            return InklingResources.validate(rootURL: rootURL, fileManager: fileManager)
        case .museGlimmer:
            return MuseGlimmerResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .museGlimmerAssistant:
            return MuseGlimmerResources.validateAssistant(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .nemotronH:
            return NemotronHResources.missingTargetFiles(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .nemotronHDSpark:
            return NemotronHResources.missingDSparkFiles(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .nemotronOmni:
            return NemotronOmniResources.missingTargetFiles(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .sam31:
            return SAM31Resources(modelRootURL: rootURL).missingRequiredPaths(fileManager: fileManager)
        case .falconPerception:
            return FalconPerceptionResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .terramindFlood:
            return TerraMindFloodResources.missingSourcePaths(in: rootURL, fileManager: fileManager)
        case .terramindFire:
            return TerraMindFireResources.missingSourcePaths(in: rootURL, fileManager: fileManager)
        case .tessera:
            return TESSERAResources.missingSourcePaths(for: id, in: rootURL, fileManager: fileManager)
        case .olmoEarth:
            return OlmoEarthResources.missingSourcePaths(in: rootURL, fileManager: fileManager)
        case .insightFaceBuffaloL:
            return FaceAnalysisResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR:
            guard let pin = GeometryModelPins.pin(for: id) else { return [rootURL] }
            return Self.invalidPinnedArtifacts(pin.runtimeArtifacts, in: rootURL, fileManager: fileManager)
        case .instantMesh:
            if Self.invalidInstantMeshNativeArtifacts(in: rootURL, fileManager: fileManager).isEmpty {
                return []
            }
            guard let pin = GeometryModelPins.pin(for: id) else { return [rootURL] }
            return Self.invalidPinnedArtifacts(pin.artifacts, in: rootURL, fileManager: fileManager)
        case .trellis2:
            return Trellis2Resources.validate(rootURL: rootURL, fileManager: fileManager)
        case .qwen3TTS:
            return Self.missingQwen3TTSPaths(in: rootURL, fileManager: fileManager)
        case .qwen3ASR:
            return Self.missingQwen3ASRPaths(in: rootURL, fileManager: fileManager)
        case .parakeet:
            return Self.missingParakeetPaths(in: rootURL, fileManager: fileManager)
        case .sortformer:
            return Self.missingSortformerPaths(in: rootURL, fileManager: fileManager)
        case .qwen3Embedding:
            return Qwen3EmbeddingResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .privacyFilter:
            return OpenAIPrivacyFilterResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .codegenGGUF:
            return Self.missingCodeGenPaths(
                preferredRelativePath: hubFallback?.filePath,
                in: rootURL,
                fileManager: fileManager
            )
        case .deepseekV4FlashIMatrixGGUF:
            return Self.missingDeepseekV4FlashIMatrixPaths(in: rootURL, fileManager: fileManager)
        case .lightOnOCR:
            return Self.missingLightOnOCRPaths(in: rootURL, fileManager: fileManager)
        case .aceStep:
            return Self.missingACEStepPaths(modelID: id, in: rootURL, fileManager: fileManager)
        case .aceStepLM:
            return ACEStep5HzLMResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .miniMaxMusic3:
            return MiniMaxMusic3Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .magentaRT2:
            return Self.missingMagentaRT2Paths(modelID: id, in: rootURL, fileManager: fileManager)
        case .muScriptor:
            return MuScriptorResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .roFormer:
            if let resources = try? RoFormerResources(rootURL: rootURL, modelID: id) {
                return resources.validate(fileManager: fileManager)
            }
            if let resources = try? MelBandRoFormerResources(rootURL: rootURL, modelID: id) {
                return resources.validate(fileManager: fileManager)
            }
            return [rootURL]
        case .apBWE:
            return APBWEResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .univerSR:
            return UniverSRResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .woosh:
            let checkpointsRoot = WooshResources.normalizeRoot(rootURL, fileManager: fileManager)
            let variant = WooshVariant.resolve(model: id, rootURL: checkpointsRoot, fileManager: fileManager) ?? .dflow
            return WooshModelResources(checkpointsRootURL: checkpointsRoot, variant: variant)
                .missingFiles(fileManager: fileManager)
        case .wooshClap:
            let checkpointsRoot = WooshResources.normalizeRoot(rootURL, fileManager: fileManager)
            return WooshCLAPResources(checkpointsRootURL: checkpointsRoot)
                .missingFiles(fileManager: fileManager)
        case .wooshSynchformer:
            return WooshSynchformerResources(rootURL: rootURL)
                .missingFiles(fileManager: fileManager)
        case .mmaudio:
            return MMAudioModelResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo23MLX:
            return Self.missingLTXVideo23MLXPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo23FullMLX:
            return Self.missingLTXVideo23FullMLXPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo23A2VMLX:
            return Self.missingLTXVideo23A2VMLXPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo25:
            let resources = LTX25Resources(rootURL: rootURL)
            return id == ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
                ? resources.validateFull(fileManager: fileManager)
                : resources.validate(fileManager: fileManager)
        case .wan22TI2VMLX:
            return Wan2Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .miniMaxH3MLX:
            return MiniMaxH3Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .cosmos3EdgeMLX:
            let resources = Cosmos3Resources(rootURL: rootURL)
            return resources.validate(fileManager: fileManager)
                + resources.validateReasoner(fileManager: fileManager)
        case .scail2MLX:
            return SCAIL2Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .dreamXCausalMLX:
            return Wan2DreamXCausalResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .hfTextChat:
            return Self.missingHFTextRootPaths(in: rootURL, fileManager: fileManager)
        }
    }

    func validationMessages(in rootURL: URL, fileManager: FileManager = .default) -> [String] {
        switch validationKind {
        case .nemotronOmni:
            return NemotronOmniResources.validationMessages(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            )
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR:
            guard let pin = GeometryModelPins.pin(for: id) else {
                return ["Missing exact artifact pin for managed model \(id)."]
            }
            return Self.pinnedArtifactValidationMessages(
                pin.runtimeArtifacts,
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            )
        case .instantMesh:
            let normalized = normalizedRootURL(rootURL, fileManager: fileManager)
            if Self.invalidInstantMeshNativeArtifacts(in: normalized, fileManager: fileManager).isEmpty {
                return []
            }
            guard let pin = GeometryModelPins.pin(for: id) else {
                return ["Missing exact artifact pin for managed model \(id)."]
            }
            return Self.pinnedArtifactValidationMessages(
                pin.artifacts,
                in: normalized,
                fileManager: fileManager
            )
        case .trellis2:
            return Trellis2Resources.validationMessages(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            )
        case .sam31:
            return SAM31Resources.validateRoot(normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .falconPerception:
            return FalconPerceptionResources.validateRoot(normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
                .map { "Missing required LTX file: \($0.path)" }
        case .ltxVideo23MLX:
            return Self.missingLTXVideo23MLXPaths(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required LTX 2.3 MLX file: \($0.path)" }
        case .ltxVideo23FullMLX:
            return Self.missingLTXVideo23FullMLXPaths(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required LTX 2.3 full MLX file: \($0.path)" }
        case .ltxVideo23A2VMLX:
            return Self.missingLTXVideo23A2VMLXPaths(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required LTX 2.3 A2Vid MLX file: \($0.path)" }
        case .ltxVideo25:
            let resources = LTX25Resources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            )
            return (id == ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
                ? resources.validateFull(fileManager: fileManager)
                : resources.validate(fileManager: fileManager))
                .map { "Missing required LTX 2.5 file: \($0.path)" }
        case .wan22TI2VMLX:
            let resources = Wan2Resources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager))
            let missing = resources.validate(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required Wan2.2 TI2V MLX file: \($0.path)" }
            }
            return []
        case .miniMaxH3MLX:
            let resources = MiniMaxH3Resources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            )
            let missing = resources.validate(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required MiniMax-H3 MLX file: \($0.path)" }
            }
            do {
                let configuration = try resources.loadConfiguration()
                let expectedTask = id == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue ? "ref2va" : "fl2va"
                guard configuration.task == expectedTask else {
                    return ["MiniMax-H3 model \(id) requires partition \(expectedTask), got \(configuration.task)."]
                }
                if id == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue {
                    let expectedQuantization = MiniMaxH3QuantizationConfiguration(
                        bits: 8,
                        groupSize: 64,
                        mode: "affine"
                    )
                    guard configuration.quantization == expectedQuantization,
                          configuration.textEncoderQuantization == expectedQuantization else {
                        return ["MiniMax-H3 Ref2VA requires MLX affine INT8/group-64 transformer and conditioner weights."]
                    }
                    return resources.validateManagedRef2VAArtifact(fileManager: fileManager)
                }
                if id == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue {
                    guard try resources.transformerStorage() == .compactBF16,
                          configuration.quantization == nil else {
                        return ["MiniMax-H3 BF16 requires an unquantized compact BF16 transformer."]
                    }
                    return resources.validateCompactCachePack(fileManager: fileManager)
                }
                if id == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue {
                    let expectedQuantization = MiniMaxH3QuantizationConfiguration(
                        bits: 8,
                        groupSize: 64,
                        mode: "affine"
                    )
                    guard try resources.transformerStorage() == .affineQ8,
                          configuration.quantization == expectedQuantization,
                          configuration.textEncoderQuantization == expectedQuantization else {
                        return ["MiniMax-H3 Q8 requires MLX affine INT8/group-64 transformer and conditioner weights."]
                    }
                    return resources.validateCompactCachePack(fileManager: fileManager)
                }
                return []
            } catch {
                return [error.localizedDescription]
            }
        case .cosmos3EdgeMLX:
            let resources = Cosmos3Resources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            )
            let missing = resources.validate(fileManager: fileManager)
                + resources.validateReasoner(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required Cosmos3-Edge file: \($0.path)" }
            }
            do {
                _ = try resources.loadTransformerConfiguration()
                _ = try resources.loadVAEConfiguration()
                _ = try resources.loadReasonerConfiguration()
                return []
            } catch {
                return [error.localizedDescription]
            }
        case .scail2MLX:
            let resources = SCAIL2Resources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager))
            let missing = resources.validate(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required SCAIL-2 MLX file: \($0.path)" }
            }
            do {
                _ = try resources.loadConfiguration()
                return []
            } catch {
                return [error.localizedDescription]
            }
        case .dreamXCausalMLX:
            return Wan2DreamXCausalResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            ).validate(fileManager: fileManager).map {
                "Missing required DreamX causal MLX file: \($0.path)"
            }
        case .magentaRT2:
            return Self.missingMagentaRT2Paths(
                modelID: id,
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required Magenta RT2 file: \($0.path)" }
        case .roFormer:
            if let resources = try? RoFormerResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager),
                modelID: id
            ) {
                return resources.validationMessages(fileManager: fileManager)
            }
            if let resources = try? MelBandRoFormerResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager),
                modelID: id
            ) {
                return resources.validationMessages(fileManager: fileManager)
            }
            return ["Unsupported managed RoFormer model id: \(id)"]
        case .apBWE:
            return APBWEResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            ).validationMessages(fileManager: fileManager)
        case .univerSR:
            return UniverSRResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            ).validationMessages(fileManager: fileManager)
        default:
            return missingPaths(in: rootURL, fileManager: fileManager).map { "Missing required file: \($0.path)" }
        }
    }

    func isManagedRootComplete(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        missingPaths(in: rootURL, fileManager: fileManager).isEmpty
            && managedSourceMatches(rootURL, fileManager: fileManager)
    }

    func isManagedRuntimeReady(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        let normalized = normalizedRootURL(rootURL, fileManager: fileManager)
        switch validationKind {
        case .instantMesh:
            return Self.invalidInstantMeshNativeArtifacts(
                in: normalized,
                fileManager: fileManager
            ).isEmpty
        case .terramindFlood:
            return (try? TerraMindFloodResources.inspect(normalized)) != nil
        case .terramindFire:
            return (try? TerraMindFireResources.inspect(normalized)) != nil
        case .tessera:
            return (try? TESSERAResources.inspect(normalized))?.source.modelID == id
        case .olmoEarth:
            return (try? OlmoEarthResources.inspect(normalized))?.source.modelID == id
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR:
            guard let pin = GeometryModelPins.pin(for: id) else { return false }
            return Self.invalidPinnedArtifacts(
                pin.runtimeArtifacts,
                in: normalized,
                fileManager: fileManager
            ).isEmpty
        default:
            return isManagedRootComplete(normalized, fileManager: fileManager)
        }
    }

    func managedSourceMatches(_ rootURL: URL, fileManager: FileManager) -> Bool {
        let requiresPinnedSource = id == ModelResolver.ModelID.zetaNano.rawValue
            || id == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
        guard requiresPinnedSource, let expectedRepo = upstreamRepoId else {
            return true
        }
        let normalized = normalizedRootURL(rootURL, fileManager: fileManager)
        guard let manifest = try? MereRunModelManifest.loadIfPresent(from: normalized, fileManager: fileManager),
              let installedRepo = manifest.upstreamRepoId else {
            return id != ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
                && id != ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
        }

        let requiresExactRevision = id == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
        if installedRepo == expectedRepo, !requiresExactRevision {
            return true
        }
        if let expectedWithRevision = upstreamRevision.map({ "\(expectedRepo)@\($0)" }),
           installedRepo == expectedWithRevision {
            return true
        }
        if upstreamRevision == nil, installedRepo == expectedRepo {
            return true
        }
        if id == ModelResolver.ModelID.zetaNano.rawValue,
           installedRepo == "\(expectedRepo)@\(ZImageTurboRepository.revision)" {
            return true
        }
        return false
    }

    func managedInstallRootURL() -> URL {
        MereRunModelPaths.modelDir(id)
    }

    func managedRuntimeURL(fileManager: FileManager = .default) -> URL? {
        switch installShape {
        case .directoryRoot, .structuredRoot:
            if let modelID {
                guard let resolved = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) else {
                    return nil
                }
                return resolved.rootURL
            }
            let root = normalizedRootURL(managedInstallRootURL(), fileManager: fileManager)
            return validateRuntimeURL(root, fileManager: fileManager).isEmpty ? root : nil
        case .singleFile(let relativePath):
            if let modelID,
               let resolved = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID),
               let externalFile = Self.findFirstGGUFFile(
                   in: resolved.rootURL,
                   fileManager: fileManager
               ),
               validateRuntimeURL(externalFile, fileManager: fileManager).isEmpty {
                return externalFile
            }
            let aliasURL = MereRunModelPaths.resolveModelFile(relativePath: relativePath) { candidate in
                self.validateRuntimeURL(candidate, fileManager: fileManager).isEmpty
            }
            if validateRuntimeURL(aliasURL, fileManager: fileManager).isEmpty {
                return aliasURL
            }

            let root = managedInstallRootURL()
            let normalizedRoot = normalizedRootURL(root, fileManager: fileManager)
            if let gguf = Self.findFirstGGUFFile(in: normalizedRoot, fileManager: fileManager),
               validateRuntimeURL(gguf, fileManager: fileManager).isEmpty {
                return gguf
            }
            return nil
        }
    }

    func validateRuntimeURL(_ url: URL, fileManager: FileManager = .default) -> [URL] {
        switch installShape {
        case .singleFile:
            switch validationKind {
            case .codegenGGUF:
                return fileManager.fileExists(atPath: url.path) ? [] : [url]
            case .deepseekV4FlashIMatrixGGUF:
                // Accept any GGUF whose name marks it as imatrix-tuned.
                if fileManager.fileExists(atPath: url.path),
                   url.lastPathComponent.lowercased().contains("imatrix") {
                    return []
                }
                return [url]
            default:
                return fileManager.fileExists(atPath: url.path) ? [] : [url]
            }
        case .directoryRoot, .structuredRoot:
            return missingPaths(in: url, fileManager: fileManager)
        }
    }

    private static func missingFiles(
        _ relativePaths: [String],
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        relativePaths
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func invalidPinnedArtifacts(
        _ artifacts: [ModelArtifactPin],
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        artifacts.compactMap { artifact in
            do {
                _ = try artifact.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return rootURL.appendingPathComponent(artifact.filename)
            }
        }
    }

    private static func pinnedArtifactValidationMessages(
        _ artifacts: [ModelArtifactPin],
        in rootURL: URL,
        fileManager: FileManager
    ) -> [String] {
        artifacts.compactMap { artifact in
            do {
                _ = try artifact.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return error.localizedDescription
            }
        }
    }

    private static func invalidInstantMeshNativeArtifacts(
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        let native = rootURL.appendingPathComponent(
            InstantMeshResources.managedConvertedDirectoryName,
            isDirectory: true
        )
        return invalidPinnedArtifacts(
            [
                InstantMeshResources.convertedWeightsPin,
                InstantMeshResources.convertedConfigurationPin,
                InstantMeshResources.convertedSourceManifestPin,
                InstantMeshResources.convertedLicensePin,
            ],
            in: native,
            fileManager: fileManager
        )
    }

    private static func missingQwen3TTSPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let speechTokenizerDir = rootURL.appendingPathComponent("speech_tokenizer", isDirectory: true)
        let speechTokenizerConfig = speechTokenizerDir.appendingPathComponent("config.json")
        let tokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let vocab = rootURL.appendingPathComponent("vocab.json")
        let merges = rootURL.appendingPathComponent("merges.txt")
        let tokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")

        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        if !fileManager.fileExists(atPath: speechTokenizerConfig.path) { missing.append(speechTokenizerConfig) }
        let tokenizerWeights = (try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: speechTokenizerDir,
            includingPropertiesForKeys: nil
        ))?.filter {
            $0.pathExtension == "safetensors"
        } ?? []
        if tokenizerWeights.isEmpty { missing.append(speechTokenizerDir) }
        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSON.path)
        let hasVocab = fileManager.fileExists(atPath: vocab.path)
        let hasMerges = fileManager.fileExists(atPath: merges.path)
        if !hasTokenizerJSON && !(hasVocab && hasMerges) { missing.append(tokenizerJSON) }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) { missing.append(tokenizerConfig) }
        return missing
    }

    private static func missingQwen3ASRPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let vocab = rootURL.appendingPathComponent("vocab.json")
        let merges = rootURL.appendingPathComponent("merges.txt")
        let tokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")

        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSON.path)
        let hasVocab = fileManager.fileExists(atPath: vocab.path)
        let hasMerges = fileManager.fileExists(atPath: merges.path)
        if !hasTokenizerJSON && !(hasVocab && hasMerges) { missing.append(tokenizerJSON) }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) { missing.append(tokenizerConfig) }
        return missing
    }

    private static func missingParakeetPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerModel = rootURL.appendingPathComponent("tokenizer.model")
        let tokenizerVocab = rootURL.appendingPathComponent("tokenizer.vocab")
        let vocabTxt = rootURL.appendingPathComponent("vocab.txt")

        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        let hasTokenizer = fileManager.fileExists(atPath: tokenizerModel.path)
            || fileManager.fileExists(atPath: tokenizerVocab.path)
            || fileManager.fileExists(atPath: vocabTxt.path)
        if !hasTokenizer { missing.append(tokenizerModel) }
        return missing
    }

    private static func missingSortformerPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        Self.missingFiles(
            ["config.json", "model.safetensors"],
            in: rootURL,
            fileManager: fileManager
        )
    }

    private static func missingHFTextRootPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let tokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")
        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        if hasIndex {
            missing.append(contentsOf: missingShardPaths(indexURL: modelIndexURL, fileManager: fileManager))
        }
        if !fileManager.fileExists(atPath: tokenizerJSON.path) { missing.append(tokenizerJSON) }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) { missing.append(tokenizerConfig) }
        return missing
    }

    private static func missingShardPaths(indexURL: URL, fileManager: FileManager) -> [URL] {
        guard let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
            return []
        }

        let rootURL = indexURL.deletingLastPathComponent()
        return index.shardFilenames
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingCodeGenPaths(
        preferredRelativePath: String?,
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        if let preferredRelativePath {
            let preferredURL = rootURL.appendingPathComponent(preferredRelativePath, isDirectory: false)
            if isRegularFileOrSymlinkTarget(preferredURL, fileManager: fileManager) {
                return []
            }
        }
        return findFirstGGUFFile(in: rootURL, fileManager: fileManager) == nil
            ? [rootURL.appendingPathComponent("*.gguf")]
            : []
    }

    /// DeepSeek V4 Flash explicitly prefers the imatrix-tuned GGUF (per the
    /// upstream README's "USE THE IMATRIX VERSIONS" note). A directory that
    /// only contains the legacy non-imatrix GGUF is considered *not* complete,
    /// so `mere.run model pull` will fetch the preferred variant instead of
    /// reporting "already installed."
    private static func missingDeepseekV4FlashIMatrixPaths(
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        guard let enumerator = fileManager.enumeratorResolvingSymlinks(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return [rootURL.appendingPathComponent("*imatrix*.gguf")]
        }
        while let candidate = enumerator.nextObject() as? URL {
            guard candidate.pathExtension.lowercased() == "gguf",
                  candidate.lastPathComponent.lowercased().contains("imatrix") else {
                continue
            }
            if isRegularFileOrSymlinkTarget(candidate, fileManager: fileManager) {
                return []
            }
        }
        return [rootURL.appendingPathComponent("*imatrix*.gguf")]
    }

    private static func missingLightOnOCRPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerURL = rootURL.appendingPathComponent("tokenizer")
        let tokenizerJSON = tokenizerURL.appendingPathComponent("tokenizer.json")
        let tokenizerConfig = tokenizerURL.appendingPathComponent("tokenizer_config.json")
        let rootTokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let rootTokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")
        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        if !fileManager.fileExists(atPath: modelWeightsURL.path) { missing.append(modelWeightsURL) }
        let hasTokenizer = fileManager.fileExists(atPath: tokenizerJSON.path)
            || fileManager.fileExists(atPath: rootTokenizerJSON.path)
        if !hasTokenizer { missing.append(rootTokenizerJSON) }
        let hasTokenizerConfig = fileManager.fileExists(atPath: tokenizerConfig.path)
            || fileManager.fileExists(atPath: rootTokenizerConfig.path)
        if !hasTokenizerConfig { missing.append(rootTokenizerConfig) }
        return missing
    }

    private static func missingACEStepPaths(modelID: String, in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let decoderSubdirectories: [String]
        switch modelID {
        case ModelResolver.ModelID.aceStep.rawValue:
            decoderSubdirectories = ["acestep-v15-turbo", "music-acestep-v15-turbo"]
        case ModelResolver.ModelID.aceStepXLBase.rawValue:
            decoderSubdirectories = ["acestep-v15-xl-base"]
        case ModelResolver.ModelID.aceStepXLSFT.rawValue:
            decoderSubdirectories = ["acestep-v15-xl-sft"]
        default:
            decoderSubdirectories = ["acestep-v15-xl-turbo"]
        }
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let textDir = rootURL.appendingPathComponent("Qwen3-Embedding-0.6B", isDirectory: true)
        if !decoderSubdirectories.contains(where: {
            fileManager.fileExists(atPath: rootURL.appendingPathComponent($0, isDirectory: true).path)
        }) {
            missing.append(rootURL.appendingPathComponent(decoderSubdirectories[0], isDirectory: true))
        }
        if !fileManager.fileExists(atPath: vaeDir.path) { missing.append(vaeDir) }
        if !fileManager.fileExists(atPath: textDir.path) { missing.append(textDir) }
        if modelID == ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue {
            let lmDir = rootURL.appendingPathComponent("acestep-5Hz-lm-4B", isDirectory: true)
            let lmMissing = ACEStep5HzLMResources(rootURL: lmDir).validate(fileManager: fileManager)
            if !lmMissing.isEmpty {
                missing.append(lmDir)
            }
        }
        return missing
    }

    private static func missingMagentaRT2Paths(
        modelID: String,
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        let modelName = modelID == ModelResolver.ModelID.magentaRT2Base.rawValue ? "mrt2_base" : "mrt2_small"
        let relativePaths = [
            "models/\(modelName)/\(modelName).mlxfn",
            "models/\(modelName)/\(modelName)_state.safetensors",
            "resources/musiccoca/audio_preprocessor.tflite",
            "resources/musiccoca/mapper.tflite",
            "resources/musiccoca/music_encoder.tflite",
            "resources/musiccoca/pretrained_vector_quantizer.tflite",
            "resources/musiccoca/spm.model",
            "resources/musiccoca/text_encoder.tflite",
            "resources/spectrostream/decoder.safetensors",
            "resources/spectrostream/encoder.safetensors",
            "resources/spectrostream/quantizer.safetensors",
            "resources/spectrostream/spectrostream_encoder.mlxfn",
        ]
        return relativePaths
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingLTXVideoPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let textEncoderConfig = rootURL.appendingPathComponent("text_encoder/config.json")
        let textEncoderWeights = rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json")
        let tokenizerDir = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        if !fileManager.fileExists(atPath: textEncoderConfig.path) { missing.append(textEncoderConfig) }
        if !fileManager.fileExists(atPath: textEncoderWeights.path) { missing.append(textEncoderWeights) }
        if !fileManager.fileExists(atPath: tokenizerDir.path) { missing.append(tokenizerDir) }
        let entries = (try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: rootURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        let hasTransformer = entries.contains { $0.lastPathComponent.hasPrefix("ltx-2-19") && $0.pathExtension == "safetensors" }
        let hasUpsampler = entries.contains { $0.lastPathComponent.hasPrefix("ltx-2-spatial-upscaler") && $0.pathExtension == "safetensors" }
        if !hasTransformer { missing.append(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors")) }
        if !hasUpsampler { missing.append(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors")) }
        return missing
    }

    private static func missingLTXVideo23MLXPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let relativePaths = [
            "config.json",
            "embedded_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-distilled.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x1_5_v1_0.safetensors",
            "spatial_upscaler_x1_5_v1_0_config.json",
            "temporal_upscaler_x2_v1_0.safetensors",
            "temporal_upscaler_x2_v1_0_config.json",
        ]
        return relativePaths
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingLTXVideo23A2VMLXPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let relativePaths = [
            "config.json",
            "embedded_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
        ]
        return relativePaths
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingLTXVideo23FullMLXPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing = missingLTXVideo23A2VMLXPaths(in: rootURL, fileManager: fileManager)
        let vocoder = rootURL.appendingPathComponent("vocoder.safetensors", isDirectory: false)
        if !fileManager.fileExists(atPath: vocoder.path) {
            missing.append(vocoder)
        }
        return missing
    }

    private static func missingDiffusersImagePaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let tokenizerDir = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoderDir = rootURL.appendingPathComponent("text_encoder", isDirectory: true)
        let transformerDir = rootURL.appendingPathComponent("transformer", isDirectory: true)
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let schedulerDir = rootURL.appendingPathComponent("scheduler", isDirectory: true)

        var missing: [URL] = []
        let required: [URL] = [
            rootURL.appendingPathComponent("model_index.json"),
            tokenizerDir.appendingPathComponent("tokenizer_config.json"),
            textEncoderDir.appendingPathComponent("config.json"),
            transformerDir.appendingPathComponent("config.json"),
            vaeDir.appendingPathComponent("config.json"),
            schedulerDir.appendingPathComponent("scheduler_config.json"),
        ]
        for path in required where !fileManager.fileExists(atPath: path.path) {
            missing.append(path)
        }

        let textWeights = textEncoderDir.appendingPathComponent("model.safetensors")
        let textWeightsIndex = textEncoderDir.appendingPathComponent("model.safetensors.index.json")
        if !fileManager.fileExists(atPath: textWeights.path) && !fileManager.fileExists(atPath: textWeightsIndex.path) {
            missing.append(textWeightsIndex)
        }

        let transformerWeights = transformerDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let transformerWeightsIndex = transformerDir.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        if !fileManager.fileExists(atPath: transformerWeights.path) && !fileManager.fileExists(atPath: transformerWeightsIndex.path) {
            missing.append(transformerWeightsIndex)
        }

        let vaeWeights = vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if !fileManager.fileExists(atPath: vaeWeights.path) {
            missing.append(vaeWeights)
        }

        return missing
    }

    private static func missingBonsaiImagePaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let tokenizerDir = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoderDir = rootURL.appendingPathComponent("text_encoder-mlx-4bit", isDirectory: true)
        let transformerDir = rootURL.appendingPathComponent("transformer-packed-mflux", isDirectory: true)
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let schedulerDir = rootURL.appendingPathComponent("scheduler", isDirectory: true)

        var missing: [URL] = []
        let required: [URL] = [
            rootURL.appendingPathComponent("manifest.json"),
            tokenizerDir.appendingPathComponent("tokenizer_config.json"),
            textEncoderDir.appendingPathComponent("config.json"),
            transformerDir.appendingPathComponent("config.json"),
            transformerDir.appendingPathComponent("quantization_config.json"),
            vaeDir.appendingPathComponent("config.json"),
            schedulerDir.appendingPathComponent("scheduler_config.json"),
        ]
        for path in required where !fileManager.fileExists(atPath: path.path) {
            missing.append(path)
        }

        let textWeights = textEncoderDir.appendingPathComponent("model.safetensors")
        let textWeightsIndex = textEncoderDir.appendingPathComponent("model.safetensors.index.json")
        if !fileManager.fileExists(atPath: textWeights.path) && !fileManager.fileExists(atPath: textWeightsIndex.path) {
            missing.append(textWeightsIndex)
        }

        let transformerWeights = transformerDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if !fileManager.fileExists(atPath: transformerWeights.path) {
            missing.append(transformerWeights)
        }

        let vaeWeights = vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if !fileManager.fileExists(atPath: vaeWeights.path) {
            missing.append(vaeWeights)
        }

        return missing
    }

    static func findFirstGGUFFile(in rootURL: URL, fileManager: FileManager = .default) -> URL? {
        let enumerator = fileManager.enumeratorResolvingSymlinks(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        while let candidate = enumerator?.nextObject() as? URL {
            guard candidate.pathExtension.lowercased() == "gguf" else { continue }
            if isRegularFileOrSymlinkTarget(candidate, fileManager: fileManager) {
                return candidate
            }
        }
        return nil
    }

    private static func isRegularFileOrSymlinkTarget(_ url: URL, fileManager: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && !isDirectory.boolValue
    }
}
