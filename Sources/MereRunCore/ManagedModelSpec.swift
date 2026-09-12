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

public enum ManagedModelInstallShape: Hashable, Sendable {
    case directoryRoot
    case singleFile(relativePath: String)
    case structuredRoot
}

public enum ManagedModelValidationKind: String, Hashable, Sendable {
    case flux1
    case flux2Klein
    case bonsaiImage
    case zimageTurbo
    case hidreamO1
    case senseNovaU15
    case krea2
    case qwenImageEdit
    case ideogram4SDNQ
    case gemma4
    case diffusionGemma
    case gemma4Unified
    case gemma4MTPAssistant
    case laguna
    case lagunaDFlash
    case q35
    case q35MTPAssistant
    case lfm2
    case lfm2DSpark
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
    case qwen3VLEmbedding
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

public extension ManagedModelSpec {
    var modelID: ModelResolver.ModelID? {
        ModelResolver.ModelID(rawValue: id)
    }

    var canBePulledWithoutConfiguration: Bool {
        hubFallback != nil || !mountedHubFallbacks.isEmpty
    }

    func hasAnyManagedDownloadSource() -> Bool {
        canBePulledWithoutConfiguration
    }
}
