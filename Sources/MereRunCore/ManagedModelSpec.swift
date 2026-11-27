import Foundation

public enum ManagedModelCategory: String, CaseIterable, Hashable, Sendable {
    case image = "image"
    case textChat = "text-chat"
    case textCode = "text-code"
    case textEmbed = "text-embed"
    case textAnonymize = "text-anonymize"
    case textDecide = "text-decide"
    case textClassify = "text-classify"
    case textExtract = "text-extract"
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
    case qwenImage21
    case nemotronImage
    case miniMaxH3Image
    case flux3
    case qwenImageEdit
    case ideogram4SDNQ
    case gemma4
    case flint
    case inkling2
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
    case nemotronBolt
    case nemotronHDSpark
    case nemotronOmni
    case qwen3TTS
    case breezeTTS
    case haru4B
    case drama3TTS
    case phononTTS
    case qwen3ASR
    case parakeet
    case whistle
    case nemotronTranscribe
    case sovTranscribe
    case sovTranscribeDFlash
    case sortformer
    case qwen3Embedding
    case embeddingGemma2
    case embeddingGemma2CoreML
    case pplxEmbedV2
    case qwen3VLEmbedding
    case privacyFilter
    case laya
    case clef
    case clairic
    case d1
    case gliner25Decide
    case codegenGGUF
    case deepseekV4FlashIMatrixGGUF
    case lightOnOCR
    case unlimitedOCR
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
    case marigoldV2
    case tripoSR
    case instantMesh
    case trellis2
    case aceStep
    case aceStepLM
    case miniMaxMusic3
    case yue2
    case yue2VAE
    case sheetSage2
    case magentaRT2
    case muScriptor
    case roFormer
    case apBWE
    case univerSR
    case auk
    case aukThinker
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
    case cosmos3SuperTextToImage
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

/// Whether `api serve` lists and routes a managed model.
public enum ManagedModelAPIAvailability: String, Hashable, Sendable {
    /// Served through its explicit profile or the category's companion profile.
    case served
    /// Runs only through its CLI command; API discovery never lists it and no
    /// companion profile is derived from its category.
    case cliOnly
}

/// Declarative manifest metadata owned by a `ManagedModelSpec`.
///
/// Materialized into a `MereRunModelManifest` by `MereRunModelManifest.template(for:createdAt:)`,
/// keeping a model's download spec, validation kind, capability metadata, and manifest template
/// together in one catalog entry.
public struct ManagedModelManifestTemplate: Hashable, Sendable {
    public enum UpstreamRepoSource: Hashable, Sendable {
        /// Inherit `spec.upstreamRepoId` verbatim.
        case spec
        /// Inherit `spec.upstreamRepoId` with `@<spec.upstreamRevision>` when a revision is pinned.
        case specPinned
        /// Use an explicit upstream string verbatim (or `nil`).
        case explicit(String?)

        func resolve(for spec: ManagedModelSpec) -> String? {
            switch self {
            case .spec:
                spec.upstreamRepoId
            case .specPinned:
                if let repo = spec.upstreamRepoId, let rev = spec.upstreamRevision, !rev.isEmpty {
                    "\(repo)@\(rev)"
                } else {
                    spec.upstreamRepoId
                }
            case .explicit(let value):
                value
            }
        }
    }

    public let engine: MereRunModelManifest.Engine?
    public let family: MereRunModelManifest.Family?
    public let tier: MereRunModelManifest.Tier?
    public let variant: MereRunModelManifest.Variant?
    public let precision: MereRunModelManifest.Precision?
    public let quantization: MereRunModelManifest.Quantization?
    public let defaults: MereRunModelManifest.Defaults?
    public let supports: [MereRunModelManifest.Capability]?
    public let components: MereRunModelManifest.Components?
    public let upstream: UpstreamRepoSource

    public init(
        engine: MereRunModelManifest.Engine? = nil,
        family: MereRunModelManifest.Family? = nil,
        tier: MereRunModelManifest.Tier? = nil,
        variant: MereRunModelManifest.Variant? = nil,
        precision: MereRunModelManifest.Precision? = nil,
        quantization: MereRunModelManifest.Quantization? = nil,
        defaults: MereRunModelManifest.Defaults? = nil,
        supports: [MereRunModelManifest.Capability]? = nil,
        components: MereRunModelManifest.Components? = nil,
        upstream: UpstreamRepoSource = .specPinned
    ) {
        self.engine = engine
        self.family = family
        self.tier = tier
        self.variant = variant
        self.precision = precision
        self.quantization = quantization
        self.defaults = defaults
        self.supports = supports
        self.components = components
        self.upstream = upstream
    }

    public func makeManifest(
        for spec: ManagedModelSpec,
        sources: [MereRunModelManifest.SourceProvenance]?,
        createdAt: Date = Date()
    ) -> MereRunModelManifest {
        MereRunModelManifest(
            id: spec.id,
            engine: engine,
            family: family,
            tier: tier,
            variant: variant,
            precision: precision,
            quantization: quantization,
            defaults: defaults,
            supports: supports,
            components: components,
            upstreamRepoId: upstream.resolve(for: spec),
            sources: sources,
            usageTerms: spec.usageRestriction?.terms,
            createdAt: createdAt
        )
    }
}

/// User-facing title, summary, and unified-memory sizing owned by a `ManagedModelSpec`.
public struct ManagedModelCapabilityMetadata: Hashable, Sendable {
    public let title: String
    public let summary: String
    public let minimumUnifiedMemoryGB: Int
    public let recommendedUnifiedMemoryGB: Int
    public let isRecommendedForSetup: Bool

    public init(
        _ title: String,
        _ summary: String,
        minimum: Int,
        recommended: Int,
        setup: Bool = false
    ) {
        self.title = title
        self.summary = summary
        self.minimumUnifiedMemoryGB = minimum
        self.recommendedUnifiedMemoryGB = recommended
        self.isRecommendedForSetup = setup
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
    public let apiAvailability: ManagedModelAPIAvailability
    public let apiProfile: ManagedModelAPIProfile?
    public let manifestTemplate: ManagedModelManifestTemplate?
    public let capability: ManagedModelCapabilityMetadata?

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
        apiAvailability: ManagedModelAPIAvailability = .served,
        apiProfile: ManagedModelAPIProfile? = nil,
        manifestTemplate: ManagedModelManifestTemplate? = nil,
        capability: ManagedModelCapabilityMetadata? = nil
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
        self.apiAvailability = apiAvailability
        switch apiAvailability {
        case .served:
            self.apiProfile = apiProfile ?? ManagedModelAPIProfile.companion(
                modelID: id,
                category: category
            )
        case .cliOnly:
            self.apiProfile = nil
        }
        self.manifestTemplate = manifestTemplate
        self.capability = capability
    }
}

public extension ManagedModelSpec {
    var modelID: ModelResolver.ModelID? {
        ModelResolver.ModelID(rawValue: id)
    }

    var capabilityDescriptor: ManagedModelCapabilityDescriptor? {
        capability.map {
            ManagedModelCapabilityDescriptor(
                modelID: id,
                title: $0.title,
                summary: $0.summary,
                minimumUnifiedMemoryGB: $0.minimumUnifiedMemoryGB,
                recommendedUnifiedMemoryGB: $0.recommendedUnifiedMemoryGB,
                isRecommendedForSetup: $0.isRecommendedForSetup
            )
        }
    }

    var canBePulledWithoutConfiguration: Bool {
        hubFallback != nil || !mountedHubFallbacks.isEmpty
    }

    func hasAnyManagedDownloadSource() -> Bool {
        canBePulledWithoutConfiguration
    }
}
