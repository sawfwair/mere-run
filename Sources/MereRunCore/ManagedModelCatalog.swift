import Foundation

public enum ManagedModelCategory: String, CaseIterable, Hashable, Sendable {
    case image = "image"
    case textChat = "text-chat"
    case textCode = "text-code"
    case textEmbed = "text-embed"
    case speechTTS = "speech-tts"
    case speechASR = "speech-asr"
    case visionOCR = "vision-ocr"
    case visionSegment = "vision-segment"
    case music = "music"
    case video = "video"
}

public enum ManagedModelInstallShape: Hashable, Sendable {
    case directoryRoot
    case singleFile(relativePath: String)
    case structuredRoot
}

public struct ManagedModelArchiveSource: Hashable, Sendable {
    public let key: String
    public let size: Int64
    public let packagedKey: String?

    public init(key: String, size: Int64, packagedKey: String? = nil) {
        self.key = key
        self.size = size
        self.packagedKey = packagedKey
    }
}

public enum ManagedModelValidationKind: String, Hashable, Sendable {
    case flux2Klein
    case zimageTurbo
    case gemma4
    case q35
    case qwen3TTS
    case qwen3ASR
    case parakeet
    case qwen3Embedding
    case codegenGGUF
    case lightOnOCR
    case sam31
    case aceStep
    case ltxVideo
    case hfTextChat
}

public enum ManagedModelNormalizationKind: String, Hashable, Sendable {
    case none
    case qwen3ASRNested
    case parakeetNested
    case musicACEStep
}

public enum ManagedModelAliasKind: String, Hashable, Sendable {
    case none
    case codegenGGUF
}

public struct ManagedModelSpec: Hashable, Sendable {
    public let id: String
    public let category: ManagedModelCategory
    public let installShape: ManagedModelInstallShape
    public let archiveSource: ManagedModelArchiveSource?
    public let hubFallback: HubFallbackConfig?
    public let upstreamRepoId: String?
    public let upstreamRevision: String?
    public let validationKind: ManagedModelValidationKind
    public let normalizationKind: ManagedModelNormalizationKind
    public let aliasKind: ManagedModelAliasKind
    public let runtimeAutoDownloadAllowed: Bool
    public let resolutionFallbackIDs: [String]
    public let defaultCLICommands: [String]

    public init(
        id: String,
        category: ManagedModelCategory,
        installShape: ManagedModelInstallShape,
        archiveSource: ManagedModelArchiveSource?,
        hubFallback: HubFallbackConfig? = nil,
        upstreamRepoId: String? = nil,
        upstreamRevision: String? = nil,
        validationKind: ManagedModelValidationKind,
        normalizationKind: ManagedModelNormalizationKind = .none,
        aliasKind: ManagedModelAliasKind = .none,
        runtimeAutoDownloadAllowed: Bool = true,
        resolutionFallbackIDs: [String] = [],
        defaultCLICommands: [String] = []
    ) {
        self.id = id
        self.category = category
        self.installShape = installShape
        self.archiveSource = archiveSource
        self.hubFallback = hubFallback
        self.upstreamRepoId = upstreamRepoId
        self.upstreamRevision = upstreamRevision
        self.validationKind = validationKind
        self.normalizationKind = normalizationKind
        self.aliasKind = aliasKind
        self.runtimeAutoDownloadAllowed = runtimeAutoDownloadAllowed
        self.resolutionFallbackIDs = resolutionFallbackIDs
        self.defaultCLICommands = defaultCLICommands
    }
}

public enum ManagedModelCatalog {
    public static let allSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: "image-klein-nano",
            category: .image,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/image-klein-nano.tar.gz",
                size: 7_289_504_269,
                packagedKey: "models/zero-nano.tar.gz"
            ),
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-klein-max",
            category: .image,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/image-klein-max.tar.gz",
                size: 18_991_145_972,
                packagedKey: "models/zero-max.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: "black-forest-labs/FLUX.2-klein-4B",
                patterns: [
                    "model_index.json",
                    "tokenizer/*",
                    "text_encoder/*",
                    "transformer/*",
                    "vae/*",
                    "scheduler/*",
                ]
            ),
            upstreamRepoId: "black-forest-labs/FLUX.2-klein-4B",
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-klein-base",
            category: .image,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/image-klein-base.tar.gz",
                size: 0,
                packagedKey: "models/zero-base.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: "black-forest-labs/FLUX.2-klein-base-4B",
                patterns: [
                    "model_index.json",
                    "tokenizer/*",
                    "text_encoder/*",
                    "transformer/*",
                    "vae/*",
                    "scheduler/*",
                ]
            ),
            upstreamRepoId: "black-forest-labs/FLUX.2-klein-base-4B",
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-klein-shared",
            category: .image,
            installShape: .directoryRoot,
            archiveSource: nil,
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false
        ),
        ManagedModelSpec(
            id: "image-zimage-nano",
            category: .image,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/image-zimage-nano.tar.gz",
                size: 5_980_168_600,
                packagedKey: "models/zeta-nano.tar.gz"
            ),
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-zimage-max",
            category: .image,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/image-zimage-max.tar.gz",
                size: 18_092_604_557,
                packagedKey: "models/zeta-max.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: "Tongyi-MAI/Z-Image-Turbo",
                revision: "main",
                patterns: [
                    "model_index.json",
                    "tokenizer/*",
                    "text_encoder/*",
                    "transformer/*",
                    "vae/*",
                    "scheduler/*",
                ]
            ),
            upstreamRepoId: "Tongyi-MAI/Z-Image-Turbo",
            upstreamRevision: "main",
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "image-zimage-base",
            category: .image,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/image-zimage-base.tar.gz",
                size: 0
            ),
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "text-chat-mebot",
            category: .textChat,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-chat-mebot.tar.gz",
                size: 2_052_847_048,
                packagedKey: "models/mebot-instruct.tar.gz"
            ),
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false,
            resolutionFallbackIDs: ["image-klein-nano", "image-klein-max"],
            defaultCLICommands: ["api serve"]
        ),
        ManagedModelSpec(
            id: "text-chat-psi-agent",
            category: .textChat,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-chat-psi-agent.tar.gz",
                size: 30_257_308_652,
                packagedKey: "models/psi-agent.tar.gz"
            ),
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false
        ),
        ManagedModelSpec(
            id: "text-chat-gemma4",
            category: .textChat,
            installShape: .directoryRoot,
            archiveSource: nil,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.defaultUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.defaultUpstreamModelId,
            validationKind: .gemma4,
            resolutionFallbackIDs: ["text-chat-gemma4-max", "text-chat-gemma4-nano"],
            defaultCLICommands: ["api serve"]
        ),
        ManagedModelSpec(
            id: "text-chat-gemma4-nano",
            category: .textChat,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-chat-gemma4-nano.tar.gz",
                size: 0,
                packagedKey: "models/gemma4-nano.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.nanoUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.nanoUpstreamModelId,
            validationKind: .gemma4,
            defaultCLICommands: ["api serve"]
        ),
        ManagedModelSpec(
            id: "text-chat-gemma4-max",
            category: .textChat,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-chat-gemma4-max.tar.gz",
                size: 0,
                packagedKey: "models/gemma4-max.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.maxUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.maxUpstreamModelId,
            validationKind: .gemma4,
            defaultCLICommands: ["api serve"]
        ),
        ManagedModelSpec(
            id: "text-chat-q35",
            category: .textChat,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-chat-q35.tar.gz",
                size: 0,
                packagedKey: "models/q35.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: Q35Resources.upstreamRepoId,
                revision: Q35Resources.upstreamRevision,
                patterns: [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: Q35Resources.upstreamRepoId,
            upstreamRevision: Q35Resources.upstreamRevision,
            validationKind: .q35,
            defaultCLICommands: ["chat", "api serve"]
        ),
        ManagedModelSpec(
            id: "text-chat-q35-nano",
            category: .textChat,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-chat-q35-nano.tar.gz",
                size: 0
            ),
            hubFallback: HubFallbackConfig(
                repoId: Q35Resources.nanoUpstreamRepoId,
                revision: Q35Resources.nanoUpstreamRevision,
                patterns: [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: Q35Resources.nanoUpstreamRepoId,
            upstreamRevision: Q35Resources.nanoUpstreamRevision,
            validationKind: .q35,
            defaultCLICommands: ["chat", "api serve"]
        ),
        ManagedModelSpec(
            id: "speech-tts-qwen3-nano",
            category: .speechTTS,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/speech-tts-qwen3-nano.tar.gz",
                size: 3_691_910_698,
                packagedKey: "models/talk-nano.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
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
            upstreamRepoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            upstreamRevision: "main",
            validationKind: .qwen3TTS,
            defaultCLICommands: ["speech synthesize"]
        ),
        ManagedModelSpec(
            id: "speech-tts-qwen3-customvoice",
            category: .speechTTS,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/speech-tts-qwen3-customvoice.tar.gz",
                size: 0
            ),
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
            defaultCLICommands: ["speech synthesize"]
        ),
        ManagedModelSpec(
            id: "speech-asr-qwen3",
            category: .speechASR,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/speech-asr-qwen3.tar.gz",
                size: 0,
                packagedKey: "models/asr.tar.gz"
            ),
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
            defaultCLICommands: ["speech transcribe"]
        ),
        ManagedModelSpec(
            id: "speech-asr-parakeet",
            category: .speechASR,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/speech-asr-parakeet.tar.gz",
                size: 2_332_340_210,
                packagedKey: "models/asr-parakeet.tar.gz"
            ),
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
            defaultCLICommands: ["speech transcribe"]
        ),
        ManagedModelSpec(
            id: "text-code-qwen3",
            category: .textCode,
            installShape: .singleFile(relativePath: CodeGenResources.managedRelativePath),
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-code-qwen3.tar.gz",
                size: CodeGenResources.r2ArchiveSize,
                packagedKey: "models/qwen3-coder-next.tar.gz"
            ),
            hubFallback: CodeGenResources.hubFallbackConfig,
            upstreamRepoId: CodeGenResources.defaultRepoId,
            upstreamRevision: CodeGenResources.defaultRevision,
            validationKind: .codegenGGUF,
            aliasKind: .codegenGGUF,
            defaultCLICommands: ["text code"]
        ),
        ManagedModelSpec(
            id: "text-embed-qwen3-0.6b",
            category: .textEmbed,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/text-embed-qwen3-0.6b.tar.gz",
                size: 0,
                packagedKey: "models/qwen3-embedding-0.6b.tar.gz"
            ),
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
                    "added_tokens.json",
                    "merges.txt",
                    "vocab.json",
                    "1_Pooling/*",
                ]
            ),
            upstreamRepoId: "Qwen/Qwen3-Embedding-0.6B",
            upstreamRevision: "main",
            validationKind: .qwen3Embedding,
            defaultCLICommands: ["text embed"]
        ),
        ManagedModelSpec(
            id: "vision-ocr-lighton",
            category: .visionOCR,
            installShape: .structuredRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/vision-ocr-lighton.tar.gz",
                size: 0,
                packagedKey: "models/ocr.tar.gz"
            ),
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
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: "vision-segment-sam31",
            category: .visionSegment,
            installShape: .directoryRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/vision-segment-sam31.tar.gz",
                size: 0
            ),
            hubFallback: HubFallbackConfig(
                repoId: "facebook/sam3.1",
                patterns: [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "tokenizer/*",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "facebook/sam3.1",
            validationKind: .sam31,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["vision segment"]
        ),
        ManagedModelSpec(
            id: "music-acestep",
            category: .music,
            installShape: .structuredRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/music-acestep.tar.gz",
                size: 0,
                packagedKey: "models/acestep-v15-full.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: "main",
                patterns: [
                    "config.json",
                    "acestep-v15-turbo/*",
                    "acestep-5Hz-lm-1.7B/*",
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            upstreamRepoId: "ACE-Step/Ace-Step1.5",
            upstreamRevision: "main",
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            defaultCLICommands: ["music generate"]
        ),
        ManagedModelSpec(
            id: "video-ltx-av",
            category: .video,
            installShape: .structuredRoot,
            archiveSource: ManagedModelArchiveSource(
                key: "models/video-ltx-av.tar.gz",
                size: 0,
                packagedKey: "models/ltx-video-av.tar.gz"
            ),
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/LTX-2-distilled-bf16",
                revision: "main",
                patterns: [
                    "ltx-2-19b-distilled.safetensors",
                    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
                    "text_encoder/*",
                    "tokenizer/*",
                ]
            ),
            upstreamRepoId: "mlx-community/LTX-2-distilled-bf16",
            upstreamRevision: "main",
            validationKind: .ltxVideo,
            defaultCLICommands: ["video generate"]
        ),
    ]

    public static var allModelIDs: [String] {
        allSpecs.map(\.id)
    }

    public static func spec(for id: String) -> ManagedModelSpec? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allSpecs.first {
            $0.id == normalized || $0.upstreamRepoId?.lowercased() == normalized
        }
    }
}

public extension ManagedModelSpec {
    var modelID: ModelResolver.ModelID? {
        ModelResolver.ModelID(rawValue: id)
    }

    var canBePulledWithoutConfiguration: Bool {
        hubFallback != nil
    }

    func hasAnyManagedDownloadSource(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        if hubFallback != nil {
            return true
        }
        guard archiveSource != nil else {
            return false
        }
        return MereRunModelSourceConfiguration.hasAnyDownloadSource(environment: environment)
    }

    func normalizedRootURL(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let base = rootURL.standardizedFileURL
        switch normalizationKind {
        case .none, .musicACEStep:
            return base
        case .qwen3ASRNested:
            let direct = missingPaths(in: base, fileManager: fileManager)
            if direct.isEmpty {
                return base
            }
            let nested = base.appendingPathComponent(id, isDirectory: true)
            return missingPaths(in: nested, fileManager: fileManager).isEmpty ? nested : base
        case .parakeetNested:
            let direct = missingPaths(in: base, fileManager: fileManager)
            if direct.isEmpty {
                return base
            }
            let nested = base.appendingPathComponent(id, isDirectory: true)
            return missingPaths(in: nested, fileManager: fileManager).isEmpty ? nested : base
        }
    }

    func missingPaths(in rootURL: URL, fileManager: FileManager = .default) -> [URL] {
        switch validationKind {
        case .flux2Klein:
            return Self.missingDiffusersImagePaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .zimageTurbo:
            return ZImageTurboResources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager)).validate(fileManager: fileManager)
        case .gemma4:
            return Gemma4Resources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager)).validate(fileManager: fileManager)
        case .q35:
            return Q35Resources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager)).validate(fileManager: fileManager)
        case .sam31:
            return SAM31Resources(modelRootURL: normalizedRootURL(rootURL, fileManager: fileManager)).missingRequiredPaths(fileManager: fileManager)
        case .qwen3TTS:
            return Self.missingQwen3TTSPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .qwen3ASR:
            return Self.missingQwen3ASRPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .parakeet:
            return Self.missingParakeetPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .qwen3Embedding:
            return Qwen3EmbeddingResources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager)).validate(fileManager: fileManager)
        case .codegenGGUF:
            return Self.missingCodeGenPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .lightOnOCR:
            return Self.missingLightOnOCRPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .aceStep:
            return Self.missingACEStepPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .hfTextChat:
            return Self.missingHFTextRootPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        }
    }

    func validationMessages(in rootURL: URL, fileManager: FileManager = .default) -> [String] {
        switch validationKind {
        case .sam31:
            return SAM31Resources.validateRoot(normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
                .map { "Missing required LTX file: \($0.path)" }
        default:
            return missingPaths(in: rootURL, fileManager: fileManager).map { "Missing required file: \($0.path)" }
        }
    }

    func isManagedRootComplete(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        missingPaths(in: rootURL, fileManager: fileManager).isEmpty
    }

    func managedInstallRootURL() -> URL {
        MereRunModelPaths.modelDir(id)
    }

    func managedRuntimeURL(fileManager: FileManager = .default) -> URL? {
        switch installShape {
        case .directoryRoot, .structuredRoot:
            guard let modelID, let resolved = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) else {
                return nil
            }
            return resolved.rootURL
        case .singleFile(let relativePath):
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
            default:
                return fileManager.fileExists(atPath: url.path) ? [] : [url]
            }
        case .directoryRoot, .structuredRoot:
            return missingPaths(in: url, fileManager: fileManager)
        }
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
        let tokenizerWeights = (try? fileManager.contentsOfDirectory(at: speechTokenizerDir, includingPropertiesForKeys: nil))?.filter {
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
        if !fileManager.fileExists(atPath: tokenizerJSON.path) { missing.append(tokenizerJSON) }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) { missing.append(tokenizerConfig) }
        return missing
    }

    private static func missingCodeGenPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        findFirstGGUFFile(in: rootURL, fileManager: fileManager) == nil ? [rootURL.appendingPathComponent("*.gguf")] : []
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

    private static func missingACEStepPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let turboDir = rootURL.appendingPathComponent("music-acestep-v15-turbo", isDirectory: true)
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let textDir = rootURL.appendingPathComponent("Qwen3-Embedding-0.6B", isDirectory: true)
        if !fileManager.fileExists(atPath: turboDir.path) { missing.append(turboDir) }
        if !fileManager.fileExists(atPath: vaeDir.path) { missing.append(vaeDir) }
        if !fileManager.fileExists(atPath: textDir.path) { missing.append(textDir) }
        return missing
    }

    private static func missingLTXVideoPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let textEncoderConfig = rootURL.appendingPathComponent("text_encoder/config.json")
        let textEncoderWeights = rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json")
        let tokenizerDir = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        if !fileManager.fileExists(atPath: textEncoderConfig.path) { missing.append(textEncoderConfig) }
        if !fileManager.fileExists(atPath: textEncoderWeights.path) { missing.append(textEncoderWeights) }
        if !fileManager.fileExists(atPath: tokenizerDir.path) { missing.append(tokenizerDir) }
        let entries = (try? fileManager.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        let hasTransformer = entries.contains { $0.lastPathComponent.hasPrefix("ltx-2-19") && $0.pathExtension == "safetensors" }
        let hasUpsampler = entries.contains { $0.lastPathComponent.hasPrefix("ltx-2-spatial-upscaler") && $0.pathExtension == "safetensors" }
        if !hasTransformer { missing.append(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors")) }
        if !hasUpsampler { missing.append(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors")) }
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

    static func findFirstGGUFFile(in rootURL: URL, fileManager: FileManager = .default) -> URL? {
        let enumerator = fileManager.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        while let candidate = enumerator?.nextObject() as? URL {
            guard candidate.pathExtension.lowercased() == "gguf" else { continue }
            let values = try? candidate.resourceValues(forKeys: [.isRegularFileKey])
            if values?.isRegularFile == true {
                return candidate
            }
        }
        return nil
    }
}
