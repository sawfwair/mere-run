import Foundation

public enum ManagedModelCategory: String, CaseIterable, Hashable, Sendable {
    case image = "image"
    case textChat = "text-chat"
    case textCode = "text-code"
    case textEmbed = "text-embed"
    case textAnonymize = "text-anonymize"
    case speechTTS = "speech-tts"
    case speechASR = "speech-asr"
    case visionOCR = "vision-ocr"
    case visionSegment = "vision-segment"
    case visionGround = "vision-ground"
    case music = "music"
    case video = "video"
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
    case gemma4
    case q35
    case lfm2
    case qwen3TTS
    case qwen3ASR
    case parakeet
    case qwen3Embedding
    case privacyFilter
    case codegenGGUF
    case deepseekV4FlashIMatrixGGUF
    case lightOnOCR
    case sam31
    case falconPerception
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
    public let hubFallback: HubFallbackConfig?
    public let upstreamRepoId: String?
    public let upstreamRevision: String?
    public let validationKind: ManagedModelValidationKind
    public let normalizationKind: ManagedModelNormalizationKind
    public let aliasKind: ManagedModelAliasKind
    public let runtimeAutoDownloadAllowed: Bool
    public let resolutionFallbackIDs: [String]
    public let estimatedDownloadBytes: Int64?
    public let defaultCLICommands: [String]

    public init(
        id: String,
        category: ManagedModelCategory,
        installShape: ManagedModelInstallShape,
        hubFallback: HubFallbackConfig? = nil,
        upstreamRepoId: String? = nil,
        upstreamRevision: String? = nil,
        validationKind: ManagedModelValidationKind,
        normalizationKind: ManagedModelNormalizationKind = .none,
        aliasKind: ManagedModelAliasKind = .none,
        runtimeAutoDownloadAllowed: Bool = true,
        resolutionFallbackIDs: [String] = [],
        estimatedDownloadBytes: Int64? = nil,
        defaultCLICommands: [String] = []
    ) {
        self.id = id
        self.category = category
        self.installShape = installShape
        self.hubFallback = hubFallback
        self.upstreamRepoId = upstreamRepoId
        self.upstreamRevision = upstreamRevision
        self.validationKind = validationKind
        self.normalizationKind = normalizationKind
        self.aliasKind = aliasKind
        self.runtimeAutoDownloadAllowed = runtimeAutoDownloadAllowed
        self.resolutionFallbackIDs = resolutionFallbackIDs
        self.estimatedDownloadBytes = estimatedDownloadBytes
        self.defaultCLICommands = defaultCLICommands
    }
}

public enum ManagedModelCatalog {
    private static let diffusersImageSnapshotPatterns = [
        "model_index.json",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
        "scheduler/*",
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
    private static let kleinNanoUpstreamRepoId = "stereovoid/flux2-klein-4b-4bit"
    private static let bonsaiBinaryUpstreamRepoId = "prism-ml/bonsai-image-binary-4B-mlx-1bit"
    private static let bonsaiTernaryUpstreamRepoId = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"

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
            defaultCLICommands: ["image generate"]
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
                revision: "main",
                patterns: zImageNanoSnapshotPatterns
            ),
            upstreamRepoId: zImageNanoUpstreamRepoId,
            upstreamRevision: "main",
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
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
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: "text-chat-mebot",
            category: .textChat,
            installShape: .directoryRoot,
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["api serve"]
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
            defaultCLICommands: ["api serve"]
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
            defaultCLICommands: ["text chat", "api serve"]
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
            defaultCLICommands: ["api serve"]
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
            defaultCLICommands: ["api serve"]
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
            defaultCLICommands: ["chat", "api serve"]
        ),
        ManagedModelSpec(
            id: AgentModelResources.qwen35NineBModelId,
            category: .textCode,
            installShape: .singleFile(relativePath: AgentModelResources.qwen35NineBRelativePath),
            hubFallback: AgentModelResources.qwen35NineBHubFallbackConfig,
            upstreamRepoId: AgentModelResources.qwen35NineBRepoId,
            upstreamRevision: AgentModelResources.qwen35NineBRevision,
            validationKind: .codegenGGUF,
            defaultCLICommands: ["api serve", "text code"]
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
                patterns: ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"]
            ),
            upstreamRepoId: "unsloth/Qwen3.6-35B-A3B-GGUF",
            upstreamRevision: "main",
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: 22 * 1_073_741_824,
            defaultCLICommands: ["text chat", "api serve"]
        ),
        ManagedModelSpec(
            id: DeepseekV4FlashResources.defaultModelId,
            category: .textChat,
            installShape: .singleFile(relativePath: DeepseekV4FlashResources.managedRelativePath),
            hubFallback: DeepseekV4FlashResources.hubFallbackConfig,
            upstreamRepoId: DeepseekV4FlashResources.defaultRepoId,
            upstreamRevision: DeepseekV4FlashResources.defaultRevision,
            validationKind: .deepseekV4FlashIMatrixGGUF,
            defaultCLICommands: ["api serve", "agent"]
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
            validationKind: .lfm2,
            estimatedDownloadBytes: 10 * 1_073_741_824,
            defaultCLICommands: ["text chat", "api serve"]
        ),
        ManagedModelSpec(
            id: "speech-tts-qwen3-nano",
            category: .speechTTS,
            installShape: .directoryRoot,
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
            id: "text-code-qwen3",
            category: .textCode,
            installShape: .singleFile(relativePath: CodeGenResources.managedRelativePath),
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
            defaultCLICommands: ["text anonymize"]
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
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: "vision-segment-sam31",
            category: .visionSegment,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/sam3.1-bf16",
                revision: "main",
                patterns: [
                    "config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "mlx-community/sam3.1-bf16",
            upstreamRevision: "main",
            validationKind: .sam31,
            runtimeAutoDownloadAllowed: false,
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
            defaultCLICommands: ["vision ground"]
        ),
        ManagedModelSpec(
            id: "music-acestep",
            category: .music,
            installShape: .structuredRoot,
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

    public static func missingHubSourceMessage(for modelId: String) -> String {
        "Model \(modelId) does not have a Hugging Face Hub source in this public build. Install it from a local path or choose a model listed by `mere.run model capabilities --recommended`."
    }
}

public extension ManagedModelSpec {
    var modelID: ModelResolver.ModelID? {
        ModelResolver.ModelID(rawValue: id)
    }

    var canBePulledWithoutConfiguration: Bool {
        hubFallback != nil
    }

    func hasAnyManagedDownloadSource() -> Bool {
        hubFallback != nil
    }

    func normalizedRootURL(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let base = rootURL.resolvingSymlinksInPath()
        switch normalizationKind {
        case .none, .musicACEStep:
            return base
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
        case .gemma4:
            return Gemma4Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .q35:
            return Q35Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .lfm2:
            return LFM2Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .sam31:
            return SAM31Resources(modelRootURL: rootURL).missingRequiredPaths(fileManager: fileManager)
        case .falconPerception:
            return FalconPerceptionResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .qwen3TTS:
            return Self.missingQwen3TTSPaths(in: rootURL, fileManager: fileManager)
        case .qwen3ASR:
            return Self.missingQwen3ASRPaths(in: rootURL, fileManager: fileManager)
        case .parakeet:
            return Self.missingParakeetPaths(in: rootURL, fileManager: fileManager)
        case .qwen3Embedding:
            return Qwen3EmbeddingResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .privacyFilter:
            return OpenAIPrivacyFilterResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .codegenGGUF:
            return Self.missingCodeGenPaths(in: rootURL, fileManager: fileManager)
        case .deepseekV4FlashIMatrixGGUF:
            return Self.missingDeepseekV4FlashIMatrixPaths(in: rootURL, fileManager: fileManager)
        case .lightOnOCR:
            return Self.missingLightOnOCRPaths(in: rootURL, fileManager: fileManager)
        case .aceStep:
            return Self.missingACEStepPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: rootURL, fileManager: fileManager)
        case .hfTextChat:
            return Self.missingHFTextRootPaths(in: rootURL, fileManager: fileManager)
        }
    }

    func validationMessages(in rootURL: URL, fileManager: FileManager = .default) -> [String] {
        switch validationKind {
        case .sam31:
            return SAM31Resources.validateRoot(normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .falconPerception:
            return FalconPerceptionResources.validateRoot(normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
                .map { "Missing required LTX file: \($0.path)" }
        default:
            return missingPaths(in: rootURL, fileManager: fileManager).map { "Missing required file: \($0.path)" }
        }
    }

    func isManagedRootComplete(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        missingPaths(in: rootURL, fileManager: fileManager).isEmpty
            && managedSourceMatches(rootURL, fileManager: fileManager)
    }

    private func managedSourceMatches(_ rootURL: URL, fileManager: FileManager) -> Bool {
        guard id == ModelResolver.ModelID.zetaNano.rawValue,
              let expectedRepo = upstreamRepoId else {
            return true
        }
        let normalized = normalizedRootURL(rootURL, fileManager: fileManager)
        guard let manifest = try? MereRunModelManifest.loadIfPresent(from: normalized, fileManager: fileManager),
              let installedRepo = manifest.upstreamRepoId else {
            return true
        }

        if installedRepo == expectedRepo {
            return true
        }
        if let expectedWithRevision = upstreamRevision.map({ "\(expectedRepo)@\($0)" }),
           installedRepo == expectedWithRevision {
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

    /// DeepSeek V4 Flash explicitly prefers the imatrix-tuned GGUF (per the
    /// upstream README's "USE THE IMATRIX VERSIONS" note). A directory that
    /// only contains the legacy non-imatrix GGUF is considered *not* complete,
    /// so `mere.run model pull` will fetch the preferred variant instead of
    /// reporting "already installed."
    private static func missingDeepseekV4FlashIMatrixPaths(
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        let resolved = rootURL.resolvingSymlinksInPath()
        guard let enumerator = fileManager.enumerator(
            at: resolved,
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

    private static func missingACEStepPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let upstreamTurboDir = rootURL.appendingPathComponent("acestep-v15-turbo", isDirectory: true)
        let compatibilityTurboDir = rootURL.appendingPathComponent("music-acestep-v15-turbo", isDirectory: true)
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let textDir = rootURL.appendingPathComponent("Qwen3-Embedding-0.6B", isDirectory: true)
        if !fileManager.fileExists(atPath: upstreamTurboDir.path) && !fileManager.fileExists(atPath: compatibilityTurboDir.path) {
            missing.append(upstreamTurboDir)
        }
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
        let resolvedRoot = rootURL.resolvingSymlinksInPath()
        let enumerator = fileManager.enumerator(
            at: resolvedRoot,
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
