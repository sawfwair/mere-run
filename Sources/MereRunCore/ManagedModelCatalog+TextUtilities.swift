import Foundation

extension ManagedModelCatalog {
    static let textUtilitySpecs: [ManagedModelSpec] =
        baseTextUtilitySpecs + pplxEmbedV2Specs + glinerSpecs + layaSpecs + d1Specs + clefSpecs + clairicSpecs

    private static let clairicSpecs = ClairicCatalog.modelIDs.map { id in
        let isFlash = id == ClairicCatalog.flashModelID
        return ManagedModelSpec(
            id: id, category: .textExtract, installShape: .directoryRoot,
            hubFallback: ClairicCatalog.fallback(id), upstreamRepoId: ClairicCatalog.repository(id),
            upstreamRevision: ClairicCatalog.revision(id), validationKind: .clairic, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: isFlash ? 19_330_000_000 : 55_590_000_000,
            defaultCLICommands: ["text extract"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .clairic,
                family: .clairic,
                tier: .base,
                variant: .standard,
                precision: .bf16,
                supports: [.textExtraction],
                components: .rootText,
                upstream: .spec
            ),
            capability: isFlash
                ? .init(
                    "Clairic Flash extraction",
                    "Native Qwen3.5 BF16 field extraction from text, images, or PDF pages.",
                    minimum: 24,
                    recommended: 32
                )
                : .init(
                    "Clairic extraction",
                    "Native Qwen3.5 BF16 field extraction from text, images, or PDF pages.",
                    minimum: 64,
                    recommended: 96
                )
        )
    }

    private static let baseTextUtilitySpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: EmbeddingGemma2Catalog.modelID, category: .textEmbed, installShape: .directoryRoot,
            hubFallback: EmbeddingGemma2Catalog.hubFallback,
            upstreamRepoId: EmbeddingGemma2Catalog.repository, upstreamRevision: EmbeddingGemma2Catalog.revision,
            validationKind: .embeddingGemma2, estimatedDownloadBytes: 1_488_915_288,
            defaultCLICommands: ["text embed"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .embeddingGemma2,
                family: .embed,
                tier: .small,
                variant: .standard,
                precision: .bf16,
                supports: [.textEmbedding, .multimodalEmbedding],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "EmbeddingGemma 2 text and code embeddings",
                "Native bidirectional Gemma encoder with task prefixes and normalized 128/256/512/768-dimensional vectors.",
                minimum: 8,
                recommended: 16
            )
        ),
        ManagedModelSpec(
            id: EmbeddingGemma2CoreMLCatalog.modelID, category: .textEmbed, installShape: .directoryRoot,
            hubFallback: EmbeddingGemma2CoreMLCatalog.hubFallback,
            upstreamRepoId: EmbeddingGemma2CoreMLCatalog.repository, upstreamRevision: EmbeddingGemma2CoreMLCatalog.revision,
            validationKind: .embeddingGemma2CoreML, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_481_911_235,
            defaultCLICommands: ["text embed"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .embeddingGemma2CoreML,
                family: .embed,
                tier: .small,
                variant: .standard,
                precision: .fp16,
                supports: [.textEmbedding, .multimodalEmbedding],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "EmbeddingGemma 2 Core ML",
                "Native Core ML text, image, and audio embeddings on macOS 15 or later; 512-token limit.",
                minimum: 8,
                recommended: 16
            )
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
            apiProfile: .textCode(),
            manifestTemplate: .init(
                engine: .qwen3Coder,
                family: .code,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                supports: [.chat, .codeGeneration],
                upstream: .spec
            ),
            capability: .init(
                "Qwen3-Coder Next",
                "Runs the GGUF Qwen3-Coder Next model through llama.cpp for local coding and agent sessions.",
                minimum: 64,
                recommended: 96
            )
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
            defaultCLICommands: ["text embed"],
            manifestTemplate: .init(
                engine: .qwen3Embedding,
                family: .embed,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.textEmbedding],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Text embeddings",
                "Creates Qwen3 embedding vectors for local search and retrieval workflows.",
                minimum: 8,
                recommended: 16,
                setup: true
            )
        ),
        ManagedModelSpec(
            id: Qwen3VLEmbeddingCatalog.modelID,
            category: .visionEmbed,
            installShape: .directoryRoot,
            hubFallback: Qwen3VLEmbeddingCatalog.hubFallbackConfig,
            upstreamRepoId: Qwen3VLEmbeddingCatalog.defaultRepoID,
            upstreamRevision: Qwen3VLEmbeddingCatalog.defaultRevision,
            validationKind: .qwen3VLEmbedding,
            estimatedDownloadBytes: 4_266_564_017,
            defaultCLICommands: ["vision embed"],
            manifestTemplate: .init(
                engine: .qwen3Embedding,
                family: .embed,
                tier: .small,
                variant: .standard,
                precision: .bf16,
                supports: [.textEmbedding, .multimodalEmbedding],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Multimodal embeddings",
                "Creates shared Qwen3-VL text and image vectors for local retrieval and ranking.",
                minimum: 16,
                recommended: 24
            )
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
            defaultCLICommands: ["text anonymize"],
            manifestTemplate: .init(
                engine: .openAIPrivacyFilter,
                family: .privacy,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.textAnonymization],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Text anonymization",
                "Detects and replaces private text spans before prompts or logs leave a local workflow.",
                minimum: 8,
                recommended: 16,
                setup: true
            )
        ),
    ]

    private static let pplxEmbedV2Specs: [ManagedModelSpec] = PPLXEmbedV2Catalog.modelIDs.map { modelID in
        let isContext = modelID == PPLXEmbedV2Catalog.contextID
            || modelID == PPLXEmbedV2Catalog.contextQuantizedID
        let precision: MereRunModelManifest.Precision = switch modelID {
        case PPLXEmbedV2Catalog.contextQuantizedID: .int8
        case PPLXEmbedV2Catalog.lateQuantizedID: .int4
        default: .fp32
        }
        let capability: ManagedModelCapabilityMetadata = switch modelID {
        case PPLXEmbedV2Catalog.contextID:
            .init(
                "PPLX Embed v2 Context 9B Preview",
                "Encodes document chunks together and returns contextual int8 vectors for retrieval.",
                minimum: 64,
                recommended: 96
            )
        case PPLXEmbedV2Catalog.contextQuantizedID:
            .init(
                "PPLX Embed v2 Context 9B Preview Q8",
                "Native Q8 weights with contextual int8 chunk output; short-input parity validated.",
                minimum: 24,
                recommended: 36
            )
        case PPLXEmbedV2Catalog.lateSmallID:
            .init(
                "PPLX Embed v2 Late 0.6B",
                "Creates native 128-dimensional token vectors for text and image document retrieval with MaxSim.",
                minimum: 16,
                recommended: 24
            )
        case PPLXEmbedV2Catalog.lateQuantizedID:
            .init(
                "PPLX Embed v2 Late 9B Mixed Q4/Q8",
                "Native packed token vectors; short text and synthetic image parity validated.",
                minimum: 24,
                recommended: 36
            )
        default:
            .init(
                "PPLX Embed v2 Late 9B",
                "Creates native FP32 token vectors for text and image retrieval with the large late encoder.",
                minimum: 64,
                recommended: 96
            )
        }
        return ManagedModelSpec(
            id: modelID, category: .textEmbed, installShape: .directoryRoot,
            hubFallback: PPLXEmbedV2Catalog.hubFallback(modelID),
            upstreamRepoId: PPLXEmbedV2Catalog.repository(modelID), upstreamRevision: PPLXEmbedV2Catalog.revision(modelID),
            validationKind: .pplxEmbedV2, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: PPLXEmbedV2Catalog.estimatedDownloadBytes(modelID),
            defaultCLICommands: ["text embed"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .pplxEmbedV2,
                family: .embed,
                tier: .latest,
                variant: .standard,
                precision: precision,
                supports: isContext ? [.textEmbedding] : [.textEmbedding, .multimodalEmbedding],
                components: .rootText,
                upstream: .spec
            ),
            capability: capability
        )
    }

    private static let glinerSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: GLiNERCatalog.modelID, category: .textClassify, installShape: .directoryRoot,
            hubFallback: GLiNERCatalog.hubFallback,
            upstreamRepoId: GLiNERCatalog.repository, upstreamRevision: GLiNERCatalog.revision,
            validationKind: .gliner25Decide, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_945_828_140,
            defaultCLICommands: ["text classify", "text extract"],
            manifestTemplate: .init(
                engine: .gliner25Decide,
                family: .gliner,
                tier: .base,
                variant: .standard,
                precision: .fp32,
                supports: [.textClassification, .textExtraction],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "GLiNER2.5 Decide classification and extraction",
                "Native DeBERTa classification, entities, relations, and structured extraction; English checkpoint.",
                minimum: 8,
                recommended: 16
            )
        )
    ]

    private static let layaSpecs: [ManagedModelSpec] = LayaCatalog.modelIDs.map { modelID in
        let prefix = LayaCatalog.subfolder(modelID: modelID)
        let path = prefix.isEmpty ? "." : prefix
        let tokenizerPath = prefix.isEmpty ? "tokenizer" : "\(prefix)/tokenizer"
        let capability: ManagedModelCapabilityMetadata = switch modelID {
        case LayaCatalog.multilingualID:
            .init(
                "Laya multilingual decisions",
                "Native mmBERT decisions for multilingual text.",
                minimum: 8,
                recommended: 16
            )
        case LayaCatalog.typedDecisionsID:
            .init(
                "Laya typed decisions",
                "Native English checkpoint fine-tuned on typed-decision workflows.",
                minimum: 8,
                recommended: 16
            )
        default:
            .init(
                "Laya English decisions",
                "Native ModernBERT choice, score, and boolean decisions; English checkpoint.",
                minimum: 8,
                recommended: 16
            )
        }
        return ManagedModelSpec(
            id: modelID, category: .textDecide, installShape: .structuredRoot,
            hubFallback: LayaCatalog.hubFallback(modelID: modelID),
            upstreamRepoId: LayaCatalog.repository, upstreamRevision: LayaCatalog.revision,
            validationKind: .laya, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: modelID == LayaCatalog.multilingualID ? 678_201_636
                : modelID == LayaCatalog.typedDecisionsID ? 846_195_716 : 846_195_574,
            defaultCLICommands: ["text decide"],
            manifestTemplate: .init(
                engine: .laya,
                family: .laya,
                tier: .base,
                variant: .standard,
                precision: .fp16,
                supports: [.textDecision],
                components: .init(
                    tokenizer: .local(path: tokenizerPath),
                    textEncoder: .local(path: path),
                    transformer: .local(path: path)
                ),
                upstream: .spec
            ),
            capability: capability
        )
    }

    private static let d1Specs: [ManagedModelSpec] = D1Catalog.modelIDs.map { modelID in
        let isOmni = modelID == D1Catalog.omniModelID
        let repository = isOmni ? D1Catalog.omniRepository : D1Catalog.repository
        let revision = isOmni ? D1Catalog.omniRevision : D1Catalog.revision
        return ManagedModelSpec(
            id: modelID, category: .textDecide, installShape: .directoryRoot,
            hubFallback: D1Catalog.hubFallback(modelID: modelID),
            upstreamRepoId: repository,
            upstreamRevision: revision,
            usageRestriction: usageRestriction(
                summary: "D1 uses the custom LFM Open License v1.0; review its terms before use.",
                license: "LFM Open License v1.0",
                sourceRepoId: repository,
                sourceRevision: revision,
                licenseURL: "https://huggingface.co/\(repository)/blob/\(revision)/LICENSE"
            ),
            validationKind: .d1, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: isOmni ? 2_400_000_000 : 6_300_000_000,
            defaultCLICommands: ["text decide"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .d1,
                family: .d1,
                tier: .base,
                variant: .standard,
                precision: isOmni ? .fp32 : .bf16,
                supports: [.textDecision],
                components: .rootText,
                upstream: .spec
            ),
            capability: isOmni
                ? .init(
                    "D1 omni decisions",
                    "Native bidirectional decisions from text, JSON, images, or audio.",
                    minimum: 8,
                    recommended: 16
                )
                : .init(
                    "D1-3B decisions",
                    "Native LFM2 choice, score, and yes/no decisions from text, JSON, and images.",
                    minimum: 12,
                    recommended: 16
                )
        )
    }

    private static let clefSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: ClefCatalog.modelID, category: .textDecide, installShape: .directoryRoot,
            hubFallback: ClefCatalog.hubFallback,
            upstreamRepoId: ClefCatalog.repository, upstreamRevision: ClefCatalog.revision,
            validationKind: .clef, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 16_310_666_373,
            defaultCLICommands: ["text decide"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .clef,
                family: .clef,
                tier: .base,
                variant: .standard,
                precision: .int4,
                supports: [.textDecision],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Clef structured decisions",
                "Native Qwen3.5 joint schema decisions from text, JSON, images, or video frames.",
                minimum: 32,
                recommended: 64
            )
        ),
        ManagedModelSpec(
            id: ClefCatalog.flashModelID, category: .textDecide, installShape: .directoryRoot,
            hubFallback: ClefCatalog.flashHubFallback,
            upstreamRepoId: ClefCatalog.flashRepository, upstreamRevision: ClefCatalog.flashRevision,
            validationKind: .clef, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 6_213_894_567,
            defaultCLICommands: ["text decide"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .clef,
                family: .clef,
                tier: .base,
                variant: .standard,
                precision: .int4,
                supports: [.textDecision],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Clef Flash structured decisions",
                "Native 9B Qwen3.5 joint schema decisions from text, JSON, images, or video frames.",
                minimum: 16,
                recommended: 32
            )
        )
    ] + [
        ManagedModelSpec(
            id: ClefOmniCatalog.modelID, category: .textDecide, installShape: .directoryRoot,
            hubFallback: ClefOmniCatalog.hubFallback,
            upstreamRepoId: ClefOmniCatalog.repository, upstreamRevision: ClefOmniCatalog.revision,
            validationKind: .clef, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 70_800_000_000,
            defaultCLICommands: ["text decide"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .clef,
                family: .clef,
                tier: .base,
                variant: .standard,
                precision: .bf16,
                supports: [.textDecision],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Clef Omni structured decisions",
                "Native BF16 MoE decisions over text, images, audio, and video.",
                minimum: 96,
                recommended: 128
            )
        )
    ]
}
