import Foundation

extension ManagedModelCatalog {
    static let textUtilitySpecs: [ManagedModelSpec] = baseTextUtilitySpecs + layaSpecs + d1Specs + clefSpecs

    private static let baseTextUtilitySpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: EmbeddingGemma2Catalog.modelID, category: .textEmbed, installShape: .directoryRoot,
            hubFallback: EmbeddingGemma2Catalog.hubFallback,
            upstreamRepoId: EmbeddingGemma2Catalog.repository, upstreamRevision: EmbeddingGemma2Catalog.revision,
            validationKind: .embeddingGemma2, estimatedDownloadBytes: 1_488_915_288,
            defaultCLICommands: ["text embed"], apiAvailability: .cliOnly
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
            id: Qwen3VLEmbeddingCatalog.modelID,
            category: .visionEmbed,
            installShape: .directoryRoot,
            hubFallback: Qwen3VLEmbeddingCatalog.hubFallbackConfig,
            upstreamRepoId: Qwen3VLEmbeddingCatalog.defaultRepoID,
            upstreamRevision: Qwen3VLEmbeddingCatalog.defaultRevision,
            validationKind: .qwen3VLEmbedding,
            estimatedDownloadBytes: 4_266_564_017,
            defaultCLICommands: ["vision embed"]
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
    ] + PPLXEmbedV2Catalog.modelIDs.map { modelID in
        ManagedModelSpec(
            id: modelID, category: .textEmbed, installShape: .directoryRoot,
            hubFallback: PPLXEmbedV2Catalog.hubFallback(modelID),
            upstreamRepoId: PPLXEmbedV2Catalog.repository(modelID), upstreamRevision: PPLXEmbedV2Catalog.revision(modelID),
            validationKind: .pplxEmbedV2, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: PPLXEmbedV2Catalog.estimatedDownloadBytes(modelID),
            defaultCLICommands: ["text embed"], apiAvailability: .cliOnly
        )
    } + [
        ManagedModelSpec(
            id: GLiNERCatalog.modelID, category: .textClassify, installShape: .directoryRoot,
            hubFallback: GLiNERCatalog.hubFallback,
            upstreamRepoId: GLiNERCatalog.repository, upstreamRevision: GLiNERCatalog.revision,
            validationKind: .gliner25Decide, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_945_828_140,
            defaultCLICommands: ["text classify", "text extract"]
        )
    ]

    private static let layaSpecs: [ManagedModelSpec] = LayaCatalog.modelIDs.map { modelID in
        ManagedModelSpec(
            id: modelID, category: .textDecide, installShape: .structuredRoot,
            hubFallback: LayaCatalog.hubFallback(modelID: modelID),
            upstreamRepoId: LayaCatalog.repository, upstreamRevision: LayaCatalog.revision,
            validationKind: .laya, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: modelID == LayaCatalog.multilingualID ? 678_201_636
                : modelID == LayaCatalog.typedDecisionsID ? 846_195_716 : 846_195_574,
            defaultCLICommands: ["text decide"]
        )
    }

    private static let d1Specs: [ManagedModelSpec] = D1Catalog.modelIDs.map { modelID in
        let repository = modelID == D1Catalog.omniModelID ? D1Catalog.omniRepository : D1Catalog.repository
        let revision = modelID == D1Catalog.omniModelID ? D1Catalog.omniRevision : D1Catalog.revision
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
            estimatedDownloadBytes: modelID == D1Catalog.omniModelID ? 2_400_000_000 : 6_300_000_000,
            defaultCLICommands: ["text decide"], apiAvailability: .cliOnly
        )
    }

    private static let clefSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: ClefCatalog.modelID, category: .textDecide, installShape: .directoryRoot,
            hubFallback: ClefCatalog.hubFallback,
            upstreamRepoId: ClefCatalog.repository, upstreamRevision: ClefCatalog.revision,
            validationKind: .clef, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 16_310_666_373,
            defaultCLICommands: ["text decide"], apiAvailability: .cliOnly
        ),
        ManagedModelSpec(
            id: ClefCatalog.flashModelID, category: .textDecide, installShape: .directoryRoot,
            hubFallback: ClefCatalog.flashHubFallback,
            upstreamRepoId: ClefCatalog.flashRepository, upstreamRevision: ClefCatalog.flashRevision,
            validationKind: .clef, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 6_213_894_567,
            defaultCLICommands: ["text decide"], apiAvailability: .cliOnly
        )
    ]
}
