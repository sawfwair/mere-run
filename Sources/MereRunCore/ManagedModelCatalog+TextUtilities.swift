import Foundation

extension ManagedModelCatalog {
    static let textUtilitySpecs: [ManagedModelSpec] = [
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
    ]
}
