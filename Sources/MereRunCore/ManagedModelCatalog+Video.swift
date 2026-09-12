import Foundation

extension ManagedModelCatalog {
    static let videoSpecs: [ManagedModelSpec] = [
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
                repoId: LTX25Resources.managedRepository,
                revision: LTX25Resources.managedRevision,
                patterns: LTX25Resources.snapshotPatterns
            ),
            upstreamRepoId: LTX25Resources.managedRepository,
            upstreamRevision: LTX25Resources.managedRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: LTX25Resources.managedRepository,
                sourceRevision: LTX25Resources.managedRevision,
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
                repoId: LTX25Resources.fullManagedRepository,
                revision: LTX25Resources.fullManagedRevision,
                patterns: LTX25Resources.fullSnapshotPatterns
            ),
            upstreamRepoId: LTX25Resources.fullManagedRepository,
            upstreamRevision: LTX25Resources.fullManagedRevision,
            usageRestriction: ltxUsageRestriction(
                sourceRepoId: LTX25Resources.fullManagedRepository,
                sourceRevision: LTX25Resources.fullManagedRevision,
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
            estimatedDownloadBytes: 77_094_088_403,
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
            estimatedDownloadBytes: 58_308_237_969,
            defaultCLICommands: ["video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue,
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MiniMaxH3Resources.fastH3ArtifactRepository,
                revision: MiniMaxH3Resources.fastH3ArtifactRevision,
                patterns: MiniMaxH3Resources.fastH3ArtifactFiles
            ),
            upstreamRepoId: MiniMaxH3Resources.fastH3ArtifactRepository,
            upstreamRevision: MiniMaxH3Resources.fastH3ArtifactRevision,
            usageRestriction: miniMaxH3UsageRestriction,
            validationKind: .miniMaxH3MLX,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 57_559_079_710,
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
    ]

    private static let ltx2DistilledRevision = "c38acc2729229140f083c3a834041e8735ee5260"

    private static let ltx23MLXRevision = "baa5f235ea04fd9c95899d751295c4fd825ee4e2"

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

    private static let miniMaxH3UsageRestriction = usageRestriction(
        summary: "MiniMax-H3 weights may not be used, distributed, or displayed in the United States, European Union, United Kingdom, or Republic of Korea; downstream distribution also requires the Community License agreement, notice, and safeguards.",
        license: "MiniMax-H3 Community License",
        sourceRepoId: MiniMaxH3Resources.sourceRepository,
        sourceRevision: MiniMaxH3Resources.sourceRevision,
        licenseURL: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/ec19cc6daf5d8add9417c18e86b6b58cc6c55027/LICENSE"
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
        license: "Apache License 2.0",
        summary: "The packed LTX 2.5 text encoder includes Gemma 4 weights distributed under the Apache License 2.0.",
        sourceRepoId: LTX25Resources.sourceRepository,
        sourceRevision: LTX25Resources.sourceRevision,
        licenseURL: "https://ai.google.dev/gemma/apache_2"
    )
}
