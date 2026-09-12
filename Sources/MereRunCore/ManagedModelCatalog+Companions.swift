import Foundation

extension ManagedModelCatalog {
    private static func lfm2DSparkSpec(
        id: String,
        repoId: String,
        revision: String,
        estimatedDownloadBytes: Int64
    ) -> ManagedModelSpec {
        ManagedModelSpec(
            id: id,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: repoId,
                revision: revision,
                patterns: LFM2Resources.dsparkSnapshotPatterns
            ),
            upstreamRepoId: repoId,
            upstreamRevision: revision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: repoId,
                sourceRevision: revision,
                licenseURL: "https://huggingface.co/\(repoId)/blob/\(revision)/LICENSE"
            ),
            validationKind: .lfm2DSpark,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: estimatedDownloadBytes
        )
    }

    static var companionSpecs: [ManagedModelSpec] {
        [
            ManagedModelSpec(
                id: Q35Resources.ornith35BMTPModelId,
                category: .textCode,
                installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(
                    repoId: Q35Resources.ornith35BMTPUpstreamRepoId,
                    revision: Q35Resources.ornith35BMTPUpstreamRevision,
                    patterns: Q35Resources.ornith35BMTPSnapshotPatterns
                ),
                upstreamRepoId: Q35Resources.ornith35BMTPUpstreamRepoId,
                upstreamRevision: Q35Resources.ornith35BMTPUpstreamRevision,
                validationKind: .q35MTPAssistant,
                runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: Q35Resources.ornith35BMTPEstimatedDownloadBytes
            ),
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
            lfm2DSparkSpec(
                id: LFM2Resources.defaultDSparkModelId,
                repoId: LFM2Resources.defaultDSparkRepoId,
                revision: LFM2Resources.defaultDSparkRevision,
                estimatedDownloadBytes: LFM2Resources.defaultDSparkEstimatedDownloadBytes
            ),
            lfm2DSparkSpec(
                id: LFM2Resources.smallDSparkModelId,
                repoId: LFM2Resources.smallDSparkRepoId,
                revision: LFM2Resources.smallDSparkRevision,
                estimatedDownloadBytes: LFM2Resources.smallDSparkEstimatedDownloadBytes
            ),
            lfm2DSparkSpec(
                id: LFM2Resources.denseDSparkModelId,
                repoId: LFM2Resources.denseDSparkRepoId,
                revision: LFM2Resources.denseDSparkRevision,
                estimatedDownloadBytes: LFM2Resources.denseDSparkEstimatedDownloadBytes
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
}
