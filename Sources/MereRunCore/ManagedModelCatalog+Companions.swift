import Foundation

extension ManagedModelCatalog {
    private static func lfm2DSparkSpec(
        id: String,
        repoId: String,
        revision: String,
        estimatedDownloadBytes: Int64,
        isVision: Bool = false,
        capability: ManagedModelCapabilityMetadata
    ) -> ManagedModelSpec {
        ManagedModelSpec(
            id: id,
            category: isVision ? .visionChat : .textChat,
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
            estimatedDownloadBytes: estimatedDownloadBytes,
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: []
            ),
            capability: capability
        )
    }

    static var companionSpecs: [ManagedModelSpec] {
        [
            ManagedModelSpec(
                id: SOVTranscribeResources.draftModelID, category: .speechASR, installShape: .directoryRoot,
                hubFallback: HubFallbackConfig(repoId: SOVTranscribeResources.draftRepository, revision: SOVTranscribeResources.draftRevision,
                                               patterns: ["config.json", "model*.safetensors", "model.safetensors.index.json", "README.md", "LICENSE*"]),
                upstreamRepoId: SOVTranscribeResources.draftRepository, upstreamRevision: SOVTranscribeResources.draftRevision,
                validationKind: .sovTranscribeDFlash, runtimeAutoDownloadAllowed: false,
                estimatedDownloadBytes: 860_000_000, apiAvailability: .cliOnly,
                manifestTemplate: .init(
                    engine: .sovTranscribeASR,
                    family: .asr,
                    tier: .latest,
                    variant: .standard,
                    precision: .bf16,
                    supports: [],
                    components: .rootText
                ),
                capability: .init(
                    "SOV-Transcribe-1 DFlash companion",
                    "Auxiliary draft for SOV-Transcribe-1; select it with --draft-model when transcribing with the target.",
                    minimum: 64,
                    recommended: 96
                )
            ),
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
                estimatedDownloadBytes: Q35Resources.ornith35BMTPEstimatedDownloadBytes,
                manifestTemplate: .init(
                    engine: .qwen35HybridMoE,
                    family: .code,
                    tier: .small,
                    variant: .standard,
                    precision: .bf16,
                    supports: []
                ),
                capability: .init(
                    "Ornith 1.5 35B-A3B MTP assistant",
                    "Installs the shared BF16 next-token prediction layer used for verified Ornith speculative decode.",
                    minimum: 32,
                    recommended: 48
                )
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
                estimatedDownloadBytes: 4 * 1_073_741_824,
                manifestTemplate: .init(
                    engine: .gemma4,
                    family: .gemma,
                    tier: .latest,
                    variant: .standard,
                    precision: .bf16,
                    supports: []
                ),
                capability: .init(
                    "Gemma 4 12B MTP assistant",
                    "Installs the companion drafter used for verified Gemma 4 12B decode-tail speculation.",
                    minimum: 24,
                    recommended: 32
                )
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
                estimatedDownloadBytes: LagunaResources.dflashEstimatedDownloadBytes,
                manifestTemplate: .init(
                    engine: .laguna,
                    family: .laguna,
                    tier: .small,
                    variant: .standard,
                    precision: .bf16,
                    supports: []
                ),
                capability: .init(
                    "Laguna S 2.1 DFlash assistant",
                    "Installs the companion drafter used for verified Laguna S 2.1 speculative decoding.",
                    minimum: 96,
                    recommended: 128
                )
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
                estimatedDownloadBytes: MuseGlimmerResources.dflash2EstimatedDownloadBytes,
                manifestTemplate: .init(
                    engine: .museGlimmer,
                    family: .muse,
                    tier: .small,
                    variant: .standard,
                    precision: .bf16,
                    supports: []
                ),
                capability: .init(
                    "Muse Glimmer 30B DFlash2 assistant",
                    "Installs z-lab's grouped-convolution DFlash2 drafter for Muse Glimmer speculative decoding.",
                    minimum: 32,
                    recommended: 64
                )
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
                estimatedDownloadBytes: MuseGlimmerResources.assistantEstimatedDownloadBytes,
                manifestTemplate: .init(
                    engine: .museGlimmer,
                    family: .muse,
                    tier: .small,
                    variant: .standard,
                    precision: .bf16,
                    supports: []
                ),
                capability: .init(
                    "Muse Glimmer 30B DFlash assistant",
                    "Preserves the original Muse Glimmer DFlash drafter for compatible existing installations.",
                    minimum: 32,
                    recommended: 64
                )
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
                estimatedDownloadBytes: NemotronHResources.dsparkEstimatedDownloadBytes,
                manifestTemplate: .init(
                    engine: .nemotronH,
                    family: .nemotron,
                    tier: .small,
                    variant: .standard,
                    precision: .int4,
                    quantization: .init(bits: 4, groupSize: 16, scheme: "modelopt-nvfp4-mlx"),
                    supports: [],
                    upstream: .explicit("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark@e3af76fbff445ef795958bee96bc1126af70fd57")
                ),
                capability: .init(
                    "Nemotron 3.5 DSpark assistant",
                    "Installs the companion drafter used for verified Nemotron 3.5 speculative decoding.",
                    minimum: 32,
                    recommended: 64
                )
            ),
            lfm2DSparkSpec(
                id: LFM2Resources.defaultDSparkModelId,
                repoId: LFM2Resources.defaultDSparkRepoId,
                revision: LFM2Resources.defaultDSparkRevision,
                estimatedDownloadBytes: LFM2Resources.defaultDSparkEstimatedDownloadBytes,
                capability: .init(
                    "LFM2.5 8B-A1B DSpark assistant",
                    "Installs the verified speculative-decoding assistant for LFM2.5 8B-A1B.",
                    minimum: 16,
                    recommended: 24
                )
            ),
            lfm2DSparkSpec(
                id: LFM2Resources.smallDSparkModelId,
                repoId: LFM2Resources.smallDSparkRepoId,
                revision: LFM2Resources.smallDSparkRevision,
                estimatedDownloadBytes: LFM2Resources.smallDSparkEstimatedDownloadBytes,
                capability: .init(
                    "LFM2.5 1.2B DSpark assistant",
                    "Installs the verified speculative-decoding assistant for LFM2.5 1.2B Instruct.",
                    minimum: 8,
                    recommended: 12
                )
            ),
            lfm2DSparkSpec(
                id: LFM2Resources.denseDSparkModelId,
                repoId: LFM2Resources.denseDSparkRepoId,
                revision: LFM2Resources.denseDSparkRevision,
                estimatedDownloadBytes: LFM2Resources.denseDSparkEstimatedDownloadBytes,
                capability: .init(
                    "LFM2.5 2.6B DSpark assistant",
                    "Installs the verified speculative-decoding assistant for LFM2.5 2.6B.",
                    minimum: 8,
                    recommended: 12
                )
            ),
            lfm2DSparkSpec(
                id: LFM2Resources.visionDSparkModelId,
                repoId: LFM2Resources.visionDSparkRepoId,
                revision: LFM2Resources.visionDSparkRevision,
                estimatedDownloadBytes: LFM2Resources.visionDSparkEstimatedDownloadBytes,
                isVision: true,
                capability: .init(
                    "LFM2.5-VL 3B DSpark assistant",
                    "Installs the speculative-decoding assistant for LFM2.5-VL 3B BF16.",
                    minimum: 16,
                    recommended: 24
                )
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
                estimatedDownloadBytes: 8 * 1_073_741_824,
                manifestTemplate: .init(
                    engine: .gemma4,
                    family: .gemma,
                    tier: .latest,
                    variant: .standard,
                    precision: .int4,
                    quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                    supports: [],
                    components: .rootText,
                    upstream: .spec
                ),
                capability: .init(
                    "LTX Gemma 3 12B text encoder",
                    "Installs the shared 4-bit Gemma text encoder used by managed LTX video models.",
                    minimum: 96,
                    recommended: 128
                )
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
