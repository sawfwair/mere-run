import Foundation

extension ManagedModelCatalog {
    static let soundEffectSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshDFlow.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshDFlowSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerA/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5 * 1_073_741_824,
            defaultCLICommands: ["sfx generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshFlow.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshFlowSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerA/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5 * 1_073_741_824,
            defaultCLICommands: ["sfx generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshClap.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshCLAPSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/Woosh-CLAP/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .wooshClap,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 2 * 1_073_741_824,
            defaultCLICommands: ["sfx clap"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshSynchformer.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.synchformerRepoId,
                revision: MMAudioResources.convertedWeightsRevision,
                patterns: wooshSynchformerSnapshotPatterns
            ),
            upstreamRepoId: WooshResources.synchformerRepoId,
            upstreamRevision: MMAudioResources.convertedWeightsRevision,
            usageRestriction: wooshSynchformerUsageRestriction,
            validationKind: .wooshSynchformer,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 475 * 1_048_576,
            defaultCLICommands: ["sfx video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshVFlow8s.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshVFlow8sSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerV/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 6 * 1_073_741_824,
            defaultCLICommands: ["sfx video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.wooshDVFlow8s.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: WooshResources.huggingFaceMirrorRepoId,
                revision: wooshWeightsRevision,
                patterns: wooshDVFlow8sSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "checkpoints/TextConditionerV/tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: WooshResources.robertaTokenizerRepoId,
                        revision: WooshResources.robertaTokenizerRevision,
                        patterns: wooshRobertaTokenizerPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
            upstreamRevision: WooshResources.upstreamRelease,
            usageRestriction: wooshUsageRestriction,
            validationKind: .woosh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 6 * 1_073_741_824,
            defaultCLICommands: ["sfx video generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.mmaudioLarge44kV2.rawValue,
            category: .sfx,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MMAudioResources.convertedWeightsRepoID,
                revision: MMAudioResources.convertedWeightsRevision,
                patterns: mmaudioSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "clip",
                    hubFallback: HubFallbackConfig(
                        repoId: MMAudioResources.clipRepoID,
                        revision: MMAudioResources.clipRevision,
                        patterns: mmaudioCLIPTokenizerPatterns
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "bigvgan",
                    hubFallback: HubFallbackConfig(
                        repoId: MMAudioResources.bigVGANRepoID,
                        revision: MMAudioResources.bigVGANRevision,
                        patterns: mmaudioBigVGANPatterns
                    )
                ),
            ],
            upstreamRepoId: "\(MMAudioResources.upstreamRepoID)@\(MMAudioResources.upstreamRevision)",
            upstreamRevision: MMAudioResources.upstreamRevision,
            usageRestriction: mmaudioUsageRestriction,
            validationKind: .mmaudio,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5_700_000_000,
            defaultCLICommands: ["sfx generate", "sfx video generate"]
        ),
    ]

    private static let wooshWeightsRevision = "f7b524db359f95b2b0bdc4afce12120b72e68bff"

    private static let wooshDFlowSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-DFlow/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerA/*",
    ]

    private static let wooshFlowSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-Flow/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerA/*",
    ]

    private static let wooshCLAPSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-CLAP/*",
    ]

    private static let wooshSynchformerSnapshotPatterns = [
        WooshResources.synchformerFilename,
    ]

    private static let wooshVFlow8sSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-VFlow-8s/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerV/*",
    ]

    private static let wooshDVFlow8sSnapshotPatterns = [
        "README.md",
        "checkpoints/Woosh-DVFlow-8s/*",
        "checkpoints/Woosh-AE/*",
        "checkpoints/TextConditionerV/*",
    ]

    private static let wooshRobertaTokenizerPatterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    private static let mmaudioSnapshotPatterns = [
        "README.md",
        MMAudioResources.networkFilename,
        MMAudioResources.clipFilename,
        MMAudioResources.synchformerFilename,
        MMAudioResources.vaeFilename,
    ]

    private static let mmaudioCLIPTokenizerPatterns = [
        "LICENSE",
        "open_clip_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    private static let mmaudioBigVGANPatterns = [
        "LICENSE",
        "config.json",
        MMAudioResources.bigVGANPyTorchFilename,
    ]

    private static let wooshUsageRestriction = usageRestriction(
        summary: "Woosh model weights are licensed CC BY-NC 4.0 for non-commercial use.",
        license: "CC BY-NC 4.0",
        sourceRepoId: WooshResources.huggingFaceMirrorRepoId,
        sourceRevision: wooshWeightsRevision,
        licenseURL: "https://github.com/SonyResearch/Woosh/blob/v1.0.0/LICENSE"
    )


    private static let wooshSynchformerUsageRestriction = usageRestriction(
        summary: "The MMAudio Synchformer checkpoint used by Woosh is licensed CC BY-NC 4.0 for non-commercial use.",
        component: "synchformer",
        license: "CC BY-NC 4.0",
        sourceRepoId: WooshResources.synchformerRepoId,
        sourceRevision: MMAudioResources.convertedWeightsRevision,
        licenseURL: "https://github.com/hkchengrex/MMAudio#pre-trained-weights"
    )


    private static let mmaudioUsageRestriction = ManagedModelUsageRestriction(
        summary: "MMAudio combines non-commercial checkpoint terms with an Apple research-only visual encoder; each component's terms apply independently.",
        terms: [
            ManagedModelUsageTerm(
                component: "MMAudio checkpoints",
                license: "CC BY-NC 4.0",
                summary: "Published MMAudio checkpoints are limited to non-commercial use.",
                sourceRepoId: MMAudioResources.convertedWeightsRepoID,
                sourceRevision: MMAudioResources.convertedWeightsRevision,
                licenseURL: "https://github.com/hkchengrex/MMAudio#pre-trained-weights"
            ),
            ManagedModelUsageTerm(
                component: "Apple DFN5B CLIP visual encoder",
                license: "Apple Machine Learning Research Model License Agreement",
                summary: "Apple licenses this component exclusively for non-commercial scientific research and academic development.",
                sourceRepoId: MMAudioResources.clipRepoID,
                sourceRevision: MMAudioResources.clipRevision,
                licenseURL: "https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/blob/main/LICENSE"
            ),
        ]
    )
}
