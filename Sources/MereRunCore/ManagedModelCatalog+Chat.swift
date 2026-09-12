import Foundation

extension ManagedModelCatalog {
    static let chatSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: "text-chat-mebot",
            category: .textChat,
            installShape: .directoryRoot,
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["api serve"],
            apiProfile: .klein()
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
            estimatedDownloadBytes: 62_578_654_199,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: DiffusionGemmaResources.modelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: DiffusionGemmaResources.upstreamModelID,
                revision: DiffusionGemmaResources.upstreamRevision,
                patterns: DiffusionGemmaResources.snapshotPatterns
            ),
            upstreamRepoId: DiffusionGemmaResources.upstreamModelID,
            upstreamRevision: DiffusionGemmaResources.upstreamRevision,
            validationKind: .diffusionGemma,
            estimatedDownloadBytes: DiffusionGemmaResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .diffusionGemma()
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
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: Gemma4Resources.twelveBModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.twelveBUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.twelveBUpstreamModelId,
            validationKind: .gemma4,
            estimatedDownloadBytes: 25 * 1_073_741_824,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            companionModelIDs: [Gemma4MTPResources.modelId],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: Gemma4Resources.twelveB4BitModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.twelveB4BitUpstreamModelId,
                revision: Gemma4Resources.twelveB4BitUpstreamRevision,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.twelveB4BitUpstreamModelId,
            upstreamRevision: Gemma4Resources.twelveB4BitUpstreamRevision,
            validationKind: .gemma4,
            estimatedDownloadBytes: 6_773_374_762,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            companionModelIDs: [Gemma4MTPResources.modelId],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: Gemma4Resources.visionTwelveBModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.twelveBUpstreamModelId,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.twelveBUpstreamModelId,
            validationKind: .gemma4Unified,
            estimatedDownloadBytes: 25 * 1_073_741_824,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            companionModelIDs: [Gemma4MTPResources.modelId],
            apiProfile: .gemma4(inputModalities: [.text, .image])
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
            estimatedDownloadBytes: 16_024_791_983,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
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
            estimatedDownloadBytes: 62_578_654_199,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4()
        ),
        ManagedModelSpec(
            id: LagunaResources.modelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LagunaResources.upstreamModelID,
                revision: LagunaResources.upstreamRevision,
                patterns: LagunaResources.snapshotPatterns
            ),
            upstreamRepoId: LagunaResources.upstreamModelID,
            upstreamRevision: LagunaResources.upstreamRevision,
            validationKind: .laguna,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LagunaResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            companionModelIDs: [LagunaResources.dflashModelID],
            apiProfile: .laguna()
        ),
        ManagedModelSpec(
            id: LagunaResources.xsModelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LagunaResources.xsUpstreamModelID,
                revision: LagunaResources.xsUpstreamRevision,
                patterns: LagunaResources.snapshotPatterns
            ),
            upstreamRepoId: LagunaResources.xsUpstreamModelID,
            upstreamRevision: LagunaResources.xsUpstreamRevision,
            validationKind: .laguna,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LagunaResources.xsEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "text train-lora",
                "api serve",
                "model benchmark chat",
            ],
            apiProfile: .laguna()
        ),
        ManagedModelSpec(
            id: InklingResources.modelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: InklingResources.hubFallbackConfig,
            upstreamRepoId: InklingResources.artifactRepoID,
            upstreamRevision: InklingResources.artifactRevision,
            validationKind: .inkling,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: InklingResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "text train-lora"]
        ),
        ManagedModelSpec(
            id: MuseGlimmerResources.modelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: MuseGlimmerResources.artifactRepoId,
                revision: MuseGlimmerResources.artifactRevision,
                patterns: MuseGlimmerResources.snapshotPatterns
            ),
            upstreamRepoId: MuseGlimmerResources.artifactRepoId,
            upstreamRevision: MuseGlimmerResources.artifactRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Apache-2.0 model subject to Meta's bundled usage policy; upstream states it is not intended for download or use by people under 18.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Muse Glimmer 30B",
                        license: "Apache-2.0 with upstream usage policy",
                        summary: "Review LICENSE and USAGE_POLICY.md before installing or deploying.",
                        sourceRepoId: MuseGlimmerResources.upstreamRepoId,
                        sourceRevision: MuseGlimmerResources.upstreamRevision,
                        licenseURL: "https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/\(MuseGlimmerResources.upstreamRevision)/USAGE_POLICY.md"
                    ),
                ]
            ),
            validationKind: .museGlimmer,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: MuseGlimmerResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            companionModelIDs: [MuseGlimmerResources.dflash2ModelId],
            apiProfile: .museGlimmer()
        ),
        ManagedModelSpec(
            id: NemotronHResources.modelID,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: NemotronHResources.artifactRepoID,
                revision: NemotronHResources.artifactRevision,
                patterns: NemotronHResources.snapshotPatterns
            ),
            upstreamRepoId: NemotronHResources.artifactRepoID,
            upstreamRevision: NemotronHResources.artifactRevision,
            validationKind: .nemotronH,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: NemotronHResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            companionModelIDs: [NemotronHResources.dsparkModelID],
            apiProfile: .nemotronH()
        ),
        ManagedModelSpec(
            id: NemotronOmniResources.modelID,
            category: .omniChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: NemotronOmniResources.nativeRepoID,
                revision: NemotronOmniResources.nativeRevision,
                patterns: NemotronOmniResources.snapshotPatterns
            ),
            upstreamRepoId: NemotronOmniResources.upstreamRepoID,
            upstreamRevision: NemotronOmniResources.upstreamRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Use is governed by the NVIDIA Open Model Agreement; review and acknowledge the governing terms before download.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Nemotron 3 Nano Omni 30B-A3B Reasoning BF16",
                        license: "NVIDIA Open Model Agreement",
                        summary: "Review the NVIDIA Open Model Agreement before installing or deploying.",
                        sourceRepoId: NemotronOmniResources.upstreamRepoID,
                        sourceRevision: NemotronOmniResources.upstreamRevision,
                        licenseURL: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/"
                    ),
                ]
            ),
            validationKind: .nemotronOmni,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: NemotronOmniResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .nemotronOmni()
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
            defaultCLICommands: ["chat", "api serve"],
            apiProfile: .q36(contextWindow: Q35Resources.defaultContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38TwentySevenBModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.q38TwentySevenBModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.q38TwentySevenBUpstreamRepoId,
            upstreamRevision: Q35Resources.q38TwentySevenBUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38TwentySevenBEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
                "model benchmark vlm",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38TwentySevenB4BitModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(
                for: Q35Resources.q38TwentySevenB4BitModelId
            )?.hubFallbackConfig,
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: Q35Resources.q38MTPComponentPath,
                    hubFallback: HubFallbackConfig(
                        repoId: Q35Resources.q38MTP4BitUpstreamRepoId,
                        revision: Q35Resources.q38MTP4BitUpstreamRevision,
                        patterns: Q35Resources.q38MTPComponentSnapshotPatterns
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: Q35Resources.q38VisionComponentPath,
                    hubFallback: HubFallbackConfig(
                        repoId: Q35Resources.q38TwentySevenBUpstreamRepoId,
                        revision: Q35Resources.q38TwentySevenBUpstreamRevision,
                        patterns: Q35Resources.q38VisionComponentSnapshotPatterns
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: Q35Resources.q38LicenseComponentPath,
                    hubFallback: HubFallbackConfig(
                        repoId: Q35Resources.q38TwentySevenBUpstreamRepoId,
                        revision: Q35Resources.q38TwentySevenBUpstreamRevision,
                        patterns: Q35Resources.q38LicenseComponentSnapshotPatterns
                    )
                ),
            ],
            upstreamRepoId: Q35Resources.q38TwentySevenB4BitUpstreamRepoId,
            upstreamRevision: Q35Resources.q38TwentySevenB4BitUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38TwentySevenB4BitEstimatedDownloadBytes
                + Q35Resources.q38VisionComponentEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
                "model benchmark vlm",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38FlashNextMixedModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(
                for: Q35Resources.q38FlashNextMixedModelId
            )?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.q38FlashNextMixedUpstreamRepoId,
            upstreamRevision: Q35Resources.q38FlashNextMixedUpstreamRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Use is governed by the Qwen Community License 1.0; review and acknowledge the terms before download.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Qwen3.8-Flash-Next mixed Q2/Q4 MLX",
                        license: "Qwen Community License 1.0",
                        summary: "Review the upstream redistribution, attribution, and restricted-use terms before installing or deploying.",
                        sourceRepoId: "Qwen/Qwen3.8-Flash-Next",
                        sourceRevision: "f5d08274bafd880402bd16f5e3e6c514136ec06c",
                        licenseURL: "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE"
                    ),
                ]
            ),
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38FlashNextMixedEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38FlashNext3BitModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(
                for: Q35Resources.q38FlashNext3BitModelId
            )?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.q38FlashNext3BitUpstreamRepoId,
            upstreamRevision: Q35Resources.q38FlashNext3BitUpstreamRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Use is governed by the Qwen Community License 1.0; review and acknowledge the terms before download.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Qwen3.8-Flash-Next activation-weighted Q3 MLX",
                        license: "Qwen Community License 1.0",
                        summary: "Review the upstream redistribution, attribution, and restricted-use terms before installing or deploying.",
                        sourceRepoId: "Qwen/Qwen3.8-Flash-Next",
                        sourceRevision: "f5d08274bafd880402bd16f5e3e6c514136ec06c",
                        licenseURL: "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE"
                    ),
                ]
            ),
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38FlashNext3BitEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38FlashNext3BitNativePLEModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(
                for: Q35Resources.q38FlashNext3BitNativePLEModelId
            )?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.q38FlashNext3BitNativePLEUpstreamRepoId,
            upstreamRevision: Q35Resources.q38FlashNext3BitNativePLEUpstreamRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Use is governed by the Qwen Community License 1.0; review and acknowledge the terms before download.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Qwen3.8-Flash-Next activation-weighted Q3 MLX native PLE pack",
                        license: "Qwen Community License 1.0",
                        summary: "Review the upstream redistribution, attribution, and restricted-use terms before installing or deploying.",
                        sourceRepoId: "Qwen/Qwen3.8-Flash-Next",
                        sourceRevision: "f5d08274bafd880402bd16f5e3e6c514136ec06c",
                        licenseURL: "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE"
                    ),
                ]
            ),
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38FlashNext3BitNativePLEEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.q38FlashNext4BitModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(
                for: Q35Resources.q38FlashNext4BitModelId
            )?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.q38FlashNext4BitUpstreamRepoId,
            upstreamRevision: Q35Resources.q38FlashNext4BitUpstreamRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Use is governed by the Qwen Community License 1.0; review and acknowledge the terms before download.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "Qwen3.8-Flash-Next Q4 MLX",
                        license: "Qwen Community License 1.0",
                        summary: "Review the upstream redistribution, attribution, and restricted-use terms before installing or deploying.",
                        sourceRepoId: "Qwen/Qwen3.8-Flash-Next",
                        sourceRevision: "f5d08274bafd880402bd16f5e3e6c514136ec06c",
                        licenseURL: "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE"
                    ),
                ]
            ),
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.q38FlashNext4BitEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength)
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai27B1BitModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai27B1BitModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.bonsai27B1BitUpstreamRepoId,
            upstreamRevision: Q35Resources.bonsai27B1BitUpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.bonsai27B1BitEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(
                contextWindow: Q35Resources.bonsai27B1BitContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai27B2BitModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai27B2BitModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.bonsai27B2BitUpstreamRepoId,
            upstreamRevision: Q35Resources.bonsai27B2BitUpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.bonsai27B2BitEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(
                contextWindow: Q35Resources.bonsai27B2BitContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith9BModelId,
            category: .textCode,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith9BModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith9BUpstreamRepoId,
            upstreamRevision: Q35Resources.ornith9BUpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.ornith9BEstimatedDownloadBytes,
            defaultCLICommands: ["chat", "api serve", "agent start"],
            apiProfile: .q36(
                contextWindow: Q35Resources.defaultContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith35BMLX4BitModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith35BMLX4BitModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith35BMLX4BitBundleRepoId,
            upstreamRevision: Q35Resources.ornith35BMLX4BitBundleRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.ornith35BMLX4BitBundleEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "agent start",
                "model benchmark chat",
                "model benchmark code",
                "model benchmark vlm",
            ],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith35BMLX6BitModelId,
            category: .textCode,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith35BMLX6BitModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith35BMLX6BitUpstreamRepoId,
            upstreamRevision: Q35Resources.ornith35BMLX6BitUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.ornith35BMLX6BitEstimatedDownloadBytes,
            defaultCLICommands: ["chat", "api serve", "agent start", "model benchmark code"],
            companionModelIDs: [Q35Resources.ornith35BMTPModelId],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith35BMLX8BitModelId,
            category: .textCode,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith35BMLX8BitModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith35BMLX8BitUpstreamRepoId,
            upstreamRevision: Q35Resources.ornith35BMLX8BitUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.ornith35BMLX8BitEstimatedDownloadBytes,
            defaultCLICommands: ["chat", "api serve", "agent start", "model benchmark code"],
            companionModelIDs: [Q35Resources.ornith35BMTPModelId],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith35BMLXModelId,
            category: .textCode,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith35BMLXModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith35BMLXUpstreamRepoId,
            upstreamRevision: Q35Resources.ornith35BMLXUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.ornith35BMLXEstimatedDownloadBytes,
            defaultCLICommands: ["chat", "api serve", "agent start", "model benchmark code"],
            companionModelIDs: [Q35Resources.ornith35BMTPModelId],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.ornith35BVisionModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.ornith35BVisionModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.ornith35BVisionUpstreamRepoId,
            upstreamRevision: Q35Resources.ornith35BVisionUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Q35Resources.ornith35BVisionEstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "agent start",
                "model benchmark chat",
                "model benchmark code",
                "model benchmark vlm",
            ],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            )
        ),
        ManagedModelSpec(
            id: AgentModelResources.qwen35NineBModelId,
            category: .textCode,
            installShape: .singleFile(relativePath: AgentModelResources.qwen35NineBRelativePath),
            hubFallback: AgentModelResources.qwen35NineBHubFallbackConfig,
            upstreamRepoId: AgentModelResources.qwen35NineBRepoId,
            upstreamRevision: AgentModelResources.qwen35NineBRevision,
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: 5_680_522_464,
            defaultCLICommands: ["api serve", "text code"],
            apiProfile: .textCode()
        ),
        ManagedModelSpec(
            id: NorthMiniCodeResources.modelId,
            category: .textCode,
            installShape: .singleFile(relativePath: NorthMiniCodeResources.managedRelativePath),
            hubFallback: NorthMiniCodeResources.hubFallbackConfig,
            upstreamRepoId: NorthMiniCodeResources.upstreamRepoId,
            upstreamRevision: NorthMiniCodeResources.upstreamRevision,
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: NorthMiniCodeResources.estimatedDownloadBytes,
            defaultCLICommands: ["text code", "api serve", "agent start"],
            apiProfile: .textCode(
                contextWindow: NorthMiniCodeResources.runtimeContextLength,
                maximumOutputTokens: NorthMiniCodeResources.maxOutputTokens
            )
        ),
        ManagedModelSpec(
            id: Ornith35BCodeResources.modelId,
            category: .textCode,
            installShape: .singleFile(relativePath: Ornith35BCodeResources.managedRelativePath),
            hubFallback: Ornith35BCodeResources.hubFallbackConfig,
            upstreamRepoId: Ornith35BCodeResources.upstreamRepoId,
            upstreamRevision: Ornith35BCodeResources.upstreamRevision,
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: Ornith35BCodeResources.estimatedDownloadBytes,
            defaultCLICommands: ["text code", "api serve", "agent start"],
            apiProfile: .textCode(
                contextWindow: Ornith35BCodeResources.runtimeContextLength,
                maximumOutputTokens: Ornith35BCodeResources.maxOutputTokens
            )
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
                patterns: ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"],
                filePath: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
            ),
            upstreamRepoId: "unsloth/Qwen3.6-35B-A3B-GGUF",
            upstreamRevision: "main",
            validationKind: .codegenGGUF,
            estimatedDownloadBytes: 22 * 1_073_741_824,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .textCode()
        ),
        ManagedModelSpec(
            id: DeepseekV4FlashResources.defaultModelId,
            category: .textChat,
            installShape: .singleFile(relativePath: DeepseekV4FlashResources.managedRelativePath),
            hubFallback: DeepseekV4FlashResources.hubFallbackConfig,
            upstreamRepoId: DeepseekV4FlashResources.defaultRepoId,
            upstreamRevision: DeepseekV4FlashResources.defaultRevision,
            validationKind: .deepseekV4FlashIMatrixGGUF,
            estimatedDownloadBytes: DeepseekV4FlashResources.defaultGGUFByteCount,
            defaultCLICommands: ["api serve", "agent"],
            apiProfile: .deepseekV4Flash()
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
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.upstreamRepoId,
                sourceRevision: LFM2Resources.upstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-8bit/blob/\(LFM2Resources.upstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 10 * 1_073_741_824,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.a1bBF16ModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.a1bBF16UpstreamRepoId,
                revision: LFM2Resources.a1bBF16UpstreamRevision,
                patterns: LFM2Resources.snapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.a1bBF16UpstreamRepoId,
            upstreamRevision: LFM2Resources.a1bBF16UpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.a1bBF16UpstreamRepoId,
                sourceRevision: LFM2Resources.a1bBF16UpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/blob/\(LFM2Resources.a1bBF16UpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.a1bBF16EstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            companionModelIDs: [LFM2Resources.defaultDSparkModelId],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.smallModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.smallUpstreamRepoId,
                revision: LFM2Resources.smallUpstreamRevision,
                patterns: LFM2Resources.snapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.smallUpstreamRepoId,
            upstreamRevision: LFM2Resources.smallUpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.smallUpstreamRepoId,
                sourceRevision: LFM2Resources.smallUpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/\(LFM2Resources.smallUpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.smallEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            companionModelIDs: [LFM2Resources.smallDSparkModelId],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.smallQADModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.smallQADRepoId,
                revision: LFM2Resources.smallQADRevision,
                patterns: LFM2Resources.qadSnapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.smallQADRepoId,
            upstreamRevision: LFM2Resources.smallQADRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.smallQADRepoId,
                sourceRevision: LFM2Resources.smallQADRevision,
                licenseURL: "https://huggingface.co/\(LFM2Resources.smallQADRepoId)/blob/\(LFM2Resources.smallQADRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.smallQADEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.denseModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.denseUpstreamRepoId,
                revision: LFM2Resources.denseUpstreamRevision,
                patterns: LFM2Resources.denseSnapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.denseUpstreamRepoId,
            upstreamRevision: LFM2Resources.denseUpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.denseUpstreamRepoId,
                sourceRevision: LFM2Resources.denseUpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-2.6B-MLX/blob/\(LFM2Resources.denseUpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_601_108_788,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.denseQADModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.denseQADRepoId,
                revision: LFM2Resources.denseQADRevision,
                patterns: LFM2Resources.qadSnapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.denseQADRepoId,
            upstreamRevision: LFM2Resources.denseQADRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.denseQADRepoId,
                sourceRevision: LFM2Resources.denseQADRevision,
                licenseURL: "https://huggingface.co/\(LFM2Resources.denseQADRepoId)/blob/\(LFM2Resources.denseQADRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.denseQADEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.denseBF16ModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.denseBF16UpstreamRepoId,
                revision: LFM2Resources.denseBF16UpstreamRevision,
                patterns: LFM2Resources.snapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.denseBF16UpstreamRepoId,
            upstreamRevision: LFM2Resources.denseBF16UpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.denseBF16UpstreamRepoId,
                sourceRevision: LFM2Resources.denseBF16UpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/\(LFM2Resources.denseBF16UpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.denseBF16EstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            companionModelIDs: [LFM2Resources.denseDSparkModelId],
            apiProfile: .lfm2()
        ),
        ManagedModelSpec(
            id: LFM2Resources.visionModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.visionUpstreamRepoId,
                revision: LFM2Resources.visionUpstreamRevision,
                patterns: LFM2Resources.visionSnapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.visionUpstreamRepoId,
            upstreamRevision: LFM2Resources.visionUpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.visionUpstreamRepoId,
                sourceRevision: LFM2Resources.visionUpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-VL-3B-MLX-8bit/blob/\(LFM2Resources.visionUpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.visionEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2(inputModalities: [.text, .image])
        ),
    ]
}
