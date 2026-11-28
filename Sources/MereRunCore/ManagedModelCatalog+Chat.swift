import Foundation

extension ManagedModelCatalog {
    static let chatSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: Inkling2Resources.modelID, category: .visionChat, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: Inkling2Resources.repository, revision: Inkling2Resources.revision, patterns: Inkling2Resources.snapshotPatterns),
            upstreamRepoId: Inkling2Resources.repository, upstreamRevision: Inkling2Resources.revision,
            validationKind: .inkling2, runtimeAutoDownloadAllowed: false, estimatedDownloadBytes: 393_500_000_000,
            defaultCLICommands: ["text chat", "api serve"], apiProfile: .inkling2(),
            manifestTemplate: .init(
                engine: .inkling2,
                family: .inkling,
                tier: .small,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 16, scheme: "mixed-nvfp4-mxfp8"),
                supports: [.chat, .codeGeneration, .visionChat]
            ),
            capability: .init(
                "Inkling 2 Small 594B-A17B",
                "Native Swift/MLX reference inference with selected experts streamed from the original checkpoint; requires about 394 GB of disk space.",
                minimum: 32,
                recommended: 64
            )
        ),
        ManagedModelSpec(
            id: Inkling2Resources.miniModelID, category: .visionChat, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Inkling2Resources.miniRepository, revision: Inkling2Resources.miniRevision,
                patterns: Inkling2Resources.miniSpeedSnapshotPatterns
            ),
            upstreamRepoId: Inkling2Resources.miniRepository, upstreamRevision: Inkling2Resources.miniRevision,
            validationKind: .inkling2, runtimeAutoDownloadAllowed: false, estimatedDownloadBytes: 18_430_000_000,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"], apiProfile: .inkling2(),
            manifestTemplate: .init(
                engine: .inkling2,
                family: .inkling,
                tier: .tiny,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 16, scheme: "mixed-nvfp4-mxfp8-with-affine-q8-attention-and-head"),
                supports: [.chat, .codeGeneration, .visionChat]
            ),
            capability: .init(
                "Inkling 2 Mini 26.4B Speed",
                "Default Mini checkpoint with packed experts, affine 8-bit attention, and a bundled DSpark draft head.",
                minimum: 32,
                recommended: 32
            )
        ),
        ManagedModelSpec(
            id: Inkling2Resources.miniOfficialModelID, category: .visionChat, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Inkling2Resources.miniOfficialRepository, revision: Inkling2Resources.miniOfficialRevision,
                patterns: Inkling2Resources.miniSnapshotPatterns
            ),
            upstreamRepoId: Inkling2Resources.miniOfficialRepository, upstreamRevision: Inkling2Resources.miniOfficialRevision,
            validationKind: .inkling2, runtimeAutoDownloadAllowed: false, estimatedDownloadBytes: 20_200_000_000,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"], apiProfile: .inkling2(),
            manifestTemplate: .init(
                engine: .inkling2,
                family: .inkling,
                tier: .tiny,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 16, scheme: "mixed-nvfp4-mxfp8"),
                supports: [.chat, .codeGeneration, .visionChat]
            ),
            capability: .init(
                "Inkling 2 Mini 26.4B Official",
                "Uses the original Thinking Machines Lab checkpoint with a BF16 output head.",
                minimum: 32,
                recommended: 32
            )
        ),
        ManagedModelSpec(
            id: Inkling2Resources.miniQ8HeadModelID, category: .visionChat, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Inkling2Resources.miniQ8HeadRepository, revision: Inkling2Resources.miniQ8HeadRevision,
                patterns: Inkling2Resources.miniQ8HeadSnapshotPatterns
            ),
            upstreamRepoId: Inkling2Resources.miniQ8HeadRepository, upstreamRevision: Inkling2Resources.miniQ8HeadRevision,
            validationKind: .inkling2, runtimeAutoDownloadAllowed: false, estimatedDownloadBytes: 19_809_895_923,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"], apiProfile: .inkling2(),
            manifestTemplate: .init(
                engine: .inkling2,
                family: .inkling,
                tier: .tiny,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 16, scheme: "mixed-nvfp4-mxfp8-with-affine-q8-head"),
                supports: [.chat, .codeGeneration, .visionChat]
            ),
            capability: .init(
                "Inkling 2 Mini 26.4B Q8 Head",
                "Shares a prepacked 8-bit output head between native target verification and DSpark. Includes text and vision weights in one checkpoint.",
                minimum: 32,
                recommended: 32
            )
        ),
        ManagedModelSpec(
            id: FlintResources.modelID, category: .textChat, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: FlintResources.repository, revision: FlintResources.revision, patterns: FlintResources.snapshotPatterns),
            upstreamRepoId: FlintResources.repository, upstreamRevision: FlintResources.revision,
            validationKind: .flint, runtimeAutoDownloadAllowed: false, estimatedDownloadBytes: 22_580_000_000,
            defaultCLICommands: ["text chat", "api serve"], apiProfile: .flint(),
            manifestTemplate: .init(
                engine: .flint,
                family: .flint,
                tier: .large,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine-int4-int8-embedding"),
                supports: [.chat, .codeGeneration],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Recursive Flint INT4",
                "Runs the pinned group-64 INT4 export through native Swift/MLX with BF16 attention caches.",
                minimum: 48,
                recommended: 64
            )
        ),
        ManagedModelSpec(
            id: FlintResources.bf16ModelID, category: .textChat, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: FlintResources.bf16Repository, revision: FlintResources.bf16Revision, patterns: FlintResources.snapshotPatterns),
            upstreamRepoId: FlintResources.bf16Repository, upstreamRevision: FlintResources.bf16Revision,
            validationKind: .flint, runtimeAutoDownloadAllowed: false, estimatedDownloadBytes: 79_590_000_000,
            defaultCLICommands: ["text chat", "api serve"], apiProfile: .flint(),
            manifestTemplate: .init(
                engine: .flint,
                family: .flint,
                tier: .large,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .codeGeneration],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Recursive Flint BF16",
                "Loads the original BF16 checkpoint through native Swift/MLX.",
                minimum: 128,
                recommended: 128
            )
        ),
        ManagedModelSpec(
            id: "text-chat-mebot",
            category: .textChat,
            installShape: .directoryRoot,
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["api serve"],
            apiProfile: .klein(),
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .unknown,
                supports: [.chat, .loraInference],
                components: .mebotText
            ),
            capability: .init(
                "Local personal chat",
                "Adds a compact local chat model for API serving and personal-agent style workflows.",
                minimum: 8,
                recommended: 16
            )
        ),
        ManagedModelSpec(
            id: "text-chat-psi-agent",
            category: .textChat,
            installShape: .directoryRoot,
            validationKind: .hfTextChat,
            runtimeAutoDownloadAllowed: false,
            defaultCLICommands: ["text chat"],
            manifestTemplate: .init(
                engine: .psiChat,
                family: .psi,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Agent chat, large",
                "Adds the larger Psi agent chat model for local reasoning-heavy assistant workflows.",
                minimum: 64,
                recommended: 96
            )
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
            apiProfile: .gemma4(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 dense bf16 alias",
                "Resolves to dense bf16 Gemma 4 chat models for native Swift text chat; use the TurboQuant tier on 32 GB Macs.",
                minimum: 48,
                recommended: 64
            )
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
            apiProfile: .diffusionGemma(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .max,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "optiq-mixed-affine"),
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "DiffusionGemma 26B-A4B OptiQ",
                "Runs the mixed 4-bit and 8-bit DiffusionGemma checkpoint through native Swift/MLX parallel block denoising.",
                minimum: 48,
                recommended: 64
            )
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
            apiProfile: .gemma4(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .turbo,
                variant: .standard,
                precision: .int4,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 NVFP4 MoE",
                "Installs the MLX NVFP4 Gemma 4 26B-A4B-it MoE snapshot for the native Swift 32 GB tier.",
                minimum: 24,
                recommended: 32,
                setup: true
            )
        ),
        ManagedModelSpec(
            id: Gemma4Resources.a4bBF16ModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Gemma4Resources.a4bBF16UpstreamModelId,
                revision: Gemma4Resources.a4bBF16UpstreamRevision,
                patterns: Gemma4Resources.snapshotPatterns
            ),
            upstreamRepoId: Gemma4Resources.a4bBF16UpstreamModelId,
            upstreamRevision: Gemma4Resources.a4bBF16UpstreamRevision,
            validationKind: .gemma4,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 51_644_318_400,
            defaultCLICommands: ["text chat", "text train-lora", "api serve"],
            apiProfile: .gemma4(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .large,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 26B-A4B BF16",
                "Runs the pinned MLX BF16 Gemma 4 26B-A4B instruction checkpoint through the native text runtime.",
                minimum: 64,
                recommended: 96
            )
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
            apiProfile: .gemma4(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 12B chat",
                "Runs the dense Gemma 4 12B instruction model through the native Swift text runtime.",
                minimum: 24,
                recommended: 32
            )
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
            apiProfile: .gemma4(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 12B 4-bit chat",
                "Installs the MLX 4-bit Gemma 4 12B instruction snapshot for faster native Swift text chat.",
                minimum: 16,
                recommended: 24,
                setup: true
            )
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
            apiProfile: .gemma4(inputModalities: [.text, .image]),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 12B vision chat",
                "Runs the dense Gemma 4 12B unified model for single-image chat through the native Swift runtime.",
                minimum: 32,
                recommended: 48
            )
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
            apiProfile: .gemma4(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 chat nano",
                "Runs the smaller Gemma 4 native Swift chat model for lower-memory machines.",
                minimum: 16,
                recommended: 24,
                setup: true
            )
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
            apiProfile: .gemma4(),
            manifestTemplate: .init(
                engine: .gemma4,
                family: .gemma,
                tier: .max,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Gemma 4 chat max",
                "Runs the larger Gemma 4 native Swift chat model for higher-quality local chat.",
                minimum: 48,
                recommended: 64
            )
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
            apiProfile: .laguna(),
            manifestTemplate: .init(
                engine: .laguna,
                family: .laguna,
                tier: .max,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 16, scheme: "mlx-nvfp4"),
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Laguna S 2.1 agentic chat",
                "Runs Poolside's 118B-A8B Laguna S 2.1 NVFP4 model with verified DFlash speculative decoding through the native Swift runtime.",
                minimum: 96,
                recommended: 128
            )
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
            apiProfile: .laguna(),
            manifestTemplate: .init(
                engine: .laguna,
                family: .laguna,
                tier: .small,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 16, scheme: "mlx-nvfp4"),
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Laguna XS 2.1 agentic chat",
                "Runs Poolside's released 33B-A3B Laguna XS 2.1 through its NVFP4 MLX serialization and the native Swift runtime.",
                minimum: 36,
                recommended: 48
            )
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
            defaultCLICommands: ["text chat", "text train-lora"],
            manifestTemplate: .init(
                engine: .inkling,
                family: .inkling,
                tier: .small,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "affine-routed-experts"),
                supports: [.chat, .codeGeneration]
            ),
            capability: .init(
                "Inkling-Small 276B-A12B",
                "Runs Inkling-Small through native Swift/MLX with affine 2-bit/group-128 routed experts and BF16 non-routed weights.",
                minimum: 128,
                recommended: 128
            )
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
            apiProfile: .museGlimmer(),
            manifestTemplate: .init(
                engine: .museGlimmer,
                family: .muse,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Muse Glimmer 30B vision agent",
                "Runs Sawfwair's pinned selective MLX Q4 conversion of Meta Muse Glimmer with its managed DFlash2 speculative companion through the native Swift/MLX stack.",
                minimum: 32,
                recommended: 64
            )
        ),
        ManagedModelSpec(
            id: NemotronBoltResources.modelID,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: NemotronBoltResources.upstreamRepoID,
                revision: NemotronBoltResources.upstreamRevision,
                patterns: NemotronBoltResources.snapshotPatterns
            ),
            upstreamRepoId: NemotronBoltResources.upstreamRepoID,
            upstreamRevision: NemotronBoltResources.upstreamRevision,
            validationKind: .nemotronBolt,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: NemotronBoltResources.estimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .nemotronBolt(),
            manifestTemplate: .init(
                engine: .nemotronH,
                family: .nemotron,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 256, scheme: "ggml-iq2-xxs-q8-0-nvfp4-mixed"),
                supports: [.chat, .codeGeneration, .visionChat, .visionOCR],
                components: .rootText
            ),
            capability: .init(
                "Nemotron 3.5 Bolt 74B-A9B 24GB",
                "Text and image chat from NVIDIA's original packed checkpoint. Full-checkpoint quality and peak memory remain unqualified.",
                minimum: 32,
                recommended: 48
            )
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
            apiProfile: .nemotronH(),
            manifestTemplate: .init(
                engine: .nemotronH,
                family: .nemotron,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 16, scheme: "modelopt-nvfp4-mlx"),
                supports: [.chat, .codeGeneration],
                components: .rootText,
                upstream: .explicit("\(NemotronHResources.upstreamRepoID)@\(NemotronHResources.upstreamRevision)")
            ),
            capability: .init(
                "Nemotron 3.5 Lightning 30B-A3B",
                "Runs NVIDIA's hybrid Nemotron-H NVFP4 checkpoint with its verified DSpark speculative companion through native Swift/MLX.",
                minimum: 32,
                recommended: 64
            )
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
            apiProfile: .nemotronOmni(),
            manifestTemplate: .init(
                engine: .nemotronOmni,
                family: .nemotron,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .codeGeneration, .visionChat, .visionOCR, .audioUnderstanding, .videoUnderstanding, .documentUnderstanding],
                components: .rootText
            ),
            capability: .init(
                "Nemotron 3 Nano Omni 30B-A3B BF16",
                "Runs NVIDIA's pinned BF16 text, image, video, and audio understanding checkpoint through native Swift/MLX media towers and Nemotron-H language runtime.",
                minimum: 112,
                recommended: 128
            )
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
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .q36(contextWindow: Q35Resources.defaultContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .nano,
                variant: .base,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-optiq-mixed-affine"),
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.6 A3B chat nano",
                "Runs the Qwen3.6 35B-A3B OptiQ 4-bit MLX/MTP snapshot through the native Qwen-family runtime.",
                minimum: 24,
                recommended: 32,
                setup: true
            )
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
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .codeGeneration, .visionChat, .loraInference],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.8 27B vision chat",
                "Runs Qwen's official dense 27B BF16 vision-language checkpoint through the native Qwen-family runtime.",
                minimum: 64,
                recommended: 96
            )
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
                    destinationPath: Q35Resources.q38DFlash2ComponentPath,
                    hubFallback: HubFallbackConfig(
                        repoId: Q35Resources.q38DFlash2UpstreamRepoId,
                        revision: Q35Resources.q38DFlash2UpstreamRevision,
                        patterns: Q35Resources.q38DFlash2SnapshotPatterns
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
                + Q35Resources.q38VisionComponentEstimatedDownloadBytes + Q35Resources.q38DFlash2EstimatedDownloadBytes,
            defaultCLICommands: [
                "text chat",
                "api serve",
                "model benchmark chat",
                "model benchmark code",
                "model benchmark vlm",
            ],
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .codeGeneration, .visionChat, .loraInference],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.8 27B 4-bit vision chat",
                "Runs LM Studio's pinned MLX 4-bit conversion with Qwen's official MTP shard available for explicit speculative decode.",
                minimum: 32,
                recommended: 48
            )
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
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "mlx-mixed-q2-q4-bf16-affine"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.8 Flash Next mixed Q2/Q4 vision chat",
                "Runs Sawfwair's pinned 128 GB Mac profile with routed experts at Q2, core matrices at Q4, and sensitive paths in BF16.",
                minimum: 112,
                recommended: 128
            )
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
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int3,
                quantization: .init(bits: 3, groupSize: 64, scheme: "mlx-mixed-q3-q4-activation-refit-affine"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.8 Flash Next activation-weighted Q3 vision chat",
                "Runs Sawfwair's pinned Q3 expert profile with Q4 core matrices and a measured 62.1 GB peak on the 128 GiB qualification host.",
                minimum: 96,
                recommended: 128
            )
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
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int3,
                quantization: .init(bits: 3, groupSize: 64, scheme: "mlx-mixed-q3-q4-activation-refit-affine-native-ple"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.8 Flash Next activation-weighted Q3 native PLE",
                "Runs the complete pinned Q3 checkpoint and automatically stages its disk-backed PLE shards on the internal SSD when the model store is external.",
                minimum: 96,
                recommended: 128
            )
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
            apiProfile: .q38(contextWindow: Q35Resources.q38TwentySevenBContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.8 Flash Next Q4 vision chat",
                "Runs Sawfwair's pinned all-Q4-compatible profile for higher quality on Macs with more than 128 GB unified memory.",
                minimum: 160,
                recommended: 192
            )
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
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int1,
                quantization: .init(bits: 1, groupSize: 128, scheme: "prism-packed-affine-binary"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Bonsai 27B 1-bit vision chat",
                "Runs Prism ML's dense Qwen3.6 27B reasoning and vision model from packed 1-bit MLX weights.",
                minimum: 12,
                recommended: 16
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
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "prism-packed-affine-ternary"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ternary Bonsai 27B 2-bit vision chat",
                "Runs Prism ML's dense Qwen3.6 27B reasoning and vision model from packed ternary 2-bit MLX weights.",
                minimum: 16,
                recommended: 24
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai2TwoBModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai2TwoBModelId)?.hubFallbackConfig,
            upstreamRepoId: "prism-ml/Ternary-Bonsai-2-2B-mlx-2bit",
            upstreamRevision: "1c6cbae21f8c38830ce337972b46fdae7b18eaaa",
            validationKind: .q35,
            estimatedDownloadBytes: 1_213_254_984,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(contextWindow: Q35Resources.bonsai2ContextLength, fixedReasoning: true),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "prism-hadamard-affine-ternary"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ternary Bonsai 2 2B 2-bit vision chat",
                "Runs Prism ML's Qwen3.5 2B vision model from Hadamard-transformed ternary 2-bit MLX weights.",
                minimum: 8,
                recommended: 12
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai2FourBModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai2FourBModelId)?.hubFallbackConfig,
            upstreamRepoId: "prism-ml/Ternary-Bonsai-2-4B-mlx-2bit",
            upstreamRevision: "dff2caaa23fa8706164c44c10f1b58feb405f76a",
            validationKind: .q35,
            estimatedDownloadBytes: 1_884_421_442,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(contextWindow: Q35Resources.bonsai2ContextLength, fixedReasoning: true),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "prism-hadamard-affine-ternary"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ternary Bonsai 2 4B 2-bit vision chat",
                "Runs Prism ML's Qwen3.5 4B vision model from Hadamard-transformed ternary 2-bit MLX weights.",
                minimum: 8,
                recommended: 16
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai2NineBModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai2NineBModelId)?.hubFallbackConfig,
            upstreamRepoId: "prism-ml/Ternary-Bonsai-2-9B-mlx-2bit",
            upstreamRevision: "f4e28a59531731c41b6dfac88cf271bd58f0a287",
            validationKind: .q35,
            estimatedDownloadBytes: 3_475_209_704,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(contextWindow: Q35Resources.bonsai2ContextLength, fixedReasoning: true),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "prism-hadamard-affine-ternary"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ternary Bonsai 2 9B 2-bit vision chat",
                "Runs Prism ML's Qwen3.5 9B vision model from Hadamard-transformed ternary 2-bit MLX weights.",
                minimum: 12,
                recommended: 16
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai2ModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai2ModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.bonsai2UpstreamRepoId,
            upstreamRevision: Q35Resources.bonsai2UpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.bonsai2EstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q36(
                contextWindow: Q35Resources.bonsai2ContextLength,
                fixedReasoning: true
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "prism-hadamard-affine-ternary"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ternary Bonsai 2 27B 2-bit vision chat",
                "Runs Prism ML's dense Qwen3.8 27B reasoning and vision model from Hadamard-transformed ternary 2-bit MLX weights.",
                minimum: 16,
                recommended: 24
            )
        ),
        ManagedModelSpec(
            id: Q35Resources.bonsai25ModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.bonsai25ModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.bonsai25UpstreamRepoId,
            upstreamRevision: Q35Resources.bonsai25UpstreamRevision,
            validationKind: .q35,
            estimatedDownloadBytes: Q35Resources.bonsai25EstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve", "model benchmark chat"],
            apiProfile: .q38(contextWindow: Q35Resources.bonsai2ContextLength),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "prism-hadamard-affine-ternary"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ternary Bonsai 2.5 27B vision chat",
                "Runs Prism ML's dense Qwen3.8 27B vision and tool-calling model from 2-bit Hadamard MLX weights.",
                minimum: 16,
                recommended: 24
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
            defaultCLICommands: ["text chat", "api serve", "agent start"],
            apiProfile: .q36(
                contextWindow: Q35Resources.defaultContextLength,
                fixedReasoning: true
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .nano,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-optiq-mixed-affine"),
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Ornith 1.0 9B OptiQ",
                "Runs DeepReinforce's Ornith 1.0 9B agentic coding model through the native Qwen-family MLX runtime.",
                minimum: 16,
                recommended: 24
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
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .large,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ornith 1.5 35B-A3B 4-bit vision agent",
                "Runs the official 4-bit Ornith target with its BF16 vision tower and MTP head in one bundle.",
                minimum: 32,
                recommended: 48
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
            defaultCLICommands: ["text chat", "api serve", "agent start", "model benchmark code"],
            companionModelIDs: [Q35Resources.ornith35BMTPModelId],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .large,
                variant: .standard,
                precision: .int6,
                quantization: .init(bits: 6, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Ornith 1.5 35B-A3B MLX 6-bit",
                "Balanced official Ornith 35B MLX tier with the shared BF16 MTP companion for verified speculative decode.",
                minimum: 48,
                recommended: 64
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
            defaultCLICommands: ["text chat", "api serve", "agent start", "model benchmark code"],
            companionModelIDs: [Q35Resources.ornith35BMTPModelId],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .large,
                variant: .standard,
                precision: .int8,
                quantization: .init(bits: 8, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Ornith 1.5 35B-A3B MLX 8-bit",
                "Highest-precision official quantized Ornith 35B MLX tier with the shared BF16 MTP companion.",
                minimum: 64,
                recommended: 96
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
            defaultCLICommands: ["text chat", "api serve", "agent start", "model benchmark code"],
            companionModelIDs: [Q35Resources.ornith35BMTPModelId],
            apiProfile: .q36(
                contextWindow: Q35Resources.ornith35BMLXContextLength,
                fixedReasoning: true
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .large,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .codeGeneration],
                components: .rootText
            ),
            capability: .init(
                "Ornith 1.5 35B-A3B MLX BF16",
                "Runs Ornith's official unquantized BF16 coding agent with the shared BF16 MTP companion.",
                minimum: 96,
                recommended: 128
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
            ),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .large,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .codeGeneration, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "Ornith 1.5 35B-A3B vision agent",
                "Runs Ornith's full BF16 agentic coding and vision checkpoint through the native Qwen-family runtime.",
                minimum: 96,
                recommended: 128
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
            apiProfile: .textCode(),
            manifestTemplate: .init(
                engine: .qwen3Coder,
                family: .code,
                tier: .nano,
                variant: .standard,
                precision: .int4,
                supports: [.chat, .codeGeneration]
            ),
            capability: .init(
                "Qwen3.5 9B agent",
                "Runs the Qwen3.5 9B Q4 GGUF setup agent for lower-memory Macs.",
                minimum: 16,
                recommended: 16,
                setup: true
            )
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
            ),
            manifestTemplate: .init(
                engine: .northMiniCode,
                family: .code,
                tier: .small,
                variant: .standard,
                precision: .int4,
                supports: [.chat, .codeGeneration]
            ),
            capability: .init(
                "North Mini Code Q4",
                "Runs Cohere's compact 30B-A3B coding model through the native llama.cpp/GGUF code runtime.",
                minimum: 24,
                recommended: 32
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
            ),
            manifestTemplate: .init(
                engine: .qwen3Coder,
                family: .code,
                tier: .small,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "gguf-q4-k-m"),
                supports: [.chat, .codeGeneration]
            ),
            capability: .init(
                "Ornith 1.0 35B Q4",
                "Runs DeepReinforce's larger Ornith coding-agent GGUF through the native llama.cpp/GGUF code runtime.",
                minimum: 32,
                recommended: 64
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
            apiProfile: .textCode(),
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .nano,
                variant: .base,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "gguf-q4-k-m"),
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "Qwen3.6 A3B chat nano (GGUF)",
                "Runs Qwen3.6 35B-A3B GGUF through llama.cpp; the default chat model on Linux CUDA hosts.",
                minimum: 24,
                recommended: 32
            )
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
            defaultCLICommands: ["api serve", "agent start"],
            apiProfile: .deepseekV4Flash(),
            manifestTemplate: .init(
                engine: .deepseekV4Flash,
                family: .deepseek,
                tier: .max,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 2, groupSize: 32, scheme: "iq2-xxs-imatrix"),
                supports: [.chat]
            ),
            capability: .init(
                "DeepSeek V4 Flash IQ2 imatrix",
                "Premier 284B MoE agent tier served by the bundled ds4-server engine (imatrix-tuned q2 GGUF, ~81 GB) for 96 GB+ Apple Silicon Macs.",
                minimum: 96,
                recommended: 128,
                setup: true
            )
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
            apiProfile: .lfm2(),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .int8,
                quantization: .init(bits: 8, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5 A1B 8-bit",
                "Runs the LiquidAI LFM2.5 8B-A1B MLX 8-bit snapshot through the native Swift LFM2 runtime.",
                minimum: 16,
                recommended: 24,
                setup: true
            )
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
            apiProfile: .lfm2(),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5 A1B BF16 + DSpark",
                "Runs the exact LiquidAI LFM2.5 8B-A1B BF16 target with its native DSpark accelerator.",
                minimum: 32,
                recommended: 48,
                setup: true
            )
        ),
        ManagedModelSpec(
            id: LFM2Resources.a2b24BBF16ModelId,
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.a2b24BUpstreamRepoId,
                revision: LFM2Resources.a2b24BUpstreamRevision,
                patterns: LFM2Resources.snapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.a2b24BUpstreamRepoId,
            upstreamRevision: LFM2Resources.a2b24BUpstreamRevision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.a2b24BUpstreamRepoId,
                sourceRevision: LFM2Resources.a2b24BUpstreamRevision,
                licenseURL: "https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/\(LFM2Resources.a2b24BUpstreamRevision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.a2b24BEstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            apiProfile: .lfm2(contextWindow: LFM2Resources.a2b24BContextLength, thinkingLevels: [.high]),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .medium,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5 24B-A2B BF16",
                "Runs LiquidAI's reasoning and tool-use MoE directly from original BF16 safetensors.",
                minimum: 64,
                recommended: 96
            )
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
            apiProfile: .lfm2(),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5 1.2B BF16",
                "Runs the compact LiquidAI LFM2.5 1.2B Instruct checkpoint with its optional native DSpark accelerator.",
                minimum: 8,
                recommended: 12,
                setup: true
            )
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
            apiProfile: .lfm2(),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 32, scheme: "liquid-qad-q4_0-mlx-affine-g32"),
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5 1.2B QAD 4-bit",
                "Runs the compact QAD-trained LFM2.5 1.2B checkpoint in native MLX; choose it when memory matters more than BF16 or DSpark throughput.",
                minimum: 4,
                recommended: 8,
                setup: true
            )
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
            apiProfile: .lfm2(),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat],
                components: .denseLFM2Text
            ),
            capability: .init(
                "LFM2.5 2.6B 4-bit",
                "Runs the LiquidAI LFM2.5 2.6B dense MLX 4-bit snapshot through the native Swift LFM2 runtime.",
                minimum: 8,
                recommended: 12,
                setup: true
            )
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
            apiProfile: .lfm2(),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 32, scheme: "liquid-qad-q4_0-mlx-affine-g32"),
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5 2.6B QAD 4-bit",
                "Runs the QAD-trained LFM2.5 2.6B checkpoint in native MLX for quantized quality recovery; the standard MLX 4-bit model remains the faster compact lane.",
                minimum: 8,
                recommended: 12
            )
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
            apiProfile: .lfm2(),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: [.chat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5 2.6B BF16 + DSpark",
                "Runs the exact LiquidAI LFM2.5 2.6B BF16 target with its native DSpark accelerator.",
                minimum: 16,
                recommended: 24,
                setup: true
            )
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
            apiProfile: .lfm2(inputModalities: [.text, .image]),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .int8,
                quantization: .init(bits: 8, groupSize: 64, scheme: "mlx-affine"),
                supports: [.chat, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5-VL 3B 8-bit",
                "Runs LiquidAI's compact LFM2.5-VL vision-language model through the native Swift LFM2 runtime.",
                minimum: 8,
                recommended: 16,
                setup: true
            )
        ),
        ManagedModelSpec(
            id: LFM2Resources.visionBF16ModelId,
            category: .visionChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: LFM2Resources.visionBF16RepoId,
                revision: LFM2Resources.visionBF16Revision,
                patterns: LFM2Resources.visionSnapshotPatterns
            ),
            upstreamRepoId: LFM2Resources.visionBF16RepoId,
            upstreamRevision: LFM2Resources.visionBF16Revision,
            usageRestriction: usageRestriction(
                summary: "LFM uses a custom open license; commercial use by entities with at least USD 10M annual revenue is not licensed under its community terms.",
                license: "LFM Open License v1.0",
                sourceRepoId: LFM2Resources.visionBF16RepoId,
                sourceRevision: LFM2Resources.visionBF16Revision,
                licenseURL: "https://huggingface.co/\(LFM2Resources.visionBF16RepoId)/blob/\(LFM2Resources.visionBF16Revision)/LICENSE"
            ),
            validationKind: .lfm2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: LFM2Resources.visionBF16EstimatedDownloadBytes,
            defaultCLICommands: ["text chat", "api serve"],
            companionModelIDs: [LFM2Resources.visionDSparkModelId],
            apiProfile: .lfm2(inputModalities: [.text, .image]),
            manifestTemplate: .init(
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .visionChat],
                components: .rootText
            ),
            capability: .init(
                "LFM2.5-VL 3B BF16",
                "Runs LiquidAI's BF16 vision-language model with its native DSpark companion.",
                minimum: 16,
                recommended: 24,
                setup: true
            )
        ),
    ]
}
