import Foundation

extension ManagedModelCatalog {
    static let imageSpecs: [ManagedModelSpec] = [
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
            estimatedDownloadBytes: 4_627_979_498,
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
            estimatedDownloadBytes: 15_980_131_745,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            // FLUX.2 [klein] 9B — 9B flow + 8B Qwen3 text embedder, step-distilled to 4 steps,
            // native single/multi-reference editing. bf16 diffusers format (same loader path as
            // klein-max). Pulled from the mlx-community mirror, which is UNGATED (gated:false) —
            // no HF token required — unlike black-forest-labs/FLUX.2-klein-9B which is gated.
            // diffusersImageSnapshotPatterns skips the redundant single-file root checkpoint.
            id: "image-klein-9b",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/FLUX.2-klein-9B",
                revision: klein9BUpstreamRevision,
                patterns: diffusersImageSnapshotPatterns
            ),
            upstreamRepoId: "mlx-community/FLUX.2-klein-9B",
            upstreamRevision: klein9BUpstreamRevision,
            usageRestriction: flux9BUsageRestriction(
                sourceRepoId: "mlx-community/FLUX.2-klein-9B",
                sourceRevision: klein9BUpstreamRevision
            ),
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 34_722_771_551,
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
            estimatedDownloadBytes: 15_980_131_711,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            // FLUX.2 [klein] Base 9B — undistilled bf16 9B transformer for LoRA/fine-tuning.
            // BFL publishes the base transformer separately from reusable 9B text/VAE
            // components in common training recipes, so mount the shared components from the
            // ungated mlx-community mirror while keeping the gated Base 9B transformer source
            // explicit.
            id: "image-klein-base-9b",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "black-forest-labs/FLUX.2-klein-base-9B",
                revision: kleinBase9BUpstreamRevision,
                patterns: kleinBase9BTransformerSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "text_encoder",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["text_encoder/*"]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["tokenizer/*"]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "vae",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["vae/*"]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "scheduler",
                    hubFallback: HubFallbackConfig(
                        repoId: "mlx-community/FLUX.2-klein-9B",
                        revision: klein9BUpstreamRevision,
                        patterns: ["scheduler/*"]
                    )
                ),
            ],
            upstreamRepoId: "black-forest-labs/FLUX.2-klein-base-9B",
            upstreamRevision: kleinBase9BUpstreamRevision,
            usageRestriction: flux9BUsageRestriction(
                sourceRepoId: "black-forest-labs/FLUX.2-klein-base-9B",
                sourceRevision: kleinBase9BUpstreamRevision,
                component: "Base 9B transformer",
                additionalTerms: [
                    ManagedModelUsageTerm(
                        component: "shared 9B text encoder, tokenizer, VAE, and scheduler",
                        license: "FLUX Non-Commercial License v2.1",
                        summary: "The shared FLUX.2 Klein 9B components are licensed only for non-commercial, non-production use.",
                        sourceRepoId: "mlx-community/FLUX.2-klein-9B",
                        sourceRevision: klein9BUpstreamRevision,
                        licenseURL: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md"
                    ),
                ]
            ),
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 48 * 1_073_741_824,
            defaultCLICommands: ["image generate", "image train-lora"]
        ),
        ManagedModelSpec(
            id: "image-klein-shared",
            category: .image,
            installShape: .directoryRoot,
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false
        ),
        ManagedModelSpec(
            // Keep the gated, LoRA-compatible FLUX.2-dev image stack authoritative while
            // replacing only its identical Mistral Small 3.2 text encoder with a pinned
            // Apple-Silicon 4-bit conversion. This avoids about 35 GB of resident and
            // downloaded weights without changing the transformer targeted by Dev LoRAs.
            id: "image-flux2-dev",
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "black-forest-labs/FLUX.2-dev",
                revision: flux2DevUpstreamRevision,
                patterns: flux2DevSnapshotPatterns
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "text_encoder",
                    hubFallback: HubFallbackConfig(
                        repoId: flux2DevTextEncoderRepoId,
                        revision: flux2DevTextEncoderRevision,
                        patterns: [
                            "README.md",
                            "config.json",
                            "model-*.safetensors",
                            "model.safetensors.index.json",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "black-forest-labs/FLUX.2-dev",
            upstreamRevision: flux2DevUpstreamRevision,
            usageRestriction: flux2DevUsageRestriction,
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 78_100_000_000,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: Flux1Resources.modelID,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Flux1Resources.upstreamRepoID,
                revision: Flux1Resources.upstreamRevision,
                patterns: Flux1Resources.snapshotPatterns
            ),
            upstreamRepoId: Flux1Resources.upstreamRepoID,
            upstreamRevision: Flux1Resources.upstreamRevision,
            usageRestriction: flux1DevUsageRestriction,
            validationKind: .flux1,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Flux1Resources.estimatedDownloadBytes,
            defaultCLICommands: ["image generate"]
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
                revision: zImageNanoUpstreamRevision,
                patterns: zImageNanoSnapshotPatterns
            ),
            upstreamRepoId: zImageNanoUpstreamRepoId,
            upstreamRevision: zImageNanoUpstreamRevision,
            validationKind: .zimageTurbo,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 20_538_488_559,
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
            estimatedDownloadBytes: 32_848_305_533,
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
            estimatedDownloadBytes: 5_907_438_792,
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
            estimatedDownloadBytes: 35_231_213_079,
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
            estimatedDownloadBytes: 35_231_213_079,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: SenseNovaU15Resources.modelID,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: SenseNovaU15Resources.repository,
                revision: SenseNovaU15Resources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "README_CN.md",
                    "added_tokens.json",
                    "config.json",
                    "merges.txt",
                    "model.safetensors.index.json",
                    "model-*.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ]
            ),
            upstreamRepoId: SenseNovaU15Resources.repository,
            upstreamRevision: SenseNovaU15Resources.revision,
            validationKind: .senseNovaU15,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 50_227_451_138,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: Krea2RawResources.modelId,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Krea2RawResources.upstreamRepoId,
                revision: Krea2RawResources.upstreamRevision,
                patterns: Krea2RawResources.snapshotPatterns
            ),
            upstreamRepoId: Krea2RawResources.upstreamRepoId,
            upstreamRevision: Krea2RawResources.upstreamRevision,
            usageRestriction: krea2UsageRestriction(
                sourceRepoId: Krea2RawResources.upstreamRepoId,
                sourceRevision: Krea2RawResources.upstreamRevision
            ),
            validationKind: .krea2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Krea2RawResources.estimatedDownloadBytes,
            defaultCLICommands: ["image train-lora"]
        ),
        ManagedModelSpec(
            id: Krea2Resources.modelId,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Krea2Resources.upstreamRepoId,
                revision: Krea2Resources.upstreamRevision,
                patterns: Krea2Resources.snapshotPatterns
            ),
            upstreamRepoId: Krea2Resources.upstreamRepoId,
            upstreamRevision: Krea2Resources.upstreamRevision,
            usageRestriction: krea2UsageRestriction(
                sourceRepoId: Krea2Resources.upstreamRepoId,
                sourceRevision: Krea2Resources.upstreamRevision
            ),
            validationKind: .krea2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Krea2Resources.estimatedDownloadBytes,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: QwenImageEditRepository.model2511Id,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: QwenImageEditRepository.hubFallback2511Config,
            upstreamRepoId: QwenImageEditRepository.id2511,
            upstreamRevision: QwenImageEditRepository.revision2511,
            validationKind: .qwenImageEdit,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 57_720_463_453,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: QwenImageEditRepository.lightning2511Id,
            category: .image,
            installShape: .structuredRoot,
            hubFallback: QwenImageEditRepository.hubFallback2511Config,
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "lightning",
                    hubFallback: QwenImageEditRepository.lightningHubFallbackConfig
                ),
            ],
            upstreamRepoId: QwenImageEditRepository.id2511,
            upstreamRevision: QwenImageEditRepository.revision2511,
            validationKind: .qwenImageEdit,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 58_570_071_749,
            defaultCLICommands: ["image generate"]
        ),
        ManagedModelSpec(
            id: Ideogram4Resources.modelId,
            category: .image,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Ideogram4Resources.upstreamRepoId,
                revision: Ideogram4Resources.upstreamRevision,
                patterns: Ideogram4Resources.snapshotPatterns
            ),
            upstreamRepoId: Ideogram4Resources.upstreamRepoId,
            upstreamRevision: Ideogram4Resources.upstreamRevision,
            usageRestriction: usageRestriction(
                summary: "Ideogram 4 weights are licensed only for non-commercial purposes.",
                license: "Ideogram Non-Commercial Model Agreement",
                sourceRepoId: Ideogram4Resources.upstreamRepoId,
                sourceRevision: Ideogram4Resources.upstreamRevision,
                licenseURL: "https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md"
            ),
            validationKind: .ideogram4SDNQ,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Ideogram4Resources.estimatedDownloadBytes,
            defaultCLICommands: []
        ),
    ]

    private static let diffusersImageSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "model_index.json",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
        "scheduler/*",
    ]

    private static let kleinBase9BTransformerSnapshotPatterns = [
        "model_index.json",
        "transformer/*",
    ]

    private static let flux2DevSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "model_index.json",
        "tokenizer/*",
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
        "LICENSE*",
        "README.md",
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

    private static let zImageNanoUpstreamRevision = "b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b"

    private static let klein9BUpstreamRevision = "b0f0826a36667ec7c58253e50557ba76f8c0255e"

    private static let kleinBase9BUpstreamRevision = "32773329fbe7e81a90ef971740e8ba4b0364ecf3"

    private static let flux2DevUpstreamRevision = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"

    private static let flux2DevTextEncoderRepoId = "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit"

    private static let flux2DevTextEncoderRevision = "2a1d5eabfc504747bdc24178394821a1efc0edde"

    private static let kleinNanoUpstreamRepoId = "stereovoid/flux2-klein-4b-4bit"

    private static let bonsaiBinaryUpstreamRepoId = "prism-ml/bonsai-image-binary-4B-mlx-1bit"

    private static let bonsaiTernaryUpstreamRepoId = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"

    private static func flux9BUsageRestriction(
        sourceRepoId: String,
        sourceRevision: String,
        component: String = "model",
        additionalTerms: [ManagedModelUsageTerm] = []
    ) -> ManagedModelUsageRestriction {
        let summary = "FLUX.2 Klein 9B weights are licensed only for non-commercial, non-production use."
        let term = ManagedModelUsageTerm(
            component: component,
            license: "FLUX Non-Commercial License v2.1",
            summary: summary,
            sourceRepoId: sourceRepoId,
            sourceRevision: sourceRevision,
            licenseURL: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md"
        )
        return ManagedModelUsageRestriction(
            summary: summary,
            terms: [term] + additionalTerms
        )
    }


    private static let flux2DevUsageRestriction = ManagedModelUsageRestriction(
        summary: "FLUX.2-dev is limited to non-commercial, non-production use; BFL also requires filters or manual review and compliance with its Acceptable Use Policy.",
        terms: [
            ManagedModelUsageTerm(
                component: "FLUX.2-dev transformer, VAE, and tokenizer",
                license: "FLUX Non-Commercial License",
                summary: "Weights are limited to non-commercial, non-production use; commercial use requires a separate BFL license, and generated content requires filters or manual review.",
                sourceRepoId: "black-forest-labs/FLUX.2-dev",
                sourceRevision: flux2DevUpstreamRevision,
                licenseURL: "https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/\(flux2DevUpstreamRevision)/LICENSE.md"
            ),
            ManagedModelUsageTerm(
                component: "FLUX.2-dev use",
                license: "BFL Acceptable Use Policy",
                summary: "BFL's prohibited-use conditions apply independently of the non-commercial license.",
                sourceRepoId: "black-forest-labs/FLUX.2-dev",
                sourceRevision: flux2DevUpstreamRevision,
                licenseURL: "https://bfl.ai/legal/usage-policy"
            ),
        ]
    )


    private static let flux1DevUsageRestriction = usageRestriction(
        summary: "FLUX.1-dev weights and derivatives are limited to non-commercial, non-production use; content filtering or review and the license's prohibited-use conditions also apply.",
        license: "FLUX.1 dev Non-Commercial License v1.1.1",
        sourceRepoId: Flux1Resources.upstreamRepoID,
        sourceRevision: Flux1Resources.upstreamRevision,
        licenseURL: "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/\(Flux1Resources.upstreamRevision)/LICENSE.md"
    )


    private static func krea2UsageRestriction(
        sourceRepoId: String,
        sourceRevision: String
    ) -> ManagedModelUsageRestriction {
        usageRestriction(
            summary: "Krea 2 uses a custom community license; commercial use is limited to entities below USD 1M in trailing annual revenue and remains subject to its use and distribution conditions.",
            license: "Krea 2 Community License Agreement",
            sourceRepoId: sourceRepoId,
            sourceRevision: sourceRevision,
            licenseURL: "https://huggingface.co/\(sourceRepoId)/blob/\(sourceRevision)/LICENSE.pdf"
        )
    }
}
