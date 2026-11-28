import Foundation

extension ManagedModelCatalog {
    private static func nemotronImageManifest(
        precision: MereRunModelManifest.Precision,
        quantization: MereRunModelManifest.Quantization? = nil,
        decoderPath: String
    ) -> ManagedModelManifestTemplate {
        .init(
            engine: .nemotronImage, family: .nemotronImage, tier: .latest, variant: .base,
            precision: precision, quantization: quantization,
            defaults: .init(steps: 1, cfg: 1), supports: [.txt2img],
            components: .init(
                tokenizer: .local(path: "."), textEncoder: .local(path: "."),
                transformer: .local(path: "."), vae: .local(path: decoderPath), scheduler: nil
            )
        )
    }

    private static func nemotronPack(_ id: ManagedModelID, _ variant: NemotronImageCommunityPack.Variant,
                                     bytes: Int64, capability: ManagedModelCapabilityMetadata) -> ManagedModelSpec {
        ManagedModelSpec(
            id: id.rawValue, category: .image, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: variant.repository, revision: variant.revision,
                patterns: NemotronImageCommunityPack.files + ["LICENSE", "NOTICE", "README.md"]),
            upstreamRepoId: variant.repository, upstreamRevision: variant.revision,
            usageRestriction: usageRestriction(summary: "Licensed for non-commercial research and evaluation.",
                license: "NVIDIA License", sourceRepoId: NemotronImageResources.repository, sourceRevision: NemotronImageResources.revision,
                licenseURL: "https://huggingface.co/\(NemotronImageResources.repository)/blob/\(NemotronImageResources.revision)/LICENSE"),
            validationKind: .nemotronImage, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: bytes, defaultCLICommands: ["image generate"],
            manifestTemplate: nemotronImageManifest(
                precision: variant == .mixed ? .int6 : .int8,
                quantization: .init(bits: variant.rawValue, groupSize: 64, scheme: variant.scheme),
                decoderPath: "vq"
            ),
            capability: capability)
    }

    static let imageSpecs: [ManagedModelSpec] = [
        nemotronPack(.nemotronImageQ6, .mixed, bytes: 30_751_843_293, capability: .init(
            "Image, Nemotron Labs Image 30B-A3B Q6",
            "Recommended Nemotron Image checkpoint with 6-bit routed experts and 8-bit dense projections. The 36 GB fit is estimated.",
            minimum: 36,
            recommended: 36
        )),
        nemotronPack(.nemotronImageQ8, .q8, bytes: 40_317_799_235, capability: .init(
            "Image, Nemotron Labs Image 30B-A3B Q8",
            "Native raster generation with 8-bit routed experts. Memory requirements are estimates.",
            minimum: 48,
            recommended: 64
        )),
        ManagedModelSpec(
            id: ManagedModelID.nemotronImage.rawValue, category: .image, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: NemotronImageResources.repository, revision: NemotronImageResources.revision,
                patterns: NemotronImageResources.files + ["LICENSE", "NOTICE", "README.md"]),
            upstreamRepoId: NemotronImageResources.repository, upstreamRevision: NemotronImageResources.revision,
            usageRestriction: usageRestriction(summary: "Licensed for non-commercial research and evaluation.",
                license: "NVIDIA License", sourceRepoId: NemotronImageResources.repository, sourceRevision: NemotronImageResources.revision,
                licenseURL: "https://huggingface.co/\(NemotronImageResources.repository)/blob/\(NemotronImageResources.revision)/LICENSE"),
            validationKind: .nemotronImage, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 133_127_898_754, defaultCLICommands: ["image generate"],
            manifestTemplate: nemotronImageManifest(
                precision: .bf16, decoderPath: NemotronImageResources.decoderDirectory
            ),
            capability: .init(
                "Image, Nemotron Labs Image 30B-A3B",
                "Samples raster images as visual tokens with native BF16 inference. Memory requirements are estimates.",
                minimum: 80,
                recommended: 96
            )),
        ManagedModelSpec(
            id: MiniMaxH3ImageResources.modelID, category: .image, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: MiniMaxH3ImageResources.repository,
                revision: MiniMaxH3ImageResources.revision, patterns: MiniMaxH3ImageResources.patterns),
            upstreamRepoId: MiniMaxH3ImageResources.repository, upstreamRevision: MiniMaxH3ImageResources.revision,
            usageRestriction: usageRestriction(
                summary: "MiniMax-H3-Image excludes the US, EU, UK and Republic of Korea; commercial revenue above USD 1 million or hosted services require separate authorization.",
                license: "MiniMax-H3-Image Community License Agreement",
                sourceRepoId: MiniMaxH3ImageResources.repository, sourceRevision: MiniMaxH3ImageResources.revision,
                licenseURL: "https://huggingface.co/\(MiniMaxH3ImageResources.repository)/blob/\(MiniMaxH3ImageResources.revision)/LICENSE"),
            validationKind: .miniMaxH3Image, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 135_400_000_000, defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .miniMaxH3Image,
                family: .h3Image,
                tier: .latest,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 6, cfg: 1.0),
                supports: [.txt2img, .referenceEdit],
                components: .init(tokenizer: .local(path: "tokenizer"), textEncoder: .local(path: "text_encoder"), transformer: .local(path: "transformer"), vae: .local(path: "vae"))
            ),
            capability: .init(
                "Image, MiniMax-H3-Image",
                "Runs native BF16 six-step generation and up to 16 reference images; checkpoint quality and peak memory remain unqualified.",
                minimum: 96,
                recommended: 128
            )
        ),
        ManagedModelSpec(
            id: Cosmos3Resources.superTextToImageModelID,
            category: .image,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: Cosmos3Resources.superTextToImageMLXRepository,
                revision: Cosmos3Resources.superTextToImageMLXRevision,
                patterns: Cosmos3Resources.superTextToImageSnapshotPatterns
            ),
            upstreamRepoId: Cosmos3Resources.superTextToImageMLXRepository,
            upstreamRevision: Cosmos3Resources.superTextToImageMLXRevision,
            validationKind: .cosmos3SuperTextToImage,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 37_032_673_013,
            defaultCLICommands: ["image generate", "video cosmos3 --mode text-to-image"],
            manifestTemplate: .init(
                engine: .cosmos3Super,
                family: .cosmos,
                tier: .max,
                variant: .distilled,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                defaults: .init(steps: 4, cfg: 1.0),
                supports: [.txt2img],
                components: .init(tokenizer: .local(path: "text_tokenizer"), textEncoder: .local(path: "."), transformer: .local(path: "transformer"), vae: .local(path: "vae"), scheduler: .local(path: "scheduler"))
            ),
            capability: .init(
                "Cosmos3-Super 4-Step MLX 4-bit",
                "Generates images with the distilled four-step Cosmos3-Super export through video cosmos3 text-to-image mode.",
                minimum: 64,
                recommended: 128
            )
        ),
        ManagedModelSpec(
            id: Flux3Resources.modelID, category: .image, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: Flux3Resources.repoID, revision: Flux3Resources.revision,
                patterns: Flux3Resources.patterns),
            upstreamRepoId: Flux3Resources.repoID, upstreamRevision: Flux3Resources.revision,
            usageRestriction: usageRestriction(
                summary: "FLUX 3 Klein uses the FLUX Kommunity License, with use restrictions, a Qualifying User commercial-output exception, and filtering or output-review conditions.",
                license: "FLUX Kommunity License v1.0", sourceRepoId: Flux3Resources.repoID,
                sourceRevision: Flux3Resources.revision,
                licenseURL: "https://huggingface.co/\(Flux3Resources.repoID)/blob/\(Flux3Resources.revision)/LICENSE.md"),
            validationKind: .flux3, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 20_320_953_181, defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux3,
                family: .flux3,
                tier: .latest,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 16, cfg: 1.0),
                supports: [.txt2img, .referenceEdit],
                components: .init(tokenizer: .local(path: "text_encoder"), textEncoder: .local(path: "text_encoder"), transformer: .local(path: "."), vae: .local(path: "."))
            ),
            capability: .init(
                "Image, FLUX 3 Klein",
                "Installs experimental native BF16 FLUX 3 Klein generation and reference editing; released-checkpoint quality and memory remain unqualified.",
                minimum: 32,
                recommended: 48
            )
        ),
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-quantized-linear"),
                defaults: .init(steps: 4, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: .diffusers
            ),
            capability: .init(
                "Image, fast artistic",
                "Creates fast FLUX.2 Klein images for prompts, drafts, and lightweight image workflows.",
                minimum: 8,
                recommended: 16,
                setup: true
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .max,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 4, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: .kleinHybrid
            ),
            capability: .init(
                "Image, artistic quality",
                "Creates higher-fidelity FLUX.2 Klein images when the machine has enough memory for the larger stack.",
                minimum: 24,
                recommended: 32
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .max,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 4, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: .kleinHybrid,
                upstream: .explicit("black-forest-labs/FLUX.2-klein-9B")
            ),
            capability: .init(
                "Image, large artistic quality",
                "Installs the license-restricted FLUX.2 Klein 9B image model for higher-capacity prompt and reference-image workflows.",
                minimum: 48,
                recommended: 64
            )
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
            defaultCLICommands: ["image generate", "image train-lora"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .base,
                variant: .base,
                precision: .unknown,
                defaults: .init(steps: 50, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference, .loraTraining],
                components: .kleinHybrid
            ),
            capability: .init(
                "Image training base",
                "Installs the FLUX.2 Klein base layout used by image-training and compatibility workflows.",
                minimum: 24,
                recommended: 32
            )
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
            defaultCLICommands: ["image generate", "image train-lora"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .max,
                variant: .base,
                precision: .bf16,
                defaults: .init(steps: 50, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference, .loraTraining],
                components: .kleinHybrid,
                upstream: .spec
            ),
            capability: .init(
                "Image LoRA training base",
                "Installs the undistilled FLUX.2 Klein Base 9B checkpoint for higher-capacity LoRA training and research workflows.",
                minimum: 64,
                recommended: 96
            )
        ),
        ManagedModelSpec(
            id: "image-klein-shared",
            category: .image,
            installShape: .directoryRoot,
            validationKind: .flux2Klein,
            runtimeAutoDownloadAllowed: false,
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                precision: .unknown,
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: .kleinShared
            ),
            capability: .init(
                "Shared Klein components",
                "Represents shared FLUX.2 Klein components resolved from another installed Klein image model.",
                minimum: 8,
                recommended: 16
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .max,
                variant: .standard,
                precision: .bf16,
                defaults: .init(steps: 50, cfg: 4.0),
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: .diffusers,
                upstream: .spec
            ),
            capability: .init(
                "Image, FLUX.2-dev LoRA",
                "Installs gated FLUX.2-dev with a compatible 4-bit Mistral conditioner for high-quality generation and existing Dev LoRAs.",
                minimum: 96,
                recommended: 128
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux1,
                family: .flux1,
                tier: .max,
                variant: .standard,
                precision: .bf16,
                defaults: .init(steps: 28, cfg: 3.5),
                supports: [.txt2img, .loraInference],
                components: .diffusers
            ),
            capability: .init(
                "Image, FLUX.1-dev LoRA",
                "Installs gated FLUX.1-dev for native text-to-image generation and existing FLUX.1 adapters.",
                minimum: 48,
                recommended: 64
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .int1,
                quantization: .init(bits: 1, groupSize: 128, scheme: "prism-packed-affine-binary"),
                defaults: .init(steps: 4, cfg: 1.0, sigmaShift: 3.0),
                supports: [.txt2img],
                components: .bonsaiImage
            ),
            capability: .init(
                "Image, binary Klein",
                "Runs the Bonsai Image binary FLUX.2 Klein deployment through the native Swift MLX image runtime.",
                minimum: 16,
                recommended: 24
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .int2,
                quantization: .init(bits: 2, groupSize: 128, scheme: "mlx-packed-affine-ternary"),
                defaults: .init(steps: 4, cfg: 1.0, sigmaShift: 3.0),
                supports: [.txt2img],
                components: .bonsaiImage
            ),
            capability: .init(
                "Image, compact Klein",
                "Runs the Bonsai Image ternary FLUX.2 Klein deployment through the native Swift MLX image runtime.",
                minimum: 12,
                recommended: 16
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .zimageTurbo,
                family: .zimage,
                tier: .nano,
                variant: .distilled,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-quantized-linear"),
                defaults: .init(steps: 4, cfg: 1.0),
                supports: [.txt2img, .img2img, .loraInference],
                components: .diffusers
            ),
            capability: .init(
                "Image Nano",
                "Creates fast 4-bit Z-Image Turbo outputs for photorealistic and general image generation.",
                minimum: 12,
                recommended: 16,
                setup: true
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .zimageTurbo,
                family: .zimage,
                tier: .max,
                variant: .distilled,
                precision: .unknown,
                defaults: .init(steps: 4, cfg: 1.0),
                supports: [.txt2img, .img2img, .loraInference],
                components: .diffusers,
                upstream: .spec
            ),
            capability: .init(
                "Image, realistic quality",
                "Creates higher-fidelity Z-Image outputs from the larger Z-Image Turbo model.",
                minimum: 48,
                recommended: 64
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .zimageTurbo,
                family: .zimage,
                tier: .base,
                variant: .base,
                precision: .unknown,
                supports: [.txt2img, .img2img, .loraInference, .loraTraining],
                components: .zetaBaseHybrid
            ),
            capability: .init(
                "Image realistic base",
                "Installs the larger Z-Image base layout for advanced image workflows.",
                minimum: 48,
                recommended: 64
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .hidreamO1,
                family: .hidream,
                tier: .max,
                variant: .base,
                precision: .bf16,
                defaults: .init(steps: 50, cfg: 5.0),
                supports: [.txt2img, .referenceEdit, .subjectPersonalization],
                components: .rootTextTransformer,
                upstream: .spec
            ),
            capability: .init(
                "Image, HiDream full",
                "Runs the full HiDream O1 image model for high-quality text, reference editing, and personalization workflows.",
                minimum: 96,
                recommended: 128
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .hidreamO1,
                family: .hidream,
                tier: .max,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 28, cfg: 0.0),
                supports: [.txt2img, .referenceEdit, .subjectPersonalization],
                components: .rootTextTransformer,
                upstream: .spec
            ),
            capability: .init(
                "Image, HiDream dev",
                "Runs the distilled HiDream O1 image model for text, reference editing, and subject personalization.",
                minimum: 48,
                recommended: 64
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .senseNovaU15,
                family: .senseNova,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: .init(steps: 50, cfg: 4.0, sigmaShift: 3.0),
                supports: [.txt2img, .img2img, .referenceEdit, .subjectPersonalization],
                components: .rootTextTransformer,
                upstream: .explicit("OpenSenseNova/SenseNova-U1")
            ),
            capability: .init(
                "Image, SenseNova U1.5",
                "Runs the native raw-pixel SenseNova U1.5 model for text-to-image and multi-image editing.",
                minimum: 64,
                recommended: 96
            )
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
            defaultCLICommands: ["image generate", "image train-lora"],
            manifestTemplate: .init(
                engine: .krea2,
                family: .krea,
                tier: .max,
                variant: .base,
                precision: .bf16,
                defaults: .init(steps: 52, cfg: 3.5, sigmaShift: 1.149999976158142),
                supports: [.txt2img, .loraTraining],
                components: .diffusers
            ),
            capability: .init(
                "Image training, Krea 2 Raw",
                "Installs the Krea 2 Raw base checkpoint for LoRA training and post-training workflows.",
                minimum: 96,
                recommended: 128
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .krea2,
                family: .krea,
                tier: .turbo,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 8, cfg: 0.0, sigmaShift: 1.149999976158142),
                supports: [.txt2img, .loraInference],
                components: .diffusers
            ),
            capability: .init(
                "Image, Krea 2 Turbo",
                "Runs the Krea 2 Turbo 8-step text-to-image model and Raw-trained LoRAs through the native Swift MLX image runtime.",
                minimum: 96,
                recommended: 128
            )
        ),
        ManagedModelSpec(
            id: QwenImage21Resources.modelID,
            category: .image, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: QwenImage21Resources.repoID,
                revision: QwenImage21Resources.revision, patterns: QwenImage21Resources.patterns),
            upstreamRepoId: QwenImage21Resources.repoID, upstreamRevision: QwenImage21Resources.revision,
            usageRestriction: usageRestriction(
                summary: "Qwen Image 2.1 is restricted to non-commercial research and evaluation; commercial use requires a separate license.",
                license: "Qwen Research License Agreement",
                sourceRepoId: QwenImage21Resources.repoID, sourceRevision: QwenImage21Resources.revision,
                licenseURL: "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE"
            ),
            validationKind: .qwenImage21, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 33_131_616_240, defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .qwenImage21,
                family: .qwen,
                tier: .latest,
                variant: .base,
                precision: .bf16,
                defaults: .init(steps: 40, cfg: 1.0),
                supports: [.txt2img, .referenceEdit],
                components: .init(tokenizer: .local(path: "processor"), textEncoder: .local(path: "text_encoder"), transformer: .local(path: "transformer"), vae: .local(path: "vae"), scheduler: .local(path: "scheduler"))
            ),
            capability: .init(
                "Image, Qwen 2.1 RGBA",
                "Experimental native generation and editing with ten references and RGBA output. Bounded qualification used an M4 Max with 128 GiB; memory estimates for smaller hosts remain provisional.",
                minimum: 48,
                recommended: 96
            )
        ),
        ManagedModelSpec(
            id: QwenImage21Resources.turboModelID,
            category: .image, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: QwenImage21Resources.turboRepoID,
                revision: QwenImage21Resources.turboRevision, patterns: QwenImage21Resources.patterns),
            upstreamRepoId: QwenImage21Resources.turboRepoID, upstreamRevision: QwenImage21Resources.turboRevision,
            usageRestriction: usageRestriction(
                summary: "Qwen Image 2.1 Turbo is restricted to non-commercial research and evaluation; commercial use requires a separate license.",
                license: "Qwen Research License Agreement",
                sourceRepoId: QwenImage21Resources.turboRepoID, sourceRevision: QwenImage21Resources.turboRevision,
                licenseURL: "https://huggingface.co/\(QwenImage21Resources.turboRepoID)/blob/\(QwenImage21Resources.turboRevision)/LICENSE"
            ),
            validationKind: .qwenImage21, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 32_451_607_098, defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .qwenImage21,
                family: .qwen,
                tier: .turbo,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 8, cfg: 1.0),
                supports: [.txt2img, .referenceEdit],
                components: .init(tokenizer: .local(path: "processor"), textEncoder: .local(path: "text_encoder"), transformer: .local(path: "transformer"), vae: .local(path: "vae"), scheduler: .local(path: "scheduler"))
            ),
            capability: .init(
                "Image, Qwen 2.1 Turbo RGBA",
                "Experimental native eight-step generation and editing with original BF16 safetensors and the checkpoint's saved schedule. Trained Turbo checkpoint inference and smaller-host memory remain unqualified.",
                minimum: 48,
                recommended: 96
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .qwenImageEdit,
                family: .qwen,
                tier: .latest,
                variant: .base,
                precision: .bf16,
                defaults: .init(steps: 40, cfg: 4.0),
                supports: [.referenceEdit],
                components: .diffusers
            ),
            capability: .init(
                "Image edit, Qwen 2511 quality",
                "Runs the pinned BF16 Qwen Image Edit 2511 stack for one-to-three-image editing with the 40-step guided recipe.",
                minimum: 96,
                recommended: 128
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .qwenImageEdit,
                family: .qwen,
                tier: .turbo,
                variant: .distilled,
                precision: .bf16,
                defaults: .init(steps: 4, cfg: 1.0),
                supports: [.referenceEdit],
                components: .diffusers
            ),
            capability: .init(
                "Image edit, Qwen 2511 Lightning",
                "Runs the pinned BF16 Qwen Image Edit 2511 stack with its exact four-step Lightning adapter.",
                minimum: 96,
                recommended: 128
            )
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
            defaultCLICommands: ["image generate"],
            manifestTemplate: .init(
                engine: .ideogram4,
                family: .ideogram,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "sdnq-uint4"),
                defaults: .init(steps: 20, cfg: 7.0),
                supports: [.txt2img],
                components: .ideogram4
            ),
            capability: .init(
                "Image, Ideogram 4 typography",
                "Installs the SDNQ uint4 Ideogram 4 text-to-image stack with dedicated positive and unconditional transformers.",
                minimum: 48,
                recommended: 64
            )
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
