import Foundation

extension ManagedModelCatalog {
    static let musicSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: "music-acestep",
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "config.json",
                    "acestep-v15-turbo/*",
                    "acestep-5Hz-lm-1.7B/*",
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            upstreamRepoId: "ACE-Step/Ace-Step1.5",
            upstreamRevision: aceStepSharedRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 10_092_095_357,
            defaultCLICommands: ["music generate", "music analyze"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLBase.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-base",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-base",
                        revision: aceStepXLBaseRevision,
                        patterns: [
                            "apg_guidance.py",
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_base.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-base",
            upstreamRevision: aceStepXLBaseRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 23 * 1_073_741_824,
            defaultCLICommands: ["music generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLSFT.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-sft",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-sft",
                        revision: aceStepXLSFTRevision,
                        patterns: [
                            "apg_guidance.py",
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_base.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-sft",
            upstreamRevision: aceStepXLSFTRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 23 * 1_073_741_824,
            defaultCLICommands: ["music generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLTurbo.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-turbo",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-turbo",
                        revision: aceStepXLTurboRevision,
                        patterns: [
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_turbo.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-turbo",
            upstreamRevision: aceStepXLTurboRevision,
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 23 * 1_073_741_824,
            defaultCLICommands: ["music generate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "Qwen3-Embedding-0.6B/*",
                    "vae/*",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "acestep-v15-xl-turbo",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-v15-xl-turbo",
                        revision: aceStepXLTurboRevision,
                        patterns: [
                            "config.json",
                            "configuration_acestep_v15.py",
                            "model*.safetensors",
                            "model.safetensors.index.json",
                            "modeling_acestep_v15_xl_turbo.py",
                            "silence_latent.pt",
                        ]
                    )
                ),
                MountedHubFallbackConfig(
                    destinationPath: "acestep-5Hz-lm-4B",
                    hubFallback: HubFallbackConfig(
                        repoId: "ACE-Step/acestep-5Hz-lm-4B",
                        revision: aceStepLM4BRevision,
                        patterns: [
                            "*.json",
                            "*.safetensors",
                            "*.jinja",
                            "merges.txt",
                            "vocab.json",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "ACE-Step/acestep-v15-xl-turbo + ACE-Step/acestep-5Hz-lm-4B",
            upstreamRevision: "\(aceStepXLTurboRevision)+\(aceStepLM4BRevision)",
            validationKind: .aceStep,
            normalizationKind: .musicACEStep,
            estimatedDownloadBytes: 32 * 1_073_741_824,
            defaultCLICommands: ["music generate", "music analyze"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepLM17B.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/Ace-Step1.5",
                revision: aceStepSharedRevision,
                patterns: [
                    "acestep-5Hz-lm-1.7B/*",
                ]
            ),
            upstreamRepoId: "ACE-Step/Ace-Step1.5",
            upstreamRevision: aceStepSharedRevision,
            validationKind: .aceStepLM,
            normalizationKind: .musicACEStepLM,
            estimatedDownloadBytes: 4 * 1_073_741_824,
            defaultCLICommands: ["music generate", "music analyze", "music serve"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aceStepLM4B.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "ACE-Step/acestep-5Hz-lm-4B",
                revision: aceStepLM4BRevision,
                patterns: [
                    "*.json",
                    "*.safetensors",
                    "*.jinja",
                    "merges.txt",
                    "vocab.json",
                ]
            ),
            upstreamRepoId: "ACE-Step/acestep-5Hz-lm-4B",
            upstreamRevision: aceStepLM4BRevision,
            validationKind: .aceStepLM,
            normalizationKind: .musicACEStepLM,
            estimatedDownloadBytes: 9 * 1_073_741_824,
            defaultCLICommands: ["music generate", "music analyze", "music serve"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.miniMaxMusic3.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: MiniMaxMusic3Resources.repository,
                revision: MiniMaxMusic3Resources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "config.json",
                    "modular_model_index.json",
                    "condition_encoder/*",
                    "language_model/*",
                    "rvq_depth_decoder/*",
                    "scheduler/*",
                    "tokenizer/*",
                    "transformer/*",
                    "vocoder/*",
                ]
            ),
            upstreamRepoId: MiniMaxMusic3Resources.repository,
            upstreamRevision: MiniMaxMusic3Resources.revision,
            usageRestriction: usageRestriction(
                summary: "MiniMax Music 3 uses the custom MiniMax-Music3 Community License, including product attribution, revenue-threshold authorization, and hosted-generation safeguards.",
                component: "MiniMax Music 3 weights",
                license: "MiniMax-Music3 Community License",
                sourceRepoId: MiniMaxMusic3Resources.repository,
                sourceRevision: MiniMaxMusic3Resources.revision,
                licenseURL: "https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/\(MiniMaxMusic3Resources.revision)/LICENSE"
            ),
            validationKind: .miniMaxMusic3,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: MiniMaxMusic3Resources.estimatedDownloadBytes,
            defaultCLICommands: ["music generate", "music serve"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.magentaRT2Small.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: magentaRT2UpstreamRepoId,
                revision: magentaRT2UpstreamRevision,
                patterns: [
                    "models/mrt2_small/mrt2_small.mlxfn",
                    "models/mrt2_small/mrt2_small_state.safetensors",
                ] + magentaRT2ResourcePatterns
            ),
            upstreamRepoId: magentaRT2UpstreamRepoId,
            upstreamRevision: magentaRT2UpstreamRevision,
            validationKind: .magentaRT2,
            estimatedDownloadBytes: 1_840_072_891,
            defaultCLICommands: ["music generate", "music realtime"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.magentaRT2Base.rawValue,
            category: .music,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: magentaRT2UpstreamRepoId,
                revision: magentaRT2UpstreamRevision,
                patterns: [
                    "models/mrt2_base/mrt2_base.mlxfn",
                    "models/mrt2_base/mrt2_base_state.safetensors",
                ] + magentaRT2ResourcePatterns
            ),
            upstreamRepoId: magentaRT2UpstreamRepoId,
            upstreamRevision: magentaRT2UpstreamRevision,
            validationKind: .magentaRT2,
            estimatedDownloadBytes: 4_164_096_058,
            defaultCLICommands: ["music generate", "music realtime"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.muScriptorSmall.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "MuScriptor/muscriptor-small",
                revision: "8c127f603b807520fa465c838e9bfee8a91ada4e",
                patterns: ["LICENSE*", "README.md", "config.json", "model.safetensors"]
            ),
            upstreamRepoId: "MuScriptor/muscriptor-small",
            upstreamRevision: "8c127f603b807520fa465c838e9bfee8a91ada4e",
            usageRestriction: muScriptorUsageRestriction(
                sourceRepoId: "MuScriptor/muscriptor-small",
                sourceRevision: "8c127f603b807520fa465c838e9bfee8a91ada4e"
            ),
            validationKind: .muScriptor,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 412_000_000,
            defaultCLICommands: ["music transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.muScriptorMedium.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "MuScriptor/muscriptor-medium",
                revision: "f32236969308476e01fd3aae67357de5feb05a2d",
                patterns: ["LICENSE*", "README.md", "config.json", "model.safetensors"]
            ),
            upstreamRepoId: "MuScriptor/muscriptor-medium",
            upstreamRevision: "f32236969308476e01fd3aae67357de5feb05a2d",
            usageRestriction: muScriptorUsageRestriction(
                sourceRepoId: "MuScriptor/muscriptor-medium",
                sourceRevision: "f32236969308476e01fd3aae67357de5feb05a2d"
            ),
            validationKind: .muScriptor,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_230_000_000,
            defaultCLICommands: ["music transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.muScriptorLarge.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "MuScriptor/muscriptor-large",
                revision: "8809fdfbed2affa7ade94a7059e746e3880720e7",
                patterns: ["LICENSE*", "README.md", "config.json", "model.safetensors"]
            ),
            upstreamRepoId: "MuScriptor/muscriptor-large",
            upstreamRevision: "8809fdfbed2affa7ade94a7059e746e3880720e7",
            usageRestriction: muScriptorUsageRestriction(
                sourceRepoId: "MuScriptor/muscriptor-large",
                sourceRevision: "8809fdfbed2affa7ade94a7059e746e3880720e7"
            ),
            validationKind: .muScriptor,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 5_620_000_000,
            defaultCLICommands: ["music transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.roFormerViperX1297.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "bs_roformer/vocals_viperx/config.yaml",
                    "bs_roformer/vocals_viperx/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 639_114_645,
            defaultCLICommands: ["music separate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.roFormerFourStem.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "bs_roformer/multistem/config.yaml",
                    "bs_roformer/multistem/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 526_973_074,
            defaultCLICommands: ["music separate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.melRoFormerDereverb.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "mel_band_roformer/dereverb/config.yaml",
                    "mel_band_roformer/dereverb/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 912_891_178,
            defaultCLICommands: ["music separate"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.melRoFormerDenoise.rawValue,
            category: .music,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                patterns: [
                    "LICENSE",
                    "README.md",
                    "mel_band_roformer/denoise/config.yaml",
                    "mel_band_roformer/denoise/model.safetensors",
                ]
            ),
            upstreamRepoId: RoFormerResources.repository,
            upstreamRevision: RoFormerResources.revision,
            validationKind: .roFormer,
            estimatedDownloadBytes: 912_890_953,
            defaultCLICommands: ["music separate"]
        ),
    ]

    private static let magentaRT2UpstreamRepoId = "google/magenta-realtime-2"

    private static let magentaRT2UpstreamRevision = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"

    private static let aceStepSharedRevision = "19671f406d603126926c1b7e2adc169acbcade22"

    private static let aceStepXLBaseRevision = "220c1166efbdd9583eafcb12eb160594bbfcb241"

    private static let aceStepXLSFTRevision = "d06de46b4622f781cf07f4a013a67d591ca52819"

    private static let aceStepXLTurboRevision = "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee"

    private static let aceStepLM4BRevision = "0a3ec94b557aea7d508da38b31cfe7341f6ff737"

    private static let magentaRT2ResourcePatterns = [
        "resources/musiccoca/audio_preprocessor.tflite",
        "resources/musiccoca/mapper.tflite",
        "resources/musiccoca/music_encoder.tflite",
        "resources/musiccoca/pretrained_vector_quantizer.tflite",
        "resources/musiccoca/spm.model",
        "resources/musiccoca/text_encoder.tflite",
        "resources/spectrostream/decoder.safetensors",
        "resources/spectrostream/encoder.safetensors",
        "resources/spectrostream/quantizer.safetensors",
        "resources/spectrostream/spectrostream_encoder.mlxfn",
    ]

    private static func muScriptorUsageRestriction(
        sourceRepoId: String,
        sourceRevision: String
    ) -> ManagedModelUsageRestriction {
        usageRestriction(
            summary: "MuScriptor weights are licensed CC BY-NC 4.0 for non-commercial use.",
            license: "CC BY-NC 4.0",
            sourceRepoId: sourceRepoId,
            sourceRevision: sourceRevision,
            licenseURL: "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en"
        )
    }
}
