import Foundation

extension ManagedModelCatalog {
    static let speechSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: ManagedModelID.breezeTTS2.rawValue,
            category: .speechTTS,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: breezeTTS2Repository,
                revision: breezeTTS2Revision,
                patterns: ["LICENSE", "README.md", "config.json", "generation_config.json",
                           "model.safetensors.index.json", "model-*.safetensors",
                           "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                           "audio_tokenizer/*"]
            ),
            upstreamRepoId: breezeTTS2Repository,
            upstreamRevision: breezeTTS2Revision,
            usageRestriction: usageRestriction(
                summary: "Breeze TTS 2 weights and self-hosted outputs are for research and non-commercial use only.",
                license: "BreezeBlue Research and Non-Commercial License",
                sourceRepoId: breezeTTS2Repository,
                sourceRevision: breezeTTS2Revision,
                licenseURL: "https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/\(breezeTTS2Revision)/LICENSE"
            ),
            validationKind: .breezeTTS,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 7_648_850_000,
            defaultCLICommands: ["speech synthesize"]
        ),
        ManagedModelSpec(
            id: "speech-tts-qwen3-nano",
            category: .speechTTS,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                revision: "main",
                patterns: [
                    "LICENSE*",
                    "README.md",
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model.safetensors",
                    "speech_tokenizer/*",
                    "tokenizer_config.json",
                    "vocab.json",
                ]
            ),
            upstreamRepoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            upstreamRevision: "main",
            validationKind: .qwen3TTS,
            estimatedDownloadBytes: 4_520_158_972,
            defaultCLICommands: ["speech synthesize"]
        ),
        ManagedModelSpec(
            id: "speech-tts-qwen3-customvoice",
            category: .speechTTS,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                revision: "main",
                patterns: [
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model.safetensors",
                    "speech_tokenizer/*",
                    "tokenizer_config.json",
                    "vocab.json",
                ]
            ),
            upstreamRepoId: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            upstreamRevision: "main",
            validationKind: .qwen3TTS,
            estimatedDownloadBytes: 4_520_159_459,
            defaultCLICommands: ["speech synthesize"]
        ),
        ManagedModelSpec(
            id: "speech-asr-qwen3",
            category: .speechASR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/Qwen3-ASR-1.7B-8bit",
                revision: "main",
                patterns: [
                    "config.json",
                    "generation_config.json",
                    "preprocessor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "added_tokens.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "mlx-community/Qwen3-ASR-1.7B-8bit",
            upstreamRevision: "main",
            validationKind: .qwen3ASR,
            normalizationKind: .qwen3ASRNested,
            estimatedDownloadBytes: 2_467_855_342,
            defaultCLICommands: ["speech transcribe", "speech listen"]
        ),
        ManagedModelSpec(
            id: "speech-asr-parakeet",
            category: .speechASR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/parakeet-tdt-0.6b-v3",
                revision: "main",
                patterns: [
                    "config.json",
                    "tokenizer.model",
                    "tokenizer.vocab",
                    "vocab.txt",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "mlx-community/parakeet-tdt-0.6b-v3",
            upstreamRevision: "main",
            validationKind: .parakeet,
            normalizationKind: .parakeetNested,
            estimatedDownloadBytes: 2 * 1_073_741_824,
            defaultCLICommands: ["speech transcribe"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.sortformerDiarization.rawValue,
            category: .speechDiarization,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
                revision: sortformerUpstreamRevision,
                patterns: [
                    "README.md",
                    "config.json",
                    "model.safetensors",
                ]
            ),
            upstreamRepoId: "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
            upstreamRevision: sortformerUpstreamRevision,
            validationKind: .sortformer,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 236_108_132,
            defaultCLICommands: ["speech diarize"]
        ),
        ManagedModelSpec(
            id: Nemotron3DiarizationResources.modelID,
            category: .speechDiarization,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Nemotron3DiarizationResources.repository,
                revision: Nemotron3DiarizationResources.revision,
                patterns: ["README.md", "Nemotron-3-Diarization.nemo"]
            ),
            upstreamRepoId: Nemotron3DiarizationResources.repository,
            upstreamRevision: Nemotron3DiarizationResources.revision,
            usageRestriction: usageRestriction(
                summary: "NVIDIA Nemotron 3 Diarization is governed by OpenMDW-1.1; review its use and redistribution terms.",
                license: "OpenMDW-1.1",
                sourceRepoId: Nemotron3DiarizationResources.repository,
                sourceRevision: Nemotron3DiarizationResources.revision,
                licenseURL: "https://openmdw.ai/license/1-1/"
            ),
            validationKind: .sortformer,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 198_676_480,
            defaultCLICommands: ["speech diarize", "speech diarize-live"]
        ),
    ]

    private static let sortformerUpstreamRevision = "e23e6404bd9859e93edbf94a740eb1c7fc58f12e"
    private static let breezeTTS2Repository = "BreezeBlue/Breeze-TTS-2"
    private static let breezeTTS2Revision = "3e28c5151381a722f1d8661b4118c298caa77aa4"
}
