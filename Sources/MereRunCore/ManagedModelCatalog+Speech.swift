import Foundation

extension ManagedModelCatalog {
    static let speechSpecs: [ManagedModelSpec] = [
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
            defaultCLICommands: ["speech transcribe"]
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
    ]

    private static let sortformerUpstreamRevision = "e23e6404bd9859e93edbf94a740eb1c7fc58f12e"
}
