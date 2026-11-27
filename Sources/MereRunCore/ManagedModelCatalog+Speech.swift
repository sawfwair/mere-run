import Foundation

extension ManagedModelCatalog {
    static let speechSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: HaruResources.modelID,
            category: .audio,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: HaruResources.repository,
                revision: HaruResources.revision,
                patterns: HaruResources.files + ["README.md"]
            ),
            upstreamRepoId: HaruResources.repository,
            upstreamRevision: HaruResources.revision,
            validationKind: .haru4B,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 10_560_110_179,
            defaultCLICommands: ["speech converse"],
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .haru,
                family: .audio,
                tier: .base,
                variant: .standard,
                precision: .bf16,
                supports: [.speechConversation],
                components: .rootText
            ),
            capability: .init(
                "Haru-4B",
                "Full-duplex spoken conversations with a character card and optional voice reference.",
                minimum: 16,
                recommended: 24
            )
        ),
        ManagedModelSpec(
            id: SOVTranscribeResources.modelID, category: .speechASR, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: SOVTranscribeResources.repository, revision: SOVTranscribeResources.revision,
                                           patterns: SOVTranscribeResources.patterns),
            upstreamRepoId: SOVTranscribeResources.repository, upstreamRevision: SOVTranscribeResources.revision,
            validationKind: .sovTranscribe, runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 53_283_115_580, defaultCLICommands: ["speech transcribe"], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .sovTranscribeASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.speechRecognition],
                components: .rootText
            ),
            capability: .init(
                "SOV-Transcribe-1",
                "Native BF16 file transcription in seven languages with instructions and optional DFlash decoding; checkpoint quality remains unqualified.",
                minimum: 64,
                recommended: 96
            )
        ),
        ManagedModelSpec(
            id: NemotronTranscribeResources.modelID,
            category: .speechASR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: NemotronTranscribeResources.quantizedRepository, revision: NemotronTranscribeResources.quantizedRevision,
                patterns: NemotronTranscribeResources.quantizedSnapshotPatterns
            ),
            upstreamRepoId: NemotronTranscribeResources.quantizedRepository,
            upstreamRevision: NemotronTranscribeResources.quantizedRevision,
            usageRestriction: usageRestriction(
                summary: "NVIDIA Nemotron 3.5 Transcribe is governed by OpenMDW-1.1; review its use and redistribution terms.",
                license: "OpenMDW-1.1", sourceRepoId: NemotronTranscribeResources.repository,
                sourceRevision: NemotronTranscribeResources.revision, licenseURL: "https://openmdw.ai/license/1-1/"
            ),
            validationKind: .nemotronTranscribe,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: NemotronTranscribeResources.quantizedDownloadBytes,
            defaultCLICommands: ["speech transcribe", "speech listen"],
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .nemotronTranscribeASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: .init(bits: 4, groupSize: 64, scheme: "mlx-affine-routed-experts"),
                supports: [.speechRecognition],
                components: .rootText
            ),
            capability: .init(
                "Nemotron 3.5 Transcribe",
                "Native offline transcription and speech translation with 4-bit routed experts.",
                minimum: 32,
                recommended: 48
            )
        ),
        ManagedModelSpec(
            id: NemotronTranscribeResources.bf16ModelID,
            category: .speechASR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: NemotronTranscribeResources.repository, revision: NemotronTranscribeResources.revision,
                patterns: NemotronTranscribeResources.files + ["README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/*.txt"]
            ),
            upstreamRepoId: NemotronTranscribeResources.repository,
            upstreamRevision: NemotronTranscribeResources.revision,
            usageRestriction: usageRestriction(
                summary: "NVIDIA Nemotron 3.5 Transcribe is governed by OpenMDW-1.1; review its use and redistribution terms.",
                license: "OpenMDW-1.1", sourceRepoId: NemotronTranscribeResources.repository,
                sourceRevision: NemotronTranscribeResources.revision, licenseURL: "https://openmdw.ai/license/1-1/"
            ),
            validationKind: .nemotronTranscribe,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: NemotronTranscribeResources.weightBytes,
            defaultCLICommands: ["speech transcribe", "speech listen"],
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .nemotronTranscribeASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.speechRecognition],
                components: .rootText
            ),
            capability: .init(
                "Nemotron 3.5 Transcribe BF16",
                "Original BF16 offline transcription and speech translation in 40 languages.",
                minimum: 96,
                recommended: 128
            )
        ),
        ManagedModelSpec(
            id: Drama3Resources.modelID,
            category: .speechTTS,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: Drama3Resources.quantizedRepository, revision: Drama3Resources.quantizedRevision,
                patterns: Drama3Resources.quantizedSnapshotPatterns
            ),
            upstreamRepoId: Drama3Resources.repository,
            upstreamRevision: Drama3Resources.revision,
            usageRestriction: usageRestriction(
                summary: "Built with Fish Audio. Review the Community License revenue, hosted-service, attribution, and voice-consent terms.",
                license: "Fish Audio Community License 1.0",
                sourceRepoId: Drama3Resources.repository, sourceRevision: Drama3Resources.revision,
                licenseURL: "https://huggingface.co/fishaudio/drama-3/blob/\(Drama3Resources.revision)/LICENSE"
            ),
            validationKind: .drama3TTS,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: Drama3Resources.quantizedDownloadBytes,
            defaultCLICommands: ["speech synthesize"],
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .drama3TTS,
                family: .tts,
                tier: .latest,
                variant: .standard,
                precision: .unknown,
                supports: [.speechSynthesis],
                components: .rootText
            ),
            capability: .init(
                "DRAMA-3 Q4 experts",
                "Native multilingual and multi-speaker speech, event tags, and voice cloning. Built with Fish Audio.",
                minimum: 32,
                recommended: 36
            )
        ),
        ManagedModelSpec(
            id: ManagedModelID.phonon31m.rawValue,
            category: .speechTTS,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: PhononResources.repository,
                revision: PhononResources.revision,
                patterns: ["README.md", "config.json", "model.safetensors", "tokenizer.json", "default-voice.safetensors"]
            ),
            upstreamRepoId: PhononResources.repository,
            upstreamRevision: PhononResources.revision,
            validationKind: .phononTTS,
            estimatedDownloadBytes: 103_400_000,
            defaultCLICommands: ["speech synthesize"],
            manifestTemplate: .init(
                engine: .phononTTS,
                family: .tts,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: [.speechSynthesis],
                components: .rootText
            ),
            capability: .init(
                "Phonon 31m",
                "Synthesizes English speech locally with Gradium Phonon and a precomputed voice.",
                minimum: 4,
                recommended: 8
            )
        ),
        ManagedModelSpec(
            id: ManagedModelID.whistleASR.rawValue,
            category: .speechASR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Cactus-Compute/whistle",
                revision: "ca5287601bef25af26dcf2e1b2bdc0843a7c19e5",
                patterns: ["LICENSE", "README.md", "config.json", "whistle.cact"]
            ),
            upstreamRepoId: "Cactus-Compute/whistle",
            upstreamRevision: "ca5287601bef25af26dcf2e1b2bdc0843a7c19e5",
            validationKind: .whistle,
            estimatedDownloadBytes: 17_000_000,
            defaultCLICommands: ["speech transcribe", "speech listen"],
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .whistleASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                supports: [.speechRecognition],
                components: .init(tokenizer: .local(path: "whistle.cact"), transformer: .local(path: "whistle.cact"))
            ),
            capability: .init(
                "Whistle speech recognition",
                "Native packed CQ2/CQ4 transcription in seven European languages with beam search and live audio.",
                minimum: 8,
                recommended: 16
            )
        ),
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
            defaultCLICommands: ["speech synthesize"],
            manifestTemplate: .init(
                engine: .breezeTTS,
                family: .tts,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.speechSynthesis],
                components: .rootText
            ),
            capability: .init(
                "Breeze TTS 2",
                "Synthesizes English and Chinese speech with voice design, cloning, and direction. Research and non-commercial use only.",
                minimum: 16,
                recommended: 24
            )
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
            defaultCLICommands: ["speech synthesize"],
            manifestTemplate: .init(
                engine: .qwen3TTS,
                family: .tts,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                supports: [.speechSynthesis],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Text to speech",
                "Synthesizes speech with Qwen3 TTS, including style prompts and voice-design workflows.",
                minimum: 8,
                recommended: 16,
                setup: true
            )
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
            defaultCLICommands: ["speech synthesize"],
            manifestTemplate: .init(
                engine: .qwen3TTS,
                family: .tts,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.speechSynthesis],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Custom voice TTS",
                "Synthesizes speech as one of nine named speakers with Qwen3-TTS CustomVoice.",
                minimum: 12,
                recommended: 16
            )
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
            defaultCLICommands: ["speech transcribe", "speech listen"],
            manifestTemplate: .init(
                engine: .qwen3ASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .int8,
                supports: [.speechRecognition],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Speech recognition, multilingual",
                "Transcribes and translates speech with Qwen3 ASR.",
                minimum: 12,
                recommended: 16
            )
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
            defaultCLICommands: ["speech transcribe", "speech listen"],
            manifestTemplate: .init(
                engine: .parakeetASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.speechRecognition],
                components: .rootText,
                upstream: .spec
            ),
            capability: .init(
                "Speech recognition, fast",
                "Transcribes speech quickly with the Parakeet ASR runtime.",
                minimum: 8,
                recommended: 16,
                setup: true
            )
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
            defaultCLICommands: ["speech diarize"],
            manifestTemplate: .init(
                engine: .sortformer,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .fp16,
                supports: [.speakerDiarization],
                upstream: .spec
            ),
            capability: .init(
                "Speaker diarization",
                "Identifies up to four speakers and their active time ranges with native MLX Sortformer.",
                minimum: 8,
                recommended: 16
            )
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
            defaultCLICommands: ["speech diarize", "speech diarize-live"],
            manifestTemplate: .init(
                engine: .sortformer,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.speakerDiarization],
                upstream: .spec
            ),
            capability: .init(
                "Speaker diarization, eight speakers",
                "Identifies up to eight speakers with native MLX Nemotron 3 Diarization.",
                minimum: 8,
                recommended: 16
            )
        ),
    ]

    private static let sortformerUpstreamRevision = "e23e6404bd9859e93edbf94a740eb1c7fc58f12e"
    private static let breezeTTS2Repository = "BreezeBlue/Breeze-TTS-2"
    private static let breezeTTS2Revision = "3e28c5151381a722f1d8661b4118c298caa77aa4"
}
