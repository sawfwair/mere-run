import Foundation

extension MereRunCapabilityCatalog {
    enum SpeechListenFamily: String, MereRunFamilyID {
        case nemotronTranscribe = "nemotron-transcribe"
        case whistle
        case parakeet
        case qwen3ASR = "qwen3-asr"
    }

    enum SpeechDiarizeLiveFamily: String, MereRunFamilyID {
        case nemotron3 = "nemotron3"
    }

    enum SpeechTranscribeFamily: String, MereRunFamilyID {
        case nemotronTranscribe = "nemotron-transcribe"
        case parakeet
        case whistle
        case qwen3ASR = "qwen3-asr"
    }

    enum SpeechDiarizeFamily: String, MereRunFamilyID {
        case sortformer
        case nemotron3
    }

    enum SpeechSynthesizeFamily: String, MereRunFamilyID {
        case style
        case customVoice = "custom-voice"
        case breeze
        case phonon
        case breezeClone = "breeze-clone"
        case drama3
        case drama3Clone = "drama3-clone"
        case clone
    }

    /// Explicit Whistle selection requires a supported transcription request. For the other
    /// backends, the CLI's precedence (`SpeechTranscriptionResolver.route`): translation needs Qwen3-ASR and
    /// outranks everything, an explicit `--backend` comes next, then a named model, and Parakeet
    /// otherwise. A named model runs only when the flags leave it its own family, so the
    /// selectors outrank it: `--backend qwen --model speech-asr-parakeet` runs Qwen3-ASR's default
    /// and warns. A `--language` that Parakeet's router does not recognize also picks Qwen3-ASR,
    /// ahead of an explicit backend or model. Only its literal `auto` is declared here: the rest
    /// depends on the CLI's normalization and the installed checkpoint, so the CLI's gate asks the
    /// resolver itself (`routedFamily`), and these rules are what shells without it see.
    static let speechTranscribeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [
            .init(whenAny: [.init(flag: "--backend", values: ["nemotron"])], models: ["speech-asr-nemotron-35-transcribe"]),
            .init(whenAny: [.init(flag: "--backend", values: ["whistle"])], models: ["speech-asr-whistle"]),
            .init(
                whenAny: [
                    .init(flag: "--task", values: ["translate"]),
                    .init(flag: "--language", values: ["auto"]),
                    .init(flag: "--backend", values: ["qwen"])
                ],
                models: ["speech-asr-qwen3"]
            ),
            .always("speech-asr-parakeet")
        ],
        families: [
            .init(SpeechTranscribeFamily.nemotronTranscribe, title: "Nemotron 3.5 Transcribe",
                  models: ["speech-asr-nemotron-35-transcribe", "speech-asr-nemotron-35-transcribe-bf16"],
                  selectors: [.init(flag: "--backend", values: ["auto", "nemotron"])]),
            .init(
                SpeechTranscribeFamily.whistle, title: "Whistle", models: ["speech-asr-whistle"],
                selectors: [.init(flag: "--backend", values: ["auto", "whistle"])]
            ),
            .init(
                SpeechTranscribeFamily.parakeet, title: "Parakeet", models: ["speech-asr-parakeet"],
                selectors: [
                    .init(flag: "--task", values: ["transcribe"]),
                    .init(flag: "--backend", values: ["auto", "parakeet"])
                ]
            ),
            .init(
                SpeechTranscribeFamily.qwen3ASR, title: "Qwen3-ASR", models: ["speech-asr-qwen3"],
                selectors: [.init(flag: "--backend", values: ["auto", "qwen"])]
            )
        ],
        selectorsOverrideModel: true,
        // Which languages leave Parakeet depends on its checkpoint's vocabulary and on how the
        // language is spelled; `SpeechTranscriptionResolver.route` decides.
        routedByCommand: true
    )

    /// An exact managed id picks the family; a local folder is Nemotron 3 when it holds the NeMo
    /// archive (`SpeechDiarizeCommand.swift` via `Nemotron3DiarizationResources.isNemotron3`).
    static let speechDiarizeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("speech-diarization-sortformer")],
        families: [
            .init(SpeechDiarizeFamily.sortformer, title: "Sortformer", models: ["speech-diarization-sortformer"]),
            .init(SpeechDiarizeFamily.nemotron3, title: "Nemotron 3 Diarization", models: ["speech-diarization-nemotron3"])
        ]
    )

    /// `--mode` picks the Qwen3-TTS generator path, and in style mode the checkpoint picks the
    /// rest: CustomVoice speaks as a named `--speaker`, the other checkpoints have none. The CLI
    /// runs either managed checkpoint in either mode. A local folder is identified by its config
    /// (`ModelFamilyIdentifier`'s synthesize probe); `--mode clone` runs any checkpoint.
    static let speechSynthesizeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [
            .init(whenAny: [.init(flag: "--mode", values: ["clone"])], models: ["speech-tts-qwen3-nano"],
                  family: SpeechSynthesizeFamily.clone.rawValue),
            .init(models: ["speech-tts-qwen3-nano"], family: SpeechSynthesizeFamily.style.rawValue)
        ],
        families: [
            .init(SpeechSynthesizeFamily.drama3, title: "DRAMA-3",
                  models: ["speech-tts-drama-3"], selectors: [.init(flag: "--mode", values: ["style"])]),
            .init(SpeechSynthesizeFamily.drama3Clone, title: "DRAMA-3 clone",
                  models: ["speech-tts-drama-3"], selectors: [.init(flag: "--mode", values: ["clone"])]),
            .init(
                SpeechSynthesizeFamily.style, title: "Qwen3-TTS style", models: ["speech-tts-qwen3-nano"],
                selectors: [.init(flag: "--mode", values: ["style"])]
            ),
            .init(
                SpeechSynthesizeFamily.customVoice, title: "Qwen3-TTS CustomVoice",
                models: ["speech-tts-qwen3-customvoice"], selectors: [.init(flag: "--mode", values: ["style"])]
            ),
            .init(
                SpeechSynthesizeFamily.phonon, title: "Phonon 31m",
                models: ["speech-tts-phonon-31m"]
            ),
            .init(
                SpeechSynthesizeFamily.breeze, title: "Breeze TTS 2",
                models: ["speech-tts-breeze-2"], selectors: [.init(flag: "--mode", values: ["style"])]
            ),
            .init(
                SpeechSynthesizeFamily.clone, title: "Qwen3-TTS clone",
                models: ["speech-tts-qwen3-nano", "speech-tts-qwen3-customvoice"],
                selectors: [.init(flag: "--mode", values: ["clone"])]
            ),
            .init(
                SpeechSynthesizeFamily.breezeClone, title: "Breeze TTS 2 clone and direction",
                models: ["speech-tts-breeze-2"], selectors: [.init(flag: "--mode", values: ["clone"])]
            )
        ]
    )

    static let speechListenRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("speech-asr-qwen3")],
        families: [
            .init(SpeechListenFamily.qwen3ASR, title: "Qwen3-ASR", models: ["speech-asr-qwen3"]),
            .init(SpeechListenFamily.parakeet, title: "Parakeet", models: ["speech-asr-parakeet"]),
            .init(SpeechListenFamily.whistle, title: "Whistle", models: ["speech-asr-whistle"]),
            .init(SpeechListenFamily.nemotronTranscribe, title: "Nemotron 3.5 Transcribe",
                  models: ["speech-asr-nemotron-35-transcribe", "speech-asr-nemotron-35-transcribe-bf16"])
        ],
        listingFlags: ["--list-devices"]
    )

    static let speechDiarizeLiveRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("speech-diarization-nemotron3")],
        families: [
            .init(
                SpeechDiarizeLiveFamily.nemotron3,
                title: "Nemotron 3 Diarization",
                models: ["speech-diarization-nemotron3"]
            )
        ],
        excludedModels: .models(
            ["speech-diarization-sortformer"],
            reason: "Sortformer diarizes recorded audio only; use `speech diarize`."
        ),
        listingFlags: ["--list-devices"]
    )
}
