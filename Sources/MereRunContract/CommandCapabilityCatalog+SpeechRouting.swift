import Foundation

extension MereRunCapabilityCatalog {
    enum SpeechListenFamily: String, MereRunFamilyID {
        case qwen3ASR = "qwen3-asr"
    }

    enum SpeechDiarizeLiveFamily: String, MereRunFamilyID {
        case nemotron3 = "nemotron3"
    }

    enum SpeechTranscribeFamily: String, MereRunFamilyID {
        case parakeet
        case qwen3ASR = "qwen3-asr"
    }

    enum SpeechDiarizeFamily: String, MereRunFamilyID {
        case sortformer
        case nemotron3
    }

    enum SpeechSynthesizeFamily: String, MereRunFamilyID {
        case style
        case clone
    }

    /// The CLI's precedence (`SpeechTranscriptionResolver.route`): translation needs Qwen3-ASR and
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

    /// `--mode` picks the Qwen3-TTS generator path and so the option surface. The model does not:
    /// the CLI runs either managed checkpoint, or any local folder, in either mode.
    static let speechSynthesizeRouting = MereRunCapabilityRouting(
        modelFlags: [],
        defaultModels: [
            .init(whenAny: [.init(flag: "--mode", values: ["clone"])], models: ["speech-tts-qwen3-nano"],
                  family: SpeechSynthesizeFamily.clone.rawValue),
            .init(models: ["speech-tts-qwen3-nano"], family: SpeechSynthesizeFamily.style.rawValue)
        ],
        families: [
            .init(
                SpeechSynthesizeFamily.style, title: "Qwen3-TTS style",
                models: ["speech-tts-qwen3-nano", "speech-tts-qwen3-customvoice"],
                selectors: [.init(flag: "--mode", values: ["style"])], modelFlag: "--model"
            ),
            .init(
                SpeechSynthesizeFamily.clone, title: "Qwen3-TTS clone",
                models: ["speech-tts-qwen3-nano", "speech-tts-qwen3-customvoice"],
                selectors: [.init(flag: "--mode", values: ["clone"])], modelFlag: "--model"
            )
        ]
    )

    static let speechListenRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("speech-asr-qwen3")],
        families: [.init(SpeechListenFamily.qwen3ASR, title: "Qwen3-ASR", models: ["speech-asr-qwen3"])],
        // The live loader takes any id and runs Qwen3-ASR (`CLIQwenASRLoader`), so a Parakeet id
        // runs and has no effect.
        excludedModels: .models(
            ["speech-asr-parakeet"],
            reason: "Parakeet transcribes recorded audio only; use `speech transcribe`.",
            severity: .warning
        ),
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
