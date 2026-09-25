import Foundation

extension MereRunCapabilityCatalog {
    enum SpeechListenFamily: String, MereRunFamilyID {
        case qwen3ASR = "qwen3-asr"
    }

    enum SpeechDiarizeLiveFamily: String, MereRunFamilyID {
        case nemotron3 = "nemotron3"
    }

    static let speechListenRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("speech-asr-qwen3")],
        families: [.init(SpeechListenFamily.qwen3ASR, title: "Qwen3-ASR", models: ["speech-asr-qwen3"])],
        excludedModels: .models(
            ["speech-asr-parakeet"],
            reason: "Parakeet transcribes recorded audio only; use `speech transcribe`."
        )
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
        )
    )
}
