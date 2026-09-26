import Foundation

extension MereRunCapabilityCatalog {
    enum AudioEnhanceFamily: String, MereRunFamilyID {
        case apBWE = "ap-bwe"
        case univerSR = "universr"
    }

    enum AudioEditFamily: String, MereRunFamilyID {
        case aukBase = "auk-base"
        case aukFlash = "auk-flash"
    }

    private static let aukModels = ["audio-auk-base", "audio-auk-flash"]
    private static let aukThinker = "audio-auk-thinker"
    private static let audioEnhanceModels = ["audio-enhance-ap-bwe-16kto48k", "audio-enhance-universr-audio"]

    /// The exact `--model` id picks the runtime; `--model-path` only relocates its weights.
    static let audioEnhanceRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("audio-enhance-ap-bwe-16kto48k")],
        families: [
            .init(AudioEnhanceFamily.apBWE, title: "AP-BWE", models: ["audio-enhance-ap-bwe-16kto48k"]),
            .init(AudioEnhanceFamily.univerSR, title: "UniverSR", models: ["audio-enhance-universr-audio"])
        ],
        excludedModels: .models(aukModels, reason: "AuK generates and edits speech; use `audio edit`.")
            .and([aukThinker], reason: "It is the Qwen2.5-Omni-3B encoder that `audio edit` loads.")
    )

    /// The exact `--model` id picks the variant; `--model-path` only relocates its checkpoint.
    static let audioEditRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("audio-auk-base")],
        families: [
            .init(AudioEditFamily.aukBase, title: "AuK Base", models: ["audio-auk-base"]),
            .init(AudioEditFamily.aukFlash, title: "AuK Flash", models: ["audio-auk-flash"])
        ],
        excludedModels: .models(
            [aukThinker],
            reason: "It is the Qwen2.5-Omni-3B encoder AuK loads alongside a base or Flash model; "
                + "pass a local copy as `--thinker-path`."
        ).and(audioEnhanceModels, reason: "It extends audio bandwidth; use `audio enhance`.")
    )
}
