import Foundation

extension MereRunCapabilityCatalog {
    enum ACEStepFamily: String, MereRunFamilyID {
        case aceStep = "ace-step"
    }

    enum MuScriptorFamily: String, MereRunFamilyID {
        case muScriptor = "muscriptor"
    }

    enum MagentaRealtimeFamily: String, MereRunFamilyID {
        case magentaRT2 = "magenta-rt2"
    }

    private static let aceStepLanguageModels = ["music-acestep-lm-1.7b", "music-acestep-lm-4b"]

    static let musicAnalyzeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("music-acestep")],
        families: [
            .init(ACEStepFamily.aceStep, title: "ACE-Step", models: ["music-acestep", "music-acestep-xl-turbo-lm4b"])
        ],
        excludedModels: .models(
            aceStepLanguageModels,
            reason: "It is an ACE-Step language model; pass it as `--lm-model`."
        )
    )

    static let musicTranscribeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-path", "--model"],
        defaultModels: [.always("music-muscriptor-medium")],
        families: [
            .init(
                MuScriptorFamily.muScriptor,
                title: "MuScriptor",
                models: ["music-muscriptor-small", "music-muscriptor-medium", "music-muscriptor-large"]
            )
        ]
    )

    static let musicRealtimeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("music-magenta-rt2-small")],
        families: [
            .init(
                MagentaRealtimeFamily.magentaRT2,
                title: "Magenta RealTime 2",
                models: ["music-magenta-rt2-small", "music-magenta-rt2-base"]
            )
        ],
        excludedModels: .models(
            ["music-acestep", "music-acestep-xl-base", "music-acestep-xl-sft", "music-acestep-xl-turbo",
             "music-acestep-xl-turbo-lm4b"],
            reason: "ACE-Step renders whole songs, not a live stream; use `music generate`."
        ).and(
            aceStepLanguageModels,
            reason: "It is an ACE-Step language model; `music realtime` streams Magenta RealTime 2 only."
        )
    )

    static let musicTrainAdapterRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("music-acestep")],
        families: [.init(ACEStepFamily.aceStep, title: "ACE-Step", models: ["music-acestep"])],
        excludedModels: .models(
            aceStepLanguageModels,
            reason: "It is an ACE-Step language model; adapters train on the ACE-Step DiT (`music-acestep`)."
        )
    )
}
