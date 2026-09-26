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

    /// `music generate` runtimes, in the CLI's routing order: YuE2, MiniMax Music 3, and Magenta
    /// RealTime 2 by managed id or local layout, and ACE-Step otherwise. The ACE-Step families
    /// follow the decoder checkpoint the CLI loads: Turbo forces guidance off, and only Base runs
    /// the extract, lego, and complete tasks.
    enum MusicGenerateFamily: String, MereRunFamilyID {
        case aceStepTurbo = "ace-step-turbo"
        case aceStepSFT = "ace-step-sft"
        case aceStepBase = "ace-step-base"
        case miniMaxMusic3 = "minimax-music3"
        case yue2
        case magentaRT2 = "magenta-rt2"

        /// The ACE-Step checkpoint families, for `ignoredBy` lists.
        static let aceStep: [Self] = [.aceStepTurbo, .aceStepSFT, .aceStepBase]
    }

    enum MusicSeparateFamily: String, MereRunFamilyID {
        case bsRoFormer2Stem = "bs-roformer-2stem"
        case bsRoFormer4Stem = "bs-roformer-4stem"
        case melRoFormerDereverb = "mel-roformer-dereverb"
        case melRoFormerDenoise = "mel-roformer-denoise"
    }

    enum MusicServeFamily: String, MereRunFamilyID {
        case aceStep = "ace-step"
        case miniMaxMusic3 = "minimax-music3"
    }

    private static let aceStepLanguageModels = ["music-acestep-lm-1.7b", "music-acestep-lm-4b"]

    private static let aceStepGenerateModels = [
        "music-acestep", "music-acestep-xl-turbo", "music-acestep-xl-turbo-lm4b", "music-acestep-xl-sft",
        "music-acestep-xl-base"
    ]

    static let musicGenerateRouting = MereRunCapabilityRouting(
        // ACE-Step loads the checkpoint under `--checkpoints-root` before the model's own, and the
        // decoder `--decoder-subdirectory` names; the CLI identifies either from the files.
        modelFlags: ["--checkpoints-root", "--decoder-subdirectory", "--model"],
        defaultModels: [.always("music-acestep")],
        families: [
            .init(MusicGenerateFamily.aceStepTurbo, title: "ACE-Step Turbo",
                  models: ["music-acestep", "music-acestep-xl-turbo", "music-acestep-xl-turbo-lm4b"]),
            .init(MusicGenerateFamily.aceStepSFT, title: "ACE-Step SFT", models: ["music-acestep-xl-sft"]),
            .init(MusicGenerateFamily.aceStepBase, title: "ACE-Step Base", models: ["music-acestep-xl-base"]),
            .init(MusicGenerateFamily.miniMaxMusic3, title: "MiniMax Music 3", models: ["music-minimax-music3"]),
            .init(MusicGenerateFamily.yue2, title: "YuE2", models: ["music-yue2"]),
            .init(MusicGenerateFamily.magentaRT2, title: "Magenta RealTime 2",
                  models: ["music-magenta-rt2-small", "music-magenta-rt2-base"])
        ],
        excludedModels: .models(
            aceStepLanguageModels,
            reason: "It is an ACE-Step language model; pass it as `--lm-model`."
        ),
        // `MERERUN_MUSIC_ACESTEP_ROOT` wins over a managed ACE-Step id's own install, so the CLI
        // identifies the checkpoint it would load before trusting the id. The same root runs in
        // place of a language model id, which is excluded only when no root loads first.
        identifiedModels: aceStepGenerateModels + aceStepLanguageModels
    )

    static let musicSeparateRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("music-separate-bs-roformer-viperx-1297")],
        families: [
            .init(MusicSeparateFamily.bsRoFormer2Stem, title: "BS-RoFormer ViperX",
                  models: ["music-separate-bs-roformer-viperx-1297"]),
            .init(MusicSeparateFamily.bsRoFormer4Stem, title: "BS-RoFormer 4-stem",
                  models: ["music-separate-bs-roformer-4stem"]),
            .init(MusicSeparateFamily.melRoFormerDereverb, title: "MelBand RoFormer Dereverb",
                  models: ["music-separate-mel-roformer-dereverb"]),
            .init(MusicSeparateFamily.melRoFormerDenoise, title: "MelBand RoFormer Denoise",
                  models: ["music-separate-mel-roformer-denoise"])
        ]
    )

    static let musicServeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("music-acestep")],
        families: [
            .init(MusicServeFamily.aceStep, title: "ACE-Step", models: aceStepGenerateModels),
            .init(MusicServeFamily.miniMaxMusic3, title: "MiniMax Music 3", models: ["music-minimax-music3"])
        ],
        excludedModels: .models(
            aceStepLanguageModels,
            reason: "It is an ACE-Step language model; pass it as `--lm-model`."
        ),
        identifiedModels: aceStepLanguageModels
    )

    /// Every chunk overlap a RoFormer model accepts: the positive divisors of its chunk size in
    /// samples (`Resources/RoFormer/*.json` in Core), so each hop starts on a whole sample.
    static func roFormerOverlaps(chunkSize: Int) -> [String] {
        (1...chunkSize).filter { chunkSize.isMultiple(of: $0) }.map(String.init)
    }

    /// Analysis runs any ACE-Step checkpoint, and adds the 1.7B planner when the checkpoint has
    /// no language model of its own.
    static let musicAnalyzeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("music-acestep")],
        families: [.init(ACEStepFamily.aceStep, title: "ACE-Step", models: aceStepGenerateModels)],
        excludedModels: .models(
            aceStepLanguageModels,
            reason: "It is an ACE-Step language model; pass it as `--lm-model`."
        ),
        identifiedModels: aceStepLanguageModels
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
        ],
        listingFlags: ["--list-instruments"]
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
        ),
        listingFlags: ["--list-midi-inputs"]
    )

    /// Adapters train on any ACE-Step DiT checkpoint.
    static let musicTrainAdapterRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("music-acestep")],
        families: [.init(ACEStepFamily.aceStep, title: "ACE-Step", models: aceStepGenerateModels)],
        excludedModels: .models(
            aceStepLanguageModels,
            reason: "It is an ACE-Step language model; adapters train on the ACE-Step DiT (`music-acestep`)."
        ),
        identifiedModels: aceStepLanguageModels
    )
}

extension MereRunCapabilityOption {
    /// `scoped(scope, rules...)`, where each of `defaultOnly` (families the scope lists as ignoring
    /// the option) accepts only the option's `default_value` and refuses every other value, the
    /// way MiniMax Music 3 and YuE2 treat ACE-Step's options.
    func scoped<Family: MereRunFamilyID>(
        _ scope: MereRunOptionScope<Family>,
        onlyDefaultFor defaultOnly: [Family],
        _ rules: MereRunOptionRule<Family>...
    ) -> Self {
        let scoped = self.scoped(scope)
        let defaults = defaultOnly.map { family in
            MereRunOptionFamilyRule(family: family.rawValue, values: defaultValue.map { [$0] })
        }
        return scoped.with(families: scoped.families, ignoredBy: scoped.ignoredBy, rules: rules.map(\.rule) + defaults)
    }
}
