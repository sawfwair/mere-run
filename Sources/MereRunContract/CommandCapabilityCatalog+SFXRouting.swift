import Foundation

extension MereRunCapabilityCatalog {
    enum SFXGenerateFamily: String, MereRunFamilyID {
        case wooshDFlow = "woosh-dflow"
        case wooshFlow = "woosh-flow"
        case mmaudio
    }

    enum SFXVideoGenerateFamily: String, MereRunFamilyID {
        case wooshDVFlow = "woosh-dvflow"
        case wooshVFlow = "woosh-vflow"
        case mmaudio
    }

    enum WooshCLAPFamily: String, MereRunFamilyID {
        case wooshCLAP = "woosh-clap"
    }

    private static let mmaudioModel = "sfx-mmaudio-large-44k-v2"
    private static let wooshCLAPReason = "Woosh CLAP scores audio against a prompt; use `sfx clap score`."

    /// An exact id, or a local folder: MMAudio when it holds the MMAudio network weights, else the
    /// Woosh variant whose `checkpoints/<variant>/config.yaml` it holds.
    static let sfxGenerateRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("sfx-woosh-dflow")],
        families: [
            .init(SFXGenerateFamily.wooshDFlow, title: "Woosh DFlow", models: ["sfx-woosh-dflow"]),
            .init(SFXGenerateFamily.wooshFlow, title: "Woosh Flow", models: ["sfx-woosh-flow"]),
            .init(SFXGenerateFamily.mmaudio, title: "MMAudio", models: [mmaudioModel])
        ],
        excludedModels: .models(
            ["sfx-woosh-vflow-8s", "sfx-woosh-dvflow-8s"],
            reason: "It generates sound from video; use `sfx video generate`."
        ).and(["sfx-woosh-clap"], reason: wooshCLAPReason)
            .and(
                ["sfx-woosh-synchformer"],
                reason: "It extracts video features for `sfx video generate`, which takes it as `--synchformer-model`."
            )
    )

    /// Resolved like `sfx generate`; only the video-to-audio Woosh variants and MMAudio run it.
    static let sfxVideoGenerateRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("sfx-woosh-dvflow-8s")],
        families: [
            .init(SFXVideoGenerateFamily.wooshDVFlow, title: "Woosh DVFlow", models: ["sfx-woosh-dvflow-8s"]),
            .init(SFXVideoGenerateFamily.wooshVFlow, title: "Woosh VFlow", models: ["sfx-woosh-vflow-8s"]),
            .init(SFXVideoGenerateFamily.mmaudio, title: "MMAudio", models: [mmaudioModel])
        ],
        excludedModels: .models(
            ["sfx-woosh-dflow", "sfx-woosh-flow"],
            reason: "It generates sound from text; use `sfx generate`."
        ).and(["sfx-woosh-clap"], reason: wooshCLAPReason)
            .and(["sfx-woosh-synchformer"], reason: "It extracts video features; pass it as `--synchformer-model`.")
    )

    static let sfxCLAPScoreRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("sfx-woosh-clap")],
        families: [.init(WooshCLAPFamily.wooshCLAP, title: "Woosh CLAP", models: ["sfx-woosh-clap"])],
        excludedModels: .models(
            ["sfx-woosh-dflow", "sfx-woosh-flow"],
            reason: "It generates sound from text; use `sfx generate`."
        ).and(
            ["sfx-woosh-vflow-8s", "sfx-woosh-dvflow-8s", mmaudioModel],
            reason: "It generates sound effects; use `sfx generate` or `sfx video generate`."
        ).and(["sfx-woosh-synchformer"], reason: "It extracts video features for `sfx video generate`.")
    )
}
