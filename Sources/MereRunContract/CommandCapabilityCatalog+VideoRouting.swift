import Foundation

extension MereRunCapabilityCatalog {
    enum SCAIL2Family: String, MereRunFamilyID {
        case scail2
    }

    enum Cosmos3Family: String, MereRunFamilyID {
        case cosmos3
    }

    enum DubItFamily: String, MereRunFamilyID {
        case ltx25Distilled = "ltx25-distilled"
    }

    enum PrepareMasksFamily: String, MereRunFamilyID {
        case sam31
    }

    enum ExportLatentsFamily: String, MereRunFamilyID {
        case ltxMerged = "ltx-merged"
    }

    /// Managed models `video generate` runs, which no other video command accepts.
    private static let videoGenerationModels = [
        "video-ltx-av", "video-ltx23-av-mlx", "video-ltx23-full-mlx", "video-ltx23-a2vid-mlx",
        "video-ltx25-distilled-bf16", "video-ltx25-full-bf16", "video-wan22-ti2v-5b-mlx",
        "video-minimax-h3-fl2va-mlx", "video-minimax-h3-fl2va-bf16-mlx", "video-minimax-h3-fl2va-8bit-mlx",
        "video-minimax-h3-fasth3-vsa-datafree-mlx", "video-minimax-h3-ref2va-mlx"
    ]

    static let videoAnimateRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [.always("video-scail2-14b-mlx")],
        families: [.init(SCAIL2Family.scail2, title: "SCAIL-2", models: ["video-scail2-14b-mlx"])],
        excludedModels: .models(
            videoGenerationModels,
            reason: "It generates video from a prompt; use `video generate`."
        ).and(
            ["video-cosmos3-edge-mlx"],
            reason: "Cosmos 3 Edge has its own command; use `video cosmos3`."
        ).and(
            ["video-dreamx-world-5b-ar-mlx"],
            reason: "DreamX World is an interactive world model; use `world serve`."
        )
    )

    static let videoCosmos3Routing = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("video-cosmos3-edge-mlx")],
        families: [.init(Cosmos3Family.cosmos3, title: "Cosmos 3 Edge", models: ["video-cosmos3-edge-mlx"])]
    )

    static let videoDubItRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [.always("video-ltx25-distilled-bf16")],
        families: [
            .init(DubItFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: ["video-ltx25-distilled-bf16"])
        ]
    )

    static let videoPrepareMasksRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-segment-sam31")],
        families: [.init(PrepareMasksFamily.sam31, title: "SAM 3.1", models: ["vision-segment-sam31"])]
    )

    static let videoExportLatentsRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [.always("video-ltx-av")],
        families: [.init(ExportLatentsFamily.ltxMerged, title: "LTX (merged)", models: ["video-ltx-av"])],
        excludedModels: .models(
            ["video-ltx23-av-mlx", "video-ltx23-full-mlx", "video-ltx23-a2vid-mlx",
             "video-ltx25-distilled-bf16", "video-ltx25-full-bf16"],
            reason: "`video export-latents` requires the merged `video-ltx-av` layout."
        )
    )
}
