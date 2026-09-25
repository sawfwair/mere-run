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

// MARK: - video generate, retake, and session

extension MereRunCapabilityCatalog {
    /// The checkpoint layouts `video generate` runs (`VideoGenerationModelProfile`). FastH3 shares
    /// FL2VA's layout; its managed id selects the embedded FastH3 adapter.
    enum VideoGenerateFamily: String, MereRunFamilyID {
        case ltxMerged = "ltx-merged"
        case ltx23Distilled = "ltx23-distilled"
        case ltx23Full = "ltx23-full"
        case ltx23A2Vid = "ltx23-a2vid"
        case ltx25Distilled = "ltx25-distilled"
        case ltx25Full = "ltx25-full"
        case wan = "wan22-ti2v"
        case h3FL2VA = "h3-fl2va"
        case fastH3 = "h3-fast"
        case h3Ref2VA = "h3-ref2va"
    }

    enum VideoRetakeFamily: String, MereRunFamilyID {
        case ltx25Distilled = "ltx25-distilled"
        case ltx25Full = "ltx25-full"
    }

    enum VideoSessionFamily: String, MereRunFamilyID {
        case ltx23Distilled = "ltx23-distilled"
        case ltx23Full = "ltx23-full"
        case ltx25Distilled = "ltx25-distilled"
        case ltx25Full = "ltx25-full"
    }

    /// Video models with a command of their own.
    private static let otherVideoModels = [
        "video-cosmos3-edge-mlx", "video-scail2-14b-mlx", "video-dreamx-world-5b-ar-mlx"
    ]
    private static let otherVideoModelsReason =
        "It has its own command: `video cosmos3`, `video animate`, or `world serve`."

    /// `--model-root` names a folder whose files decide the family, and wins over `--model`. With
    /// neither, the options pick the checkpoint (`VideoGenerationOptions.resolvedRequestedModel`),
    /// in order: an LTX-2.5 Full recipe, an LTX-2.5 workflow, source audio or final quality, and
    /// otherwise the LTX-2.3 draft checkpoint.
    static let videoGenerateRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [
            MereRunDefaultModelRule(
                whenAny: [
                    .init(flag: "--dfr"),
                    .init(flag: "--ltx-preset", values: ["hq"]),
                    .init(flag: "--ltx-pipeline", values: ["keyframe-interpolation", "dev-one-stage"]),
                    .init(flag: "--ltx-sampler"),
                    .init(flag: "--distilled-lora-strength-stage-1"),
                    .init(flag: "--distilled-lora-strength-stage-2")
                ],
                models: ["video-ltx25-full-bf16"]
            ),
            MereRunDefaultModelRule(
                whenAny: [
                    .init(flag: "--hdr"),
                    .init(flag: "--high-quality-hdr"),
                    .init(flag: "--text-embeddings"),
                    .init(flag: "--enhance-prompt"),
                    .init(flag: "--auto-duration"),
                    .init(flag: "--video-decoder"),
                    .init(flag: "--image-conditioning"),
                    .init(flag: "--num-generated-keyframes", values: (1...16).map(String.init)),
                    .init(flag: "--generated-keyframe"),
                    .init(flag: "--video-conditioning")
                ],
                models: ["video-ltx25-distilled-bf16"]
            ),
            MereRunDefaultModelRule(
                whenAny: [
                    .init(flag: "--audio"),
                    .init(flag: "--quality", values: ["final"]),
                    .init(flag: "--variant", values: ["unified-av"])
                ],
                models: ["video-ltx23-full-mlx"]
            ),
            .always("video-ltx23-av-mlx")
        ],
        families: [
            .init(VideoGenerateFamily.ltxMerged, title: "LTX (merged)", models: ["video-ltx-av"]),
            .init(VideoGenerateFamily.ltx23Distilled, title: "LTX-2.3 Distilled", models: ["video-ltx23-av-mlx"]),
            .init(VideoGenerateFamily.ltx23Full, title: "LTX-2.3 Full", models: ["video-ltx23-full-mlx"]),
            .init(VideoGenerateFamily.ltx23A2Vid, title: "LTX-2.3 A2Vid", models: ["video-ltx23-a2vid-mlx"]),
            .init(VideoGenerateFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: ["video-ltx25-distilled-bf16"]),
            .init(VideoGenerateFamily.ltx25Full, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"]),
            .init(VideoGenerateFamily.wan, title: "Wan 2.2 TI2V", models: ["video-wan22-ti2v-5b-mlx"]),
            .init(
                VideoGenerateFamily.h3FL2VA, title: "MiniMax-H3 FL2VA",
                models: ["video-minimax-h3-fl2va-mlx", "video-minimax-h3-fl2va-bf16-mlx", "video-minimax-h3-fl2va-8bit-mlx"]
            ),
            .init(VideoGenerateFamily.fastH3, title: "MiniMax-H3 FastH3", models: ["video-minimax-h3-fasth3-vsa-datafree-mlx"]),
            .init(VideoGenerateFamily.h3Ref2VA, title: "MiniMax-H3 Ref2VA", models: ["video-minimax-h3-ref2va-mlx"])
        ],
        excludedModels: .models(otherVideoModels, reason: otherVideoModelsReason)
    )

    /// Retake runs any official LTX-2.5 folder, on the full lane when the full checkpoint
    /// validates. Every other checkpoint fails after resolution (`VideoRetakeCommand.run`).
    static let videoRetakeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [.always("video-ltx25-distilled-bf16")],
        families: [
            .init(VideoRetakeFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: ["video-ltx25-distilled-bf16"]),
            .init(VideoRetakeFamily.ltx25Full, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"])
        ],
        excludedModels: .models(
            videoGenerationModels.filter { !$0.hasPrefix("video-ltx25-") },
            reason: "`video retake` needs an official LTX-2.5 checkpoint."
        ).and(otherVideoModels, reason: otherVideoModelsReason)
    )

    /// The session keeps the split or full LTX-2.3 runtime, or either LTX-2.5 runtime, resident
    /// (`VideoSessionCommand.run`). `video-ltx-av` and the A2Vid id are neither listed nor
    /// excluded: both can fall back to an installed LTX-2.3 Full folder, which the session runs.
    static let videoSessionRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [.always("video-ltx23-av-mlx")],
        families: [
            .init(VideoSessionFamily.ltx23Distilled, title: "LTX-2.3 Distilled", models: ["video-ltx23-av-mlx"]),
            .init(VideoSessionFamily.ltx23Full, title: "LTX-2.3 Full", models: ["video-ltx23-full-mlx"]),
            .init(VideoSessionFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: ["video-ltx25-distilled-bf16"]),
            .init(VideoSessionFamily.ltx25Full, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"])
        ],
        excludedModels: .models(
            videoGenerationModels.filter { $0.hasPrefix("video-wan") || $0.hasPrefix("video-minimax-h3-") },
            reason: "The resident session runs LTX-2.3 and LTX-2.5 checkpoints; use `video generate`."
        ).and(otherVideoModels, reason: otherVideoModelsReason)
    )
}
