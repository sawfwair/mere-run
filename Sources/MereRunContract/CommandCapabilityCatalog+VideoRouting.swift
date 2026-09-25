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
        case ltx25Full = "ltx25-full"
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
            .init(DubItFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: ["video-ltx25-distilled-bf16"]),
            .init(DubItFamily.ltx25Full, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"])
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
        /// An LTX-2.5 Distilled folder that also holds the diffusion video decoder, which the
        /// managed checkpoint does not install: `--video-decoder diffusion` runs it.
        case ltx25DistilledDiffusion = "ltx25-distilled-diffusion"
        case ltx25Full = "ltx25-full"
        case wan = "wan22-ti2v"
        case h3FL2VA = "h3-fl2va"
        /// The legacy 4-bit FL2VA checkpoint: FL2VA without Turbo adapters, which need BF16 or Q8.
        case h3FL2VAQ4 = "h3-fl2va-q4"
        case fastH3 = "h3-fast"
        /// The FastH3 id with an explicit `--h3-adapter`, which replaces the embedded adapter and
        /// its fixed recipe; it runs like FL2VA.
        case fastH3Adapter = "h3-fast-adapter"
        case h3Ref2VA = "h3-ref2va"
    }

    enum VideoRetakeFamily: String, MereRunFamilyID {
        case ltx25Distilled = "ltx25-distilled"
        /// An LTX-2.5 Distilled folder that also holds the diffusion video decoder, which the
        /// managed checkpoint does not install: `--video-decoder diffusion` runs it.
        case ltx25DistilledDiffusion = "ltx25-distilled-diffusion"
        case ltx25Full = "ltx25-full"
    }

    enum VideoSessionFamily: String, MereRunFamilyID {
        case ltx23Distilled = "ltx23-distilled"
        case ltx23Full = "ltx23-full"
        case ltx25Distilled = "ltx25-distilled"
        /// An LTX-2.5 Distilled folder that also holds the diffusion video decoder, which the
        /// managed checkpoint does not install: `--video-decoder diffusion` runs it.
        case ltx25DistilledDiffusion = "ltx25-distilled-diffusion"
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
                    .atLeast("--num-generated-keyframes", 1),
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
            .init(VideoGenerateFamily.ltxMerged, title: "LTX (merged)", models: []),
            .init(VideoGenerateFamily.ltx23Distilled, title: "LTX-2.3 Distilled", models: ["video-ltx23-av-mlx"]),
            .init(VideoGenerateFamily.ltx23Full, title: "LTX-2.3 Full", models: []),
            .init(VideoGenerateFamily.ltx23A2Vid, title: "LTX-2.3 A2Vid", models: []),
            .init(VideoGenerateFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: [ltx25DistilledModel]),
            .init(VideoGenerateFamily.ltx25DistilledDiffusion, title: "LTX-2.5 Distilled with the diffusion decoder", models: []),
            .init(VideoGenerateFamily.ltx25Full, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"]),
            .init(VideoGenerateFamily.wan, title: "Wan 2.2 TI2V", models: ["video-wan22-ti2v-5b-mlx"]),
            .init(
                VideoGenerateFamily.h3FL2VA, title: "MiniMax-H3 FL2VA",
                models: ["video-minimax-h3-fl2va-bf16-mlx", "video-minimax-h3-fl2va-8bit-mlx"]
            ),
            .init(VideoGenerateFamily.h3FL2VAQ4, title: "MiniMax-H3 FL2VA 4-bit", models: ["video-minimax-h3-fl2va-mlx"]),
            .init(
                VideoGenerateFamily.fastH3, title: "MiniMax-H3 FastH3", models: [fastH3Model],
                selectors: [.absent("--h3-adapter")]
            ),
            .init(
                VideoGenerateFamily.fastH3Adapter, title: "MiniMax-H3 FastH3 with an adapter", models: [fastH3Model],
                selectors: [.init(flag: "--h3-adapter")]
            ),
            .init(VideoGenerateFamily.h3Ref2VA, title: "MiniMax-H3 Ref2VA", models: ["video-minimax-h3-ref2va-mlx"])
        ],
        excludedModels: .models(otherVideoModels, reason: otherVideoModelsReason),
        identifiedModels: ltx23InstallDependentModels + [ltx25DistilledModel]
    )

    private static let fastH3Model = "video-minimax-h3-fasth3-vsa-datafree-mlx"

    /// The ids whose checkpoint depends on what is installed: `video-ltx-av` can run a suggested
    /// LTX 2.3 folder (`VideoGenerationModelResolver.location`), and the LTX 2.3 Full and A2Vid
    /// ids fall back to each other's installs. The CLI identifies the folder each will run.
    private static let ltx23InstallDependentModels = ["video-ltx-av", "video-ltx23-full-mlx", "video-ltx23-a2vid-mlx"]

    /// The managed LTX-2.5 Distilled checkpoint installs only the convolutional decoder, but its
    /// installed folder runs the diffusion decoder once it holds one. The id is listed and also
    /// identified: the CLI places an install that holds the decoder in the diffusion family, and
    /// without its answer (nothing installed, or a shell that has not asked) the listed family
    /// stands.
    private static let ltx25DistilledModel = "video-ltx25-distilled-bf16"

    /// Retake runs any official LTX-2.5 folder, on the full lane when the full checkpoint
    /// validates. Every other checkpoint fails after resolution (`VideoRetakeCommand.run`).
    /// `video-ltx-av` resolves to a suggested folder, which `MERERUN_VIDEO_LTX_MODEL_ROOT` can
    /// point at an LTX-2.5 install, so the CLI identifies it, as it does the LTX-2.5 Distilled id;
    /// the LTX 2.3 ids only ever land on LTX 2.3 folders.
    static let videoRetakeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [.always("video-ltx25-distilled-bf16")],
        families: [
            .init(VideoRetakeFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: [ltx25DistilledModel]),
            .init(VideoRetakeFamily.ltx25DistilledDiffusion, title: "LTX-2.5 Distilled with the diffusion decoder", models: []),
            .init(VideoRetakeFamily.ltx25Full, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"])
        ],
        excludedModels: .models(
            videoGenerationModels.filter { !$0.hasPrefix("video-ltx25-") && $0 != "video-ltx-av" },
            reason: "`video retake` needs an official LTX-2.5 checkpoint."
        ).and(otherVideoModels, reason: otherVideoModelsReason),
        identifiedModels: ["video-ltx-av", ltx25DistilledModel]
    )

    /// The session keeps the split or full LTX-2.3 runtime, or either LTX-2.5 runtime, resident
    /// (`VideoSessionCommand.run`). The install-dependent LTX 2.3 ids are identified by the
    /// folder they resolve to, which the session may or may not run.
    static let videoSessionRouting = MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [.always("video-ltx23-av-mlx")],
        families: [
            .init(VideoSessionFamily.ltx23Distilled, title: "LTX-2.3 Distilled", models: ["video-ltx23-av-mlx"]),
            .init(VideoSessionFamily.ltx23Full, title: "LTX-2.3 Full", models: []),
            .init(VideoSessionFamily.ltx25Distilled, title: "LTX-2.5 Distilled", models: [ltx25DistilledModel]),
            .init(VideoSessionFamily.ltx25DistilledDiffusion, title: "LTX-2.5 Distilled with the diffusion decoder", models: []),
            .init(VideoSessionFamily.ltx25Full, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"])
        ],
        excludedModels: .models(
            videoGenerationModels.filter { $0.hasPrefix("video-wan") || $0.hasPrefix("video-minimax-h3-") },
            reason: "The resident session runs LTX-2.3 and LTX-2.5 checkpoints; use `video generate`."
        ).and(otherVideoModels, reason: otherVideoModelsReason),
        identifiedModels: ltx23InstallDependentModels + [ltx25DistilledModel]
    )
}
