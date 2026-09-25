import Foundation

extension MereRunCapabilityCatalog {
    enum QwenVLFamily: String, MereRunFamilyID {
        case qwen3VL = "qwen3-vl"
    }

    enum FalconPerceptionFamily: String, MereRunFamilyID {
        case falconPerception = "falcon-perception"
    }

    enum SAM31Family: String, MereRunFamilyID {
        case sam31
    }

    enum BuffaloFamily: String, MereRunFamilyID {
        case buffaloL = "buffalo-l"
    }

    enum MarigoldFamily: String, MereRunFamilyID {
        case marigoldV2 = "marigold-v2"
    }

    enum VideoDepthAnythingFamily: String, MereRunFamilyID {
        case videoDepthAnything = "video-depth-anything"
    }

    enum MoGe2Family: String, MereRunFamilyID {
        case moge2
    }

    enum DepthAnything3Family: String, MereRunFamilyID {
        case depthAnything3 = "depth-anything-3"
    }

    enum QwenVLEmbeddingFamily: String, MereRunFamilyID {
        case qwen3VLEmbedding = "qwen3-vl-embedding"
    }

    private static let videoDepthModels = ["vision-depth-vda-small", "vision-depth-vda-small-metric"]

    /// `--model` takes only a local Qwen3-VL root; without one the CLI downloads its default by
    /// repository, so the family lists no managed model.
    static let qwenVLRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.unmanaged(QwenVLFamily.qwen3VL)],
        families: [.init(QwenVLFamily.qwen3VL, title: "Qwen3-VL", models: [])]
    )

    static let visionGroundRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-ground-falcon-perception")],
        families: [
            .init(
                FalconPerceptionFamily.falconPerception,
                title: "Falcon Perception",
                models: ["vision-ground-falcon-perception"]
            )
        ]
    )

    static let sam31Routing = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-segment-sam31")],
        families: [.init(SAM31Family.sam31, title: "SAM 3.1", models: ["vision-segment-sam31"])]
    )

    static let visionFaceRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-face-buffalo-l")],
        families: [.init(BuffaloFamily.buffaloL, title: "Buffalo-L", models: ["vision-face-buffalo-l"])]
    )

    static let visionDepthRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-depth-marigold-v2")],
        families: [.init(MarigoldFamily.marigoldV2, title: "Marigold v2", models: ["vision-depth-marigold-v2"])],
        excludedModels: .models(
            videoDepthModels,
            reason: "Video Depth Anything estimates depth across video frames; use `vision depth-video`."
        )
    )

    static let visionDepthVideoRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-depth-vda-small")],
        families: [
            .init(
                VideoDepthAnythingFamily.videoDepthAnything,
                title: "Video Depth Anything",
                models: videoDepthModels
            )
        ],
        excludedModels: .models(
            ["vision-depth-marigold-v2"],
            reason: "Marigold v2 estimates single-image depth; use `vision depth`."
        )
    )

    static let visionGeometryRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-geometry-moge2-small")],
        families: [.init(MoGe2Family.moge2, title: "MoGe-2", models: ["vision-geometry-moge2-small"])],
        excludedModels: .models(
            ["vision-geometry-da3-small"],
            reason: "Depth Anything 3 reconstructs several views together; use `vision geometry-multiview`."
        )
    )

    static let visionGeometryMultiviewRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-geometry-da3-small")],
        families: [
            .init(
                DepthAnything3Family.depthAnything3,
                title: "Depth Anything 3",
                models: ["vision-geometry-da3-small"]
            )
        ],
        excludedModels: .models(
            ["vision-geometry-moge2-small"],
            reason: "MoGe-2 reconstructs one image; use `vision geometry`."
        )
    )

    static let visionEmbedRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-embed-qwen3-vl-2b")],
        families: [
            .init(
                QwenVLEmbeddingFamily.qwen3VLEmbedding,
                title: "Qwen3-VL Embedding",
                models: ["vision-embed-qwen3-vl-2b"]
            )
        ],
        excludedModels: .models(
            ["vision-embed-tessera-v2-nano", "vision-embed-tessera-v2-small", "vision-embed-tessera-v2-medium",
             "vision-embed-tessera-v2-large", "vision-embed-tessera-v2-teacher"],
            reason: "TESSERA embeds satellite time series; use `geo tessera`."
        ).and(
            ["vision-embed-olmoearth-v12-nano", "vision-embed-olmoearth-v12-tiny", "vision-embed-olmoearth-v12-small",
             "vision-embed-olmoearth-v12-base"],
            reason: "OlmoEarth embeds Earth-observation imagery; use `geo olmoearth`."
        )
    )
}
