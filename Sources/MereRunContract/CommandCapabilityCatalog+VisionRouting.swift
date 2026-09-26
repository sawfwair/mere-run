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

// MARK: - OCR

extension MereRunCapabilityCatalog {
    /// `vision ocr` runs by `--backend`, `--infinity-runtime`, and `--compare`, not by model.
    /// `--compare` runs LightOnOCR and then GLM-OCR (for `--backend lighton` or `glm`) or
    /// Infinity-Parser2, so each comparison is its own family and reads both runtimes' options.
    enum OCRFamily: String, MereRunFamilyID {
        case lightOn = "lighton"
        case glm
        case infinityNative = "infinity-native"
        case infinityExternal = "infinity-external"
        case compareGLM = "compare-glm"
        case compareInfinityNative = "compare-infinity-native"
        case compareInfinityExternal = "compare-infinity-external"

        /// `vision ocr` refuses no option another runtime reads: the families that don't read it
        /// accept it and ignore it.
        static func usedBy(_ used: [Self]) -> MereRunOptionScope<Self> {
            MereRunOptionScope(families: used, ignoredBy: allCases.filter { !used.contains($0) })
        }

        static let lightOnRuns: [Self] = [.lightOn, .compareGLM, .compareInfinityNative, .compareInfinityExternal]
        static let infinityRuns: [Self] = [
            .infinityNative, .infinityExternal, .compareInfinityNative, .compareInfinityExternal
        ]
        static let glmRuns: [Self] = [.glm, .compareGLM]
        static let externalInfinityRuns: [Self] = [.infinityExternal, .compareInfinityExternal]
    }

    private static let lightOnOCRModel = "vision-ocr-lighton"
    private static let infinityOCRModels = ["vision-ocr-infinity-pro-int8", "vision-ocr-infinity-pro"]

    /// LightOnOCR reads its model from `--model` and native Infinity-Parser2 from
    /// `--infinity-model`; GLM-OCR and external Infinity-Parser2 run external tools.
    static let visionOCRRouting: MereRunCapabilityRouting = {
        typealias F = OCRFamily
        let single = MereRunFlagCondition(flag: "--compare", values: ["false"])
        let compare = MereRunFlagCondition(flag: "--compare", values: ["true"])
        let infinity = MereRunFlagCondition(flag: "--backend", values: ["infinity"])
        let native = MereRunFlagCondition(flag: "--infinity-runtime", values: ["native"])
        let external = MereRunFlagCondition(flag: "--infinity-runtime", values: ["external"])
        return MereRunCapabilityRouting(
            modelFlags: [],
            defaultModels: F.lightOnRuns.map { MereRunDefaultModelRule(models: [lightOnOCRModel], family: $0.rawValue) }
                + [MereRunDefaultModelRule(models: [infinityOCRModels[0]], family: F.infinityNative.rawValue)],
            families: [
                .init(F.lightOn, title: "LightOnOCR", models: [lightOnOCRModel],
                      selectors: [single, .init(flag: "--backend", values: ["lighton"])], modelFlag: "--model"),
                .init(F.glm, title: "GLM-OCR", models: [], selectors: [single, .init(flag: "--backend", values: ["glm"])]),
                .init(F.infinityNative, title: "Infinity-Parser2 native", models: infinityOCRModels,
                      selectors: [single, infinity, native], modelFlag: "--infinity-model"),
                .init(F.infinityExternal, title: "Infinity-Parser2 external", models: [],
                      selectors: [single, infinity, external]),
                .init(F.compareGLM, title: "LightOnOCR vs GLM-OCR", models: [lightOnOCRModel],
                      selectors: [compare, .init(flag: "--backend", values: ["lighton", "glm"])], modelFlag: "--model"),
                .init(F.compareInfinityNative, title: "LightOnOCR vs Infinity-Parser2 native", models: [lightOnOCRModel],
                      selectors: [compare, infinity, native], modelFlag: "--model"),
                .init(F.compareInfinityExternal, title: "LightOnOCR vs Infinity-Parser2 external",
                      models: [lightOnOCRModel], selectors: [compare, infinity, external], modelFlag: "--model")
            ]
        )
    }()
}
