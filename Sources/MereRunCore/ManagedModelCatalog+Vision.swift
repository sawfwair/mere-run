import Foundation

extension ManagedModelCatalog {
    static let visionSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: Q35Resources.infinityParser2ProModelId,
            category: .visionOCR,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.infinityParser2ProModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.infinityParser2ProUpstreamRepoId,
            upstreamRevision: Q35Resources.infinityParser2ProUpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 131 * 1_073_741_824,
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: Q35Resources.infinityParser2ProInt8ModelId,
            category: .visionOCR,
            installShape: .directoryRoot,
            hubFallback: Q35Resources.profile(for: Q35Resources.infinityParser2ProInt8ModelId)?.hubFallbackConfig,
            upstreamRepoId: Q35Resources.infinityParser2ProInt8UpstreamRepoId,
            upstreamRevision: Q35Resources.infinityParser2ProInt8UpstreamRevision,
            validationKind: .q35,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 38 * 1_073_741_824,
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: "vision-ocr-lighton",
            category: .visionOCR,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "lightonai/LightOnOCR-2-1B",
                revision: "main",
                patterns: [
                    "added_tokens.json",
                    "chat_template.jinja",
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "processor_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ]
            ),
            upstreamRepoId: "lightonai/LightOnOCR-2-1B",
            upstreamRevision: "main",
            validationKind: .lightOnOCR,
            estimatedDownloadBytes: 2_022_801_518,
            defaultCLICommands: ["vision ocr"]
        ),
        ManagedModelSpec(
            id: "vision-segment-sam31",
            category: .visionSegment,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "mlx-community/sam3.1-bf16",
                revision: sam31MLXRevision,
                patterns: [
                    "LICENSE*",
                    "README.md",
                    "config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "tokenizer",
                    hubFallback: HubFallbackConfig(
                        repoId: "AEmotionStudio/sam3.1",
                        revision: sam31TokenizerRevision,
                        patterns: [
                            "LICENSE*",
                            "README.md",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "vocab.json",
                            "merges.txt",
                            "special_tokens_map.json",
                        ]
                    )
                ),
            ],
            upstreamRepoId: "mlx-community/sam3.1-bf16",
            upstreamRevision: sam31MLXRevision,
            usageRestriction: usageRestriction(
                summary: "SAM 3.1 uses Meta's custom SAM License, including trade-control, prohibited-use, redistribution, and research-attribution conditions.",
                license: "SAM License",
                sourceRepoId: "mlx-community/sam3.1-bf16",
                sourceRevision: sam31MLXRevision,
                licenseURL: "https://huggingface.co/facebook/sam3.1/blob/main/LICENSE"
            ),
            validationKind: .sam31,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 3_498_072_777,
            defaultCLICommands: ["vision segment"]
        ),
        ManagedModelSpec(
            id: "vision-ground-falcon-perception",
            category: .visionGround,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "tiiuae/Falcon-Perception",
                revision: "main",
                patterns: [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "generation_config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "*.safetensors",
                ]
            ),
            upstreamRepoId: "tiiuae/Falcon-Perception",
            upstreamRevision: "main",
            validationKind: .falconPerception,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 2_534_591_776,
            defaultCLICommands: ["vision ground"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionFloodTerraMindBase.rawValue,
            category: .visionFlood,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: TerraMindFloodResources.sourceRepository,
                revision: TerraMindFloodResources.sourceRevision,
                patterns: [
                    "README.md",
                    "LICENSE*",
                    "NOTICE*",
                    TerraMindFloodResources.sourceCheckpointFilename,
                    TerraMindFloodResources.sourceConfigurationFilename,
                ]
            ),
            upstreamRepoId: TerraMindFloodResources.sourceRepository,
            upstreamRevision: TerraMindFloodResources.sourceRevision,
            validationKind: .terramindFlood,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 673_199_088,
            defaultCLICommands: ["geo flood"]
        ),
        ManagedModelSpec(
            id: FaceAnalysisResources.modelID,
            category: .visionFace,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "deepghs/insightface",
                revision: "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb",
                patterns: [
                    "LICENSE*",
                    "README.md",
                    FaceAnalysisResources.detectorRelativePath,
                    FaceAnalysisResources.recognizerRelativePath,
                ]
            ),
            upstreamRepoId: "deepghs/insightface",
            upstreamRevision: "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb",
            usageRestriction: usageRestriction(
                summary: "InsightFace Buffalo-L pretrained weights are limited to non-commercial research use.",
                license: "InsightFace pretrained model non-commercial research terms",
                sourceRepoId: "deepghs/insightface",
                sourceRevision: "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb",
                licenseURL: "https://github.com/deepinsight/insightface#license"
            ),
            validationKind: .insightFaceBuffaloL,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: FaceAnalysisResources.detectorByteCount
                + FaceAnalysisResources.recognizerByteCount,
            defaultCLICommands: [
                "vision face detect",
                "vision face embed",
                "vision face compare",
                "vision face batch",
            ]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
            category: .visionGeometry,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "Ruicheng/moge-2-vits-normal-onnx",
                revision: "e50ffda41565591092adea54c6ac83d6212e1e23",
                patterns: ["model.onnx", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "Ruicheng/moge-2-vits-normal-onnx",
            upstreamRevision: "e50ffda41565591092adea54c6ac83d6212e1e23",
            validationKind: .moge2,
            estimatedDownloadBytes: 140_852_051,
            defaultCLICommands: ["vision geometry"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionDepthVDASmall.rawValue,
            category: .visionDepth,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "depth-anything/Video-Depth-Anything-Small",
                revision: "256875362cff76724b920335dfb4b29dd611f66e",
                patterns: ["video_depth_anything_vits.pth", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "depth-anything/Video-Depth-Anything-Small",
            upstreamRevision: "256875362cff76724b920335dfb4b29dd611f66e",
            validationKind: .videoDepthAnything,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 116_440_756,
            defaultCLICommands: ["vision depth-video"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
            category: .visionDepth,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "depth-anything/Metric-Video-Depth-Anything-Small",
                revision: "273d090f2ce17df50c2872d82c8322c45da5b4dd",
                patterns: ["metric_video_depth_anything_vits.pth", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "depth-anything/Metric-Video-Depth-Anything-Small",
            upstreamRevision: "273d090f2ce17df50c2872d82c8322c45da5b4dd",
            validationKind: .videoDepthAnything,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 116_444_063,
            defaultCLICommands: ["vision depth-video"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionGeometryDA3Small.rawValue,
            category: .visionGeometry,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "depth-anything/DA3-SMALL",
                revision: "e08cab65ca0ec38e7826075418411ab90cab4da3",
                patterns: ["config.json", "model.safetensors", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "depth-anything/DA3-SMALL",
            upstreamRevision: "e08cab65ca0ec38e7826075418411ab90cab4da3",
            validationKind: .depthAnything3,
            estimatedDownloadBytes: 137_248_940,
            defaultCLICommands: ["vision geometry-multiview"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.image3DTripoSR.rawValue,
            category: .image3D,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "stabilityai/TripoSR",
                revision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
                patterns: ["config.yaml", "model.ckpt", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "stabilityai/TripoSR",
            upstreamRevision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
            validationKind: .tripoSR,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_677_247_729,
            defaultCLICommands: ["image reconstruct-3d", "vision image-to-3d"]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.image3DInstantMeshBase.rawValue,
            category: .image3D,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: "TencentARC/InstantMesh",
                revision: "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
                patterns: ["instant_mesh_base.ckpt", "LICENSE*", "NOTICE*"]
            ),
            upstreamRepoId: "TencentARC/InstantMesh",
            upstreamRevision: "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
            validationKind: .instantMesh,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 1_253_574_354,
            defaultCLICommands: [
                "image reconstruct-3d-multiview",
                "vision image-to-3d-multiview",
            ]
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.image3DTrellis2.rawValue,
            category: .image3D,
            installShape: .structuredRoot,
            hubFallback: Trellis2Resources.primaryHubFallback,
            mountedHubFallbacks: Trellis2Resources.mountedHubFallbacks,
            upstreamRepoId: Trellis2Resources.repository,
            upstreamRevision: Trellis2Resources.revision,
            usageRestriction: usageRestriction(
                summary: "TRELLIS.2 downloads a manually gated DINOv3 component governed by Meta's custom DINOv3 License.",
                component: "DINOv3 image encoder",
                license: "DINOv3 License",
                sourceRepoId: Trellis2Resources.dinoV3Repository,
                sourceRevision: Trellis2Resources.dinoV3Revision,
                licenseURL: "https://ai.meta.com/resources/models-and-libraries/dinov3-license/"
            ),
            validationKind: .trellis2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 11_010_775_158,
            defaultCLICommands: [
                "image reconstruct-3d-trellis2",
                "vision image-to-3d-trellis2",
            ]
        ),
    ]

    static let geoExpansionSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionFireTerraMindBase.rawValue,
            category: .visionFire,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: TerraMindFireResources.sourceRepository,
                revision: TerraMindFireResources.sourceRevision,
                patterns: [
                    "README.md",
                    "LICENSE*",
                    "NOTICE*",
                    TerraMindFireResources.sourceCheckpointFilename,
                    TerraMindFireResources.sourceConfigurationFilename,
                ]
            ),
            upstreamRepoId: TerraMindFireResources.sourceRepository,
            upstreamRevision: TerraMindFireResources.sourceRevision,
            validationKind: .terramindFire,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 673_193_610,
            defaultCLICommands: ["geo fire"]
        ),
    ] + TESSERAResources.allSpecs.map { source in
        ManagedModelSpec(
            id: source.modelID,
            category: .visionEmbed,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: source.sourceRepository,
                revision: source.sourceRevision,
                patterns: ["README.md", source.sourceCheckpointFilename]
            ),
            upstreamRepoId: source.sourceRepository,
            upstreamRevision: source.sourceRevision,
            validationKind: .tessera,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: source.sourceCheckpointByteCount,
            defaultCLICommands: ["geo tessera"]
        )
    } + OlmoEarthResources.allSpecs.map { source in
        ManagedModelSpec(
            id: source.modelID,
            category: .visionEmbed,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: source.sourceRepository,
                revision: source.sourceRevision,
                patterns: [
                    "README.md",
                    "LICENSE*",
                    OlmoEarthResources.sourceWeightsFilename,
                    OlmoEarthResources.sourceConfigurationFilename,
                ]
            ),
            upstreamRepoId: source.sourceRepository,
            upstreamRevision: source.sourceRevision,
            usageRestriction: usageRestriction(
                summary: "OlmoEarth permits broad use but prohibits military and defense applications, intelligence gathering, human surveillance and policing, and extractive activities such as drilling, mining, and deforestation.",
                license: "OlmoEarth Artifact License",
                sourceRepoId: source.sourceRepository,
                sourceRevision: source.sourceRevision,
                licenseURL: "https://huggingface.co/\(source.sourceRepository)/blob/\(source.sourceRevision)/LICENSE.txt"
            ),
            validationKind: .olmoEarth,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: source.sourceWeightsByteCount,
            defaultCLICommands: ["geo olmoearth"]
        )
    }


    private static let sam31MLXRevision = "a992e302ea9b0f03f41dfd93414a4fd0e818f65b"

    private static let sam31TokenizerRevision = "694239a1479aab8fd1317c87c433c58acd7c6eab"
}
