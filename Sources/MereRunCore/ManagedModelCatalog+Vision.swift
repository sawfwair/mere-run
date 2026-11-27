import Foundation

extension ManagedModelCatalog {
    static let visionSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: UnlimitedOCRResources.modelID,
            category: .visionOCR,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: UnlimitedOCRResources.repository,
                revision: UnlimitedOCRResources.revision,
                patterns: UnlimitedOCRResources.files
            ),
            upstreamRepoId: UnlimitedOCRResources.repository,
            upstreamRevision: UnlimitedOCRResources.revision,
            validationKind: .unlimitedOCR,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: 6_700_000_000,
            defaultCLICommands: ["vision ocr"],
            manifestTemplate: .init(
                engine: .unlimitedOCR,
                family: .ocr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.visionOCR],
                components: .rootText
            ),
            capability: .init(
                "Unlimited-OCR-2",
                "Native BF16 document OCR with ordered pages and learned sparse reference attention.",
                minimum: 16,
                recommended: 24
            )
        ),
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
            defaultCLICommands: ["vision ocr"],
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .ocr,
                tier: .max,
                variant: .standard,
                precision: .bf16,
                supports: [.chat, .visionChat, .visionOCR],
                components: .rootText
            ),
            capability: .init(
                "Infinity-Parser2 Pro OCR",
                "Runs the heavyweight native Infinity-Parser2 Pro eval model for document parsing.",
                minimum: 128,
                recommended: 192
            )
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
            defaultCLICommands: ["vision ocr"],
            manifestTemplate: .init(
                engine: .qwen35HybridMoE,
                family: .ocr,
                tier: .max,
                variant: .standard,
                precision: .int8,
                quantization: .init(bits: 8, groupSize: 64, scheme: "mlx-quantized-linear"),
                supports: [.chat, .visionChat, .visionOCR],
                components: .rootText
            ),
            capability: .init(
                "Infinity-Parser2 Pro int8 OCR",
                "Runs the quantized native Infinity-Parser2 Pro model for quality-focused OCR evals.",
                minimum: 64,
                recommended: 96
            )
        ),
        ManagedModelSpec(
            id: "vision-ocr-lighton",
            category: .visionOCR,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "lightonai/LightOnOCR-3-1B",
                revision: LightOnOCRResources.oneBRevision,
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
            upstreamRepoId: "lightonai/LightOnOCR-3-1B",
            upstreamRevision: LightOnOCRResources.oneBRevision,
            validationKind: .lightOnOCR,
            estimatedDownloadBytes: 2_022_801_518,
            defaultCLICommands: ["vision ocr"],
            manifestTemplate: .init(
                engine: .lightOnOCR,
                family: .ocr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                supports: [.visionOCR],
                components: .init(tokenizer: .anyOf([.local(path: "tokenizer"), .local(path: ".")]), textEncoder: .local(path: "."))
            ),
            capability: .init(
                "OCR",
                "Reads text from images and documents with the LightOn OCR runtime.",
                minimum: 8,
                recommended: 16,
                setup: true
            )
        ),
        lightOnQwenSpec(id: .lightOnOCR3Small, downloadBytes: 1_705_971_840),
        lightOnQwenSpec(id: .lightOnOCR3FourB, downloadBytes: 9_110_359_024),
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
            defaultCLICommands: ["vision segment", "vision track", "vision track-live", "video prepare-masks"],
            manifestTemplate: .init(
                engine: .samSegmentation,
                family: .sam,
                tier: .latest,
                variant: .standard,
                precision: .unknown,
                supports: [.visionSegmentation, .visionTracking],
                components: .sam,
                upstream: .explicit("facebook/sam3.1")
            ),
            capability: .init(
                "Vision segmentation",
                "Segments and tracks objects in images or video frames with SAM 3.1.",
                minimum: 16,
                recommended: 24
            )
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
            defaultCLICommands: ["vision ground"],
            manifestTemplate: .init(
                engine: .falconPerception,
                family: .falcon,
                tier: .latest,
                variant: .standard,
                precision: .unknown,
                supports: [.visionGrounding, .visionDetection, .visionSegmentation],
                components: .falconPerception,
                upstream: .spec
            ),
            capability: .init(
                "Vision grounding",
                "Finds and labels objects in images with Falcon Perception grounding.",
                minimum: 16,
                recommended: 24
            )
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
            defaultCLICommands: ["geo flood"],
            manifestTemplate: .init(
                engine: .terramindFlood,
                family: .terramind,
                tier: .base,
                variant: .standard,
                precision: .fp32,
                supports: [.floodSegmentation]
            ),
            capability: .init(
                "Geospatial flood segmentation",
                "Runs four-timestep Sentinel-2, Sentinel-1, and DEM flood segmentation with native Swift MLX.",
                minimum: 16,
                recommended: 24
            )
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
            ],
            manifestTemplate: .init(
                engine: .insightFace,
                family: .face,
                tier: .latest,
                variant: .standard,
                precision: .fp32,
                supports: [.faceDetection, .faceLandmarks, .faceEmbedding, .faceVerification]
            ),
            capability: .init(
                "Face analysis",
                "Detects faces and landmarks, creates identity embeddings, and compares people locally.",
                minimum: 8,
                recommended: 16
            )
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
            defaultCLICommands: ["vision geometry"],
            manifestTemplate: .init(
                engine: .moge2,
                family: .geometry,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.metricDepth, .surfaceNormals, .pointMap, .cameraIntrinsics, .pointCloud]
            ),
            capability: .init(
                "Metric image geometry",
                "Predicts metric depth, point maps, normals, masks, and camera intrinsics with MoGe-2 Small.",
                minimum: 8,
                recommended: 16
            )
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
            defaultCLICommands: ["vision depth-video"],
            manifestTemplate: .init(
                engine: .videoDepthAnything,
                family: .depth,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.relativeDepth, .temporalDepth]
            ),
            capability: .init(
                "Temporal relative depth",
                "Predicts temporally consistent affine-relative video depth with Video Depth Anything Small.",
                minimum: 16,
                recommended: 32
            )
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
            defaultCLICommands: ["vision depth-video"],
            manifestTemplate: .init(
                engine: .videoDepthAnything,
                family: .depth,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.metricDepth, .temporalDepth]
            ),
            capability: .init(
                "Temporal metric depth",
                "Predicts temporally consistent metric video depth with Video Depth Anything Small.",
                minimum: 16,
                recommended: 32
            )
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
            defaultCLICommands: ["vision geometry-multiview"],
            manifestTemplate: .init(
                engine: .depthAnything3,
                family: .geometry,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.relativeDepth, .cameraIntrinsics, .cameraExtrinsics, .pointCloud]
            ),
            capability: .init(
                "Multi-view scene geometry",
                "Predicts relative depth, confidence, intrinsics, and extrinsics across multiple views with DA3 Small.",
                minimum: 12,
                recommended: 24
            )
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.visionDepthMarigoldV2.rawValue,
            category: .visionDepth,
            installShape: .structuredRoot,
            hubFallback: MarigoldV2Repository.baseHubFallback,
            mountedHubFallbacks: MarigoldV2Repository.mountedHubFallbacks,
            upstreamRepoId: MarigoldV2Repository.upstreamRepoId,
            upstreamRevision: MarigoldV2Repository.upstreamRevision,
            validationKind: .marigoldV2,
            runtimeAutoDownloadAllowed: false,
            estimatedDownloadBytes: MarigoldV2Repository.estimatedDownloadBytes,
            defaultCLICommands: ["vision depth"],
            // The depth API serves video depth; the still-image runtime has no API
            // adapter and must not advertise the category's video profile.
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .marigoldV2,
                family: .depth,
                tier: .large,
                variant: .standard,
                precision: .int4,
                supports: [.relativeDepth],
                upstream: .explicit("huawei-bayerlab/marigold-v2-0@6fd6d1ca246c9d2d99a4d8ac375a4eccc87178ad")
            ),
            capability: .init(
                "Single-image relative depth",
                "Estimates relative depth with Marigold V2. Loads the BF16 base before 4-bit quantization.",
                minimum: 64,
                recommended: 96
            )
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
            defaultCLICommands: ["image reconstruct-3d", "vision image-to-3d"],
            manifestTemplate: .init(
                engine: .tripoSR,
                family: .threeD,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.imageTo3D, .meshGeneration]
            ),
            capability: .init(
                "Single-image object mesh",
                "Reconstructs a normalized object mesh and vertex colors from one image with TripoSR.",
                minimum: 16,
                recommended: 32
            )
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
            ],
            manifestTemplate: .init(
                engine: .instantMesh,
                family: .threeD,
                tier: .base,
                variant: .standard,
                precision: .fp32,
                supports: [.multiViewReconstruction, .meshGeneration]
            ),
            capability: .init(
                "Multi-view object mesh",
                "Reconstructs a FlexiCubes object mesh from licensed user-supplied views with InstantMesh Base.",
                minimum: 24,
                recommended: 48
            )
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
            ],
            manifestTemplate: .init(
                engine: .trellis2,
                family: .threeD,
                tier: .max,
                variant: .standard,
                precision: .bf16,
                supports: [.imageTo3D, .meshGeneration]
            ),
            capability: .init(
                "PBR O-Voxel object mesh",
                "Reconstructs a 512-resolution PBR O-Voxel mesh with the native Swift MLX TRELLIS.2 runtime.",
                minimum: 64,
                recommended: 128
            )
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
            defaultCLICommands: ["geo fire"],
            manifestTemplate: .init(
                engine: .terramindFire,
                family: .terramind,
                tier: .base,
                variant: .standard,
                precision: .fp32,
                supports: [.fireSegmentation]
            ),
            capability: .init(
                "Geospatial fire segmentation",
                "Runs four-timestep Sentinel-2, Sentinel-1, and DEM fire segmentation with native Swift MLX.",
                minimum: 16,
                recommended: 24
            )
        ),
    ] + TESSERAResources.allSpecs.map { source in
        let tier: MereRunModelManifest.Tier
        let capability: ManagedModelCapabilityMetadata
        switch source.variant {
        case .nano:
            tier = .nano
            capability = .init(
                "TESSERA v2 Nano time-series embeddings",
                "Encodes Sentinel-1/2 observation sequences into 128-dimensional retrieval and analysis vectors.",
                minimum: 4,
                recommended: 6
            )
        case .small:
            tier = .small
            capability = .init(
                "TESSERA v2 Small time-series embeddings",
                "Adds temporal capacity for compact Sentinel-1/2 representation workflows.",
                minimum: 6,
                recommended: 8
            )
        case .medium:
            tier = .medium
            capability = .init(
                "TESSERA v2 Medium time-series embeddings",
                "Balances temporal representation quality and latency on entry Apple Silicon machines.",
                minimum: 8,
                recommended: 12
            )
        case .large:
            tier = .large
            capability = .init(
                "TESSERA v2 Large time-series embeddings",
                "Runs the strongest deployable TESSERA v2 student for Sentinel-1/2 time-series analysis.",
                minimum: 8,
                recommended: 16
            )
        case .teacher:
            tier = .max
            capability = .init(
                "TESSERA v2 2B Teacher embeddings",
                "Runs the full 2.06B research encoder for 1,024-dimensional Sentinel-1/2 time-series embeddings.",
                minimum: 24,
                recommended: 32
            )
        }
        return ManagedModelSpec(
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
            defaultCLICommands: ["geo tessera"],
            manifestTemplate: .init(
                engine: .tessera,
                family: .tessera,
                tier: tier,
                variant: .standard,
                precision: .fp32,
                supports: [.earthObservationEmbedding]
            ),
            capability: capability
        )
    } + OlmoEarthResources.allSpecs.map { source in
        let tier: MereRunModelManifest.Tier
        let capability: ManagedModelCapabilityMetadata
        switch source.variant {
        case .nano:
            tier = .nano
            capability = .init(
                "OlmoEarth v1.2 Nano embeddings",
                "Creates compact multisensor spatial embeddings from Sentinel-2, Sentinel-1, or Landsat imagery.",
                minimum: 4,
                recommended: 8
            )
        case .tiny:
            tier = .tiny
            capability = .init(
                "OlmoEarth v1.2 Tiny embeddings",
                "Creates efficient multisensor spatial embeddings on entry Apple Silicon machines.",
                minimum: 8,
                recommended: 12
            )
        case .small:
            tier = .small
            capability = .init(
                "OlmoEarth v1.2 Small embeddings",
                "Increases multisensor Earth-observation representation capacity for local analysis.",
                minimum: 12,
                recommended: 16
            )
        case .base:
            tier = .base
            capability = .init(
                "OlmoEarth v1.2 Base embeddings",
                "Runs the largest OlmoEarth v1.2 encoder for multisensor humanitarian and environmental analysis.",
                minimum: 16,
                recommended: 24
            )
        }
        return ManagedModelSpec(
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
            defaultCLICommands: ["geo olmoearth"],
            manifestTemplate: .init(
                engine: .olmoEarth,
                family: .olmoEarth,
                tier: tier,
                variant: .standard,
                precision: .fp32,
                supports: [.earthObservationEmbedding]
            ),
            capability: capability
        )
    }


    private static let sam31MLXRevision = "a992e302ea9b0f03f41dfd93414a4fd0e818f65b"

    private static let sam31TokenizerRevision = "694239a1479aab8fd1317c87c433c58acd7c6eab"
}

private func lightOnQwenSpec(id: ModelResolver.ModelID, downloadBytes: Int64) -> ManagedModelSpec {
    let profile = Q35Resources.profile(for: id.rawValue)!
    let isSmall = id == .lightOnOCR3Small
    return ManagedModelSpec(
        id: id.rawValue,
        category: .visionOCR,
        installShape: .directoryRoot,
        hubFallback: profile.hubFallbackConfig,
        upstreamRepoId: profile.upstreamRepoId,
        upstreamRevision: profile.upstreamRevision,
        validationKind: .q35,
        estimatedDownloadBytes: downloadBytes,
        defaultCLICommands: ["vision ocr"],
        manifestTemplate: .init(
            engine: .qwen35HybridMoE,
            family: .ocr,
            tier: isSmall ? .nano : .max,
            variant: .standard,
            precision: .bf16,
            supports: [.visionOCR],
            components: .rootText
        ),
        capability: isSmall
            ? .init(
                "LightOnOCR 3 0.8B",
                "Fast native Qwen3.5 OCR with optional document grounding.",
                minimum: 8,
                recommended: 16
            )
            : .init(
                "LightOnOCR 3 4B",
                "Quality-focused native Qwen3.5 OCR with optional document grounding.",
                minimum: 16,
                recommended: 24
            )
    )
}
