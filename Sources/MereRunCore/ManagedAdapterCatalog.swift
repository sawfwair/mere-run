import Foundation

public struct ManagedAdapterSpec: Equatable, Sendable {
    public let id: String
    public let title: String
    public let version: String
    public let summary: String
    public let baseModelID: String
    public let compatibleBaseModelIDs: Set<String>
    public let format: String
    public let license: String
    public let upstreamRevision: String?
    public let usageRestriction: ManagedModelUsageRestriction?
    public let releaseManifestURL: URL
    public let downloadURL: URL
    public let artifact: ModelArtifactPin

    public init(
        id: String,
        title: String,
        version: String,
        summary: String,
        baseModelID: String,
        compatibleBaseModelIDs: Set<String>? = nil,
        format: String,
        license: String,
        upstreamRevision: String? = nil,
        usageRestriction: ManagedModelUsageRestriction? = nil,
        releaseManifestURL: URL,
        downloadURL: URL,
        artifact: ModelArtifactPin
    ) {
        self.id = id
        self.title = title
        self.version = version
        self.summary = summary
        self.baseModelID = baseModelID
        self.compatibleBaseModelIDs = (compatibleBaseModelIDs ?? []).union([baseModelID])
        self.format = format
        self.license = license
        self.upstreamRevision = upstreamRevision
        self.usageRestriction = usageRestriction
        self.releaseManifestURL = releaseManifestURL
        self.downloadURL = downloadURL
        self.artifact = artifact
    }

    public func installDirectory(
        adaptersRoot: URL = MereRunModelPaths.adaptersDir
    ) -> URL {
        adaptersRoot
            .appendingPathComponent(id, isDirectory: true)
            .appendingPathComponent(version, isDirectory: true)
    }

    public func supports(baseModelID: String) -> Bool {
        compatibleBaseModelIDs.contains(baseModelID)
    }

    public func installedFileURL(
        adaptersRoot: URL = MereRunModelPaths.adaptersDir
    ) -> URL {
        installDirectory(adaptersRoot: adaptersRoot)
            .appendingPathComponent(artifact.filename, isDirectory: false)
    }

    public func verifiedInstalledFileURL(
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        fileManager: FileManager = .default
    ) throws -> URL {
        try artifact.verify(
            in: installDirectory(adaptersRoot: adaptersRoot),
            fileManager: fileManager
        )
    }

    public func isInstalled(
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        fileManager: FileManager = .default
    ) -> Bool {
        (try? verifiedInstalledFileURL(
            adaptersRoot: adaptersRoot,
            fileManager: fileManager
        )) != nil
    }
}

public enum ManagedAdapterCatalog {
    public static let merePlatformAssistantID = "mere-platform-assistant"
    public static let scail2LightX2VFourStepID = "scail2-lightx2v-4step"
    public static let scail2LightX2VFourStepRevision = "27ae38da91014b947dd39cc3fa78b97cd7b386dd"
    public static let miniMaxH3TurboFourStepID = "minimax-h3-turbo-4step"
    public static let miniMaxH3TurboFourStepRevision = "b604dd5fe25c4c747699f698a1e63f6c46d4a066"
    public static let miniMaxH3LightX2VFourStepID = "minimax-h3-lightx2v-4step"
    public static let miniMaxH3LightX2VFourStepRevision = "b65e359c0d128b3c5e08e0f5bf2791b794378588"
    public static let miniMaxH3LightX2VEightStepV1ID = "minimax-h3-lightx2v-8step-v1"
    public static let miniMaxH3LightX2VEightStepV1_768pID = "minimax-h3-lightx2v-8step-v1-768p"
    public static let miniMaxH3LightX2VEightStepV1_768pRevision = "05ef678438e84933c406131b59abbf86919b3aac"
    public static let miniMaxH3LightX2VFourStepV1_768pID = "minimax-h3-lightx2v-4step-v1-768p"
    public static let miniMaxH3LightX2VV1Revision = "e6346777701aa2b64d42ed058cdd71ae00e7cd52"
    public static let miniMaxH3LightX2VRef2VFourStepV01ID = "minimax-h3-lightx2v-ref2v-4step-v0.1"
    public static let miniMaxH3LightX2VRef2VFourStepV01Revision = "5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1"
    public static let ltx25PixelSpatialUpscalerID = "ltx25-pixel-spatial-upscaler-x2"
    public static let ltx25PixelSpatialUpscalerRevision = "74c4e68ee7dd99f3997d5a1bb1a3784941822222"

    public static let allSpecs: [ManagedAdapterSpec] = [
        ManagedAdapterSpec(
            id: merePlatformAssistantID,
            title: "Mere Platform Assistant",
            version: "22",
            summary: "Promoted Mere platform assistant for Gemma 4 12B 4-bit.",
            baseModelID: Gemma4Resources.twelveB4BitModelId,
            format: TextLoRATrainingManifest.format,
            license: "Apache-2.0",
            releaseManifestURL: URL(
                string: "https://releases.merekit.com/mere-run/adapters/mere-platform-assistant/v22/release.json"
            )!,
            downloadURL: URL(
                string: "https://releases.merekit.com/mere-run/adapters/mere-platform-assistant/v22/mere-platform-assistant-v22.safetensors"
            )!,
            artifact: ModelArtifactPin(
                filename: "mere-platform-assistant-v22.safetensors",
                byteCount: 128_131_022,
                sha256: "c4fec5979631b4031196c1e21c0b990437a26c5ebc52aec32f89338d64063290"
            )
        ),
        ManagedAdapterSpec(
            id: scail2LightX2VFourStepID,
            title: "LightX2V Wan 2.1 I2V 4-step",
            version: String(scail2LightX2VFourStepRevision.prefix(12)),
            summary: "Apache-2.0 four-step distilled Wan 2.1 I2V adapter for native SCAIL-2.",
            baseModelID: SCAIL2Resources.modelID,
            format: SCAIL2DistilledAdapter.format,
            license: "Apache-2.0",
            upstreamRevision: scail2LightX2VFourStepRevision,
            releaseManifestURL: URL(
                string: "https://huggingface.co/lightx2v/Wan2.1-Distill-Loras/commit/\(scail2LightX2VFourStepRevision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/lightx2v/Wan2.1-Distill-Loras/resolve/\(scail2LightX2VFourStepRevision)/wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors",
                byteCount: 739_472_104,
                sha256: "8833bd4fd7c8eabebf0bc8ee5cfaf47f4f310ce116928a02c1adf8941dd4b0f1"
            )
        ),
        ManagedAdapterSpec(
            id: miniMaxH3TurboFourStepID,
            title: "MiniMax-H3 Turbo 4-step (EMA-850)",
            version: String(miniMaxH3TurboFourStepRevision.prefix(12)),
            summary: "Four-evaluation runtime LoRA for native MiniMax-H3 compact BF16 and Q8 FL2VA models.",
            baseModelID: ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
            compatibleBaseModelIDs: [
                ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
                MiniMaxH3Resources.fl2vaQ8ModelID,
            ],
            format: MiniMaxH3TurboAdapter.format,
            license: "Apache-2.0 (adapter); MiniMax-H3 Community License (base model)",
            upstreamRevision: miniMaxH3TurboFourStepRevision,
            releaseManifestURL: URL(
                string: "https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/commit/\(miniMaxH3TurboFourStepRevision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/\(miniMaxH3TurboFourStepRevision)/minimax_h3_turbo_4step_ema_ckpt850.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
                byteCount: 779_849_816,
                sha256: "5a6eeba171cf183020a4ad48774bb2968f29f8168afd6ec17a04987f3528b4ea"
            )
        ),
        ManagedAdapterSpec(
            id: miniMaxH3LightX2VFourStepID,
            title: "MiniMax-H3 Turbo 4-step (LightX2V)",
            version: String(miniMaxH3LightX2VFourStepRevision.prefix(12)),
            summary: "LightX2V four-evaluation PEFT LoRA for native MiniMax-H3 compact BF16 and Q8 FL2VA models.",
            baseModelID: ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
            compatibleBaseModelIDs: [
                ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
                MiniMaxH3Resources.fl2vaQ8ModelID,
            ],
            format: MiniMaxH3TurboAdapter.lightX2VFormat,
            license: "Apache-2.0 (adapter); MiniMax-H3 Community License (base model)",
            upstreamRevision: miniMaxH3LightX2VFourStepRevision,
            releaseManifestURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/commit/\(miniMaxH3LightX2VFourStepRevision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/\(miniMaxH3LightX2VFourStepRevision)/minimax_h3_fl2v_turbo_4step_v0.1.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
                byteCount: 1_383_677_888,
                sha256: "5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29"
            )
        ),
        ManagedAdapterSpec(
            id: miniMaxH3LightX2VEightStepV1ID,
            title: "MiniMax-H3 Turbo 8-step v1.0 (LightX2V)",
            version: String(miniMaxH3LightX2VV1Revision.prefix(12)),
            summary: "LightX2V v1.0 eight-evaluation 544p PEFT LoRA for native MiniMax-H3 compact BF16 and Q8 FL2VA.",
            baseModelID: ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
            compatibleBaseModelIDs: [
                ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
                MiniMaxH3Resources.fl2vaQ8ModelID,
            ],
            format: MiniMaxH3TurboAdapter.lightX2VFormat,
            license: "Apache-2.0 (adapter); MiniMax-H3 Community License (base model)",
            upstreamRevision: miniMaxH3LightX2VV1Revision,
            releaseManifestURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/commit/\(miniMaxH3LightX2VV1Revision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/\(miniMaxH3LightX2VV1Revision)/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
                byteCount: 1_383_677_768,
                sha256: "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"
            )
        ),
        ManagedAdapterSpec(
            id: miniMaxH3LightX2VFourStepV1_768pID,
            title: "MiniMax-H3 Turbo 4-step v1.0 768p (LightX2V)",
            version: String(miniMaxH3LightX2VV1Revision.prefix(12)),
            summary: "LightX2V v1.0 four-evaluation 1344x768 PEFT LoRA for native MiniMax-H3 compact BF16 and Q8 FL2VA.",
            baseModelID: ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
            compatibleBaseModelIDs: [
                ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
                MiniMaxH3Resources.fl2vaQ8ModelID,
            ],
            format: MiniMaxH3TurboAdapter.lightX2VFormat,
            license: "Apache-2.0 (adapter); MiniMax-H3 Community License (base model)",
            upstreamRevision: miniMaxH3LightX2VV1Revision,
            releaseManifestURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/commit/\(miniMaxH3LightX2VV1Revision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/\(miniMaxH3LightX2VV1Revision)/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
                byteCount: 1_383_677_808,
                sha256: "1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c"
            )
        ),
        ManagedAdapterSpec(
            id: miniMaxH3LightX2VEightStepV1_768pID,
            title: "MiniMax-H3 Turbo 8-step v1.0 768p (LightX2V)",
            version: String(miniMaxH3LightX2VEightStepV1_768pRevision.prefix(12)),
            summary: "LightX2V v1.0 eight-evaluation 1344x768 PEFT LoRA for native MiniMax-H3 compact BF16 and Q8 FL2VA.",
            baseModelID: ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
            compatibleBaseModelIDs: [
                ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
                MiniMaxH3Resources.fl2vaQ8ModelID,
            ],
            format: MiniMaxH3TurboAdapter.lightX2VFormat,
            license: "Apache-2.0 (adapter); MiniMax-H3 Community License (base model)",
            upstreamRevision: miniMaxH3LightX2VEightStepV1_768pRevision,
            releaseManifestURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/commit/\(miniMaxH3LightX2VEightStepV1_768pRevision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/\(miniMaxH3LightX2VEightStepV1_768pRevision)/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
                byteCount: 1_383_677_808,
                sha256: "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4"
            )
        ),
        ManagedAdapterSpec(
            id: miniMaxH3LightX2VRef2VFourStepV01ID,
            title: "MiniMax-H3 Ref2VA Turbo 4-step v0.1 (LightX2V)",
            version: String(miniMaxH3LightX2VRef2VFourStepV01Revision.prefix(12)),
            summary: "LightX2V four-evaluation PEFT LoRA for native MiniMax-H3 Ref2VA.",
            baseModelID: ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue,
            format: MiniMaxH3TurboAdapter.lightX2VFormat,
            license: "Apache-2.0 (adapter); MiniMax-H3 Community License (base model)",
            upstreamRevision: miniMaxH3LightX2VRef2VFourStepV01Revision,
            releaseManifestURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/commit/\(miniMaxH3LightX2VRef2VFourStepV01Revision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/\(miniMaxH3LightX2VRef2VFourStepV01Revision)/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
                byteCount: 1_383_677_768,
                sha256: "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99"
            )
        ),
        ManagedAdapterSpec(
            id: ltx25PixelSpatialUpscalerID,
            title: "LTX-2.5 Pixel Spatial Upscaler x2",
            version: String(ltx25PixelSpatialUpscalerRevision.prefix(12)),
            summary: "Official x2 spatial detailing IC-LoRA for the native LTX-2.5 DFR refinement stage.",
            baseModelID: ModelResolver.ModelID.ltxVideo25FullBF16.rawValue,
            format: "ltx-2.5-ic-lora",
            license: "LTX-2 Community License",
            upstreamRevision: ltx25PixelSpatialUpscalerRevision,
            usageRestriction: ManagedModelUsageRestriction(
                summary: "Use is governed by the LTX-2 Community License and the gated Hugging Face repository terms.",
                terms: [
                    ManagedModelUsageTerm(
                        component: "LTX-2.5 Pixel Spatial Upscaler x2",
                        license: "LTX-2 Community License",
                        summary: "Official gated Lightricks IC-LoRA adapter for LTX-2.5.",
                        sourceRepoId: "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
                        sourceRevision: ltx25PixelSpatialUpscalerRevision,
                        licenseURL: "https://github.com/Lightricks/LTX-2/blob/d151147788a9284cca791edc6ce898007e727fe6/LICENSE"
                    ),
                ]
            ),
            releaseManifestURL: URL(
                string: "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/commit/\(ltx25PixelSpatialUpscalerRevision)"
            )!,
            downloadURL: URL(
                string: "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/resolve/\(ltx25PixelSpatialUpscalerRevision)/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors?download=true"
            )!,
            artifact: ModelArtifactPin(
                filename: "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
                byteCount: 327_322_640,
                sha256: "984851b769ea2bcb4c9e0a239a7676239e42c6a6001ddc69943b41ff0b283c1d"
            )
        ),
    ]

    public static func spec(for id: String) -> ManagedAdapterSpec? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allSpecs.first { $0.id == normalized }
    }

    /// Resolves a catalog id to its verified installed file. Returns `nil`
    /// when the reference is not a catalog id so callers can preserve local
    /// path behavior.
    public static func resolveInstalledReference(
        _ reference: String,
        baseModelID: String,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        fileManager: FileManager = .default
    ) throws -> URL? {
        guard let spec = spec(for: reference) else { return nil }
        guard spec.supports(baseModelID: baseModelID) else {
            throw ManagedAdapterResolutionError.incompatibleBaseModel(
                adapterID: spec.id,
                expected: spec.compatibleBaseModelIDs.sorted().joined(separator: " or "),
                actual: baseModelID
            )
        }
        do {
            return try spec.verifiedInstalledFileURL(
                adaptersRoot: adaptersRoot,
                fileManager: fileManager
            )
        } catch let error as ModelArtifactVerificationError {
            throw ManagedAdapterResolutionError.notInstalled(
                adapterID: spec.id,
                path: spec.installedFileURL(adaptersRoot: adaptersRoot).path,
                detail: error.localizedDescription
            )
        }
    }
}

public enum ManagedAdapterResolutionError: Error, Equatable, LocalizedError, Sendable {
    case incompatibleBaseModel(adapterID: String, expected: String, actual: String)
    case notInstalled(adapterID: String, path: String, detail: String)

    public var errorDescription: String? {
        switch self {
        case .incompatibleBaseModel(let adapterID, let expected, let actual):
            "Adapter \(adapterID) requires base model \(expected), not \(actual)."
        case .notInstalled(let adapterID, let path, let detail):
            "Adapter \(adapterID) is not installed and verified at \(path). Run `mere.run adapter pull \(adapterID)`. \(detail)"
        }
    }
}
