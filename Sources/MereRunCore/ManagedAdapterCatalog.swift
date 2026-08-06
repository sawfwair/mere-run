import Foundation

public struct ManagedAdapterSpec: Equatable, Sendable {
    public let id: String
    public let title: String
    public let version: String
    public let summary: String
    public let baseModelID: String
    public let format: String
    public let license: String
    public let upstreamRevision: String?
    public let releaseManifestURL: URL
    public let downloadURL: URL
    public let artifact: ModelArtifactPin

    public init(
        id: String,
        title: String,
        version: String,
        summary: String,
        baseModelID: String,
        format: String,
        license: String,
        upstreamRevision: String? = nil,
        releaseManifestURL: URL,
        downloadURL: URL,
        artifact: ModelArtifactPin
    ) {
        self.id = id
        self.title = title
        self.version = version
        self.summary = summary
        self.baseModelID = baseModelID
        self.format = format
        self.license = license
        self.upstreamRevision = upstreamRevision
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
            title: "MiniMax-H3 Turbo 4-step",
            version: String(miniMaxH3TurboFourStepRevision.prefix(12)),
            summary: "Four-evaluation runtime LoRA for the native MiniMax-H3 BF16 FL2VA model.",
            baseModelID: ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue,
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
        guard spec.baseModelID == baseModelID else {
            throw ManagedAdapterResolutionError.incompatibleBaseModel(
                adapterID: spec.id,
                expected: spec.baseModelID,
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
