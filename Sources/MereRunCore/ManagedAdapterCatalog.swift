import Foundation

public struct ManagedAdapterSpec: Equatable, Sendable {
    public let id: String
    public let title: String
    public let version: String
    public let summary: String
    public let baseModelID: String
    public let format: String
    public let license: String
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
