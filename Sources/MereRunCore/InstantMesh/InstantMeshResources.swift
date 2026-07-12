import Foundation
@preconcurrency import MLX

/// InstantMesh runtime accepts only the deterministic, non-executable package
/// emitted by `scripts/model-conversion/convert_instantmesh_base.py`.
public enum InstantMeshCheckpointFormat: String, Codable, Equatable, Hashable, Sendable {
    case convertedSafetensors = "verified-converted-safetensors"
}

public struct InstantMeshCheckpoint: Codable, Equatable, Hashable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let license: String
    public let format: InstantMeshCheckpointFormat
    public let rootURL: URL
    public let weightsURL: URL
    public let configurationURL: URL
    public let sourceManifestURL: URL
    public let weightsByteCount: Int64
    public let weightsSHA256: String
    public let sourceSHA256: String
    public let configurationSHA256: String
    public let sourceManifestSHA256: String
    public let viewGenerationIncluded: Bool

    public init(
        modelID: String,
        repository: String,
        revision: String,
        sourceRepository: String,
        sourceRevision: String,
        license: String,
        format: InstantMeshCheckpointFormat,
        rootURL: URL,
        weightsURL: URL,
        configurationURL: URL,
        sourceManifestURL: URL,
        weightsByteCount: Int64,
        weightsSHA256: String,
        sourceSHA256: String,
        configurationSHA256: String,
        sourceManifestSHA256: String,
        viewGenerationIncluded: Bool
    ) {
        self.modelID = modelID
        self.repository = repository
        self.revision = revision
        self.sourceRepository = sourceRepository
        self.sourceRevision = sourceRevision
        self.license = license
        self.format = format
        self.rootURL = rootURL.standardizedFileURL
        self.weightsURL = weightsURL.standardizedFileURL
        self.configurationURL = configurationURL.standardizedFileURL
        self.sourceManifestURL = sourceManifestURL.standardizedFileURL
        self.weightsByteCount = weightsByteCount
        self.weightsSHA256 = weightsSHA256.lowercased()
        self.sourceSHA256 = sourceSHA256.lowercased()
        self.configurationSHA256 = configurationSHA256.lowercased()
        self.sourceManifestSHA256 = sourceManifestSHA256.lowercased()
        self.viewGenerationIncluded = viewGenerationIncluded
    }
}

public enum InstantMeshResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case checkpointNotFound(String)
    case unsupportedCheckpointPath(String)
    case conversionRequired(sourcePath: String)
    case unrecognizedPinnedSource(String)
    case invalidConvertedPackage(String)
    case checkpointIdentityChanged

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let value):
            "Unsupported InstantMesh model '\(value)'. Only image-3d-instantmesh-base is permitted."
        case .checkpointNotFound(let path):
            "InstantMesh checkpoint was not found: \(path)"
        case .unsupportedCheckpointPath(let path):
            "Expected a verified converted InstantMesh safetensors package at \(path)."
        case .conversionRequired(let sourcePath):
            "InstantMesh runtime never interprets Lightning/Pickle checkpoints. The pinned source at "
                + "\(sourcePath) requires the explicit offline conversion in "
                + "scripts/model-conversion/convert_instantmesh_base.py. Write the converted package to "
                + "\(InstantMeshResources.suggestedManagedConvertedPath(for: sourcePath)) to use the managed "
                + "model id, or pass another output directory with --model."
        case .unrecognizedPinnedSource(let path):
            "InstantMesh source checkpoint does not match the audited upstream artifact: \(path)"
        case .invalidConvertedPackage(let detail):
            "Invalid converted InstantMesh package: \(detail)"
        case .checkpointIdentityChanged:
            "InstantMesh checkpoint identity changed after preflight verification."
        }
    }
}

public enum InstantMeshResources {
    public static let defaultModelID = ModelResolver.ModelID.image3DInstantMeshBase.rawValue
    /// A converted package may live here inside the managed raw-source root.
    /// This lets `model pull` retain its immutable source and model manifest
    /// while an explicit offline conversion makes the same model id runnable.
    public static let managedConvertedDirectoryName = "native"

    static func suggestedManagedConvertedPath(for sourcePath: String) -> String {
        URL(fileURLWithPath: sourcePath).standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(managedConvertedDirectoryName, isDirectory: true)
            .path
    }
    public static let convertedWeightsPin = ModelArtifactPin(
        filename: "model.safetensors",
        byteCount: InstantMeshWeights.convertedSafetensorsByteCount,
        sha256: InstantMeshWeights.convertedSafetensorsSHA256
    )
    public static let convertedConfigurationPin = ModelArtifactPin(
        filename: "config.json",
        byteCount: 486,
        sha256: "33f89581172ab2d46759a1632b6e57ca9f9f1c6c23567468157cb4b48a3bc781"
    )
    public static let convertedSourceManifestPin = ModelArtifactPin(
        filename: "SOURCE.json",
        byteCount: 1_074,
        sha256: "9fbda0d3875744353a4ca6ee9ee836182cb46f72aa0d241c30ee62b746d60061"
    )
    public static let convertedLicensePin = ModelArtifactPin(
        filename: "LICENSE",
        byteCount: 11_357,
        sha256: "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )

    public static func resolve(requestedModel: String?) async throws -> InstantMeshCheckpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspectExplicit(explicit)
            }
            if looksLikePath(requested) {
                throw InstantMeshResourceError.checkpointNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID : requested.lowercased()
        guard modelID == defaultModelID else {
            throw InstantMeshResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: defaultModelID,
            allowAutoDownload: false
        )
        return try inspectExplicit(resolution.url)
    }

    public static func inspectExplicit(_ url: URL) throws -> InstantMeshCheckpoint {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw InstantMeshResourceError.checkpointNotFound(standardized.path)
        }
        if isDirectory.boolValue {
            let convertedURL = standardized.appendingPathComponent(convertedWeightsPin.filename)
            if FileManager.default.fileExists(atPath: convertedURL.path) {
                return try inspectConvertedDirectory(standardized)
            }
            let managedConvertedRoot = standardized.appendingPathComponent(
                managedConvertedDirectoryName,
                isDirectory: true
            )
            if FileManager.default.fileExists(
                atPath: managedConvertedRoot.appendingPathComponent(convertedWeightsPin.filename).path
            ) {
                return try inspectConvertedDirectory(managedConvertedRoot)
            }
            let sourceURL = standardized.appendingPathComponent(sourceWeightsPin.filename)
            if FileManager.default.fileExists(atPath: sourceURL.path) {
                try requireConversion(for: sourceURL)
            }
            throw InstantMeshResourceError.invalidConvertedPackage(
                "missing model.safetensors, config.json, and SOURCE.json"
            )
        }

        switch standardized.pathExtension.lowercased() {
        case "ckpt":
            try requireConversion(for: standardized)
        case "safetensors":
            guard standardized.lastPathComponent == convertedWeightsPin.filename else {
                throw InstantMeshResourceError.unsupportedCheckpointPath(standardized.path)
            }
            return try inspectConvertedDirectory(standardized.deletingLastPathComponent())
        default:
            throw InstantMeshResourceError.unsupportedCheckpointPath(standardized.path)
        }
    }

    public static func loadModel(from checkpoint: InstantMeshCheckpoint) throws -> InstantMeshModel {
        let verified = try inspectConvertedDirectory(checkpoint.rootURL)
        guard verified == checkpoint else {
            throw InstantMeshResourceError.checkpointIdentityChanged
        }
        let model = InstantMeshModel()
        try InstantMeshWeights.load(
            model: model,
            safetensorsURL: verified.weightsURL,
            dtype: .float32
        )
        return model
    }

    private static var pin: GeometryModelPin { GeometryModelPins.instantMeshBase }

    private static var sourceWeightsPin: ModelArtifactPin {
        guard let artifact = pin.artifacts.first(where: { $0.filename == "instant_mesh_base.ckpt" }) else {
            fatalError("InstantMesh source checkpoint pin is missing")
        }
        return artifact
    }

    private static func inspectConvertedDirectory(_ rootURL: URL) throws -> InstantMeshCheckpoint {
        do {
            let weightsURL = try convertedWeightsPin.verify(in: rootURL)
            let configurationURL = try convertedConfigurationPin.verify(in: rootURL)
            let sourceManifestURL = try convertedSourceManifestPin.verify(in: rootURL)
            _ = try convertedLicensePin.verify(in: rootURL)
            return InstantMeshCheckpoint(
                modelID: defaultModelID,
                repository: pin.repository,
                revision: pin.revision,
                sourceRepository: pin.sourceCodeRepository,
                sourceRevision: pin.sourceCodeRevision,
                license: pin.license,
                format: .convertedSafetensors,
                rootURL: rootURL,
                weightsURL: weightsURL,
                configurationURL: configurationURL,
                sourceManifestURL: sourceManifestURL,
                weightsByteCount: convertedWeightsPin.byteCount,
                weightsSHA256: convertedWeightsPin.sha256,
                sourceSHA256: sourceWeightsPin.sha256,
                configurationSHA256: convertedConfigurationPin.sha256,
                sourceManifestSHA256: convertedSourceManifestPin.sha256,
                viewGenerationIncluded: false
            )
        } catch {
            throw InstantMeshResourceError.invalidConvertedPackage(error.localizedDescription)
        }
    }

    private static func requireConversion(for sourceURL: URL) throws -> Never {
        do {
            let actualByteCount = try ModelArtifactPin.fileByteCount(sourceURL)
            guard actualByteCount == sourceWeightsPin.byteCount else {
                throw ModelArtifactVerificationError.sizeMismatch(
                    path: sourceURL.path,
                    expected: sourceWeightsPin.byteCount,
                    actual: actualByteCount
                )
            }
            let actualSHA256 = try ModelArtifactPin.fileSHA256(
                sourceURL.standardizedFileURL.resolvingSymlinksInPath()
            )
            guard actualSHA256 == sourceWeightsPin.sha256 else {
                throw ModelArtifactVerificationError.checksumMismatch(
                    path: sourceURL.path,
                    expected: sourceWeightsPin.sha256,
                    actual: actualSHA256
                )
            }
        } catch {
            throw InstantMeshResourceError.unrecognizedPinnedSource(sourceURL.path)
        }
        throw InstantMeshResourceError.conversionRequired(sourcePath: sourceURL.path)
    }

    private static func looksLikePath(_ value: String) -> Bool {
        value.hasPrefix("/") || value.hasPrefix("~") || value.hasPrefix(".")
            || value.contains("/") || value.hasSuffix(".ckpt") || value.hasSuffix(".safetensors")
    }
}
