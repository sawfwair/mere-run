import Foundation
@preconcurrency import MLX

/// Fully verified identity for the only DA3 checkpoint accepted by the native
/// production runtime. Both URLs may be managed-install symlinks; the hashes
/// and byte counts describe their resolved immutable targets.
public struct DepthAnything3Checkpoint: Codable, Equatable, Hashable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let license: String
    public let rootURL: URL
    public let weightsURL: URL
    public let configurationURL: URL
    public let weightsByteCount: Int64
    public let weightsSHA256: String
    public let configurationByteCount: Int64
    public let configurationSHA256: String

    public init(
        modelID: String,
        repository: String,
        revision: String,
        sourceRepository: String,
        sourceRevision: String,
        license: String,
        rootURL: URL,
        weightsURL: URL,
        configurationURL: URL,
        weightsByteCount: Int64,
        weightsSHA256: String,
        configurationByteCount: Int64,
        configurationSHA256: String
    ) {
        self.modelID = modelID
        self.repository = repository
        self.revision = revision
        self.sourceRepository = sourceRepository
        self.sourceRevision = sourceRevision
        self.license = license
        self.rootURL = rootURL.standardizedFileURL
        self.weightsURL = weightsURL.standardizedFileURL
        self.configurationURL = configurationURL.standardizedFileURL
        self.weightsByteCount = weightsByteCount
        self.weightsSHA256 = weightsSHA256.lowercased()
        self.configurationByteCount = configurationByteCount
        self.configurationSHA256 = configurationSHA256.lowercased()
    }
}

public enum DepthAnything3ResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case checkpointNotFound(String)
    case unsupportedCheckpointPath(String)
    case missingCompanionConfiguration(String)
    case checkpointIdentityChanged

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let value):
            "Unsupported Depth Anything 3 model '\(value)'. Only vision-geometry-da3-small is permitted."
        case .checkpointNotFound(let path):
            "Depth Anything 3 checkpoint was not found: \(path)"
        case .unsupportedCheckpointPath(let path):
            "Expected a DA3-Small directory or exact pinned .safetensors file at \(path)."
        case .missingCompanionConfiguration(let path):
            "Pinned DA3-Small config.json is required beside the checkpoint: \(path)"
        case .checkpointIdentityChanged:
            "DA3-Small checkpoint identity changed after preflight verification."
        }
    }
}

public enum DepthAnything3Resources {
    public static let defaultModelID = ModelResolver.ModelID.visionGeometryDA3Small.rawValue

    /// Resolves either the exact managed model ID or a local path whose two
    /// artifacts match the authoritative pins. Runtime auto-download is
    /// deliberately disabled so a paid/interactive workflow can preflight
    /// model availability without mutating the cache.
    public static func resolve(requestedModel: String?) async throws -> DepthAnything3Checkpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspectExplicit(explicit)
            }
            if looksLikePath(requested) {
                throw DepthAnything3ResourceError.checkpointNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID : requested.lowercased()
        guard modelID == defaultModelID else {
            throw DepthAnything3ResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: defaultModelID,
            allowAutoDownload: false
        )
        return try inspectExplicit(resolution.url)
    }

    public static func inspectExplicit(_ url: URL) throws -> DepthAnything3Checkpoint {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw DepthAnything3ResourceError.checkpointNotFound(standardized.path)
        }

        let root: URL
        if isDirectory.boolValue {
            root = standardized
        } else {
            guard standardized.pathExtension.lowercased() == "safetensors" else {
                throw DepthAnything3ResourceError.unsupportedCheckpointPath(standardized.path)
            }
            root = standardized.deletingLastPathComponent()
        }

        let canonicalWeights = root.appendingPathComponent(DepthAnything3SmallCheckpoint.artifact.filename)
        let weightsURL: URL
        if isDirectory.boolValue || standardized.lastPathComponent == DepthAnything3SmallCheckpoint.artifact.filename {
            weightsURL = canonicalWeights
        } else {
            // Explicit files may have a descriptive local name. Verification is
            // content-addressed below; managed packages retain the canonical name.
            weightsURL = standardized
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw DepthAnything3ResourceError.checkpointNotFound(weightsURL.path)
        }

        let configurationURL = root.appendingPathComponent(
            DepthAnything3SmallCheckpoint.configurationArtifact.filename
        )
        guard FileManager.default.fileExists(atPath: configurationURL.path) else {
            throw DepthAnything3ResourceError.missingCompanionConfiguration(configurationURL.path)
        }

        try verify(
            url: weightsURL,
            expected: DepthAnything3SmallCheckpoint.artifact
        )
        try verify(
            url: configurationURL,
            expected: DepthAnything3SmallCheckpoint.configurationArtifact
        )
        return checkpoint(root: root, weights: weightsURL, configuration: configurationURL)
    }

    public static func loadModel(from checkpoint: DepthAnything3Checkpoint) throws -> DepthAnything3Model {
        // Reinspect the exact preflight artifact rather than only its parent;
        // explicit content-addressed files may intentionally use a noncanonical
        // local filename beside the pinned config.json.
        let verified = try inspectExplicit(checkpoint.weightsURL)
        guard verified == checkpoint else {
            throw DepthAnything3ResourceError.checkpointIdentityChanged
        }
        let model = DepthAnything3Model()
        try DepthAnything3Weights.load(
            model: model,
            safetensorsURL: verified.weightsURL,
            dtype: .float32,
            verifyChecksum: false // inspectExplicit verified the exact bytes immediately above.
        )
        return model
    }

    private static func checkpoint(
        root: URL,
        weights: URL,
        configuration: URL
    ) -> DepthAnything3Checkpoint {
        DepthAnything3Checkpoint(
            modelID: defaultModelID,
            repository: DepthAnything3SmallCheckpoint.repository,
            revision: DepthAnything3SmallCheckpoint.revision,
            sourceRepository: DepthAnything3SmallCheckpoint.upstreamSourceRepository,
            sourceRevision: DepthAnything3SmallCheckpoint.upstreamSourceRevision,
            license: DepthAnything3SmallCheckpoint.license,
            rootURL: root,
            weightsURL: weights,
            configurationURL: configuration,
            weightsByteCount: DepthAnything3SmallCheckpoint.artifact.byteCount,
            weightsSHA256: DepthAnything3SmallCheckpoint.artifact.sha256,
            configurationByteCount: DepthAnything3SmallCheckpoint.configurationArtifact.byteCount,
            configurationSHA256: DepthAnything3SmallCheckpoint.configurationArtifact.sha256
        )
    }

    private static func verify(url: URL, expected: ModelArtifactPin) throws {
        let actualByteCount = try ModelArtifactPin.fileByteCount(url)
        guard actualByteCount == expected.byteCount else {
            throw ModelArtifactVerificationError.sizeMismatch(
                path: url.path,
                expected: expected.byteCount,
                actual: actualByteCount
            )
        }
        let actualSHA256 = try ModelArtifactPin.fileSHA256(url.resolvingSymlinksInPath())
        guard actualSHA256 == expected.sha256 else {
            throw ModelArtifactVerificationError.checksumMismatch(
                path: url.path,
                expected: expected.sha256,
                actual: actualSHA256
            )
        }
    }

    private static func looksLikePath(_ value: String) -> Bool {
        value.contains("/") || value.hasPrefix(".")
            || value.lowercased().hasSuffix(".safetensors")
    }
}
