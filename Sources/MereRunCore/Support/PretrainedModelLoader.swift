import Foundation

public struct HubFallbackConfig: Sendable, Hashable {
    public let repoId: String
    public let revision: String
    public let patterns: [String]
    public let filePath: String?

    public init(
        repoId: String,
        revision: String = "main",
        patterns: [String],
        filePath: String? = nil
    ) {
        self.repoId = repoId
        self.revision = revision
        self.patterns = patterns
        self.filePath = filePath
    }
}

/// Shared model-loading utilities for local path resolution and managed model directories.
public enum PretrainedModelLoader {
    public enum ProgressEvent: Sendable, Hashable {
        case downloading(percent: Int)
        case extracting
    }

    public enum LoadError: LocalizedError, Sendable {
        case unsupportedModelId(String)
        case missingFiles([String])
        case downloadFailed(String)

        public var errorDescription: String? {
            switch self {
            case .unsupportedModelId(let modelId):
                return "Unsupported model id: \(modelId)"
            case .missingFiles(let files):
                return "Missing required files: \(files.joined(separator: ", "))"
            case .downloadFailed(let message):
                return "Download failed: \(message)"
            }
        }
    }

    public struct ManagedRoot: Sendable, Hashable {
        public let modelDir: URL
        public let resolvedRoot: URL
        public let isComplete: Bool

        public init(modelDir: URL, resolvedRoot: URL, isComplete: Bool) {
            self.modelDir = modelDir
            self.resolvedRoot = resolvedRoot
            self.isComplete = isComplete
        }
    }

    /// Resolve explicit `modelPath` first, then treat `modelId` as a local path if it exists.
    public static func resolveProvidedOrLocalRoot(
        modelPath: String?,
        modelId: String,
        fileManager: FileManager = .default,
        normalize: (URL, FileManager) -> URL = { base, _ in base }
    ) -> URL? {
        if let modelPath {
            let url = URL(fileURLWithPath: modelPath).standardizedFileURL
            return normalize(url, fileManager)
        }

        let candidate = URL(fileURLWithPath: modelId).standardizedFileURL
        guard fileManager.fileExists(atPath: candidate.path) else {
            return nil
        }
        return normalize(candidate, fileManager)
    }

    /// Resolve (or create) the managed model directory and evaluate whether it's already complete.
    public static func resolveManagedRoot(
        storageId: String,
        fileManager: FileManager = .default,
        normalize: (URL, FileManager) -> URL = { base, _ in base },
        isComplete: (URL, FileManager) -> Bool
    ) -> ManagedRoot {
        let modelDir = MereRunModelPaths.resolveModelDir(storageId) { root in
            let resolved = normalize(root, fileManager)
            return isComplete(resolved, fileManager)
        }
        let resolvedRoot = normalize(modelDir, fileManager)
        let complete = isComplete(resolvedRoot, fileManager)
        return ManagedRoot(modelDir: modelDir, resolvedRoot: resolvedRoot, isComplete: complete)
    }

    /// Resolve from local path, local modelId path, or a Hugging Face snapshot in mere.run storage.
    public static func fromPretrainedSnapshot(
        modelPath: String?,
        modelId: String,
        defaultModelIds: Set<String>,
        storageId: String,
        hubFallback: HubFallbackConfig? = nil,
        fileManager: FileManager = .default,
        normalize: (URL, FileManager) -> URL = { base, _ in base },
        validate: (URL, FileManager) -> [URL],
        progress: (@Sendable (ProgressEvent) -> Void)? = nil
    ) async throws -> URL {
        if let provided = resolveProvidedOrLocalRoot(
            modelPath: modelPath,
            modelId: modelId,
            fileManager: fileManager,
            normalize: normalize
        ) {
            return provided
        }

        guard defaultModelIds.contains(modelId) else {
            throw LoadError.unsupportedModelId(modelId)
        }

        let managed = resolveManagedRoot(
            storageId: storageId,
            fileManager: fileManager,
            normalize: normalize
        ) { root, manager in
            validate(root, manager).isEmpty
        }

        if managed.isComplete {
            return managed.resolvedRoot
        }

        guard let hubFallback else {
            throw LoadError.downloadFailed(
                ManagedModelCatalog.missingHubSourceMessage(for: modelId)
            )
        }

        let snapshotURL = try await downloadViaHubSnapshot(config: hubFallback, progress: progress)
        let resolvedRoot = normalize(snapshotURL, fileManager)
        let missingAfter = validate(resolvedRoot, fileManager)
        guard missingAfter.isEmpty else {
            throw LoadError.missingFiles(missingAfter.map(\.lastPathComponent))
        }

        return resolvedRoot
    }

    /// Resolve from local path, local modelId path, or managed file in mere.run storage.
    public static func fromPretrainedFile(
        modelPath: String?,
        modelId: String,
        defaultModelIds: Set<String>,
        relativePath: String,
        hubFallback: HubFallbackConfig? = nil,
        fileManager: FileManager = .default,
        validate: (URL, FileManager) -> [URL],
        progress: (@Sendable (ProgressEvent) -> Void)? = nil
    ) async throws -> URL {
        if let provided = resolveProvidedOrLocalRoot(
            modelPath: modelPath,
            modelId: modelId,
            fileManager: fileManager
        ) {
            return provided
        }

        guard defaultModelIds.contains(modelId) else {
            throw LoadError.unsupportedModelId(modelId)
        }

        let managedFile = MereRunModelPaths.resolveModelFile(relativePath: relativePath) { candidate in
            validate(candidate, fileManager).isEmpty
        }

        if validate(managedFile, fileManager).isEmpty {
            return managedFile
        }

        guard let hubFallback else {
            throw LoadError.downloadFailed(
                ManagedModelCatalog.missingHubSourceMessage(for: modelId)
            )
        }

        let fileURL = try await downloadViaHubSnapshot(
            config: hubFallback,
            relativePath: relativePath,
            progress: progress
        )
        let missingAfter = validate(fileURL, fileManager)
        guard missingAfter.isEmpty else {
            throw LoadError.missingFiles(missingAfter.map(\.lastPathComponent))
        }

        return fileURL
    }

    private static func downloadViaHubSnapshot(
        config: HubFallbackConfig,
        progress: (@Sendable (ProgressEvent) -> Void)?
    ) async throws -> URL {
        do {
            let snapshot = try makeHubSnapshot(config: config)
            return try await snapshot.prepare { snapshotProgress in
                let percent = min(100, max(0, Int(snapshotProgress.fractionCompleted * 100)))
                progress?(.downloading(percent: percent))
            }
        } catch let error as LoadError {
            throw error
        } catch {
            throw LoadError.downloadFailed(error.localizedDescription)
        }
    }

    private static func downloadViaHubSnapshot(
        config: HubFallbackConfig,
        relativePath: String,
        progress: (@Sendable (ProgressEvent) -> Void)?
    ) async throws -> URL {
        do {
            let snapshot = try makeHubSnapshot(config: config)
            let resolvedPath = config.filePath ?? relativePath
            return try await snapshot.fileURL(for: resolvedPath) { snapshotProgress in
                let percent = min(100, max(0, Int(snapshotProgress.fractionCompleted * 100)))
                progress?(.downloading(percent: percent))
            }
        } catch let error as LoadError {
            throw error
        } catch {
            throw LoadError.downloadFailed(error.localizedDescription)
        }
    }

    private static func makeHubSnapshot(config: HubFallbackConfig) throws -> HubSnapshot {
        try HubSnapshot(
            options: HubSnapshotOptions(
                repoId: config.repoId,
                revision: config.revision,
                patterns: config.patterns
            )
        )
    }
}
