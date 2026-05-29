import Foundation

public enum ManagedModelResolver {
    public enum RuntimeSource: String, Hashable, Sendable {
        case explicitPath
        case managedStore
        case hubSnapshot
    }

    public struct RuntimeResolution: Hashable, Sendable {
        public let spec: ManagedModelSpec
        public let url: URL
        public let source: RuntimeSource

        public init(spec: ManagedModelSpec, url: URL, source: RuntimeSource) {
            self.spec = spec
            self.url = url
            self.source = source
        }
    }

    public struct InstallResult: Hashable, Sendable {
        public let spec: ManagedModelSpec
        public let installURL: URL
        public let manifest: MereRunModelManifest?
        public let wasAlreadyInstalled: Bool

        public init(
            spec: ManagedModelSpec,
            installURL: URL,
            manifest: MereRunModelManifest?,
            wasAlreadyInstalled: Bool
        ) {
            self.spec = spec
            self.installURL = installURL
            self.manifest = manifest
            self.wasAlreadyInstalled = wasAlreadyInstalled
        }
    }

    public enum InstallProgress: Sendable, Hashable {
        case downloadingBytes(completed: Int64, total: Int64?)
        case downloadingPercent(percent: Int, speedBytesPerSecond: Double?)
        case extracting
    }

    public enum ResolverError: LocalizedError, Sendable {
        case unknownModelID(String)
        case invalidModelPath(String)
        case modelNotInstalled(String)
        case autoDownloadDisabled(String)
        case unsupportedInstallShape(String)
        case downloadFailed(String)
        case invalidInstalledModel(String)

        public var errorDescription: String? {
            switch self {
            case .unknownModelID(let id):
                return "Unknown canonical model id: \(id)"
            case .invalidModelPath(let path):
                return "Model path not found: \(path)"
            case .modelNotInstalled(let id):
                return "Model \(id) is not installed."
            case .autoDownloadDisabled(let id):
                return "Model \(id) is not installed and does not support auto-download in this command."
            case .unsupportedInstallShape(let id):
                return "Managed install shape is not supported for \(id)."
            case .downloadFailed(let message):
                return message
            case .invalidInstalledModel(let message):
                return message
            }
        }
    }

    public static func resolveInstalledModel(
        id: String,
        fileManager: FileManager = .default
    ) -> URL? {
        guard let spec = ManagedModelCatalog.spec(for: id) else {
            return nil
        }
        return spec.managedRuntimeURL(fileManager: fileManager)
    }

    public static func resolveForRuntime(
        requestedModel: String?,
        defaultModelID: String,
        allowAutoDownload: Bool = true,
        fileManager: FileManager = .default,
        progress: (@Sendable (PretrainedModelLoader.ProgressEvent) -> Void)? = nil
    ) async throws -> RuntimeResolution {
        if let explicit = resolveExplicitPath(from: requestedModel, fileManager: fileManager) {
            return RuntimeResolution(
                spec: try requiredSpec(id: defaultModelID),
                url: explicit,
                source: .explicitPath
            )
        }

        let effectiveID = normalizedRequestedID(requestedModel) ?? defaultModelID
        guard let spec = ManagedModelCatalog.spec(for: effectiveID) else {
            if let requestedModel, !requestedModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ResolverError.invalidModelPath(requestedModel)
            }
            throw ResolverError.unknownModelID(effectiveID)
        }

        if let installed = spec.managedRuntimeURL(fileManager: fileManager) {
            return RuntimeResolution(spec: spec, url: installed, source: .managedStore)
        }

        guard allowAutoDownload, spec.runtimeAutoDownloadAllowed else {
            throw ResolverError.autoDownloadDisabled(spec.id)
        }

        switch spec.installShape {
        case .directoryRoot:
            guard let hubFallback = spec.hubFallback else {
                throw ResolverError.downloadFailed(
                    ManagedModelCatalog.missingHubSourceMessage(for: spec.id)
                )
            }

            let snapshotURL = try await downloadHubSnapshot(config: hubFallback, progress: progress)
            let normalized = spec.normalizedRootURL(snapshotURL, fileManager: fileManager)
            let missing = spec.missingPaths(in: normalized, fileManager: fileManager)
            guard missing.isEmpty else {
                throw ResolverError.invalidInstalledModel(
                    "Missing required files after Hugging Face download: \(missing.map(\.lastPathComponent).joined(separator: ", "))"
                )
            }
            return RuntimeResolution(spec: spec, url: normalized, source: .hubSnapshot)

        case .singleFile(let relativePath):
            guard let hubFallback = spec.hubFallback else {
                throw ResolverError.downloadFailed(
                    ManagedModelCatalog.missingHubSourceMessage(for: spec.id)
                )
            }
            do {
                let fileURL = try await PretrainedModelLoader.fromPretrainedFile(
                    modelPath: nil,
                    modelId: spec.id,
                    defaultModelIds: [spec.id],
                    relativePath: relativePath,
                    hubFallback: hubFallback,
                    fileManager: fileManager,
                    validate: { file, manager in
                        spec.validateRuntimeURL(file, fileManager: manager)
                    },
                    progress: progress
                )
                return RuntimeResolution(
                    spec: spec,
                    url: fileURL,
                    source: .hubSnapshot
                )
            } catch let error as PretrainedModelLoader.LoadError {
                throw ResolverError.downloadFailed(error.localizedDescription)
            }

        case .structuredRoot:
            guard let hubFallback = spec.hubFallback else {
                throw ResolverError.downloadFailed(
                    ManagedModelCatalog.missingHubSourceMessage(for: spec.id)
                )
            }

            let snapshotURL = try await downloadHubSnapshot(config: hubFallback, progress: progress)
            try normalizeManagedLayoutIfNeeded(for: spec, in: snapshotURL, fileManager: fileManager)
            let normalized = spec.normalizedRootURL(snapshotURL, fileManager: fileManager)
            let missing = spec.missingPaths(in: normalized, fileManager: fileManager)
            guard missing.isEmpty else {
                throw ResolverError.invalidInstalledModel(
                    "Missing required files after Hugging Face download: \(missing.map(\.path).joined(separator: ", "))"
                )
            }
            return RuntimeResolution(spec: spec, url: normalized, source: .hubSnapshot)
        }
    }

    public static func installManagedModel(
        id: String,
        force: Bool = false,
        fileManager: FileManager = .default,
        progress: (@Sendable (InstallProgress) -> Void)? = nil
    ) async throws -> InstallResult {
        let spec = try requiredSpec(id: id)
        let modelDir = spec.managedInstallRootURL()

        if !force, isManagedInstallComplete(spec: spec, at: modelDir, fileManager: fileManager) {
            let manifest = try? MereRunModelManifest.loadIfPresent(from: modelDir, fileManager: fileManager)
            return InstallResult(spec: spec, installURL: modelDir, manifest: manifest, wasAlreadyInstalled: true)
        }

        if fileManager.fileExists(atPath: modelDir.path) {
            try? fileManager.removeItem(at: modelDir)
        }

        guard let hubFallback = spec.hubFallback else {
            throw ResolverError.downloadFailed(
                ManagedModelCatalog.missingHubSourceMessage(for: spec.id)
            )
        }

        let snapshotURL = try await downloadHubSnapshotWithByteProgress(
            config: hubFallback,
            progress: progress
        )
        try normalizeManagedLayoutIfNeeded(for: spec, in: snapshotURL, fileManager: fileManager)
        try fileManager.createDirectory(at: modelDir.deletingLastPathComponent(), withIntermediateDirectories: true)
        let manifest = try materializeManagedInstallRoot(
            for: spec,
            snapshotURL: snapshotURL,
            modelDir: modelDir,
            fileManager: fileManager
        )
        try installManagedAliasesIfNeeded(for: spec, rootURL: modelDir, fileManager: fileManager)

        let normalized = spec.normalizedRootURL(modelDir, fileManager: fileManager)
        let missing = spec.missingPaths(in: normalized, fileManager: fileManager)
        guard missing.isEmpty else {
            throw ResolverError.invalidInstalledModel(
                "Installed model is incomplete: \(missing.map(\.path).joined(separator: ", "))"
            )
        }
        return InstallResult(spec: spec, installURL: modelDir, manifest: manifest, wasAlreadyInstalled: false)
    }

    public static func isManagedInstallComplete(
        spec: ManagedModelSpec,
        at rootURL: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        guard spec.isManagedRootComplete(rootURL, fileManager: fileManager) else {
            return false
        }
        guard let manifest = try? MereRunModelManifest.loadIfPresent(from: rootURL, fileManager: fileManager),
              manifest.id == spec.id else {
            return false
        }
        return true
    }

    private static func normalizedRequestedID(_ requestedModel: String?) -> String? {
        guard let requestedModel else { return nil }
        let trimmed = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed.lowercased()
    }

    private static func resolveExplicitPath(
        from requestedModel: String?,
        fileManager: FileManager
    ) -> URL? {
        guard let requestedModel else { return nil }
        let trimmed = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let candidate = URL(fileURLWithPath: trimmed).standardizedFileURL
        guard fileManager.fileExists(atPath: candidate.path) else {
            return nil
        }
        return candidate
    }

    private static func requiredSpec(id: String) throws -> ManagedModelSpec {
        guard let spec = ManagedModelCatalog.spec(for: id) else {
            throw ResolverError.unknownModelID(id)
        }
        return spec
    }

    private static func downloadHubSnapshot(
        config: HubFallbackConfig,
        progress: (@Sendable (PretrainedModelLoader.ProgressEvent) -> Void)?
    ) async throws -> URL {
        do {
            let snapshot = try HubSnapshot(
                options: HubSnapshotOptions(
                    repoId: config.repoId,
                    revision: config.revision,
                    patterns: config.patterns
                )
            )
            return try await snapshot.prepare { snapshotProgress in
                let percent = min(100, max(0, Int(snapshotProgress.fractionCompleted * 100)))
                progress?(.downloading(percent: percent))
            }
        } catch {
            throw ResolverError.downloadFailed(error.localizedDescription)
        }
    }

    private static func downloadHubSnapshotWithByteProgress(
        config: HubFallbackConfig,
        progress: (@Sendable (InstallProgress) -> Void)?
    ) async throws -> URL {
        do {
            let snapshot = try HubSnapshot(
                options: HubSnapshotOptions(
                    repoId: config.repoId,
                    revision: config.revision,
                    patterns: config.patterns
                )
            )
            return try await snapshot.prepare { snapshotProgress in
                let percent = min(100, max(0, Int(snapshotProgress.fractionCompleted * 100)))
                progress?(.downloadingPercent(
                    percent: percent,
                    speedBytesPerSecond: snapshotProgress.estimatedSpeedBytesPerSecond
                ))
            }
        } catch {
            throw ResolverError.downloadFailed(error.localizedDescription)
        }
    }

    private static func normalizeManagedLayoutIfNeeded(
        for spec: ManagedModelSpec,
        in rootURL: URL,
        fileManager: FileManager
    ) throws {
        switch spec.normalizationKind {
        case .none, .qwen3ASRNested, .parakeetNested, .musicACEStep:
            return
        }
    }

    static func materializeManagedInstallRoot(
        for spec: ManagedModelSpec,
        snapshotURL: URL,
        modelDir: URL,
        fileManager: FileManager
    ) throws -> MereRunModelManifest? {
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let entries = try fileManager.contentsOfDirectory(
            at: snapshotURL,
            includingPropertiesForKeys: nil,
            options: []
        )
        for entry in entries where entry.lastPathComponent != MereRunModelManifest.filename {
            let linkURL = modelDir.appendingPathComponent(entry.lastPathComponent)
            try fileManager.createSymbolicLink(at: linkURL, withDestinationURL: entry)
        }
        return try MereRunModelManifest.writeTemplateIfKnown(modelId: spec.id, to: modelDir)
    }

    private static func installManagedAliasesIfNeeded(
        for spec: ManagedModelSpec,
        rootURL: URL,
        fileManager: FileManager
    ) throws {
        switch spec.aliasKind {
        case .none:
            return
        case .codegenGGUF:
            let aliasURL = MereRunModelPaths.modelsDir.appendingPathComponent(
                CodeGenResources.managedRelativePath,
                isDirectory: false
            )
            let preferredHubFile = rootURL.appendingPathComponent(CodeGenResources.hubGGUFPath, isDirectory: false)
            let targetURL = fileManager.fileExists(atPath: preferredHubFile.path)
                ? preferredHubFile
                : ManagedModelSpec.findFirstGGUFFile(in: rootURL, fileManager: fileManager)

            guard let targetURL else {
                throw ResolverError.invalidInstalledModel("No GGUF file found under \(rootURL.path)")
            }
            if fileManager.fileExists(atPath: aliasURL.path) {
                try? fileManager.removeItem(at: aliasURL)
            }
            try fileManager.createDirectory(at: aliasURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fileManager.createSymbolicLink(at: aliasURL, withDestinationURL: targetURL)
        }
    }

}
