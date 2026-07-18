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
        case usageTermsNotAcknowledged(String)
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
            case .usageTermsNotAcknowledged(let id):
                return "Model \(id) has third-party usage terms that must be acknowledged before download."
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

        if !spec.mountedHubFallbacks.isEmpty {
            let result = try await installManagedModel(
                id: spec.id,
                fileManager: fileManager,
                progress: { installProgress in
                    switch installProgress {
                    case .downloadingBytes(let completed, let total):
                        guard let total, total > 0 else {
                            progress?(.downloading(percent: 0))
                            return
                        }
                        let percent = min(100, max(0, Int(Double(completed) / Double(total) * 100)))
                        progress?(.downloading(percent: percent))
                    case .downloadingPercent(let percent, _):
                        progress?(.downloading(percent: percent))
                    case .extracting:
                        progress?(.extracting)
                    }
                }
            )
            return RuntimeResolution(
                spec: spec,
                url: spec.normalizedRootURL(result.installURL, fileManager: fileManager),
                source: .managedStore
            )
        }

        switch spec.installShape {
        case .directoryRoot:
            guard let hubFallback = spec.hubFallback else {
                throw ResolverError.downloadFailed(
                    ManagedModelCatalog.missingHubSourceMessage(for: spec.id)
                )
            }

            let snapshotURL = try await downloadHubSnapshot(config: hubFallback, progress: progress)
            try installBundledGeometryLicenseIfNeeded(
                for: spec,
                rootURL: snapshotURL,
                fileManager: fileManager
            )
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
        usageTermsAcknowledged: Bool = false,
        fileManager: FileManager = .default,
        progress: (@Sendable (InstallProgress) -> Void)? = nil
    ) async throws -> InstallResult {
        let spec = try requiredSpec(id: id)
        let modelDir = spec.managedInstallRootURL()

        if !force, isManagedInstallComplete(spec: spec, at: modelDir, fileManager: fileManager) {
            let manifest = try? MereRunModelManifest.loadIfPresent(from: modelDir, fileManager: fileManager)
            return InstallResult(spec: spec, installURL: modelDir, manifest: manifest, wasAlreadyInstalled: true)
        }

        if spec.usageRestriction != nil, !usageTermsAcknowledged {
            throw ResolverError.usageTermsNotAcknowledged(spec.id)
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
        let mountedSnapshots = try await downloadMountedHubSnapshots(
            for: spec,
            progress: progress
        )
        let storageLock = try ModelStorageFileLock.acquire(
            hubDirectory: HubSnapshot.resolvedDownloadBase(fileManager: fileManager),
            fileManager: fileManager
        )
        defer { storageLock.unlock() }
        try normalizeManagedLayoutIfNeeded(for: spec, in: snapshotURL, fileManager: fileManager)
        try fileManager.createDirectory(at: modelDir.deletingLastPathComponent(), withIntermediateDirectories: true)
        let manifest = try materializeManagedInstallRoot(
            for: spec,
            snapshotURL: snapshotURL,
            mountedSnapshots: mountedSnapshots,
            modelDir: modelDir,
            usageTermsAcknowledged: usageTermsAcknowledged,
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

    private static func installBundledGeometryLicenseIfNeeded(
        for spec: ManagedModelSpec,
        rootURL: URL,
        fileManager: FileManager
    ) throws {
        guard let pin = GeometryModelPins.pin(for: spec.id), pin.licenseEvidence != nil else {
            return
        }
        try pin.installBundledLicenseEvidence(in: rootURL, fileManager: fileManager)
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
                guard !config.patterns.isEmpty else {
                    let percent = min(100, max(0, Int(snapshotProgress.fractionCompleted * 100)))
                    progress?(.downloadingPercent(
                        percent: percent,
                        speedBytesPerSecond: snapshotProgress.estimatedSpeedBytesPerSecond
                    ))
                    return
                }

                let total = snapshotProgress.totalUnitCount > 0 ? snapshotProgress.totalUnitCount : nil
                progress?(.downloadingBytes(
                    completed: max(0, snapshotProgress.completedUnitCount),
                    total: total
                ))
            }
        } catch {
            throw ResolverError.downloadFailed(error.localizedDescription)
        }
    }

    private static func downloadMountedHubSnapshots(
        for spec: ManagedModelSpec,
        progress: (@Sendable (InstallProgress) -> Void)?
    ) async throws -> [(MountedHubFallbackConfig, URL)] {
        var snapshots: [(MountedHubFallbackConfig, URL)] = []
        snapshots.reserveCapacity(spec.mountedHubFallbacks.count)

        for mounted in spec.mountedHubFallbacks {
            let snapshotURL = try await downloadHubSnapshotWithByteProgress(
                config: mounted.hubFallback,
                progress: progress
            )
            snapshots.append((mounted, snapshotURL))
        }

        return snapshots
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
        mountedSnapshots: [(MountedHubFallbackConfig, URL)] = [],
        modelDir: URL,
        usageTermsAcknowledged: Bool = false,
        fileManager: FileManager
    ) throws -> MereRunModelManifest? {
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let mountedDirectories = materializedDirectories(for: mountedSnapshots.map { $0.0.destinationPath })
        let mountedDestinationPaths = Set(mountedSnapshots.map { normalizedRelativePath($0.0.destinationPath) })
        try materializeSnapshotEntries(
            from: snapshotURL,
            to: modelDir,
            fileManager: fileManager,
            includedPatterns: spec.hubFallback?.patterns ?? [],
            materializedDirectoryPaths: mountedDirectories,
            excludedRelativePaths: mountedDestinationPaths
        )

        for (mounted, mountedSnapshotURL) in mountedSnapshots {
            let destinationURL = modelDir.appendingPathComponent(mounted.destinationPath, isDirectory: true)
            let sourceURL = mountedSnapshotSourceURL(
                snapshotURL: mountedSnapshotURL,
                destinationPath: mounted.destinationPath,
                fileManager: fileManager
            )
            try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true)
            try materializeSnapshotEntries(
                from: sourceURL,
                to: destinationURL,
                fileManager: fileManager,
                includedPatterns: mountedPatterns(
                    mounted.hubFallback.patterns,
                    destinationPath: mounted.destinationPath,
                    sourceURL: sourceURL,
                    snapshotURL: mountedSnapshotURL
                )
            )
        }

        try installBundledGeometryLicenseIfNeeded(
            for: spec,
            rootURL: spec.normalizedRootURL(modelDir, fileManager: fileManager),
            fileManager: fileManager
        )

        return try MereRunModelManifest.writeTemplateIfKnown(
            modelId: spec.id,
            to: modelDir,
            usageTermsAcknowledged: usageTermsAcknowledged
        )
    }

    private static func mountedSnapshotSourceURL(
        snapshotURL: URL,
        destinationPath: String,
        fileManager: FileManager
    ) -> URL {
        let exactSourceURL = snapshotURL.appendingPathComponent(destinationPath, isDirectory: true)
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: exactSourceURL.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return exactSourceURL
        }
        return snapshotURL
    }

    private static func materializeSnapshotEntries(
        from snapshotURL: URL,
        to destinationRoot: URL,
        fileManager: FileManager,
        relativePath: String = "",
        includedPatterns: [String] = [],
        materializedDirectoryPaths: Set<String> = [],
        excludedRelativePaths: Set<String> = []
    ) throws {
        let entries = try fileManager.contentsOfDirectory(
            at: snapshotURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: []
        )
        for entry in entries where entry.lastPathComponent != MereRunModelManifest.filename {
            let entryRelativePath = relativePath.isEmpty
                ? entry.lastPathComponent
                : "\(relativePath)/\(entry.lastPathComponent)"
            if isExcludedRelativePath(entryRelativePath, excludedRelativePaths: excludedRelativePaths) {
                continue
            }

            let linkURL = destinationRoot.appendingPathComponent(entry.lastPathComponent)
            let isDirectory = (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            if isInternalSnapshotPath(entryRelativePath)
                || !shouldIncludeSnapshotEntry(
                    entryRelativePath,
                    isDirectory: isDirectory,
                    includedPatterns: includedPatterns
                ) {
                continue
            }
            if isDirectory && shouldMaterializeDirectory(entryRelativePath, materializedDirectoryPaths: materializedDirectoryPaths) {
                try fileManager.createDirectory(at: linkURL, withIntermediateDirectories: true)
                try materializeSnapshotEntries(
                    from: entry,
                    to: linkURL,
                    fileManager: fileManager,
                    relativePath: entryRelativePath,
                    includedPatterns: includedPatterns,
                    materializedDirectoryPaths: materializedDirectoryPaths,
                    excludedRelativePaths: excludedRelativePaths
                )
                continue
            }

            if fileManager.fileExists(atPath: linkURL.path) {
                try fileManager.removeItem(at: linkURL)
            }
            try fileManager.createSymbolicLink(at: linkURL, withDestinationURL: entry)
        }
    }

    private static func mountedPatterns(
        _ patterns: [String],
        destinationPath: String,
        sourceURL: URL,
        snapshotURL: URL
    ) -> [String] {
        guard sourceURL.standardizedFileURL.path != snapshotURL.standardizedFileURL.path else {
            return patterns
        }
        let prefix = normalizedRelativePath(destinationPath) + "/"
        return patterns.compactMap { pattern in
            pattern.hasPrefix(prefix) ? String(pattern.dropFirst(prefix.count)) : nil
        }
    }

    private static func isInternalSnapshotPath(_ relativePath: String) -> Bool {
        let components = relativePath.split(separator: "/")
        return components.contains(".cache")
            || relativePath == HubSnapshotReceipt.filename
    }

    private static func shouldIncludeSnapshotEntry(
        _ relativePath: String,
        isDirectory: Bool,
        includedPatterns: [String]
    ) -> Bool {
        guard !includedPatterns.isEmpty else { return true }
        if HubSnapshot.matchesPath(relativePath, patterns: includedPatterns) {
            return true
        }
        guard isDirectory else { return false }
        let prefix = relativePath + "/"
        return includedPatterns.contains { $0.hasPrefix(prefix) }
    }

    private static func materializedDirectories(for mountedDestinationPaths: [String]) -> Set<String> {
        var directories = Set<String>()
        for destinationPath in mountedDestinationPaths {
            let components = normalizedRelativePath(destinationPath)
                .split(separator: "/")
                .map(String.init)
            guard !components.isEmpty else { continue }
            var current: [String] = []
            for component in components {
                current.append(component)
                directories.insert(current.joined(separator: "/"))
            }
        }
        return directories
    }

    private static func shouldMaterializeDirectory(
        _ relativePath: String,
        materializedDirectoryPaths: Set<String>
    ) -> Bool {
        materializedDirectoryPaths.contains(relativePath)
            || materializedDirectoryPaths.contains(where: { $0.hasPrefix("\(relativePath)/") })
    }

    private static func isExcludedRelativePath(
        _ relativePath: String,
        excludedRelativePaths: Set<String>
    ) -> Bool {
        excludedRelativePaths.contains(relativePath)
            || excludedRelativePaths.contains(where: { relativePath.hasPrefix("\($0)/") })
    }

    private static func normalizedRelativePath(_ path: String) -> String {
        path.split(separator: "/").joined(separator: "/")
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
