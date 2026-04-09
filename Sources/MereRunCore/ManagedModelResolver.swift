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
            guard let archiveSource = spec.archiveSource else {
                if let hubFallback = spec.hubFallback {
                    let snapshotURL = try await downloadHubSnapshot(config: hubFallback, progress: progress)
                    let normalized = spec.normalizedRootURL(snapshotURL, fileManager: fileManager)
                    let missing = spec.missingPaths(in: normalized, fileManager: fileManager)
                    guard missing.isEmpty else {
                        throw ResolverError.invalidInstalledModel(
                            "Missing required files after Hugging Face download: \(missing.map(\.lastPathComponent).joined(separator: ", "))"
                        )
                    }
                    return RuntimeResolution(spec: spec, url: normalized, source: .hubSnapshot)
                }
                throw ResolverError.downloadFailed(
                    MereRunModelSourceConfiguration.missingConfigurationMessage(
                        purpose: "Managed model downloads for \(spec.id)"
                    )
                )
            }

            do {
                let rootURL = try await PretrainedModelLoader.fromPretrainedArchive(
                    modelPath: nil,
                    modelId: spec.id,
                    defaultModelIds: [spec.id],
                    storageId: spec.id,
                    archiveKey: archiveSource.key,
                    archiveSize: archiveSource.size,
                    hubFallback: spec.hubFallback,
                    strictArchiveSize: archiveSource.size > 0,
                    fileManager: fileManager,
                    normalize: { root, manager in
                        spec.normalizedRootURL(root, fileManager: manager)
                    },
                    validate: { root, manager in
                        spec.missingPaths(in: root, fileManager: manager)
                    },
                    progress: progress
                )
                _ = try MereRunModelManifest.writeTemplateIfKnown(modelId: spec.id, to: rootURL)
                return RuntimeResolution(spec: spec, url: rootURL, source: .managedStore)
            } catch let error as PretrainedModelLoader.LoadError {
                throw ResolverError.downloadFailed(error.localizedDescription)
            }

        case .singleFile(let relativePath):
            guard let archiveSource = spec.archiveSource else {
                throw ResolverError.unsupportedInstallShape(spec.id)
            }
            do {
                let fileURL = try await PretrainedModelLoader.fromPretrainedFile(
                    modelPath: nil,
                    modelId: spec.id,
                    defaultModelIds: [spec.id],
                    relativePath: relativePath,
                    remoteKey: archiveSource.key,
                    expectedSize: archiveSource.size,
                    hubFallback: spec.hubFallback,
                    strictSizeCheck: archiveSource.size > 0,
                    fileManager: fileManager,
                    validate: { file, manager in
                        spec.validateRuntimeURL(file, fileManager: manager)
                    },
                    progress: progress
                )
                return RuntimeResolution(
                    spec: spec,
                    url: fileURL,
                    source: fileURL.path.contains("/hub/") ? .hubSnapshot : .managedStore
                )
            } catch let error as PretrainedModelLoader.LoadError {
                throw ResolverError.downloadFailed(error.localizedDescription)
            }

        case .structuredRoot:
            if !MereRunModelSourceConfiguration.hasAnyDownloadSource() {
                guard let hubFallback = spec.hubFallback else {
                    throw ResolverError.downloadFailed(
                        MereRunModelSourceConfiguration.missingConfigurationMessage(
                            purpose: "Managed model downloads for \(spec.id)"
                        )
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

            let installed = try await installManagedModel(
                id: spec.id,
                force: false,
                fileManager: fileManager,
                progress: { installProgress in
                    switch installProgress {
                    case .downloadingBytes(let completed, let total):
                        if let total, total > 0 {
                            let percent = min(100, max(0, Int(Double(completed) / Double(total) * 100)))
                            progress?(.downloading(percent: percent))
                        }
                    case .extracting:
                        progress?(.extracting)
                    }
                }
            )
            return RuntimeResolution(spec: spec, url: installed.installURL, source: .managedStore)
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

        if !force, spec.isManagedRootComplete(modelDir, fileManager: fileManager) {
            let manifest = try? MereRunModelManifest.loadIfPresent(from: modelDir, fileManager: fileManager)
            return InstallResult(spec: spec, installURL: modelDir, manifest: manifest, wasAlreadyInstalled: true)
        }

        if fileManager.fileExists(atPath: modelDir.path) {
            try? fileManager.removeItem(at: modelDir)
        }

        if MereRunModelSourceConfiguration.hasAnyDownloadSource(),
           let archiveSource = spec.archiveSource {
            try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
            let archiveFile = MereRunModelPaths.downloadsDir.appendingPathComponent(archiveSource.key)
            try fileManager.createDirectory(at: archiveFile.deletingLastPathComponent(), withIntermediateDirectories: true)

            try await downloadArchive(
                spec: spec,
                archiveSource: archiveSource,
                to: archiveFile,
                fileManager: fileManager,
                progress: progress
            )
            progress?(.extracting)
            try extractArchive(archiveFile, to: modelDir)
            try? fileManager.removeItem(at: archiveFile)
            try normalizeManagedLayoutIfNeeded(for: spec, in: modelDir, fileManager: fileManager)
            let normalizedRoot = spec.normalizedRootURL(modelDir, fileManager: fileManager)
            let manifest = try MereRunModelManifest.writeTemplateIfKnown(modelId: spec.id, to: modelDir)
            try installManagedAliasesIfNeeded(for: spec, rootURL: modelDir, fileManager: fileManager)
            let missing = spec.missingPaths(in: normalizedRoot, fileManager: fileManager)
            guard missing.isEmpty else {
                throw ResolverError.invalidInstalledModel(
                    "Installed model is incomplete: \(missing.map(\.path).joined(separator: ", "))"
                )
            }
            return InstallResult(spec: spec, installURL: modelDir, manifest: manifest, wasAlreadyInstalled: false)
        }

        guard let hubFallback = spec.hubFallback else {
            throw ResolverError.downloadFailed(
                MereRunModelSourceConfiguration.missingConfigurationMessage(
                    purpose: "Managed model downloads for \(spec.id)"
                )
            )
        }

        let snapshotURL = try await downloadHubSnapshot(
            config: hubFallback,
            progress: { event in
                switch event {
                case .downloading(let percent):
                    progress?(.downloadingBytes(completed: Int64(percent), total: 100))
                case .extracting:
                    progress?(.extracting)
                }
            }
        )
        try normalizeManagedLayoutIfNeeded(for: spec, in: snapshotURL, fileManager: fileManager)
        let manifest = try MereRunModelManifest.writeTemplateIfKnown(modelId: spec.id, to: snapshotURL)
        try installManagedAliasesIfNeeded(for: spec, rootURL: snapshotURL, fileManager: fileManager)
        try fileManager.createDirectory(at: modelDir.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fileManager.createSymbolicLink(at: modelDir, withDestinationURL: snapshotURL)

        let normalized = spec.normalizedRootURL(modelDir, fileManager: fileManager)
        let missing = spec.missingPaths(in: normalized, fileManager: fileManager)
        guard missing.isEmpty else {
            throw ResolverError.invalidInstalledModel(
                "Installed model is incomplete: \(missing.map(\.path).joined(separator: ", "))"
            )
        }
        return InstallResult(spec: spec, installURL: modelDir, manifest: manifest, wasAlreadyInstalled: false)
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

    private static func downloadArchive(
        spec: ManagedModelSpec,
        archiveSource: ManagedModelArchiveSource,
        to archiveFile: URL,
        fileManager: FileManager,
        progress: (@Sendable (InstallProgress) -> Void)?
    ) async throws {
        let environment = ProcessInfo.processInfo.environment
        let usePackagedPublicSource =
            !MereRunModelSourceConfiguration.hasExplicitDownloadConfiguration(environment: environment)
            && MereRunModelSourceConfiguration.publicBaseURL(environment: environment) != nil

        let requestKey: String
        if usePackagedPublicSource {
            guard let packagedKey = archiveSource.packagedKey else {
                throw ResolverError.downloadFailed(
                    "No packaged public download is configured for \(spec.id)."
                )
            }
            requestKey = packagedKey
        } else {
            requestKey = archiveSource.key
        }

        let request: URLRequest
        do {
            request = try await R2DownloadRequestBuilder.makeGETRequest(
                key: requestKey,
                environment: environment
            ).request
        } catch let error as R2DownloadRequestBuilder.BuildError {
            throw ResolverError.downloadFailed(error.localizedDescription)
        }

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw ResolverError.downloadFailed("Download failed: HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }

        let totalSize = httpResponse.expectedContentLength > 0 ? httpResponse.expectedContentLength : nil
        let tempURL = archiveFile.appendingPathExtension("partial")
        try? fileManager.removeItem(at: tempURL)
        fileManager.createFile(atPath: tempURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tempURL)
        defer { try? handle.close() }

        var bytesWritten: Int64 = 0
        var buffer = Data()
        let bufferSize = 1024 * 1024
        for try await byte in asyncBytes {
            buffer.append(byte)
            if buffer.count >= bufferSize {
                try handle.write(contentsOf: buffer)
                bytesWritten += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                progress?(.downloadingBytes(completed: bytesWritten, total: totalSize))
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            bytesWritten += Int64(buffer.count)
            progress?(.downloadingBytes(completed: bytesWritten, total: totalSize))
        }
        try handle.synchronize()

        if fileManager.fileExists(atPath: archiveFile.path) {
            try? fileManager.removeItem(at: archiveFile)
        }
        try fileManager.moveItem(at: tempURL, to: archiveFile)
    }

    private static func extractArchive(_ archiveFile: URL, to destination: URL) throws {
        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archiveFile.path, "-C", destination.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ResolverError.downloadFailed("tar extraction failed (exit \(process.terminationStatus))")
        }
        #else
        throw ResolverError.downloadFailed("Archive extraction is only supported on macOS.")
        #endif
    }

    private static func normalizeManagedLayoutIfNeeded(
        for spec: ManagedModelSpec,
        in rootURL: URL,
        fileManager: FileManager
    ) throws {
        switch spec.normalizationKind {
        case .none, .qwen3ASRNested, .parakeetNested:
            return
        case .musicACEStep:
            try renameDirectoryIfPresent(
                from: rootURL.appendingPathComponent("acestep-v15-turbo", isDirectory: true),
                to: rootURL.appendingPathComponent("music-acestep-v15-turbo", isDirectory: true),
                fileManager: fileManager
            )
            let renamePairs: [(String, String)] = [
                ("acestep-5Hz-lm-1.7B", "music-acestep-5hz-lm-1.7b"),
                ("acestep-5hz-lm-1.7b", "music-acestep-5hz-lm-1.7b"),
                ("acestep-5Hz-lm", "music-acestep-5hz-lm"),
                ("acestep-5hz-lm", "music-acestep-5hz-lm"),
                ("acestep-lm", "music-acestep-lm"),
            ]
            for (fromName, toName) in renamePairs {
                try renameDirectoryIfPresent(
                    from: rootURL.appendingPathComponent(fromName, isDirectory: true),
                    to: rootURL.appendingPathComponent(toName, isDirectory: true),
                    fileManager: fileManager
                )
            }
        }
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

    private static func renameDirectoryIfPresent(
        from source: URL,
        to destination: URL,
        fileManager: FileManager
    ) throws {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: source.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return
        }
        guard !fileManager.fileExists(atPath: destination.path) else {
            throw ResolverError.invalidInstalledModel("Destination already exists at \(destination.path)")
        }
        try fileManager.moveItem(at: source, to: destination)
    }
}
