import Foundation

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
        case extractionFailed

        public var errorDescription: String? {
            switch self {
            case .unsupportedModelId(let modelId):
                return "Unsupported model id: \(modelId)"
            case .missingFiles(let files):
                return "Missing required files: \(files.joined(separator: ", "))"
            case .downloadFailed(let message):
                return "Download failed: \(message)"
            case .extractionFailed:
                return "Failed to extract model archive"
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

    /// Resolve from local path, local modelId path, or managed archive model in mere.run storage.
    public static func fromPretrainedArchive(
        modelPath: String?,
        modelId: String,
        defaultModelIds: Set<String>,
        storageId: String,
        archiveKey: String,
        archiveSize: Int64,
        strictArchiveSize: Bool = true,
        fileManager: FileManager = .default,
        normalize: (URL, FileManager) -> URL = { base, _ in base },
        validate: (URL, FileManager) -> [URL],
        progress: (@Sendable (ProgressEvent) -> Void)? = nil,
        onArchiveSizeMismatch: (@Sendable (Int64, Int64) -> Void)? = nil
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

        let archiveFile = MereRunModelPaths.downloadsDir.appendingPathComponent(archiveKey)
        try fileManager.createDirectory(at: managed.modelDir, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: archiveFile.deletingLastPathComponent(), withIntermediateDirectories: true)

        progress?(.downloading(percent: 0))
        try await downloadR2Object(
            key: archiveKey,
            to: archiveFile,
            expectedSize: archiveSize,
            strictSizeCheck: strictArchiveSize,
            sizeMismatchPrefix: "Archive size mismatch",
            fileManager: fileManager,
            progressPercent: { percent in
                progress?(.downloading(percent: percent))
            },
            onSizeMismatch: onArchiveSizeMismatch
        )

        progress?(.extracting)
        try extractArchive(archiveFile, to: managed.modelDir)
        try? fileManager.removeItem(at: archiveFile)

        let resolvedRoot = normalize(managed.modelDir, fileManager)
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
        remoteKey: String,
        expectedSize: Int64,
        strictSizeCheck: Bool = true,
        sizeMismatchPrefix: String = "File size mismatch",
        fileManager: FileManager = .default,
        validate: (URL, FileManager) -> [URL],
        progress: (@Sendable (ProgressEvent) -> Void)? = nil,
        onSizeMismatch: (@Sendable (Int64, Int64) -> Void)? = nil
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

        try fileManager.createDirectory(
            at: managedFile.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        progress?(.downloading(percent: 0))
        try await downloadR2Object(
            key: remoteKey,
            to: managedFile,
            expectedSize: expectedSize,
            strictSizeCheck: strictSizeCheck,
            sizeMismatchPrefix: sizeMismatchPrefix,
            fileManager: fileManager,
            progressPercent: { percent in
                progress?(.downloading(percent: percent))
            },
            onSizeMismatch: onSizeMismatch
        )

        let missingAfter = validate(managedFile, fileManager)
        guard missingAfter.isEmpty else {
            throw LoadError.missingFiles(missingAfter.map(\.lastPathComponent))
        }

        return managedFile
    }

    private static func downloadR2Object(
        key: String,
        to destination: URL,
        expectedSize: Int64,
        strictSizeCheck: Bool,
        sizeMismatchPrefix: String,
        fileManager: FileManager,
        progressPercent: (@Sendable (Int) -> Void)?,
        onSizeMismatch: (@Sendable (Int64, Int64) -> Void)?
    ) async throws {
        let request = try await R2DownloadRequestBuilder.makeGETRequest(key: key).request

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw LoadError.downloadFailed("HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }

        let tempURL = destination.appendingPathExtension("partial")
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

                if expectedSize > 0 {
                    let percent = min(100, Int(Double(bytesWritten) / Double(expectedSize) * 100))
                    progressPercent?(percent)
                }
            }
        }

        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            bytesWritten += Int64(buffer.count)
        }

        try handle.synchronize()

        if expectedSize > 0 {
            let percent = min(100, Int(Double(bytesWritten) / Double(expectedSize) * 100))
            progressPercent?(percent)
        }

        if fileManager.fileExists(atPath: destination.path) {
            try? fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: tempURL, to: destination)

        if expectedSize > 0 {
            let actualSize: Int64
            if let attrs = try? fileManager.attributesOfItem(atPath: destination.path),
               let size = attrs[.size] as? Int64 {
                actualSize = size
            } else {
                actualSize = bytesWritten
            }

            if actualSize != expectedSize {
                if strictSizeCheck {
                    throw LoadError.downloadFailed(
                        "\(sizeMismatchPrefix) (got \(actualSize), expected \(expectedSize))"
                    )
                }
                onSizeMismatch?(actualSize, expectedSize)
            }
        }
    }

    private static func extractArchive(_ archiveFile: URL, to modelDir: URL) throws {
        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archiveFile.path, "-C", modelDir.path]
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw LoadError.extractionFailed
        }
        #else
        throw LoadError.extractionFailed
        #endif
    }
}
