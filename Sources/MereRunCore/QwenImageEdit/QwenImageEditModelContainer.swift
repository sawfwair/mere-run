import Foundation

/// Container for Qwen-Image-Edit model resources with caching support.
public actor QwenImageEditModelContainer {
    public enum ContainerError: LocalizedError {
        case missingModelLocation
        case missingModelFiles([URL])

        public var errorDescription: String? {
            switch self {
            case .missingModelLocation:
                return "Qwen-Image-Edit model container has neither a local root URL nor a model spec."
            case .missingModelFiles(let urls):
                let list = urls.map { $0.path }.sorted().joined(separator: "\n")
                return "Qwen-Image-Edit model files missing:\n\(list)"
            }
        }
    }

    public typealias ProgressHandler = @Sendable (PretrainedModelLoader.ProgressEvent) -> Void

    private let localRootURL: URL?
    private let modelSpec: String?

    private var cachedResources: QwenImageEditResources?
    private var cachedConfigs: QwenImageEditModelConfigs?

    /// Initialize with a local directory containing the model files.
    public init(localRootURL: URL) {
        self.localRootURL = localRootURL
        self.modelSpec = nil
    }

    /// Initialize with a model spec resolved via mere.run's R2 model loader.
    public init(modelSpec: String = QwenImageEditRepository.id) {
        self.localRootURL = nil
        self.modelSpec = modelSpec
    }

    /// Create container for the default Qwen-Image-Edit model.
    public static func defaultModel() throws -> QwenImageEditModelContainer {
        QwenImageEditModelContainer(modelSpec: QwenImageEditRepository.id)
    }

    /// Get model resources, downloading if necessary.
    public func resources(
        progressHandler: ProgressHandler? = nil
    ) async throws -> QwenImageEditResources {
        if let cachedResources {
            return cachedResources
        }

        let rootURL: URL
        if let localRootURL {
            rootURL = localRootURL
        } else if let modelSpec {
            rootURL = try await resolveRemoteRoot(modelSpec: modelSpec, progressHandler: progressHandler)
        } else {
            throw ContainerError.missingModelLocation
        }

        let resources = QwenImageEditResources(rootURL: rootURL)
        let missing = resources.validate()
        if !missing.isEmpty {
            throw ContainerError.missingModelFiles(missing)
        }

        cachedResources = resources
        return resources
    }

    /// Get model configurations, downloading if necessary.
    public func configs(
        progressHandler: ProgressHandler? = nil
    ) async throws -> QwenImageEditModelConfigs {
        if let cachedConfigs {
            return cachedConfigs
        }

        let resources = try await resources(progressHandler: progressHandler)
        let configs = try QwenImageEditModelConfigs.load(from: resources)
        cachedConfigs = configs
        return configs
    }

    /// Invalidate all cached data.
    public func invalidateCache() async {
        cachedResources = nil
        cachedConfigs = nil
    }

    /// Get the root URL of the model directory.
    public func rootURL(
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let resources = try await resources(progressHandler: progressHandler)
        return resources.rootURL
    }

    private func resolveRemoteRoot(
        modelSpec: String,
        progressHandler: ProgressHandler?
    ) async throws -> URL {
        try await QwenImageEditRepository.resolveModelRoot(
            modelSpec: modelSpec,
            progress: progressHandler
        )
    }
}

// MARK: - Convenience Factory

extension QwenImageEditModelContainer {
    /// Create container from a model specification string.
    /// Supports local paths or known model aliases.
    public static func from(modelSpec: String) throws -> QwenImageEditModelContainer {
        let fm = FileManager.default
        let localURL = URL(fileURLWithPath: modelSpec).standardizedFileURL
        if fm.fileExists(atPath: localURL.path) {
            return QwenImageEditModelContainer(localRootURL: localURL)
        }
        return QwenImageEditModelContainer(modelSpec: modelSpec)
    }
}
