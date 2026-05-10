import Foundation

public actor ZImageTurboModelContainer {
    public enum ContainerError: LocalizedError {
        case missingModelLocation
        case missingModelFiles([URL])

        public var errorDescription: String? {
            switch self {
            case .missingModelLocation:
                return "Z-Image Turbo model container has neither a local root URL nor a model spec."
            case .missingModelFiles(let urls):
                let list = urls.map { $0.path }.sorted().joined(separator: "\n")
                return "Z-Image Turbo model files missing:\n\(list)"
            }
        }
    }

    public typealias ProgressHandler = @Sendable (PretrainedModelLoader.ProgressEvent) -> Void

    private let localRootURL: URL?
    private let modelSpec: String?

    private var cachedResources: ZImageTurboResources?
    private var cachedConfigs: ZImageTurboModelConfigs?

    public init(localRootURL: URL) {
        self.localRootURL = localRootURL
        self.modelSpec = nil
    }

    public init(modelSpec: String = ZImageTurboRepository.defaultModelSpec) {
        self.localRootURL = nil
        self.modelSpec = modelSpec
    }

    public func resources(
        progressHandler: ProgressHandler? = nil
    ) async throws -> ZImageTurboResources {
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

        let resources = ZImageTurboResources(rootURL: rootURL)
        let missing = resources.validate()
        if !missing.isEmpty {
            throw ContainerError.missingModelFiles(missing)
        }

        cachedResources = resources
        return resources
    }

    public func configs(
        progressHandler: ProgressHandler? = nil
    ) async throws -> ZImageTurboModelConfigs {
        if let cachedConfigs {
            return cachedConfigs
        }

        let resources = try await resources(progressHandler: progressHandler)
        let configs = try ZImageTurboModelConfigs.load(from: resources)
        cachedConfigs = configs
        return configs
    }

    public func invalidateCache() async {
        cachedResources = nil
        cachedConfigs = nil
    }

    private func resolveRemoteRoot(
        modelSpec: String,
        progressHandler: ProgressHandler?
    ) async throws -> URL {
        try await ZImageTurboRepository.resolveRemoteModelRoot(
            modelSpec: modelSpec,
            progress: progressHandler
        )
    }
}
