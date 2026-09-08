import Foundation

/// Shared model lookup utilities for mere.run's public model families.
///
/// Phase 1 scope:
/// - Resolve known model IDs (e.g. `image-klein-max`, `image-zimage-max`) to a local model root directory.
/// - Resolve from mere.run's configured local model store (default: `~/Library/Application Support/MereRun/models/...`).
///
/// This intentionally does *not* download models; it only resolves paths.
public struct ModelResolver {
    public typealias ModelID = ManagedModelID

    public enum Source: String, Hashable, Sendable {
        /// `.../models/<id>` under the configured local mere.run model store.
        case localModelStore
        /// An explicit canonical-model-id to directory binding.
        case registeredBinding
        /// A canonical `<root>/<model-id>` directory under a registered read-only root.
        case registeredSearchRoot
    }

    public struct Resolution: Hashable, Sendable {
        public let modelID: ModelID
        public let rootURL: URL
        public let source: Source
        public let catalogRootURL: URL?

        public init(
            modelID: ModelID,
            rootURL: URL,
            source: Source,
            catalogRootURL: URL? = nil
        ) {
            self.modelID = modelID
            self.rootURL = rootURL
            self.source = source
            self.catalogRootURL = catalogRootURL
        }

        public var isExternallyManaged: Bool {
            source != .localModelStore
        }
    }

    public enum ResolverError: LocalizedError, Sendable {
        case applicationSupportUnavailable
        case modelNotFound(ModelID, searched: [URL], upstreamRepoId: String?)

        public var errorDescription: String? {
            switch self {
            case .applicationSupportUnavailable:
                return "Could not locate Application Support directory."
            case .modelNotFound(let id, let searched, let upstreamRepoId):
                var lines: [String] = []
                lines.append("Model not found: \(id.rawValue)")
                if let upstreamRepoId {
                    lines.append("Upstream repo: \(upstreamRepoId)")
                }
                if !searched.isEmpty {
                    lines.append("Searched:")
                    lines.append(contentsOf: searched.map { "  - \($0.path)" })
                }
                return lines.joined(separator: "\n")
            }
        }
    }

    private let fileManager: FileManager
    private let locations: ModelLocationSnapshot

    public init(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        locations: ModelLocationSnapshot? = nil
    ) {
        self.fileManager = fileManager
        self.locations = locations ?? MereRunModelLocations.snapshot(
            fileManager: fileManager,
            environment: environment
        )
    }

    public func resolve(_ modelID: ModelID) throws -> Resolution {
        let resolver = InstalledModelResolver(fileManager: fileManager, locations: locations)
        do {
            let result = try resolver.resolve(
                modelID,
                descriptor: { id in
                    ManagedModelCatalog.spec(for: id.rawValue).map { spec in
                        InstalledModelDescriptor(
                            id: id, upstreamRepoID: spec.upstreamRepoId,
                            fallbackIDs: spec.resolutionFallbackIDs.compactMap(ModelID.init(rawValue:)),
                            requiresUsageTermsAcknowledgement: spec.usageRestriction != nil
                        )
                    }
                },
                validateRuntime: { id, root in
                    ManagedModelCatalog.spec(for: id.rawValue)?.isManagedRuntimeReady(root, fileManager: fileManager) == true
                }
            )
            return Resolution(modelID: modelID, rootURL: result.candidate.rootURL,
                              source: source(for: result.candidate.kind), catalogRootURL: result.candidate.catalogRootURL)
        } catch let error as InstalledModelResolver.ResolverError {
            throw ResolverError.modelNotFound(modelID, searched: error.searchedPaths, upstreamRepoId: error.upstreamRepoID)
        }
    }

    public func resolveIfPresent(_ modelID: ModelID) -> Resolution? {
        try? resolve(modelID)
    }

    /// The writable primary store precedes explicit bindings and registered roots.
    public func locationCandidates(for modelID: ModelID) -> [ModelLocationCandidate] {
        locations.candidates(for: modelID.rawValue)
    }

    private func source(for kind: ModelLocationKind) -> Source {
        switch kind {
        case .primaryStore: .localModelStore
        case .registeredBinding: .registeredBinding
        case .registeredSearchRoot: .registeredSearchRoot
        }
    }
}
