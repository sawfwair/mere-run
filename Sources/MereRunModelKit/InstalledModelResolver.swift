import Foundation

/// Catalog facts needed for installed-only lookup. Runtime owners provide these
/// values from their catalog and retain responsibility for checkpoint validation.
public struct InstalledModelDescriptor: Hashable, Sendable {
    public let id: ManagedModelID
    public let upstreamRepoID: String?
    public let fallbackIDs: [ManagedModelID]
    public let requiresUsageTermsAcknowledgement: Bool

    public init(id: ManagedModelID, upstreamRepoID: String? = nil, fallbackIDs: [ManagedModelID] = [],
                requiresUsageTermsAcknowledgement: Bool = false) {
        self.id = id
        self.upstreamRepoID = upstreamRepoID
        self.fallbackIDs = fallbackIDs
        self.requiresUsageTermsAcknowledgement = requiresUsageTermsAcknowledgement
    }
}

/// Installed lookup without an inference dependency or implicit download. A
/// candidate is usable only after both location policy and the runtime validator pass.
public struct InstalledModelResolver {
    public struct Resolution: Hashable, Sendable {
        public let requestedModelID: ManagedModelID
        public let installedModelID: ManagedModelID
        public let candidate: ModelLocationCandidate
    }

    public struct ResolverError: LocalizedError, Sendable {
        public let modelID: ManagedModelID
        public let searchedPaths: [URL]
        public let upstreamRepoID: String?

        public var errorDescription: String? {
            var lines = ["Model not found: \(modelID.rawValue)"]
            if let upstreamRepoID { lines.append("Upstream repo: \(upstreamRepoID)") }
            if !searchedPaths.isEmpty {
                lines.append("Searched:")
                lines += searchedPaths.map { "  - \($0.path)" }
            }
            return lines.joined(separator: "\n")
        }
    }

    private let fileManager: FileManager
    private let locations: ModelLocationSnapshot

    public init(fileManager: FileManager = .default, locations: ModelLocationSnapshot) {
        self.fileManager = fileManager
        self.locations = locations
    }

    public func resolve(
        _ modelID: ManagedModelID,
        descriptor: (ManagedModelID) -> InstalledModelDescriptor?,
        validateRuntime: (ManagedModelID, URL) -> Bool
    ) throws -> Resolution {
        let requested = descriptor(modelID)
        let ids = [modelID] + (requested?.fallbackIDs ?? [])
        for id in ids {
            guard let spec = id == modelID ? requested : descriptor(id), spec.id == id else { continue }
            for candidate in locations.candidates(for: id.rawValue) {
                if accepts(candidate, descriptor: spec), validateRuntime(id, candidate.rootURL) {
                    return Resolution(requestedModelID: modelID, installedModelID: id, candidate: candidate)
                }
            }
        }
        throw ResolverError(modelID: modelID,
                            searchedPaths: ids.flatMap { locations.candidates(for: $0.rawValue).map(\.rootURL) },
                            upstreamRepoID: requested?.upstreamRepoID)
    }

    private func accepts(_ candidate: ModelLocationCandidate, descriptor: InstalledModelDescriptor) -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: candidate.rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }
        let manifest: MereRunModelManifest?
        do {
            manifest = try MereRunModelManifest.loadIfPresent(from: candidate.rootURL, fileManager: fileManager)
        } catch {
            return false
        }
        if let manifest {
            guard manifest.id == descriptor.id.rawValue else { return false }
            if candidate.kind.isExternallyManaged, descriptor.requiresUsageTermsAcknowledgement,
               manifest.usageTermsAcknowledged != true, !candidate.usageTermsAcknowledged {
                return false
            }
            return true
        }
        return candidate.kind == .registeredBinding
            && (!descriptor.requiresUsageTermsAcknowledgement || candidate.usageTermsAcknowledged)
    }
}
