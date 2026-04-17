import Foundation
import MereRunCore

/// Backward-compatible shim over the shared managed model catalog.
enum R2ModelRegistry {
    struct ModelEntry {
        let id: String
        let category: String
        let archiveKey: String
        let packagedArchiveKey: String?
        let upstreamRepoId: String?
        let upstreamRevision: String?
    }

    static let allEntries: [ModelEntry] = ManagedModelCatalog.allSpecs.compactMap { spec in
        guard let archive = spec.archiveSource else {
            return nil
        }
        return ModelEntry(
            id: spec.id,
            category: spec.category.rawValue,
            archiveKey: archive.key,
            packagedArchiveKey: archive.packagedKey,
            upstreamRepoId: spec.upstreamRepoId,
            upstreamRevision: spec.upstreamRevision
        )
    }

    static func entry(for id: String) -> ModelEntry? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allEntries.first { $0.id == normalized }
    }

    static func archiveKey(for modelID: ModelResolver.ModelID) -> String? {
        entry(for: modelID.rawValue)?.archiveKey
    }
}
