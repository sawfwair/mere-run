import ArgumentParser
import Foundation
import MereRunCore

struct ModelRemove: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remove",
        abstract: "Remove a model from the local model store."
    )

    @Argument(help: "Canonical model id (for example: image-klein-nano).")
    var target: String

    @Flag(name: [.long], help: "Skip confirmation prompt.")
    var force: Bool = false

    @Flag(name: [.long], help: "Remove model links but retain unshared Hub payloads.")
    var keepCache: Bool = false

    @Flag(name: [.long], help: "Emit a structured JSON removal result. Requires --force.")
    var json: Bool = false

    func run() throws {
        let id = resolveID(target)
        guard let id else {
            throw ValidationError("Unknown canonical model id: \(target)")
        }
        if json && !force {
            throw ValidationError("--json requires --force because confirmation text would make stdout invalid JSON.")
        }

        let resolver = ModelResolver()
        let modelID = ModelResolver.ModelID(rawValue: id)
        let installURL: URL
        if let modelID, let resolved = resolver.resolveIfPresent(modelID) {
            installURL = resolved.rootURL
        } else {
            if let aliasFallback = gemmaAliasInstallURL(for: id) {
                installURL = aliasFallback
            } else {
                let flatDir = MereRunModelPaths.modelDir(id)
                var isDir: ObjCBool = false
                guard FileManager.default.fileExists(atPath: flatDir.path, isDirectory: &isDir), isDir.boolValue else {
                    throw ValidationError("\(id) is not installed.")
                }
                installURL = flatDir
            }
        }

        let storageManager = try ModelStorageManager()
        let storageUsage = try storageManager.report(modelIDs: [id]).models.first
        let cacheUnits = try storageManager.cacheUnitsReferenced(by: id)
        let referencedBytes = storageUsage?.referencedBytes ?? FileSystemHelper.directorySize(at: installURL)
        let expectedReclaimableBytes = storageUsage?.reclaimableBytes ?? 0
        let referencedSize = ModelStorageCommandOutput.bytes(referencedBytes)
        let reclaimableSize = ModelStorageCommandOutput.bytes(expectedReclaimableBytes)

        if !force {
            print("Remove \(id)?")
            print("  Path: \(installURL.path)")
            print("  Referenced: \(referencedSize)")
            print("  Reclaimable now: \(keepCache ? "0 bytes (--keep-cache)" : reclaimableSize)")
            if let sharedBytes = storageUsage?.sharedBytes, sharedBytes > 0 {
                print("  Shared and preserved: \(ModelStorageCommandOutput.bytes(sharedBytes))")
            }
            if let externalBytes = storageUsage?.externalBytes, externalBytes > 0 {
                print("  External and preserved: \(ModelStorageCommandOutput.bytes(externalBytes))")
            }
            print("")
            print("Confirm? [y/N] ", terminator: "")
            guard let answer = readLine()?.lowercased(), answer == "y" || answer == "yes" else {
                print("Aborted.")
                return
            }
        }

        try FileManager.default.removeItem(at: installURL)
        try removeManagedAliasesIfNeeded(for: id)
        let garbagePlan = keepCache
            ? ModelStorageGarbagePlan(
                hubPath: storageManager.hubDirectory.path,
                reclaimableBytes: 0,
                incompleteDownloadBytes: 0,
                items: []
            )
            : try storageManager.garbageCollectionPlan(limitingTo: cacheUnits)
        let garbageResult = keepCache ? nil : try storageManager.execute(garbagePlan)
        let reclaimedBytes = (garbageResult?.reclaimedBytes ?? 0) + (storageUsage?.localBytes ?? 0)
        let result = ModelRemoveOutput(
            id: id,
            installPath: installURL.path,
            referencedBytes: referencedBytes,
            expectedReclaimableBytes: expectedReclaimableBytes,
            reclaimedBytes: reclaimedBytes,
            retainedSharedBytes: storageUsage?.sharedBytes ?? 0,
            retainedExternalBytes: storageUsage?.externalBytes ?? 0,
            cacheRetained: keepCache,
            deletedCacheItemCount: garbageResult?.deletedItemCount ?? 0
        )
        if json {
            print(try ModelStorageCommandOutput.encode(result))
        } else {
            var message = "Removed \(id); reclaimed \(ModelStorageCommandOutput.bytes(reclaimedBytes))"
            if result.retainedSharedBytes > 0 {
                message += "; preserved \(ModelStorageCommandOutput.bytes(result.retainedSharedBytes)) shared"
            }
            if result.retainedExternalBytes > 0 {
                message += "; preserved \(ModelStorageCommandOutput.bytes(result.retainedExternalBytes)) external"
            }
            if keepCache {
                message += "; Hub payload retained"
            }
            print(message)
        }
    }

    /// Resolve a user-supplied string to a canonical model id.
    private func resolveID(_ raw: String) -> String? {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ManagedModelCatalog.spec(for: normalized) != nil {
            return normalized
        }
        return ModelResolver.ModelID(rawValue: normalized)?.rawValue
    }

    private func gemmaAliasInstallURL(for id: String) -> URL? {
        guard id == Gemma4Resources.defaultModelId else {
            return nil
        }

        let candidates = [
            MereRunModelPaths.modelDir(Gemma4Resources.maxModelId),
            MereRunModelPaths.modelDir(Gemma4Resources.nanoModelId),
        ]
        for candidate in candidates {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir), isDir.boolValue {
                return candidate
            }
        }
        return nil
    }

    private func removeManagedAliasesIfNeeded(for id: String) throws {
        guard let spec = ManagedModelCatalog.spec(for: id) else {
            return
        }
        switch spec.aliasKind {
        case .none:
            return
        case .codegenGGUF:
            let aliasURL = MereRunModelPaths.modelsDir.appendingPathComponent(
                CodeGenResources.managedRelativePath,
                isDirectory: false
            )
            if FileManager.default.fileExists(atPath: aliasURL.path)
                || (try? FileManager.default.destinationOfSymbolicLink(atPath: aliasURL.path)) != nil {
                try? FileManager.default.removeItem(at: aliasURL)
            }
        }
    }
}

struct ModelRemoveOutput: Codable, Equatable {
    let id: String
    let installPath: String
    let referencedBytes: Int64
    let expectedReclaimableBytes: Int64
    let reclaimedBytes: Int64
    let retainedSharedBytes: Int64
    let retainedExternalBytes: Int64
    let cacheRetained: Bool
    let deletedCacheItemCount: Int
}
