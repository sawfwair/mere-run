import Foundation

extension NemotronTranscribeResources {
    /// Preserve an original BF16 install before the default ID is replaced by INT4.
    /// The caller holds the model-storage lock. Regular payloads are hard-linked;
    /// existing cache symlinks are resolved so they survive replacement of the old root.
    static func preserveLegacyBF16Install(modelsRoot: URL, fileManager: FileManager = .default) throws {
        let legacy = modelsRoot.appendingPathComponent(modelID)
        guard var manifest = try MereRunModelManifest.loadIfPresent(from: legacy, fileManager: fileManager),
              manifest.id == modelID, manifest.engine == .nemotronTranscribeASR, manifest.precision == .bf16,
              manifest.upstreamRepoId == repository + "@" + revision,
              missingFiles(modelID: bf16ModelID, root: legacy, fileManager: fileManager).isEmpty else { return }
        let destination = modelsRoot.appendingPathComponent(bf16ModelID)
        if fileManager.fileExists(atPath: destination.path) {
            guard let existing = try MereRunModelManifest.loadIfPresent(from: destination, fileManager: fileManager),
                  existing.id == bf16ModelID, existing.engine == .nemotronTranscribeASR, existing.precision == .bf16,
                  existing.upstreamRepoId == repository + "@" + revision,
                  missingFiles(modelID: bf16ModelID, root: destination, fileManager: fileManager).isEmpty else {
                throw ManagedModelResolver.ResolverError.invalidInstalledModel(
                    "Cannot preserve the original Nemotron BF16 install: \(destination.path) already contains an incomplete model.")
            }
            return
        }
        let staging = modelsRoot.appendingPathComponent(".\(bf16ModelID).preserve.\(UUID().uuidString)")
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }
        try preserveEntries(from: legacy, to: staging, sourceRoot: legacy.resolvingSymlinksInPath(),
                            destinationRoot: staging, fileManager: fileManager)
        manifest.id = bf16ModelID
        try manifest.write(to: staging)
        try fileManager.moveItem(at: staging, to: destination)
    }

    private static func preserveEntries(from source: URL, to destination: URL, sourceRoot: URL,
                                        destinationRoot: URL, fileManager: FileManager) throws {
        let entries = try fileManager.contentsOfDirectory(at: source,
            includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles])
        for entry in entries where entry.lastPathComponent != MereRunModelManifest.filename {
            let target = destination.appendingPathComponent(entry.lastPathComponent)
            let properties = try entry.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            if properties.isSymbolicLink == true {
                let resolved = entry.resolvingSymlinksInPath()
                let prefix = sourceRoot.path + "/"
                let preserved = resolved.path.hasPrefix(prefix)
                    ? destinationRoot.appendingPathComponent(String(resolved.path.dropFirst(prefix.count))) : resolved
                try ManagedModelResolver.createRelocatableSymlink(at: target, to: preserved, fileManager: fileManager)
            } else if properties.isDirectory == true {
                try fileManager.createDirectory(at: target, withIntermediateDirectories: true)
                try preserveEntries(from: entry, to: target, sourceRoot: sourceRoot,
                                    destinationRoot: destinationRoot, fileManager: fileManager)
            } else {
                try fileManager.linkItem(at: entry, to: target)
            }
        }
    }
}
