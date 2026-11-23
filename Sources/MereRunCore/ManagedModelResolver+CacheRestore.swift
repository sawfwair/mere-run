import Foundation

extension ManagedModelResolver {
    /// Restore missing catalog installs from complete MereRun Hub snapshots. This path uses
    /// only local metadata and file sizes, then publishes the same symlink-based install as pull.
    package static func restoreCachedModels(
        specs: [ManagedModelSpec] = ManagedModelCatalog.allKnownSpecs,
        locations: ModelLocationSnapshot? = nil,
        hubCacheURL: URL? = nil,
        fileManager: FileManager = .default,
        onFailure: (String, Error) -> Void
    ) throws -> [String] {
        let cache = try HubSnapshot.resolvedDownloadBase(requested: hubCacheURL, fileManager: fileManager)
        guard fileManager.fileExists(atPath: cache.appending(path: "snapshots/models").path) else { return [] }
        let locations = locations ?? MereRunModelLocations.snapshot(fileManager: fileManager)
        let resolver = ModelResolver(fileManager: fileManager, locations: locations)
        var restored: [String] = []
        for spec in specs {
            let modelDir = locations.primaryRoot.appending(path: spec.id, directoryHint: .isDirectory)
            guard !fileManager.fileExists(atPath: modelDir.path),
                  !fileManager.fileExists(atPath: cacheRestoreExclusion(for: modelDir).path),
                  spec.usageRestriction == nil, let source = spec.hubFallback else { continue }
            let options = snapshotOptions(for: source, cache: cache)
            guard let snapshot = try? HubSnapshot.cachedMaterializedSnapshotURL(
                options: options, fileManager: fileManager
            ) else { continue }
            if let modelID = spec.modelID, resolver.resolveIfPresent(modelID) != nil { continue }
            do {
                let lock = try ModelStorageFileLock.acquire(hubDirectory: cache, fileManager: fileManager)
                defer { lock.unlock() }
                guard !fileManager.fileExists(atPath: modelDir.path),
                      (try? fileManager.destinationOfSymbolicLink(atPath: modelDir.path)) == nil,
                      !fileManager.fileExists(atPath: cacheRestoreExclusion(for: modelDir).path),
                      isCompleteCachedSnapshot(snapshot, options: options) else { continue }

                let mounted = spec.mountedHubFallbacks.compactMap { component -> (MountedHubFallbackConfig, URL)? in
                    let options = snapshotOptions(for: component.hubFallback, cache: cache)
                    guard let url = try? HubSnapshot.cachedMaterializedSnapshotURL(options: options, fileManager: fileManager),
                          isCompleteCachedSnapshot(url, options: options) else { return nil }
                    return (component, url)
                }
                guard mounted.count == spec.mountedHubFallbacks.count else { continue }
                _ = try publishManagedInstall(
                    for: spec, snapshotURL: snapshot, mountedSnapshots: mounted, modelDir: modelDir,
                    usageTermsAcknowledged: false, replaceExisting: false, fileManager: fileManager
                )
                restored.append(spec.id)
            } catch {
                onFailure(spec.id, error)
            }
        }
        return restored
    }

    /// A retained or shared snapshot must not undo an explicit removal at the next launch.
    package static func cacheRestoreExclusion(for modelDir: URL) -> URL {
        modelDir.deletingLastPathComponent()
            .appending(path: ".cache-restore-exclusions", directoryHint: .isDirectory)
            .appending(path: modelDir.lastPathComponent, directoryHint: .notDirectory)
    }

    private static func snapshotOptions(for source: HubFallbackConfig, cache: URL) -> HubSnapshotOptions {
        HubSnapshotOptions(
            repoId: source.repoId, revision: source.revision, patterns: source.patterns, cacheDirectory: cache
        )
    }

    private static func isCompleteCachedSnapshot(_ snapshot: URL, options: HubSnapshotOptions) -> Bool {
        guard let data = try? Data(contentsOf: snapshot.appending(path: HubSnapshotReceipt.filename)),
              let receipt = try? JSONDecoder().decode(HubSnapshotReceipt.self, from: data),
              receipt.schemaVersion == 1, receipt.repository == options.repoId else { return false }

        let revision = options.revision
        let isCommit = revision.utf8.count == 40 && revision.allSatisfy(\.isHexDigit)
        guard receipt.resolvedRevision == revision || (!isCommit && receipt.requestedRevision == revision) else {
            return false
        }
        let selected = receipt.files.filter { HubSnapshot.matchesPath($0.path, patterns: options.patterns) }
        guard !selected.isEmpty, options.patterns.allSatisfy({ pattern in
            selected.contains { HubSnapshot.matchesPath($0.path, patterns: [pattern]) }
        }) else { return false }

        return selected.allSatisfy { file in
            guard file.commit == receipt.resolvedRevision else { return false }
            let payload = snapshot.appending(path: file.path).resolvingSymlinksInPath()
            let values = try? payload.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            return values?.isRegularFile == true && values?.fileSize.map(Int64.init) == file.size
        }
    }
}
