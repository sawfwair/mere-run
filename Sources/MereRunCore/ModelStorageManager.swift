import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public struct ModelStorageModelUsage: Codable, Equatable, Sendable {
    public let id: String
    public let installPath: String
    public let installed: Bool
    public let referencedBytes: Int64
    public let localBytes: Int64
    public let reclaimableBytes: Int64
    public let sharedBytes: Int64
    public let externalBytes: Int64

    public init(
        id: String,
        installPath: String,
        installed: Bool,
        referencedBytes: Int64,
        localBytes: Int64,
        reclaimableBytes: Int64,
        sharedBytes: Int64,
        externalBytes: Int64 = 0
    ) {
        self.id = id
        self.installPath = installPath
        self.installed = installed
        self.referencedBytes = referencedBytes
        self.localBytes = localBytes
        self.reclaimableBytes = reclaimableBytes
        self.sharedBytes = sharedBytes
        self.externalBytes = externalBytes
    }
}

public struct ModelStorageReport: Codable, Equatable, Sendable {
    public let applicationSupportPath: String
    public let modelStorePath: String
    public let hubPath: String
    public let applicationSupportBytes: Int64
    public let modelStoreBytes: Int64
    public let hubBytes: Int64
    public let otherApplicationSupportBytes: Int64
    public let garbageCollectableBytes: Int64
    public let incompleteDownloadBytes: Int64
    public let models: [ModelStorageModelUsage]

    public init(
        applicationSupportPath: String,
        modelStorePath: String,
        hubPath: String,
        applicationSupportBytes: Int64,
        modelStoreBytes: Int64,
        hubBytes: Int64,
        otherApplicationSupportBytes: Int64,
        garbageCollectableBytes: Int64,
        incompleteDownloadBytes: Int64,
        models: [ModelStorageModelUsage]
    ) {
        self.applicationSupportPath = applicationSupportPath
        self.modelStorePath = modelStorePath
        self.hubPath = hubPath
        self.applicationSupportBytes = applicationSupportBytes
        self.modelStoreBytes = modelStoreBytes
        self.hubBytes = hubBytes
        self.otherApplicationSupportBytes = otherApplicationSupportBytes
        self.garbageCollectableBytes = garbageCollectableBytes
        self.incompleteDownloadBytes = incompleteDownloadBytes
        self.models = models
    }
}

public struct ModelStorageGarbageItem: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable {
        case cacheUnit = "cache_unit"
        case payload
        case metadata
        case incompleteDownload = "incomplete_download"
        case blob
        case reference
    }

    public let kind: Kind
    public let path: String
    public let logicalBytes: Int64

    public init(kind: Kind, path: String, logicalBytes: Int64) {
        self.kind = kind
        self.path = path
        self.logicalBytes = logicalBytes
    }
}

public struct ModelStorageGarbagePlan: Codable, Equatable, Sendable {
    public let hubPath: String
    public let reclaimableBytes: Int64
    public let incompleteDownloadBytes: Int64
    public let items: [ModelStorageGarbageItem]
    public let scopeCacheUnitPaths: [String]?

    public init(
        hubPath: String,
        reclaimableBytes: Int64,
        incompleteDownloadBytes: Int64,
        items: [ModelStorageGarbageItem],
        scopeCacheUnitPaths: [String]? = nil
    ) {
        self.hubPath = hubPath
        self.reclaimableBytes = reclaimableBytes
        self.incompleteDownloadBytes = incompleteDownloadBytes
        self.items = items
        self.scopeCacheUnitPaths = scopeCacheUnitPaths
    }
}

public struct ModelStorageGarbageResult: Codable, Equatable, Sendable {
    public let plannedBytes: Int64
    public let reclaimedBytes: Int64
    public let deletedItemCount: Int

    public init(plannedBytes: Int64, reclaimedBytes: Int64, deletedItemCount: Int) {
        self.plannedBytes = plannedBytes
        self.reclaimedBytes = reclaimedBytes
        self.deletedItemCount = deletedItemCount
    }
}

/// Builds a physical ownership graph for the model store and its Hub-backed payloads.
///
/// Model install roots may contain direct files, symlink individual payload files, or be
/// symlinks themselves. Accounting therefore follows model links for referenced bytes,
/// but counts file identities only once for physical and reclaimable bytes.
/// Directory traversal in this type intentionally does not use the runtime's
/// symlink-resolving helpers: ownership discovery and garbage collection must inspect
/// link edges without recursively treating their targets as locally owned payloads.
public final class ModelStorageManager {
    private struct FileIdentity: Hashable {
        let device: UInt64
        let inode: UInt64
    }

    private struct FileRecord {
        let url: URL
        let identity: FileIdentity
        let size: Int64
        let linkCount: Int
    }

    private struct Reference {
        let owner: String
        let target: URL
        let targetIsDirectory: Bool
        let cacheUnit: URL?
    }

    private let fileManager: FileManager
    private let unreferencedGracePeriod: TimeInterval
    private let now: () -> Date
    public let modelsDirectory: URL
    public let hubDirectory: URL
    public let applicationSupportDirectory: URL

    public init(
        modelsDirectory: URL = MereRunModelPaths.modelsDir,
        hubDirectory: URL? = nil,
        applicationSupportDirectory: URL = MereRunModelPaths.applicationSupportBase,
        fileManager: FileManager = .default,
        unreferencedGracePeriod: TimeInterval = 3_600,
        now: @escaping () -> Date = Date.init
    ) throws {
        self.fileManager = fileManager
        self.unreferencedGracePeriod = max(0, unreferencedGracePeriod)
        self.now = now
        let resolvedHubDirectory = try hubDirectory ?? HubSnapshot.resolvedDownloadBase(fileManager: fileManager)
        self.modelsDirectory = Self.canonicalFileURL(modelsDirectory)
        self.hubDirectory = Self.canonicalFileURL(resolvedHubDirectory)
        self.applicationSupportDirectory = Self.canonicalFileURL(applicationSupportDirectory)
    }

    public func report(
        modelIDs: [String] = ManagedModelCatalog.allSpecs.map(\.id)
    ) throws -> ModelStorageReport {
        let references = try symlinkReferences()
        let ownerFiles = try referencedFilesByOwner(references: references)
        let externalOwnerFiles = try referencedFilesByOwner(
            references: references.filter { !isDescendant($0.target, of: applicationSupportDirectory) },
            includeLocalFiles: false
        )
        let identityOwners = ownersByIdentity(ownerFiles)
        let garbagePlan = try garbageCollectionPlan(references: references, limitingTo: nil)

        let models = try modelIDs.map { id in
            try modelUsage(
                id: id,
                ownerFiles: ownerFiles,
                externalOwnerFiles: externalOwnerFiles,
                identityOwners: identityOwners
            )
        }
        let applicationSupportBytes = uniqueBytes(fileRecords(at: applicationSupportDirectory))
        let modelStoreBytes = uniqueBytes(fileRecords(at: modelsDirectory))
        let hubBytes = uniqueBytes(fileRecords(at: hubDirectory))
        let otherBytes = max(0, applicationSupportBytes - hubBytes - modelStoreBytes)

        return ModelStorageReport(
            applicationSupportPath: applicationSupportDirectory.path,
            modelStorePath: modelsDirectory.path,
            hubPath: hubDirectory.path,
            applicationSupportBytes: applicationSupportBytes,
            modelStoreBytes: modelStoreBytes,
            hubBytes: hubBytes,
            otherApplicationSupportBytes: otherBytes,
            garbageCollectableBytes: garbagePlan.reclaimableBytes,
            incompleteDownloadBytes: garbagePlan.incompleteDownloadBytes,
            models: models
        )
    }

    public func cacheUnitsReferenced(by owner: String) throws -> [URL] {
        let units = try symlinkReferences()
            .filter { $0.owner == owner }
            .compactMap(\.cacheUnit)
        return uniqueURLs(units)
    }

    public func garbageCollectionPlan(
        limitingTo cacheUnits: [URL]? = nil
    ) throws -> ModelStorageGarbagePlan {
        try garbageCollectionPlan(
            references: symlinkReferences(),
            limitingTo: cacheUnits.map { Set($0.map(Self.pathKey)) }
        )
    }

    public func execute(_ plan: ModelStorageGarbagePlan) throws -> ModelStorageGarbageResult {
        let storageLock = try ModelStorageFileLock.acquire(hubDirectory: hubDirectory, fileManager: fileManager)
        defer { storageLock.unlock() }
        let hubBefore = uniqueBytes(fileRecords(at: hubDirectory))
        let currentPlan = try garbageCollectionPlan(
            references: symlinkReferences(),
            limitingTo: plan.scopeCacheUnitPaths.map(Set.init)
        )
        let currentlySafePaths = Set(currentPlan.items.map(\.path))
        var deletedCount = 0

        for item in plan.items.filter({ currentlySafePaths.contains($0.path) }).sorted(by: Self.deletionOrder) {
            let url = URL(fileURLWithPath: item.path).standardizedFileURL
            guard isDescendant(url, of: hubDirectory), url.path != hubDirectory.path else {
                throw CocoaError(.fileWriteInvalidFileName)
            }
            guard fileManager.fileExists(atPath: url.path) else { continue }
            try fileManager.removeItem(at: url)
            deletedCount += 1
        }

        removeEmptyCacheDirectories()
        let hubAfter = uniqueBytes(fileRecords(at: hubDirectory))
        return ModelStorageGarbageResult(
            plannedBytes: plan.reclaimableBytes,
            reclaimedBytes: max(0, hubBefore - hubAfter),
            deletedItemCount: deletedCount
        )
    }

    private func modelUsage(
        id: String,
        ownerFiles: [String: [FileIdentity: FileRecord]],
        externalOwnerFiles: [String: [FileIdentity: FileRecord]],
        identityOwners: [FileIdentity: Set<String>]
    ) throws -> ModelStorageModelUsage {
        let installURL = modelsDirectory.appendingPathComponent(id, isDirectory: true)
        let installed = fileManager.fileExists(atPath: installURL.path)
        guard installed else {
            return ModelStorageModelUsage(
                id: id,
                installPath: installURL.path,
                installed: false,
                referencedBytes: 0,
                localBytes: 0,
                reclaimableBytes: 0,
                sharedBytes: 0,
                externalBytes: 0
            )
        }

        let referenced = ownerFiles[id] ?? [:]
        let localRecords = fileRecords(at: installURL)
        let localIdentities = Set(localRecords.map(\.identity))
        let localBytes = uniqueBytes(localRecords)
        let externalRecords = externalOwnerFiles[id] ?? [:]
        let externalIdentities = Set(externalRecords.keys)
        let externalBytes = externalRecords.values.reduce(Int64(0)) { $0 + $1.size }
        let referencedBytes = referenced.values.reduce(Int64(0)) { $0 + $1.size }
        let exclusiveCacheBytes = referenced.reduce(Int64(0)) { partial, entry in
            let (identity, record) = entry
            guard !localIdentities.contains(identity),
                  !externalIdentities.contains(identity),
                  identityOwners[identity] == [id] else {
                return partial
            }
            return partial + record.size
        }
        let reclaimable = localBytes + exclusiveCacheBytes

        return ModelStorageModelUsage(
            id: id,
            installPath: installURL.path,
            installed: true,
            referencedBytes: referencedBytes,
            localBytes: localBytes,
            reclaimableBytes: reclaimable,
            sharedBytes: max(0, referencedBytes - reclaimable - externalBytes),
            externalBytes: externalBytes
        )
    }

    private func garbageCollectionPlan(
        references: [Reference],
        limitingTo cacheUnitPaths: Set<String>?
    ) throws -> ModelStorageGarbagePlan {
        let cacheUnits = try cacheUnitURLs().filter { unit in
            cacheUnitPaths == nil || cacheUnitPaths?.contains(Self.pathKey(unit)) == true
        }
        let allHubRecords = fileRecords(at: hubDirectory)
        let hubRecordsByIdentity = Dictionary(grouping: allHubRecords, by: \.identity)
        var items: [ModelStorageGarbageItem] = []
        var plannedFilePaths = Set<String>()
        var incompleteBytes: Int64 = 0

        for unit in cacheUnits {
            let unitRecords = payloadFileRecords(at: unit)
            let unitReferences = references.filter {
                guard let referencedUnit = $0.cacheUnit else { return false }
                return Self.pathKey(referencedUnit) == Self.pathKey(unit)
            }
            if unitReferences.isEmpty {
                if cacheUnitPaths == nil, cacheUnitIsWithinGracePeriod(unit) {
                    continue
                }
                let bytes = uniqueBytes(fileRecords(at: unit))
                items.append(ModelStorageGarbageItem(kind: .cacheUnit, path: unit.path, logicalBytes: bytes))
                plannedFilePaths.formUnion(fileRecords(at: unit).map { $0.url.path })
                incompleteBytes += incompleteDownloadBytes(at: unit)
                continue
            }

            for record in unitRecords where !isLive(record.url, references: unitReferences) {
                items.append(ModelStorageGarbageItem(kind: .payload, path: record.url.path, logicalBytes: record.size))
                plannedFilePaths.insert(record.url.path)
                if let metadata = metadataURL(for: record.url, in: unit),
                   fileManager.fileExists(atPath: metadata.path) {
                    let metadataBytes = fileRecord(at: metadata)?.size ?? 0
                    items.append(ModelStorageGarbageItem(
                        kind: .metadata,
                        path: metadata.path,
                        logicalBytes: metadataBytes
                    ))
                    plannedFilePaths.insert(metadata.path)
                }
            }

            for incomplete in incompleteDownloadRecords(at: unit) {
                items.append(ModelStorageGarbageItem(
                    kind: .incompleteDownload,
                    path: incomplete.url.path,
                    logicalBytes: incomplete.size
                ))
                plannedFilePaths.insert(incomplete.url.path)
                incompleteBytes += incomplete.size
            }
        }

        let blobRoot = hubDirectory.appendingPathComponent("blobs", isDirectory: true)
        for blob in fileRecords(at: blobRoot) {
            let allLinks = hubRecordsByIdentity[blob.identity] ?? []
            let plannedLinks = allLinks.filter { plannedFilePaths.contains($0.url.path) }.count
            let canDeleteOrphan = cacheUnitPaths == nil && blob.linkCount == 1
            if canDeleteOrphan || (plannedLinks > 0 && blob.linkCount <= plannedLinks + 1) {
                items.append(ModelStorageGarbageItem(kind: .blob, path: blob.url.path, logicalBytes: blob.size))
                plannedFilePaths.insert(blob.url.path)
            }
        }

        let plannedCacheUnitKeys = Set(
            items.filter { $0.kind == .cacheUnit }.map { Self.pathKey(URL(fileURLWithPath: $0.path)) }
        )
        let refsRoot = hubDirectory.appendingPathComponent("refs", isDirectory: true)
        for reference in fileRecords(at: refsRoot) {
            guard let snapshot = snapshotURL(forReference: reference.url, refsRoot: refsRoot) else { continue }
            let snapshotKey = Self.pathKey(snapshot)
            let snapshotWillBeDeleted = plannedCacheUnitKeys.contains(snapshotKey)
            let snapshotIsMissing = !fileManager.fileExists(atPath: snapshot.path)
            guard snapshotWillBeDeleted || (cacheUnitPaths == nil && snapshotIsMissing) else { continue }
            items.append(
                ModelStorageGarbageItem(
                    kind: .reference,
                    path: reference.url.path,
                    logicalBytes: reference.size
                )
            )
            plannedFilePaths.insert(reference.url.path)
        }

        let reclaimableBytes = hubRecordsByIdentity.reduce(Int64(0)) { partial, entry in
            let (_, records) = entry
            guard let first = records.first else { return partial }
            let plannedLinks = records.filter { plannedFilePaths.contains($0.url.path) }.count
            guard plannedLinks >= first.linkCount else { return partial }
            return partial + first.size
        }

        return ModelStorageGarbagePlan(
            hubPath: hubDirectory.path,
            reclaimableBytes: reclaimableBytes,
            incompleteDownloadBytes: incompleteBytes,
            items: uniqueGarbageItems(items),
            scopeCacheUnitPaths: cacheUnitPaths?.sorted()
        )
    }

    private func symlinkReferences() throws -> [Reference] {
        var references: [Reference] = []
        for root in referenceRoots() where fileManager.fileExists(atPath: root.path) {
            guard let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isSymbolicLinkKey],
                options: []
            ) else { continue }

            for case let url as URL in enumerator {
                if url.standardizedFileURL.path == hubDirectory.path || isDescendant(url, of: hubDirectory) {
                    enumerator.skipDescendants()
                    continue
                }
                guard isSymbolicLink(url) else { continue }
                let target = Self.canonicalFileURL(url)
                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(atPath: target.path, isDirectory: &isDirectory) else {
                    continue
                }
                let targetIsInHub = isDescendant(target, of: hubDirectory)
                guard targetIsInHub || isDescendant(url, of: modelsDirectory) else { continue }
                guard url.lastPathComponent != ".cache",
                      url.lastPathComponent != HubSnapshotReceipt.filename else { continue }
                references.append(
                    Reference(
                        owner: ownerID(for: url),
                        target: target,
                        targetIsDirectory: isDirectory.boolValue,
                        cacheUnit: targetIsInHub ? cacheUnit(containing: target) : nil
                    )
                )
            }
        }
        return references
    }

    private func referencedFilesByOwner(
        references: [Reference],
        includeLocalFiles: Bool = true
    ) throws -> [String: [FileIdentity: FileRecord]] {
        var result: [String: [FileIdentity: FileRecord]] = [:]
        for reference in references {
            let records = reference.targetIsDirectory
                ? payloadFileRecords(at: reference.target)
                : [fileRecord(at: reference.target)].compactMap { $0 }
            for record in records {
                result[reference.owner, default: [:]][record.identity] = record
            }
        }

        if includeLocalFiles {
            for id in ManagedModelCatalog.allSpecs.map(\.id) {
                let root = modelsDirectory.appendingPathComponent(id, isDirectory: true)
                for record in fileRecords(at: root) {
                    result[id, default: [:]][record.identity] = record
                }
            }
        }
        return result
    }

    private func ownersByIdentity(
        _ ownerFiles: [String: [FileIdentity: FileRecord]]
    ) -> [FileIdentity: Set<String>] {
        var result: [FileIdentity: Set<String>] = [:]
        for (owner, files) in ownerFiles {
            for identity in files.keys {
                result[identity, default: []].insert(owner)
            }
        }
        return result
    }

    private func referenceRoots() -> [URL] {
        uniqueURLs([applicationSupportDirectory, modelsDirectory])
    }

    private func ownerID(for link: URL) -> String {
        if isDescendant(link, of: modelsDirectory) {
            let linkPath = Self.pathKey(link)
            let modelsPath = Self.pathKey(modelsDirectory)
            let relative = String(linkPath.dropFirst(modelsPath.count + 1))
            let first = relative.split(separator: "/").first.map(String.init) ?? relative
            if first == CodeGenResources.managedRelativePath {
                return ModelResolver.ModelID.qwen3Code.rawValue
            }
            return first
        }
        let linkPath = Self.pathKey(link)
        let applicationSupportPath = Self.pathKey(applicationSupportDirectory)
        let relative = isDescendant(link, of: applicationSupportDirectory)
            ? String(linkPath.dropFirst(applicationSupportPath.count + 1))
            : linkPath
        return "legacy:\(relative.split(separator: "/").first.map(String.init) ?? relative)"
    }

    private func cacheUnit(containing target: URL) -> URL? {
        let legacyRoot = hubDirectory.appendingPathComponent("models", isDirectory: true)
        if isDescendant(target, of: legacyRoot) {
            let relative = String(target.path.dropFirst(legacyRoot.path.count + 1))
            let parts = relative.split(separator: "/")
            guard parts.count >= 2 else { return nil }
            return legacyRoot
                .appendingPathComponent(String(parts[0]), isDirectory: true)
                .appendingPathComponent(String(parts[1]), isDirectory: true)
        }

        let snapshotsRoot = hubDirectory.appendingPathComponent("snapshots/models", isDirectory: true)
        if isDescendant(target, of: snapshotsRoot) {
            let relative = String(target.path.dropFirst(snapshotsRoot.path.count + 1))
            let parts = relative.split(separator: "/")
            guard parts.count >= 3 else { return nil }
            return snapshotsRoot
                .appendingPathComponent(String(parts[0]), isDirectory: true)
                .appendingPathComponent(String(parts[1]), isDirectory: true)
                .appendingPathComponent(String(parts[2]), isDirectory: true)
        }
        return nil
    }

    private func cacheUnitURLs() throws -> [URL] {
        var result: [URL] = []
        let legacy = hubDirectory.appendingPathComponent("models", isDirectory: true)
        for owner in directoryChildren(of: legacy) {
            result.append(contentsOf: directoryChildren(of: owner))
        }

        let snapshots = hubDirectory.appendingPathComponent("snapshots/models", isDirectory: true)
        for owner in directoryChildren(of: snapshots) {
            for repository in directoryChildren(of: owner) {
                result.append(contentsOf: directoryChildren(of: repository))
            }
        }
        return result
    }

    private func directoryChildren(of root: URL) -> [URL] {
        let urls = (try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
    }

    private func payloadFileRecords(at root: URL) -> [FileRecord] {
        fileRecords(at: root).filter { record in
            let relative = relativePath(of: record.url, under: root)
            let components = relative.split(separator: "/")
            return !components.contains(".cache")
                && record.url.lastPathComponent != HubSnapshotReceipt.filename
                && record.url.lastPathComponent != ".mererun-storage.lock"
                && !record.url.lastPathComponent.hasSuffix(".incomplete")
        }
    }

    private func incompleteDownloadRecords(at root: URL) -> [FileRecord] {
        fileRecords(at: root).filter { $0.url.lastPathComponent.hasSuffix(".incomplete") }
    }

    private func incompleteDownloadBytes(at root: URL) -> Int64 {
        uniqueBytes(incompleteDownloadRecords(at: root))
    }

    private func isLive(_ payload: URL, references: [Reference]) -> Bool {
        references.contains { reference in
            Self.pathKey(reference.target) == Self.pathKey(payload)
                || (reference.targetIsDirectory && isDescendant(payload, of: reference.target))
        }
    }

    private func metadataURL(for payload: URL, in unit: URL) -> URL? {
        guard isDescendant(payload, of: unit) else { return nil }
        let relative = relativePath(of: payload, under: unit)
        return unit
            .appendingPathComponent(".cache/huggingface/download", isDirectory: true)
            .appendingPathComponent(relative + ".metadata", isDirectory: false)
    }

    private func snapshotURL(forReference reference: URL, refsRoot: URL) -> URL? {
        let parts = relativePath(of: reference, under: refsRoot).split(separator: "/").map(String.init)
        guard parts.count == 4,
              parts[3].hasSuffix(".ref"),
              let data = try? Data(contentsOf: reference),
              let resolved = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !resolved.isEmpty else {
            return nil
        }
        return hubDirectory
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(parts[0], isDirectory: true)
            .appendingPathComponent(parts[1], isDirectory: true)
            .appendingPathComponent(parts[2], isDirectory: true)
            .appendingPathComponent(HubSnapshot.revisionKey(resolved), isDirectory: true)
    }

    private func cacheUnitIsWithinGracePeriod(_ unit: URL) -> Bool {
        guard unreferencedGracePeriod > 0,
              let attributes = try? fileManager.attributesOfItem(atPath: unit.path),
              let modified = attributes[.modificationDate] as? Date else {
            return false
        }
        return now().timeIntervalSince(modified) < unreferencedGracePeriod
    }

    private func fileRecords(at root: URL) -> [FileRecord] {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory) else { return [] }
        if isSymbolicLink(root) {
            return []
        }
        if !isDirectory.boolValue {
            return fileRecord(at: root).map { [$0] } ?? []
        }
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
            options: []
        ) else { return [] }

        var result: [FileRecord] = []
        for case let url as URL in enumerator {
            if isSymbolicLink(url) {
                continue
            }
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            if values?.isRegularFile == true, let record = fileRecord(at: url) {
                result.append(record)
            }
        }
        return result
    }

    private func isSymbolicLink(_ url: URL) -> Bool {
        (try? fileManager.destinationOfSymbolicLink(atPath: url.path)) != nil
    }

    private func fileRecord(at url: URL) -> FileRecord? {
        guard let attributes = try? fileManager.attributesOfItem(atPath: url.path),
              let device = (attributes[.systemNumber] as? NSNumber)?.uint64Value,
              let inode = (attributes[.systemFileNumber] as? NSNumber)?.uint64Value,
              let size = (attributes[.size] as? NSNumber)?.int64Value else {
            return nil
        }
        let links = (attributes[.referenceCount] as? NSNumber)?.intValue ?? 1
        return FileRecord(
            url: url.standardizedFileURL,
            identity: FileIdentity(device: device, inode: inode),
            size: size,
            linkCount: max(1, links)
        )
    }

    private func uniqueBytes(_ records: [FileRecord]) -> Int64 {
        var seen = Set<FileIdentity>()
        return records.reduce(Int64(0)) { partial, record in
            guard seen.insert(record.identity).inserted else { return partial }
            return partial + record.size
        }
    }

    private func uniqueURLs(_ urls: [URL]) -> [URL] {
        var seen = Set<String>()
        return urls.filter { seen.insert(Self.pathKey($0)).inserted }
    }

    private func uniqueGarbageItems(_ items: [ModelStorageGarbageItem]) -> [ModelStorageGarbageItem] {
        var seen = Set<String>()
        return items.filter { seen.insert($0.path).inserted }
    }

    private func relativePath(of url: URL, under root: URL) -> String {
        let path = Self.pathKey(url)
        let rootPath = Self.pathKey(root)
        return String(path.dropFirst(rootPath.count + 1))
    }

    private func isDescendant(_ url: URL, of root: URL) -> Bool {
        let path = Self.pathKey(url)
        let rootPath = Self.pathKey(root)
        return path.hasPrefix(rootPath + "/")
    }

    private func removeEmptyCacheDirectories() {
        let roots = [
            hubDirectory.appendingPathComponent("models", isDirectory: true),
            hubDirectory.appendingPathComponent("snapshots", isDirectory: true),
            hubDirectory.appendingPathComponent("blobs", isDirectory: true),
        ]
        for root in roots {
            guard let enumerator = fileManager.enumerator(at: root, includingPropertiesForKeys: nil) else { continue }
            var directories: [URL] = []
            for case let url as URL in enumerator {
                var isDirectory: ObjCBool = false
                if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                    directories.append(url)
                }
            }
            for directory in directories.sorted(by: { $0.path.count > $1.path.count }) {
                let contents = try? fileManager.contentsOfDirectory(atPath: directory.path)
                if contents?.isEmpty == true {
                    try? fileManager.removeItem(at: directory)
                }
            }
        }
    }

    private static func deletionOrder(_ lhs: ModelStorageGarbageItem, _ rhs: ModelStorageGarbageItem) -> Bool {
        if lhs.kind == .cacheUnit, rhs.kind != .cacheUnit { return true }
        if rhs.kind == .cacheUnit, lhs.kind != .cacheUnit { return false }
        return lhs.path.count > rhs.path.count
    }

    private static func canonicalFileURL(_ url: URL) -> URL {
        let resolved = url.path.withCString { realpath($0, nil) }
        guard let resolved else { return normalizedSystemAlias(url.resolvingSymlinksInPath().standardizedFileURL) }
        defer { free(resolved) }
        let canonical = URL(fileURLWithPath: String(cString: resolved)).standardizedFileURL
        return normalizedSystemAlias(canonical)
    }

    private static func normalizedSystemAlias(_ url: URL) -> URL {
        #if canImport(Darwin)
        let path = url.path
        if path == "/var" || path.hasPrefix("/var/") || path == "/tmp" || path.hasPrefix("/tmp/") {
            return URL(fileURLWithPath: "/private" + path).standardizedFileURL
        }
        #endif
        return url
    }

    private static func pathKey(_ url: URL) -> String {
        let path = url.standardizedFileURL.path
        #if canImport(Darwin)
        if path == "/private/var" || path.hasPrefix("/private/var/")
            || path == "/private/tmp" || path.hasPrefix("/private/tmp/") {
            return String(path.dropFirst("/private".count))
        }
        #endif
        return path
    }
}
