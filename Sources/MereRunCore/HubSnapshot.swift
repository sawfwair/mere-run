import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Crypto
@preconcurrency import Hub

public struct HubSnapshotReceipt: Codable, Equatable, Sendable {
    public static let filename = ".mererun_snapshot.json"

    public struct File: Codable, Equatable, Sendable {
        public let path: String
        public let size: Int64
        public let commit: String
        public let etag: String?
    }

    public let schemaVersion: Int
    public let repository: String
    public let requestedRevision: String
    public let resolvedRevision: String
    public let files: [File]
}

public struct HubSnapshotOptions: Sendable {
    public var repoId: String
    public var revision: String
    public var repoType: Hub.RepoType
    public var patterns: [String]
    public var cacheDirectory: URL?
    public var accessToken: String?
    public var offline: Bool
    public var useBackgroundSession: Bool

    public init(
        repoId: String,
        revision: String = "main",
        repoType: Hub.RepoType = .models,
        patterns: [String] = [],
        cacheDirectory: URL? = nil,
        accessToken: String? = nil,
        offline: Bool = false,
        useBackgroundSession: Bool = false
    ) {
        self.repoId = repoId
        self.revision = revision
        self.repoType = repoType
        self.patterns = patterns
        self.cacheDirectory = cacheDirectory
        self.accessToken = accessToken
        self.offline = offline
        self.useBackgroundSession = useBackgroundSession
    }
}

public struct HubSnapshotProgress: Sendable {
    public let fractionCompleted: Double
    public let completedUnitCount: Int64
    public let totalUnitCount: Int64
    public let estimatedSpeedBytesPerSecond: Double?

    public init(
        completedUnitCount: Int64,
        totalUnitCount: Int64,
        estimatedSpeedBytesPerSecond: Double? = nil
    ) {
        self.completedUnitCount = completedUnitCount
        self.totalUnitCount = totalUnitCount
        self.fractionCompleted = totalUnitCount > 0
            ? Double(completedUnitCount) / Double(totalUnitCount)
            : 0
        self.estimatedSpeedBytesPerSecond = estimatedSpeedBytesPerSecond
    }

    init(progress: Progress, estimatedSpeedBytesPerSecond: Double?) {
        let total = progress.totalUnitCount
        let completed = progress.completedUnitCount
        self.init(
            completedUnitCount: completed,
            totalUnitCount: total,
            estimatedSpeedBytesPerSecond: estimatedSpeedBytesPerSecond
        )
    }
}

public actor HubSnapshot {
    public typealias ProgressHandler = @Sendable (HubSnapshotProgress) -> Void

    private let options: HubSnapshotOptions
    private let hubApi: HubApi
    private let downloadBase: URL
    private var cachedSnapshotURL: URL?

    public init(
        options: HubSnapshotOptions,
        hubApi: HubApi? = nil
    ) throws {
        self.options = options

        let downloadBase = try Self.resolveDownloadBase(
            requested: options.cacheDirectory,
            fileManager: .default
        )
        self.downloadBase = downloadBase

        self.hubApi = hubApi ?? HubApi(
            downloadBase: downloadBase,
            hfToken: options.accessToken,
            useBackgroundSession: options.useBackgroundSession,
            useOfflineMode: options.offline ? true : nil
        )
    }

    public func prepare(progressHandler: ProgressHandler? = nil) async throws -> URL {
        if let cachedSnapshotURL, FileManager.default.fileExists(atPath: cachedSnapshotURL.path) {
            return cachedSnapshotURL
        }

        let storageLock = try ModelStorageFileLock.acquire(hubDirectory: downloadBase)
        defer { storageLock.unlock() }

        if !options.offline {
            let snapshotURL = try await prepareMaterializedSnapshot(progressHandler: progressHandler)
            cachedSnapshotURL = snapshotURL
            return snapshotURL
        }

        if let snapshotURL = offlineMaterializedSnapshotURL() {
            cachedSnapshotURL = snapshotURL
            return snapshotURL
        }

        let repo = Hub.Repo(id: options.repoId, type: options.repoType)
        let snapshotURL = try await hubApi.snapshot(
            from: repo,
            revision: options.revision,
            matching: options.patterns,
            progressHandler: { progress, speed in
                progressHandler?(HubSnapshotProgress(progress: progress, estimatedSpeedBytesPerSecond: speed))
            }
        )

        cachedSnapshotURL = snapshotURL
        return snapshotURL
    }

    public func fileURL(
        for relativePath: String,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let snapshot = try await prepare(progressHandler: progressHandler)
        let fileURL = snapshot.appending(path: relativePath)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw Hub.HubClientError.fileNotFound(relativePath)
        }
        return fileURL
    }

    public func invalidateCache() {
        cachedSnapshotURL = nil
    }

    public static func resolvedDownloadBase(
        requested: URL? = nil,
        fileManager: FileManager = .default
    ) throws -> URL {
        try resolveDownloadBase(
            requested: requested,
            fileManager: fileManager
        )
    }

    /// Returns a materialized snapshot already present for the requested
    /// revision without contacting the Hub or trusting an unverified receipt.
    public static func cachedMaterializedSnapshotURL(
        options: HubSnapshotOptions,
        fileManager: FileManager = .default
    ) throws -> URL? {
        let downloadBase = try resolveDownloadBase(
            requested: options.cacheDirectory,
            fileManager: fileManager
        )
        let repositoryRoot = downloadBase
            .appending(path: "snapshots")
            .appending(path: options.repoType.rawValue)
            .appending(path: options.repoId)
        let referenceURL = downloadBase
            .appending(path: "refs")
            .appending(path: options.repoType.rawValue)
            .appending(path: options.repoId)
            .appending(path: revisionKey(options.revision) + ".ref")

        if let data = try? Data(contentsOf: referenceURL),
           let resolved = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !resolved.isEmpty {
            let referenced = repositoryRoot.appending(path: revisionKey(resolved))
            if fileManager.fileExists(atPath: referenced.path) {
                return referenced
            }
        }

        let requested = repositoryRoot.appending(path: revisionKey(options.revision))
        return fileManager.fileExists(atPath: requested.path) ? requested : nil
    }

    private static func resolveDownloadBase(
        requested: URL?,
        fileManager: FileManager
    ) throws -> URL {
        if let requested {
            try fileManager.createDirectory(at: requested, withIntermediateDirectories: true)
            return requested
        }

        let env = ProcessInfo.processInfo.environment
        if let hubCache = env["MERERUN_HUB_CACHE"], !hubCache.isEmpty {
            let url = URL(fileURLWithPath: hubCache)
            try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        if let cacheHome = env["MERERUN_MODEL_CACHE_HOME"], !cacheHome.isEmpty {
            let url = URL(fileURLWithPath: cacheHome).appending(path: "hub")
            try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let url = appSupport.appending(path: "MereRun/hub")
            try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        if let caches = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let url = caches.appending(path: "MereRun/hub")
            try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        let url = fileManager.temporaryDirectory.appending(path: "MereRun/hub")
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func prepareMaterializedSnapshot(progressHandler: ProgressHandler?) async throws -> URL {
        let tree = try await remoteTreeEntries()
        let entries = tree.entries
            .filter { $0.type == "file" && Self.matchesPath($0.path, patterns: options.patterns) }
            .sorted { $0.path < $1.path }
        guard !entries.isEmpty else {
            throw Hub.HubClientError.fileNotFound(options.patterns.joined(separator: ", "))
        }

        var preResolvedFiles: [String: HubSnapshotRemoteFile] = [:]
        let resolvedRevision: String
        if let treeRevision = tree.resolvedRevision {
            resolvedRevision = treeRevision
        } else {
            let firstEntry = entries[0]
            let remote = try await resolveRemoteFile(
                source: resolveURL(for: firstEntry.path),
                relativePath: firstEntry.path
            )
            resolvedRevision = remote.commitHash
            preResolvedFiles[firstEntry.path] = remote
        }

        let snapshotURL = materializedSnapshotURL(resolvedRevision: resolvedRevision)
        let metadataURL = snapshotURL
            .appending(path: ".cache")
            .appending(path: "huggingface")
            .appending(path: "download")
        try FileManager.default.createDirectory(at: snapshotURL, withIntermediateDirectories: true)

        let totalBytes = max(entries.reduce(Int64(0)) { partial, entry in
            partial + max(entry.size ?? 0, 0)
        }, 1)
        var completedBytes: Int64 = 0
        var receiptFiles: [HubSnapshotReceipt.File] = []
        progressHandler?(HubSnapshotProgress(completedUnitCount: completedBytes, totalUnitCount: totalBytes))

        for entry in entries {
            try Self.validateRelativePath(entry.path)
            let expectedBytes = max(entry.size ?? 0, 0)
            let destination = snapshotURL.appending(path: entry.path)
            let metadataDestination = metadataURL.appending(path: entry.path + ".metadata")
            if Self.fileExists(at: destination, expectedBytes: expectedBytes),
               let metadata = Self.readDownloadMetadata(at: metadataDestination),
               metadata.commitHash == resolvedRevision {
                completedBytes += max(expectedBytes, Self.fileSize(at: destination))
                receiptFiles.append(
                    HubSnapshotReceipt.File(
                        path: entry.path,
                        size: Self.fileSize(at: destination),
                        commit: metadata.commitHash,
                        etag: metadata.etag
                    )
                )
                progressHandler?(HubSnapshotProgress(
                    completedUnitCount: min(completedBytes, totalBytes),
                    totalUnitCount: totalBytes
                ))
                continue
            }

            if let legacy = try importLegacyPayload(
                relativePath: entry.path,
                expectedBytes: expectedBytes,
                resolvedRevision: resolvedRevision,
                destination: destination,
                metadataDestination: metadataDestination
            ) {
                completedBytes += max(expectedBytes, Self.fileSize(at: destination))
                receiptFiles.append(
                    HubSnapshotReceipt.File(
                        path: entry.path,
                        size: Self.fileSize(at: destination),
                        commit: legacy.commitHash,
                        etag: legacy.etag
                    )
                )
                progressHandler?(HubSnapshotProgress(
                    completedUnitCount: min(completedBytes, totalBytes),
                    totalUnitCount: totalBytes
                ))
                continue
            }

            let remote: HubSnapshotRemoteFile
            if let preResolved = preResolvedFiles.removeValue(forKey: entry.path) {
                remote = preResolved
            } else {
                let source = resolveURL(for: entry.path)
                remote = try await resolveRemoteFile(source: source, relativePath: entry.path)
            }
            guard remote.commitHash == resolvedRevision else {
                throw Hub.HubClientError.downloadError(
                    "Repository revision changed while downloading \(options.repoId): "
                        + "expected \(resolvedRevision), found \(remote.commitHash)"
                )
            }
            if let adopted = try adoptExistingPayload(
                expectedBytes: expectedBytes,
                destination: destination,
                metadataDestination: metadataDestination,
                remote: remote
            ) {
                completedBytes += max(expectedBytes, Self.fileSize(at: destination))
                receiptFiles.append(
                    HubSnapshotReceipt.File(
                        path: entry.path,
                        size: Self.fileSize(at: destination),
                        commit: adopted.commitHash,
                        etag: adopted.etag
                    )
                )
                progressHandler?(HubSnapshotProgress(
                    completedUnitCount: min(completedBytes, totalBytes),
                    totalUnitCount: totalBytes
                ))
                continue
            }
            let startedAt = Date()
            let completedBeforeDownload = completedBytes
            let delegate = HubSnapshotDownloadDelegate { written, _, _ in
                let elapsed = max(Date().timeIntervalSince(startedAt), 0.001)
                let speed = Double(written) / elapsed
                progressHandler?(HubSnapshotProgress(
                    completedUnitCount: min(completedBeforeDownload + written, totalBytes),
                    totalUnitCount: totalBytes,
                    estimatedSpeedBytesPerSecond: speed
                ))
            }
            let tempURL = try await download(remote.downloadURL, delegate: delegate, relativePath: entry.path)
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? FileManager.default.removeItem(at: destination)
            try materializeDownloadedPayload(tempURL, remote: remote, destination: destination)
            try writeDownloadMetadata(remote, to: metadataDestination)
            completedBytes += max(expectedBytes, Self.fileSize(at: destination))
            receiptFiles.append(
                HubSnapshotReceipt.File(
                    path: entry.path,
                    size: Self.fileSize(at: destination),
                    commit: remote.commitHash,
                    etag: remote.etag
                )
            )
            progressHandler?(HubSnapshotProgress(
                completedUnitCount: min(completedBytes, totalBytes),
                totalUnitCount: totalBytes,
                estimatedSpeedBytesPerSecond: delegate.lastSpeedBytesPerSecond
            ))
        }

        try writeSnapshotReceipt(
            HubSnapshotReceipt(
                schemaVersion: 1,
                repository: options.repoId,
                requestedRevision: options.revision,
                resolvedRevision: resolvedRevision,
                files: receiptFiles
            ),
            to: snapshotURL
        )
        try writeRequestedRevisionReference(resolvedRevision: resolvedRevision)
        progressHandler?(HubSnapshotProgress(completedUnitCount: totalBytes, totalUnitCount: totalBytes))
        return snapshotURL
    }

    private func remoteTreeEntries() async throws -> HubSnapshotTree {
        let initialURL = hostURL()
            .appending(path: "api")
            .appending(path: options.repoType.rawValue)
            .appending(path: options.repoId)
            .appending(path: "tree")
            .appending(component: options.revision)
        var components = URLComponents(url: initialURL, resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        var nextURL: URL? = components?.url ?? initialURL

        var entries: [HubSnapshotTreeEntry] = []
        var resolvedRevision: String?
        var pageCount = 0
        while let pageURL = nextURL {
            pageCount += 1
            guard pageCount <= 1_000 else {
                throw Hub.HubClientError.downloadError("Too many Hub tree pages for \(options.repoId)")
            }
            var request = authorizedRequest(url: pageURL)
            request.httpMethod = "GET"
            let (data, response) = try await URLSession.shared.data(for: request)
            let http = try Self.validateHTTPResponse(
                response,
                data: data,
                context: "\(options.repoId)@\(options.revision)"
            )
            entries.append(contentsOf: try JSONDecoder().decode([HubSnapshotTreeEntry].self, from: data))
            resolvedRevision = resolvedRevision ?? http.value(forHTTPHeaderField: "X-Repo-Commit")
            nextURL = Self.nextPageURL(
                from: http.value(forHTTPHeaderField: "Link"),
                relativeTo: pageURL
            )
        }
        return HubSnapshotTree(entries: entries, resolvedRevision: resolvedRevision)
    }

    private func resolveRemoteFile(source: URL, relativePath: String) async throws -> HubSnapshotRemoteFile {
        var request = authorizedRequest(url: source)
        request.httpMethod = "HEAD"
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let delegate = HubSnapshotNoRedirectDelegate()
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let (data, response) = try await session.data(for: request)
        let http = try Self.validateHTTPResponse(response, data: data, context: "\(options.repoId)@\(options.revision)/\(relativePath)", allowsRedirect: true)

        let downloadURL: URL
        if (300..<400).contains(http.statusCode),
           let location = http.value(forHTTPHeaderField: "Location"),
           let resolved = Self.redirectURL(from: location, relativeTo: source) {
            downloadURL = resolved
        } else {
            downloadURL = source
        }

        let commitHash = http.value(forHTTPHeaderField: "X-Repo-Commit") ?? options.revision
        let etag = Self.normalizedETag(
            http.value(forHTTPHeaderField: "X-Linked-ETag")
                ?? http.value(forHTTPHeaderField: "ETag")
        )
        return HubSnapshotRemoteFile(commitHash: commitHash, etag: etag, downloadURL: downloadURL)
    }

    private func download(
        _ url: URL,
        delegate: HubSnapshotDownloadDelegate,
        relativePath: String
    ) async throws -> URL {
        var currentURL = url
        for _ in 0..<8 {
            var request = authorizedRequest(url: currentURL)
            request.httpMethod = "GET"
            let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
            defer { session.invalidateAndCancel() }
            let (tempURL, response) = try await session.download(for: request)
            let http = try Self.validateDownloadedResponse(
                response,
                errorBodyURL: tempURL,
                context: "\(options.repoId)@\(options.revision)/\(relativePath)"
            )
            if (200..<300).contains(http.statusCode) {
                return tempURL
            }
            try? FileManager.default.removeItem(at: tempURL)
            guard (300..<400).contains(http.statusCode),
                  let location = http.value(forHTTPHeaderField: "Location"),
                  let redirected = Self.redirectURL(from: location, relativeTo: currentURL) else {
                throw Hub.HubClientError.httpStatusCode(http.statusCode)
            }
            currentURL = redirected
        }
        throw Hub.HubClientError.downloadError("Too many redirects while downloading \(options.repoId)/\(relativePath)")
    }

    private func writeDownloadMetadata(_ remote: HubSnapshotRemoteFile, to metadataURL: URL) throws {
        try FileManager.default.createDirectory(
            at: metadataURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let content = "\(remote.commitHash)\n\(remote.etag ?? "")\n\(Date().timeIntervalSince1970)\n"
        try content.write(to: metadataURL, atomically: true, encoding: .utf8)
    }

    private func materializeDownloadedPayload(
        _ temporaryURL: URL,
        remote: HubSnapshotRemoteFile,
        destination: URL
    ) throws {
        guard let etag = remote.etag, !etag.isEmpty else {
            try FileManager.default.moveItem(at: temporaryURL, to: destination)
            return
        }

        let blobURL = contentBlobURL(etag: etag)
        try FileManager.default.createDirectory(
            at: blobURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: blobURL.path) {
            do {
                try FileManager.default.moveItem(at: temporaryURL, to: blobURL)
            } catch {
                if FileManager.default.fileExists(atPath: blobURL.path) {
                    try? FileManager.default.removeItem(at: temporaryURL)
                } else {
                    throw error
                }
            }
        } else {
            guard Self.fileSize(at: blobURL) == Self.fileSize(at: temporaryURL) else {
                try? FileManager.default.removeItem(at: temporaryURL)
                throw Hub.HubClientError.downloadError(
                    "Hub blob identity collision for \(remote.etag ?? "unknown ETag")"
                )
            }
            try? FileManager.default.removeItem(at: temporaryURL)
        }
        try FileManager.default.linkItem(at: blobURL, to: destination)
    }

    private func importLegacyPayload(
        relativePath: String,
        expectedBytes: Int64,
        resolvedRevision: String,
        destination: URL,
        metadataDestination: URL
    ) throws -> HubSnapshotDownloadMetadata? {
        let legacyRoot = legacySnapshotURL()
        let legacyPayload = legacyRoot.appending(path: relativePath)
        let legacyMetadata = legacyRoot
            .appending(path: ".cache")
            .appending(path: "huggingface")
            .appending(path: "download")
            .appending(path: relativePath + ".metadata")
        guard Self.fileExists(at: legacyPayload, expectedBytes: expectedBytes),
              let metadata = Self.readDownloadMetadata(at: legacyMetadata),
              metadata.commitHash == resolvedRevision else {
            return nil
        }

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? FileManager.default.removeItem(at: destination)
        if let etag = metadata.etag, !etag.isEmpty {
            let blobURL = contentBlobURL(etag: etag)
            try FileManager.default.createDirectory(
                at: blobURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if !FileManager.default.fileExists(atPath: blobURL.path) {
                try FileManager.default.linkItem(at: legacyPayload, to: blobURL)
            }
            try FileManager.default.linkItem(at: blobURL, to: destination)
        } else {
            try FileManager.default.linkItem(at: legacyPayload, to: destination)
        }
        try FileManager.default.createDirectory(
            at: metadataDestination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? FileManager.default.removeItem(at: metadataDestination)
        try FileManager.default.copyItem(at: legacyMetadata, to: metadataDestination)
        return metadata
    }

    /// Adopt an exact payload reconstructed by an external resumable
    /// downloader. Size alone is not sufficient: bind the bytes to the pinned
    /// Hub revision by validating its content-addressed ETag before writing the
    /// local metadata and receipt used by subsequent offline pulls.
    private func adoptExistingPayload(
        expectedBytes: Int64,
        destination: URL,
        metadataDestination: URL,
        remote: HubSnapshotRemoteFile
    ) throws -> HubSnapshotDownloadMetadata? {
        guard Self.fileExists(at: destination, expectedBytes: expectedBytes),
              let etag = remote.etag,
              try Self.payloadMatchesETag(at: destination, etag: etag, byteCount: expectedBytes) else {
            return nil
        }

        let blobURL = contentBlobURL(etag: etag)
        try FileManager.default.createDirectory(
            at: blobURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: blobURL.path) {
            guard Self.fileSize(at: blobURL) == expectedBytes else {
                throw Hub.HubClientError.downloadError(
                    "Hub blob identity collision for \(etag)"
                )
            }
        } else {
            try FileManager.default.linkItem(at: destination, to: blobURL)
        }
        try writeDownloadMetadata(remote, to: metadataDestination)
        return HubSnapshotDownloadMetadata(commitHash: remote.commitHash, etag: etag)
    }

    private func writeSnapshotReceipt(_ receipt: HubSnapshotReceipt, to snapshotURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(receipt)
        try data.write(
            to: snapshotURL.appendingPathComponent(HubSnapshotReceipt.filename, isDirectory: false),
            options: .atomic
        )
    }

    private func writeRequestedRevisionReference(resolvedRevision: String) throws {
        let referenceURL = requestedRevisionReferenceURL()
        try FileManager.default.createDirectory(
            at: referenceURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(resolvedRevision.utf8).write(to: referenceURL, options: .atomic)
    }

    private func offlineMaterializedSnapshotURL() -> URL? {
        if let data = try? Data(contentsOf: requestedRevisionReferenceURL()),
           let resolved = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !resolved.isEmpty {
            let snapshot = materializedSnapshotURL(resolvedRevision: resolved)
            if FileManager.default.fileExists(atPath: snapshot.path) {
                return snapshot
            }
        }
        let requested = materializedSnapshotURL(resolvedRevision: options.revision)
        return FileManager.default.fileExists(atPath: requested.path) ? requested : nil
    }

    private func materializedSnapshotURL(resolvedRevision: String) -> URL {
        downloadBase
            .appending(path: "snapshots")
            .appending(path: options.repoType.rawValue)
            .appending(path: options.repoId)
            .appending(path: Self.revisionKey(resolvedRevision))
    }

    private func legacySnapshotURL() -> URL {
        downloadBase
            .appending(path: options.repoType.rawValue)
            .appending(path: options.repoId)
    }

    private func requestedRevisionReferenceURL() -> URL {
        downloadBase
            .appending(path: "refs")
            .appending(path: options.repoType.rawValue)
            .appending(path: options.repoId)
            .appending(path: Self.revisionKey(options.revision) + ".ref")
    }

    private func contentBlobURL(etag: String) -> URL {
        let key = Self.contentKey(etag)
        return downloadBase
            .appending(path: "blobs")
            .appending(path: String(key.prefix(2)))
            .appending(path: key)
    }

    private func resolveURL(for relativePath: String) -> URL {
        var url = hostURL()
        if options.repoType != .models {
            url = url.appending(path: options.repoType.rawValue)
        }
        return url
            .appending(path: options.repoId)
            .appending(path: "resolve")
            .appending(component: options.revision)
            .appending(path: relativePath)
    }

    private func authorizedRequest(url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        if let token = accessToken(), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func accessToken() -> String? {
        // Precedence: explicit option > environment > persisted config file.
        if let token = options.accessToken {
            return token
        }
        let env = ProcessInfo.processInfo.environment
        if let token = env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"], !token.isEmpty {
            return token
        }
        return MereRunConfig.load().hfToken
    }

    private func hostURL() -> URL {
        // Precedence: HF_ENDPOINT env > persisted config > huggingface.co
        let endpoint = ProcessInfo.processInfo.environment["HF_ENDPOINT"]
            ?? MereRunConfig.load().hfEndpoint
        if let endpoint, let url = URL(string: endpoint), url.scheme != nil, url.host != nil {
            return url
        }
        return URL(string: "https://huggingface.co")!
    }

    static func matchesPath(_ path: String, patterns: [String]) -> Bool {
        guard !patterns.isEmpty else { return true }
        return patterns.contains { pattern in
            guard let regex = globRegex(pattern) else {
                return path == pattern
            }
            return path == pattern || regex.firstMatch(
                in: path,
                range: NSRange(path.startIndex..<path.endIndex, in: path)
            ) != nil
        }
    }

    static func redirectURL(from location: String, relativeTo source: URL) -> URL? {
        URL(string: location, relativeTo: source)?.absoluteURL
    }

    static func nextPageURL(from linkHeader: String?, relativeTo source: URL) -> URL? {
        guard let linkHeader else { return nil }
        for entry in linkHeader.split(separator: ",") {
            let parts = entry.split(separator: ";", omittingEmptySubsequences: true)
            guard let first = parts.first,
                  parts.dropFirst().contains(where: {
                      $0.trimmingCharacters(in: .whitespacesAndNewlines) == "rel=\"next\""
                  }) else {
                continue
            }
            let value = first.trimmingCharacters(in: .whitespacesAndNewlines)
            guard value.hasPrefix("<"), value.hasSuffix(">") else { continue }
            return URL(string: String(value.dropFirst().dropLast()), relativeTo: source)?.absoluteURL
        }
        return nil
    }

    static func revisionKey(_ revision: String) -> String {
        SHA256.hash(data: Data(revision.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func payloadMatchesETag(
        at url: URL,
        etag: String,
        byteCount: Int64
    ) throws -> Bool {
        let expected = etag.lowercased()
        guard expected.allSatisfy({ $0.isHexDigit }) else { return false }

        if expected.count == 64 {
            return try ModelArtifactPin.fileSHA256(url) == expected
        }
        guard expected.count == 40 else { return false }

        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = Insecure.SHA1()
        hasher.update(data: Data("blob \(byteCount)\0".utf8))
        while true {
            let chunk = try handle.read(upToCount: 1_048_576) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return actual == expected
    }

    private static func contentKey(_ etag: String) -> String {
        let safe = etag.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_.")).contains($0)
        }
        return safe && !etag.isEmpty ? etag : revisionKey(etag)
    }

    private static func globRegex(_ pattern: String) -> NSRegularExpression? {
        var regex = "^"
        for scalar in pattern.unicodeScalars {
            switch scalar {
            case "*":
                regex += ".*"
            case "?":
                regex += "."
            case ".", "\\", "+", "(", ")", "{", "}", "[", "]", "^", "$", "|":
                regex += "\\\(String(scalar))"
            default:
                regex += String(scalar)
            }
        }
        regex += "$"
        return try? NSRegularExpression(pattern: regex)
    }

    private static func validateRelativePath(_ path: String) throws {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.split(separator: "/").contains("..") else {
            throw Hub.HubClientError.downloadError("Unsafe Hub path: \(path)")
        }
    }

    private static func fileExists(at url: URL, expectedBytes: Int64) -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return false
        }
        return expectedBytes <= 0 || fileSize(at: url) == expectedBytes
    }

    private static func fileSize(at url: URL) -> Int64 {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?
            .int64Value
        return size ?? 0
    }

    private static func normalizedETag(_ etag: String?) -> String? {
        guard var etag else { return nil }
        if etag.hasPrefix("W/") {
            etag = String(etag.dropFirst(2))
        }
        return etag.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }

    private static func readDownloadMetadata(at url: URL) -> HubSnapshotDownloadMetadata? {
        guard let content = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let lines = content.components(separatedBy: .newlines)
        guard let commit = lines.first, !commit.isEmpty else { return nil }
        let etag = lines.count > 1 && !lines[1].isEmpty ? lines[1] : nil
        return HubSnapshotDownloadMetadata(commitHash: commit, etag: etag)
    }

    @discardableResult
    private static func validateDownloadedResponse(
        _ response: URLResponse,
        errorBodyURL: URL,
        context: String
    ) throws -> HTTPURLResponse {
        guard let http = response as? HTTPURLResponse else {
            throw Hub.HubClientError.unexpectedError
        }
        guard !(200..<300).contains(http.statusCode),
              !(300..<400).contains(http.statusCode) else {
            return http
        }
        return try validateHTTPResponse(
            response,
            data: try? Data(contentsOf: errorBodyURL),
            context: context,
            allowsRedirect: true
        )
    }

    @discardableResult
    private static func validateHTTPResponse(
        _ response: URLResponse,
        data: Data?,
        context: String,
        allowsRedirect: Bool = false
    ) throws -> HTTPURLResponse {
        guard let http = response as? HTTPURLResponse else {
            throw Hub.HubClientError.unexpectedError
        }
        if (200..<300).contains(http.statusCode) || (allowsRedirect && (300..<400).contains(http.statusCode)) {
            return http
        }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw Hub.HubClientError.authorizationRequired
        }
        if http.statusCode == 404 {
            throw Hub.HubClientError.fileNotFound(context)
        }
        if let data,
           let body = String(data: data, encoding: .utf8),
           !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw Hub.HubClientError.downloadError("\(context): HTTP \(http.statusCode): \(body)")
        }
        throw Hub.HubClientError.httpStatusCode(http.statusCode)
    }
}

private struct HubSnapshotTree: Sendable {
    let entries: [HubSnapshotTreeEntry]
    let resolvedRevision: String?
}

private struct HubSnapshotTreeEntry: Decodable, Sendable {
    let path: String
    let type: String
    let size: Int64?
}

private struct HubSnapshotDownloadMetadata: Sendable {
    let commitHash: String
    let etag: String?
}

private struct HubSnapshotRemoteFile: Sendable {
    let commitHash: String
    let etag: String?
    let downloadURL: URL
}

private final class HubSnapshotNoRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _: URLSession,
        task _: URLSessionTask,
        willPerformHTTPRedirection _: HTTPURLResponse,
        newRequest _: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

private final class HubSnapshotDownloadDelegate: NSObject, URLSessionTaskDelegate, URLSessionDownloadDelegate, @unchecked Sendable {
    private let startedAt = Date()
    private let progressHandler: @Sendable (Int64, Int64, Double?) -> Void
    private(set) var lastSpeedBytesPerSecond: Double?

    init(progressHandler: @escaping @Sendable (Int64, Int64, Double?) -> Void) {
        self.progressHandler = progressHandler
    }

    func urlSession(
        _: URLSession,
        task _: URLSessionTask,
        willPerformHTTPRedirection _: HTTPURLResponse,
        newRequest _: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }

    func urlSession(
        _: URLSession,
        downloadTask _: URLSessionDownloadTask,
        didFinishDownloadingTo _: URL
    ) {}

    func urlSession(
        _: URLSession,
        downloadTask _: URLSessionDownloadTask,
        didWriteData _: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let elapsed = max(Date().timeIntervalSince(startedAt), 0.001)
        let speed = Double(totalBytesWritten) / elapsed
        lastSpeedBytesPerSecond = speed
        progressHandler(totalBytesWritten, totalBytesExpectedToWrite, speed)
    }
}
