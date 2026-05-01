import Foundation
@preconcurrency import Hub

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
}
