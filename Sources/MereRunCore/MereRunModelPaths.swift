import Foundation

public enum MereRunModelPaths {
    public static let modelsDirEnvironmentKey = "MERERUN_MODELS_DIR"
    public static let modelStorageActivePathDefaultsKey = "mererun.models.storage.activePath"
    public static let modelStorageActiveBookmarkDefaultsKey = "mererun.models.storage.activeBookmark"
    public static let modelStoragePendingPathDefaultsKey = "mererun.models.storage.pendingPath"
    public static let modelStoragePendingBookmarkDefaultsKey = "mererun.models.storage.pendingBookmark"

    public enum ModelStoreSource: String, Sendable {
        case processOverride
        case environment
        case persisted
        case `default`
    }

    public struct ModelStoreResolution: Sendable {
        public let source: ModelStoreSource
        public let activeModelsDir: URL
        public let configuredModelsDir: URL?
        public let isFallbackToDefault: Bool

        public init(
            source: ModelStoreSource,
            activeModelsDir: URL,
            configuredModelsDir: URL?,
            isFallbackToDefault: Bool
        ) {
            self.source = source
            self.activeModelsDir = activeModelsDir
            self.configuredModelsDir = configuredModelsDir
            self.isFallbackToDefault = isFallbackToDefault
        }
    }

    private static let resolvedBase = resolveApplicationSupportBase()
    private static let modelOverrideQueue = DispatchQueue(label: "MereRunModelPaths.modelsOverride")
    nonisolated(unsafe) private static var processModelsDirOverride: URL?

    public static var applicationSupportBase: URL {
        resolvedBase
    }

    /// `~/Library/Application Support/MereRun/models`
    public static var defaultModelsDir: URL {
        applicationSupportBase.appendingPathComponent("models", isDirectory: true)
    }

    public static func setProcessModelsDirOverride(_ url: URL?) {
        modelOverrideQueue.sync {
            processModelsDirOverride = url?.standardizedFileURL
        }
    }

    public static var modelStoreSource: ModelStoreSource {
        modelStoreResolution().source
    }

    public static var configuredModelsDir: URL? {
        modelStoreResolution().configuredModelsDir
    }

    public static var isUsingFallbackModelsDir: Bool {
        modelStoreResolution().isFallbackToDefault
    }

    private static func resolveApplicationSupportBase() -> URL {
        let fm = FileManager.default
        do {
            let dir = try fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            return dir.appendingPathComponent("MereRun", isDirectory: true)
        } catch {
            let fallback = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            return fallback.appendingPathComponent("MereRun", isDirectory: true)
        }
    }

    public static var modelsDir: URL {
        modelStoreResolution().activeModelsDir
    }

    public static var downloadsDir: URL {
        applicationSupportBase.appendingPathComponent("downloads", isDirectory: true)
    }

    /// `~/Library/Application Support/MereRun/adapters`
    public static var adaptersDir: URL {
        applicationSupportBase.appendingPathComponent("adapters", isDirectory: true)
    }

    public static var outputDir: URL {
        applicationSupportBase.appendingPathComponent("output", isDirectory: true)
    }

    public static var modelCacheBase: URL {
        applicationSupportBase.appendingPathComponent("model-cache", isDirectory: true)
    }

    public static func modelDir(_ id: String) -> URL {
        modelsDir.appendingPathComponent(id, isDirectory: true)
    }

    public static func modelStoreResolution(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaults: UserDefaults = .standard
    ) -> ModelStoreResolution {
        let defaultModelsDir = self.defaultModelsDir.standardizedFileURL

        if let processOverride = currentProcessModelsDirOverride() {
            return resolveConfiguredModelsDir(
                processOverride,
                source: .processOverride,
                defaultModelsDir: defaultModelsDir,
                fileManager: fileManager
            )
        }

        if let envPath = environment[modelsDirEnvironmentKey], !envPath.isEmpty {
            return resolveConfiguredModelsDir(
                normalizedPathURL(envPath),
                source: .environment,
                defaultModelsDir: defaultModelsDir,
                fileManager: fileManager
            )
        }

        if let persisted = persistedActiveModelsDir(defaults: defaults) {
            return resolveConfiguredModelsDir(
                persisted,
                source: .persisted,
                defaultModelsDir: defaultModelsDir,
                fileManager: fileManager
            )
        }

        return ModelStoreResolution(
            source: .default,
            activeModelsDir: defaultModelsDir,
            configuredModelsDir: nil,
            isFallbackToDefault: false
        )
    }

    public static func outputDir(for feature: String) -> URL {
        outputDir.appendingPathComponent(feature, isDirectory: true)
    }

    /// Returns the primary `models/<id>` when no candidates validate.
    /// Callers that support nested layouts should return the nested URL from `validator`.
    public static func resolveModelDir(_ id: String, validator: (URL) -> Bool) -> URL {
        let candidates = candidateModelRootsForLookup().map {
            $0.appendingPathComponent(id, isDirectory: true)
        }

        for candidate in candidates {
            if validator(candidate) {
                return candidate
            }
        }

        return modelDir(id)
    }

    /// Resolves a file under `models/` by relative path (e.g. `"qwen3.gguf"` or `"foo/bar.bin"`).
    /// Returns the primary path when none of the lookup candidates validate.
    public static func resolveModelFile(relativePath: String, validator: (URL) -> Bool) -> URL {
        let candidates = candidateModelRootsForLookup().map {
            $0.appendingPathComponent(relativePath, isDirectory: false)
        }

        for candidate in candidates {
            if validator(candidate) {
                return candidate
            }
        }

        return modelsDir.appendingPathComponent(relativePath, isDirectory: false)
    }

    private static func currentProcessModelsDirOverride() -> URL? {
        modelOverrideQueue.sync {
            processModelsDirOverride
        }
    }

    private static func resolveConfiguredModelsDir(
        _ configured: URL,
        source: ModelStoreSource,
        defaultModelsDir: URL,
        fileManager: FileManager
    ) -> ModelStoreResolution {
        let standardized = configured.standardizedFileURL
        if standardized.path == defaultModelsDir.path {
            return ModelStoreResolution(
                source: source,
                activeModelsDir: defaultModelsDir,
                configuredModelsDir: standardized,
                isFallbackToDefault: false
            )
        }

        if source == .processOverride || source == .environment {
            return ModelStoreResolution(
                source: source,
                activeModelsDir: standardized,
                configuredModelsDir: standardized,
                isFallbackToDefault: false
            )
        }

        if isReadableDirectory(standardized, fileManager: fileManager) {
            return ModelStoreResolution(
                source: source,
                activeModelsDir: standardized,
                configuredModelsDir: standardized,
                isFallbackToDefault: false
            )
        }

        return ModelStoreResolution(
            source: source,
            activeModelsDir: defaultModelsDir,
            configuredModelsDir: standardized,
            isFallbackToDefault: true
        )
    }

    private static func persistedActiveModelsDir(defaults: UserDefaults) -> URL? {
        if let local = defaults.string(forKey: modelStorageActivePathDefaultsKey), !local.isEmpty {
            return normalizedPathURL(local)
        }

        return nil
    }

    private static func normalizedPathURL(_ raw: String) -> URL {
        let expanded = (raw as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded).standardizedFileURL
    }

    private static func isReadableDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return false
        }
        return fileManager.isReadableFile(atPath: url.path)
    }

    private static func candidateModelRootsForLookup() -> [URL] {
        let roots = [modelsDir.standardizedFileURL]

        var seen: Set<String> = []
        var unique: [URL] = []
        for root in roots where seen.insert(root.path).inserted {
            unique.append(root)
        }
        return unique
    }
}
