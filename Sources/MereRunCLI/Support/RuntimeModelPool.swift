import Foundation
import MereRunCore

struct RuntimeModelPoolStatus: Codable, Equatable, Sendable {
    let object: String
    let defaultModel: String
    let settingsPath: String
    let activeRequests: Int
    let admission: RuntimeRequestAdmissionSnapshot?
    let capabilities: RuntimeControlPlaneCapabilities
    let memory: RuntimeMemorySnapshot
    let models: [RuntimeModelPoolEntrySnapshot]
    let cacheStats: RuntimeCacheStatsSummary
    let benchmarkStats: RuntimeBenchmarkStatsSummary?
}

struct RuntimeControlPlaneCapabilities: Codable, Equatable, Sendable {
    let requestAdmission: RuntimeCapabilityStatus
    let chunkedPrefill: RuntimeCapabilityStatus
    let continuousBatching: RuntimeCapabilityStatus
    let prefixKVReuse: RuntimeCapabilityStatus
    let ssdKVCache: RuntimeCapabilityStatus

    static func current(
        gemma4PrefixKVCacheEnabled: Bool,
        gemma4ContinuousBatchingEnabled: Bool,
        q35ContinuousBatchingEnabled: Bool,
        q35PrefixKVCacheEnabled: Bool
    ) -> RuntimeControlPlaneCapabilities {
        let prefixKVCacheEnabled = gemma4PrefixKVCacheEnabled || q35PrefixKVCacheEnabled
        let continuousBatchingEnabled = gemma4ContinuousBatchingEnabled || q35ContinuousBatchingEnabled
        return RuntimeControlPlaneCapabilities(
            requestAdmission: RuntimeCapabilityStatus(
                available: true,
                enabled: true,
                detail: "Fair FIFO request admission is active."
            ),
            chunkedPrefill: RuntimeCapabilityStatus(
                available: true,
                enabled: true,
                detail: "Gemma4 and Qwen-family models prefill long prompts in cancellable chunks."
            ),
            continuousBatching: RuntimeCapabilityStatus(
                available: true,
                enabled: continuousBatchingEnabled,
                detail: continuousBatchingEnabled
                    ? "Gemma4 and Qwen-family decode rows are packed when typed cache state is compatible; Qwen linear-only rows may batch across decode positions."
                    : "Decode batching is available behind MERERUN_GEMMA4_CONTINUOUS_BATCHING=1 and MERERUN_Q35_CONTINUOUS_BATCHING=1; set --max-active-requests above 1 to allow overlap."
            ),
            prefixKVReuse: RuntimeCapabilityStatus(
                available: true,
                enabled: prefixKVCacheEnabled,
                detail: prefixKVCacheEnabled
                    ? "In-memory prefix KV reuse is enabled for matching Gemma4 or Qwen-family text token prefixes."
                    : "In-memory prefix KV reuse is available behind MERERUN_GEMMA4_PREFIX_KV_CACHE=1 and MERERUN_Q35_PREFIX_KV_CACHE=1."
            ),
            ssdKVCache: RuntimeCapabilityStatus(
                available: false,
                enabled: false,
                detail: "Unavailable until in-memory prefix reuse shows measured TTFT or throughput wins."
            )
        )
    }
}

struct RuntimeCapabilityStatus: Codable, Equatable, Sendable {
    let available: Bool
    let enabled: Bool
    let detail: String
}

struct RuntimeRequestAdmissionSnapshot: Codable, Equatable, Sendable {
    let maxActiveRequests: Int
    let activeRequests: Int
    let queuedRequests: Int
    let totalAdmittedRequests: Int
    let totalCompletedRequests: Int
    let totalCancelledRequests: Int
}

struct RuntimeFutureStats: Codable, Equatable, Sendable {
    let available: Bool
    let detail: String

    static let notImplemented = RuntimeFutureStats(
        available: false,
        detail: "not implemented"
    )
}

struct RuntimeModelBenchmarkStats: Codable, Equatable, Sendable {
    let completedRequests: Int
    let failedRequests: Int
    let generatedTokens: Int
    let averageLoadSeconds: Double?
    let averagePrefillSeconds: Double?
    let averageDecodeSeconds: Double?
    let averageTotalSeconds: Double?
    let decodeTokensPerSecond: Double?
    let lastCompletedAt: Date?

    init(
        completedRequests: Int,
        failedRequests: Int,
        generatedTokens: Int,
        totalLoadSeconds: Double,
        totalPrefillSeconds: Double,
        totalDecodeSeconds: Double,
        lastCompletedAt: Date?
    ) {
        self.completedRequests = completedRequests
        self.failedRequests = failedRequests
        self.generatedTokens = generatedTokens
        self.averageLoadSeconds = Self.average(totalLoadSeconds, count: completedRequests)
        self.averagePrefillSeconds = Self.average(totalPrefillSeconds, count: completedRequests)
        self.averageDecodeSeconds = Self.average(totalDecodeSeconds, count: completedRequests)
        self.averageTotalSeconds = Self.average(
            totalLoadSeconds + totalPrefillSeconds + totalDecodeSeconds,
            count: completedRequests
        )
        self.decodeTokensPerSecond = totalDecodeSeconds > 0
            ? Double(generatedTokens) / totalDecodeSeconds
            : nil
        self.lastCompletedAt = lastCompletedAt
    }

    private static func average(_ value: Double, count: Int) -> Double? {
        guard count > 0 else { return nil }
        return value / Double(count)
    }
}

struct RuntimeBenchmarkStatsSummary: Codable, Equatable, Sendable {
    let available: Bool
    let detail: String
    let reportedModelCount: Int
    let completedRequests: Int
    let failedRequests: Int
    let generatedTokens: Int
    let averageLoadSeconds: Double?
    let averagePrefillSeconds: Double?
    let averageDecodeSeconds: Double?
    let averageTotalSeconds: Double?
    let decodeTokensPerSecond: Double?

    init(stats: [RuntimeModelBenchmarkStats]) {
        available = true
        detail = "Aggregated from completed native chat requests observed by this runtime pool."
        reportedModelCount = stats.count
        completedRequests = stats.reduce(0) { $0 + $1.completedRequests }
        failedRequests = stats.reduce(0) { $0 + $1.failedRequests }
        generatedTokens = stats.reduce(0) { $0 + $1.generatedTokens }

        let loadSeconds = stats.weightedTotal(\.averageLoadSeconds, count: \.completedRequests)
        let prefillSeconds = stats.weightedTotal(\.averagePrefillSeconds, count: \.completedRequests)
        let decodeSeconds = stats.weightedTotal(\.averageDecodeSeconds, count: \.completedRequests)
        let totalSeconds = stats.weightedTotal(\.averageTotalSeconds, count: \.completedRequests)
        averageLoadSeconds = Self.average(loadSeconds, count: completedRequests)
        averagePrefillSeconds = Self.average(prefillSeconds, count: completedRequests)
        averageDecodeSeconds = Self.average(decodeSeconds, count: completedRequests)
        averageTotalSeconds = Self.average(totalSeconds, count: completedRequests)
        decodeTokensPerSecond = decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : nil
    }

    private static func average(_ value: Double, count: Int) -> Double? {
        guard count > 0 else { return nil }
        return value / Double(count)
    }
}

private extension Array where Element == RuntimeModelBenchmarkStats {
    func weightedTotal(
        _ value: KeyPath<RuntimeModelBenchmarkStats, Double?>,
        count: KeyPath<RuntimeModelBenchmarkStats, Int>
    ) -> Double {
        reduce(0) { partial, stats in
            guard let average = stats[keyPath: value] else { return partial }
            return partial + average * Double(stats[keyPath: count])
        }
    }
}

struct RuntimeCacheStatsSummary: Codable, Equatable, Sendable {
    let available: Bool
    let detail: String
    let prefixKVReuse: RuntimePrefixKVCacheSummary
    let decodeBatching: RuntimeDecodeBatchingSummary
    let ssdKVCache: RuntimeFutureStats

    static let empty = RuntimeCacheStatsSummary(
        prefixKVCaches: [],
        decodeBatchers: []
    )

    init(
        prefixKVCaches: [PrefixKVCacheStats],
        decodeBatchers: [RuntimeDecodeBatchingStats]
    ) {
        available = true
        detail = "Aggregated from loaded model runtime counters."
        prefixKVReuse = RuntimePrefixKVCacheSummary(stats: prefixKVCaches)
        decodeBatching = RuntimeDecodeBatchingSummary(stats: decodeBatchers)
        ssdKVCache = RuntimeFutureStats(
            available: false,
            detail: "Unavailable until in-memory prefix reuse shows measured TTFT or throughput wins."
        )
    }
}

struct RuntimePrefixKVCacheSummary: Codable, Equatable, Sendable {
    let reportedModelCount: Int
    let enabledModelCount: Int
    let entries: Int
    let maxEntries: Int
    let hits: Int
    let misses: Int
    let storedPrefixes: Int
    let reusedTokens: Int
    let storedTokens: Int

    init(stats: [PrefixKVCacheStats]) {
        reportedModelCount = stats.count
        enabledModelCount = stats.filter(\.enabled).count
        entries = stats.reduce(0) { $0 + $1.entries }
        maxEntries = stats.reduce(0) { $0 + $1.maxEntries }
        hits = stats.reduce(0) { $0 + $1.hits }
        misses = stats.reduce(0) { $0 + $1.misses }
        storedPrefixes = stats.reduce(0) { $0 + $1.storedPrefixes }
        reusedTokens = stats.reduce(0) { $0 + $1.reusedTokens }
        storedTokens = stats.reduce(0) { $0 + $1.storedTokens }
    }
}

struct RuntimeDecodeBatchingSummary: Codable, Equatable, Sendable {
    let reportedModelCount: Int
    let enabledModelCount: Int
    let activeRows: Int
    let queuedRows: Int
    let batchedDecodeSteps: Int
    let samePositionBatchedSteps: Int
    let variablePositionBatchedSteps: Int
    let singleDecodeSteps: Int
    let totalBatchedRows: Int
    let maxBatchSize: Int

    init(stats: [RuntimeDecodeBatchingStats]) {
        reportedModelCount = stats.count
        enabledModelCount = stats.filter(\.enabled).count
        activeRows = stats.reduce(0) { $0 + $1.activeRows }
        queuedRows = stats.reduce(0) { $0 + $1.queuedRows }
        batchedDecodeSteps = stats.reduce(0) { $0 + $1.batchedDecodeSteps }
        samePositionBatchedSteps = stats.reduce(0) { $0 + $1.samePositionBatchedSteps }
        variablePositionBatchedSteps = stats.reduce(0) { $0 + $1.variablePositionBatchedSteps }
        singleDecodeSteps = stats.reduce(0) { $0 + $1.singleDecodeSteps }
        totalBatchedRows = stats.reduce(0) { $0 + $1.totalBatchedRows }
        maxBatchSize = stats.map(\.maxBatchSize).max() ?? 0
    }
}

struct RuntimeMemorySnapshot: Codable, Equatable, Sendable {
    let physicalBytes: UInt64
    let activeRequests: Int
    let activeModelCount: Int
    let pressure: String
}

struct RuntimeModelPoolEntrySnapshot: Codable, Equatable, Sendable {
    let id: String
    let category: String
    let engine: RuntimeServingEngine
    let installPath: String?
    let loaded: Bool
    let activeRequests: Int
    let lastAccess: Date?
    let lastError: String?
    let pinned: Bool
    let alias: String?
    let ttlSeconds: Int?
    let maxContextTokens: Int?
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let engineOverride: RuntimeServingEngine?
    let kvCacheMode: RuntimeKVCacheMode?
    let prefixKVCache: PrefixKVCacheStats?
    let continuousBatching: RuntimeDecodeBatchingStats?
    let benchmarkStats: RuntimeModelBenchmarkStats?
}

struct RuntimeChatPlan: Sendable {
    let lease: RuntimeModelLease
    let request: ChatRequest
    let modelID: String
    let engine: RuntimeServingEngine
    let includeUsage: Bool
}

enum RuntimeModelPoolError: LocalizedError, Equatable {
    case unknownModel(String)
    case unsupportedModel(String)
    case modelNotInstalled(String)
    case incompatibleEngine(String)
    case unloadConflict(String, activeRequests: Int)
    case invalidSettings(String)
    case rawProxyUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .unknownModel(let id):
            return "Unknown model '\(id)'. Use a configured alias or a managed model id."
        case .unsupportedModel(let id):
            return "Model '\(id)' is not supported by `mere.run api serve`."
        case .modelNotInstalled(let id):
            return "Model '\(id)' is not installed. Run `mere.run model pull \(id)` first."
        case .incompatibleEngine(let detail):
            return detail
        case .unloadConflict(let id, let activeRequests):
            return "Model '\(id)' has \(activeRequests) active request(s); unload it after they finish."
        case .invalidSettings(let detail):
            return detail
        case .rawProxyUnavailable(let id):
            return "Model '\(id)' is not backed by a raw OpenAI proxy."
        }
    }
}

actor RuntimeModelPool {
    private struct MutableState {
        var activeRequests = 0
        var lastAccess: Date?
        var lastError: String?
        var completedRequests = 0
        var failedRequests = 0
        var generatedTokens = 0
        var totalLoadSeconds: Double = 0
        var totalPrefillSeconds: Double = 0
        var totalDecodeSeconds: Double = 0
        var lastCompletedAt: Date?

        var benchmarkStats: RuntimeModelBenchmarkStats {
            RuntimeModelBenchmarkStats(
                completedRequests: completedRequests,
                failedRequests: failedRequests,
                generatedTokens: generatedTokens,
                totalLoadSeconds: totalLoadSeconds,
                totalPrefillSeconds: totalPrefillSeconds,
                totalDecodeSeconds: totalDecodeSeconds,
                lastCompletedAt: lastCompletedAt
            )
        }
    }

    private struct ResolvedModel {
        let id: String
        let category: String
        let engine: RuntimeServingEngine
        let installPath: String?
        let settings: RuntimeModelSettings
        let spec: ManagedModelSpec?
    }

    private let defaultModelID: String
    private let defaultEngine: RuntimeServingEngine
    private let startupModelPath: String?
    private let settingsStore: RuntimeModelSettingsStore
    private let gemma4KVCacheQuantization: Gemma4KVCacheQuantization
    private let gemma4PrefixKVCacheEnabled: Bool
    private let gemma4ContinuousBatchingEnabled: Bool
    private let q35PrefixKVCacheEnabled: Bool
    private let q35ContinuousBatchingEnabled: Bool

    private var loadedModels: [String: RuntimeLoadedModel] = [:]
    private var states: [String: MutableState] = [:]

    init(
        defaultModelID: String,
        defaultEngine: RuntimeServingEngine,
        startupModelPath: String?,
        settingsStore: RuntimeModelSettingsStore = RuntimeModelSettingsStore(),
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        gemma4PrefixKVCacheEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_PREFIX_KV_CACHE"] == "1",
        gemma4ContinuousBatchingEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_CONTINUOUS_BATCHING"] == "1",
        q35PrefixKVCacheEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_Q35_PREFIX_KV_CACHE"] == "1",
        q35ContinuousBatchingEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_Q35_CONTINUOUS_BATCHING"] == "1"
    ) {
        self.defaultModelID = defaultModelID
        self.defaultEngine = defaultEngine
        self.startupModelPath = startupModelPath
        self.settingsStore = settingsStore
        self.gemma4KVCacheQuantization = gemma4KVCacheQuantization
        self.gemma4PrefixKVCacheEnabled = gemma4PrefixKVCacheEnabled
        self.gemma4ContinuousBatchingEnabled = gemma4ContinuousBatchingEnabled
        self.q35PrefixKVCacheEnabled = q35PrefixKVCacheEnabled
        self.q35ContinuousBatchingEnabled = q35ContinuousBatchingEnabled
    }

    func preloadDefault() async throws {
        _ = try await loadModel(idOrAlias: defaultModelID)
    }

    func modelsResponse(createdAt: Date = Date()) throws -> OpenAIModelsResponse {
        APIServerContract.modelsResponse(modelIds: try listedOpenAIModelIDs(), createdAt: createdAt)
    }

    func status() async -> RuntimeModelPoolStatus {
        await status(admission: nil)
    }

    func status(admission: RuntimeRequestAdmissionSnapshot?) async -> RuntimeModelPoolStatus {
        let settings = (try? settingsStore.load())?.models ?? [:]
        let installed = installedServableCatalogIDs()
        let ids = Set(installed + Array(loadedModels.keys) + [defaultModelID] + Array(settings.keys))
        var prefixStats: [String: PrefixKVCacheStats] = [:]
        var batchingStats: [String: RuntimeDecodeBatchingStats] = [:]
        let currentLoadedModels = loadedModels
        for (id, loaded) in currentLoadedModels {
            if let stats = await loaded.prefixKVCacheStats() {
                prefixStats[id] = stats
            }
            if let stats = await loaded.continuousBatchingStats() {
                batchingStats[id] = stats
            }
        }
        let snapshots = ids.sorted().compactMap { id in
            snapshot(
                for: id,
                settings: settings,
                prefixKVCache: prefixStats[id],
                continuousBatching: batchingStats[id]
            )
        }
        let activeRequests = snapshots.reduce(0) { $0 + $1.activeRequests }
        let loadedCount = snapshots.filter(\.loaded).count
        return RuntimeModelPoolStatus(
            object: "runtime.status",
            defaultModel: defaultModelID,
            settingsPath: settingsStore.url.path,
            activeRequests: activeRequests,
            admission: admission,
            capabilities: .current(
                gemma4PrefixKVCacheEnabled: gemma4PrefixKVCacheEnabled,
                gemma4ContinuousBatchingEnabled: gemma4ContinuousBatchingEnabled,
                q35ContinuousBatchingEnabled: q35ContinuousBatchingEnabled,
                q35PrefixKVCacheEnabled: q35PrefixKVCacheEnabled
            ),
            memory: RuntimeMemorySnapshot(
                physicalBytes: ProcessInfo.processInfo.physicalMemory,
                activeRequests: activeRequests,
                activeModelCount: loadedCount,
                pressure: "unknown"
            ),
            models: snapshots,
            cacheStats: RuntimeCacheStatsSummary(
                prefixKVCaches: Array(prefixStats.values),
                decodeBatchers: Array(batchingStats.values)
            ),
            benchmarkStats: RuntimeBenchmarkStatsSummary(
                stats: snapshots.compactMap(\.benchmarkStats).filter {
                    $0.completedRequests > 0 || $0.failedRequests > 0
                }
            )
        )
    }

    func loadModel(idOrAlias: String) async throws -> RuntimeModelPoolEntrySnapshot {
        let resolved = try resolveModel(idOrAlias)
        _ = try await ensureLoaded(resolved)
        touch(id: resolved.id, error: nil)
        return try snapshot(idOrAlias: resolved.id)
    }

    func unloadModel(idOrAlias: String) async throws -> RuntimeModelPoolEntrySnapshot {
        let resolved = try resolveModel(idOrAlias)
        let state = state(for: resolved.id)
        guard state.activeRequests == 0 else {
            throw RuntimeModelPoolError.unloadConflict(resolved.id, activeRequests: state.activeRequests)
        }
        if let loaded = loadedModels.removeValue(forKey: resolved.id) {
            await loaded.unload()
        }
        touch(id: resolved.id, error: nil)
        return try snapshot(idOrAlias: resolved.id)
    }

    func settings(idOrAlias: String) throws -> RuntimeModelSettings {
        let resolved = try resolveModel(idOrAlias)
        return resolved.settings
    }

    func updateSettings(idOrAlias: String, settings: RuntimeModelSettings) throws -> RuntimeModelSettings {
        let resolved = try resolveModel(idOrAlias, requireInstalled: false)
        guard let spec = resolved.spec else {
            throw RuntimeModelPoolError.invalidSettings("Runtime settings require a managed catalog model id.")
        }
        do {
            try settingsStore.writeSettings(settings, for: spec.id)
            return try settingsStore.settings(for: spec.id)
        } catch {
            throw RuntimeModelPoolError.invalidSettings(error.localizedDescription)
        }
    }

    func makeChatPlan(
        for openAIRequest: OpenAIChatRequest,
        fallbackLoraPath: String?,
        serverContextSize: Int
    ) async throws -> RuntimeChatPlan {
        let resolved = try resolveModel(openAIRequest.model)
        var effectiveRequest = openAIRequest
        applyDefaults(from: resolved.settings, to: &effectiveRequest)
        let contextSize = resolved.settings.maxContextTokens ?? serverContextSize
        let capabilities = resolved.engine.openAICompatibility
        var chatRequest = try APIServerContract.chatRequest(
            from: effectiveRequest,
            fallbackLoraPath: fallbackLoraPath,
            contextSize: contextSize,
            capabilities: capabilities
        )
        chatRequest.kvCacheMode = resolved.settings.kvCacheMode
        let includeUsage = try APIServerContract.includeUsageInStreaming(
            effectiveRequest,
            capabilities: capabilities
        )
        let loaded = try await ensureLoaded(resolved)
        retainLease(id: resolved.id)
        let lease = RuntimeModelLease(
            modelID: resolved.id,
            engine: resolved.engine,
            loaded: loaded,
            pool: self
        )
        return RuntimeChatPlan(
            lease: lease,
            request: chatRequest,
            modelID: resolved.id,
            engine: resolved.engine,
            includeUsage: includeUsage
        )
    }

    fileprivate func releaseLease(modelID: String) {
        var state = state(for: modelID)
        state.activeRequests = max(0, state.activeRequests - 1)
        state.lastAccess = Date()
        states[modelID] = state
    }

    fileprivate func recordChatCompletion(modelID: String, response: ChatResponse) {
        var state = state(for: modelID)
        state.completedRequests += 1
        state.generatedTokens += response.tokensGenerated
        if let timing = response.timing {
            state.totalLoadSeconds += timing.loadSeconds
            state.totalPrefillSeconds += timing.prefillSeconds
            state.totalDecodeSeconds += timing.decodeSeconds
        }
        state.lastCompletedAt = Date()
        state.lastError = nil
        states[modelID] = state
    }

    fileprivate func recordChatFailure(modelID: String, error: Error) {
        var state = state(for: modelID)
        state.failedRequests += 1
        state.lastError = error.localizedDescription
        states[modelID] = state
    }

    private func ensureLoaded(_ resolved: ResolvedModel) async throws -> RuntimeLoadedModel {
        if let loaded = loadedModels[resolved.id] {
            return loaded
        }

        let loaded = makeLoadedModel(for: resolved)
        loadedModels[resolved.id] = loaded
        do {
            try await loaded.prepare { progress in
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
            touch(id: resolved.id, error: nil)
            return loaded
        } catch {
            loadedModels.removeValue(forKey: resolved.id)
            touch(id: resolved.id, error: error.localizedDescription)
            throw error
        }
    }

    private func makeLoadedModel(for resolved: ResolvedModel) -> RuntimeLoadedModel {
        switch resolved.engine {
        case .textCode:
            return .textCode(
                CodeGenGenerator(modelId: resolved.id),
                modelPath: resolved.installPath
            )
        case .textChatKlein:
            let useStandalone = resolved.id == ModelResolver.ModelID.mebot.rawValue
                && resolved.installPath == MeBotModelCatalog.resolveModelPath()
            return .textChatKlein(
                Flux2KleinGenerator(),
                modelPath: resolved.installPath,
                useStandalone: useStandalone
            )
        case .textChatGemma4:
            return .textChatGemma4(
                Gemma4Generator(
                    modelId: resolved.id,
                    kvCacheQuantization: gemma4KVCacheQuantization,
                    prefixKVCacheEnabled: gemma4PrefixKVCacheEnabled,
                    continuousBatchingEnabled: gemma4ContinuousBatchingEnabled
                ),
                modelPath: resolved.installPath
            )
        case .textChatQ36, .textChatQ35:
            return .textChatQ35(
                Q35Generator(
                    modelId: resolved.id,
                    prefixKVCacheEnabled: q35PrefixKVCacheEnabled,
                    continuousBatchingEnabled: q35ContinuousBatchingEnabled
                ),
                modelPath: resolved.installPath
            )
        case .textChatLFM2:
            return .textChatLFM2(
                LFM2Generator(modelId: resolved.id),
                modelPath: resolved.installPath
            )
        case .textChatDeepseekV4Flash:
            return .textChatDeepseekV4Flash(
                DeepseekV4FlashGenerator(modelId: resolved.id),
                modelPath: resolved.installPath
            )
        }
    }

    private func resolveModel(
        _ requested: String?,
        requireInstalled: Bool = true
    ) throws -> ResolvedModel {
        let requestedName = requested?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty ?? defaultModelID
        let resolvedID = try resolveAliasOrID(requestedName)
        if let spec = ManagedModelCatalog.spec(for: resolvedID) {
            guard spec.isAPIServableRuntimeModel,
                  let defaultEngine = spec.defaultRuntimeServingEngine else {
                throw RuntimeModelPoolError.unsupportedModel(spec.id)
            }
            let settings = try settingsForSpec(spec)
            let engine = settings.engineOverride ?? defaultEngine
            guard engine.isCompatible(with: defaultEngine) else {
                throw RuntimeModelPoolError.incompatibleEngine(
                    "Engine '\(engine.rawValue)' is not compatible with model '\(spec.id)'."
                )
            }
            let installPath = try installPath(for: spec, requireInstalled: requireInstalled)
            return ResolvedModel(
                id: spec.id,
                category: spec.category.rawValue,
                engine: engine,
                installPath: installPath,
                settings: settings,
                spec: spec
            )
        }

        guard resolvedID == defaultModelID else {
            throw RuntimeModelPoolError.unknownModel(resolvedID)
        }
        guard let startupModelPath else {
            throw RuntimeModelPoolError.unknownModel(resolvedID)
        }
        return ResolvedModel(
            id: resolvedID,
            category: category(for: defaultEngine),
            engine: defaultEngine,
            installPath: startupModelPath,
            settings: RuntimeModelSettings(),
            spec: nil
        )
    }

    private func resolveAliasOrID(_ requestedName: String) throws -> String {
        let document = try settingsStore.load()
        if let match = document.models.first(where: { _, settings in
            settings.alias?.caseInsensitiveCompare(requestedName) == .orderedSame
        }) {
            return match.key
        }
        if let spec = ManagedModelCatalog.spec(for: requestedName) {
            return spec.id
        }
        if requestedName == defaultModelID {
            return defaultModelID
        }
        throw RuntimeModelPoolError.unknownModel(requestedName)
    }

    private func settingsForSpec(_ spec: ManagedModelSpec) throws -> RuntimeModelSettings {
        do {
            return try settingsStore.settings(for: spec.id).validated(for: spec)
        } catch {
            throw RuntimeModelPoolError.invalidSettings(error.localizedDescription)
        }
    }

    private func installPath(for spec: ManagedModelSpec, requireInstalled: Bool) throws -> String? {
        if spec.id == defaultModelID, let startupModelPath {
            return startupModelPath
        }
        if let runtimeURL = spec.managedRuntimeURL() {
            return runtimeURL.path
        }
        if spec.id == defaultModelID {
            return nil
        }
        if requireInstalled {
            throw RuntimeModelPoolError.modelNotInstalled(spec.id)
        }
        return nil
    }

    private func listedOpenAIModelIDs() throws -> [String] {
        let settings = try settingsStore.load().models
        var ids = Set(installedServableCatalogIDs())
        ids.insert(defaultModelID)
        for (modelID, modelSettings) in settings {
            guard modelSettings.alias != nil,
                  ids.contains(modelID) || modelID == defaultModelID else {
                continue
            }
            if let alias = modelSettings.alias {
                ids.insert(alias)
            }
        }
        return ids.sorted()
    }

    private func installedServableCatalogIDs() -> [String] {
        ManagedModelCatalog.allSpecs.compactMap { spec in
            guard spec.isAPIServableRuntimeModel else { return nil }
            if spec.id == defaultModelID {
                return spec.id
            }
            return spec.managedRuntimeURL() == nil ? nil : spec.id
        }
    }

    private func snapshot(idOrAlias: String) throws -> RuntimeModelPoolEntrySnapshot {
        let resolved = try resolveModel(idOrAlias, requireInstalled: false)
        let settings = try settingsStore.load().models
        guard let snapshot = snapshot(for: resolved.id, settings: settings) else {
            throw RuntimeModelPoolError.unknownModel(idOrAlias)
        }
        return snapshot
    }

    private func snapshot(
        for id: String,
        settings: [String: RuntimeModelSettings],
        prefixKVCache: PrefixKVCacheStats? = nil,
        continuousBatching: RuntimeDecodeBatchingStats? = nil
    ) -> RuntimeModelPoolEntrySnapshot? {
        let spec = ManagedModelCatalog.spec(for: id)
        let engine = settings[id]?.engineOverride
            ?? spec?.defaultRuntimeServingEngine
            ?? (id == defaultModelID ? defaultEngine : nil)
        guard let engine else { return nil }
        guard spec?.isAPIServableRuntimeModel != false else { return nil }

        let state = state(for: id)
        let modelSettings = settings[id] ?? RuntimeModelSettings()
        let installPath: String?
        if id == defaultModelID, let startupModelPath {
            installPath = startupModelPath
        } else {
            installPath = spec?.managedRuntimeURL()?.path
        }
        return RuntimeModelPoolEntrySnapshot(
            id: id,
            category: spec?.category.rawValue ?? category(for: engine),
            engine: engine,
            installPath: installPath,
            loaded: loadedModels[id] != nil,
            activeRequests: state.activeRequests,
            lastAccess: state.lastAccess,
            lastError: state.lastError,
            pinned: modelSettings.pinned,
            alias: modelSettings.alias,
            ttlSeconds: modelSettings.ttlSeconds,
            maxContextTokens: modelSettings.maxContextTokens,
            maxTokens: modelSettings.maxTokens,
            temperature: modelSettings.temperature,
            topP: modelSettings.topP,
            engineOverride: modelSettings.engineOverride,
            kvCacheMode: modelSettings.kvCacheMode,
            prefixKVCache: prefixKVCache,
            continuousBatching: continuousBatching,
            benchmarkStats: state.benchmarkStats
        )
    }

    private func applyDefaults(
        from settings: RuntimeModelSettings,
        to request: inout OpenAIChatRequest
    ) {
        if request.max_tokens == nil, request.max_completion_tokens == nil {
            request.max_tokens = settings.maxTokens
        }
        if request.temperature == nil {
            request.temperature = settings.temperature
        }
        if request.top_p == nil {
            request.top_p = settings.topP
        }
    }

    private func state(for id: String) -> MutableState {
        states[id] ?? MutableState()
    }

    private func touch(id: String, error: String?) {
        var state = state(for: id)
        state.lastAccess = Date()
        state.lastError = error
        states[id] = state
    }

    private func retainLease(id: String) {
        var state = state(for: id)
        state.activeRequests += 1
        state.lastAccess = Date()
        states[id] = state
    }

    private func category(for engine: RuntimeServingEngine) -> String {
        switch engine {
        case .textCode:
            return ManagedModelCategory.textCode.rawValue
        case .textChatKlein,
             .textChatGemma4,
             .textChatQ36,
             .textChatQ35,
             .textChatLFM2,
             .textChatDeepseekV4Flash:
            return ManagedModelCategory.textChat.rawValue
        }
    }
}

actor RuntimeRequestAdmission {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<RuntimeRequestAdmissionLease, Error>
    }

    private enum WaiterState {
        case pending
        case waiting
        case cancelled
    }

    private let maxActiveRequests: Int
    private var activeRequests = 0
    private var waiters: [Waiter] = []
    private var waiterStates: [UUID: WaiterState] = [:]
    private var totalAdmittedRequests = 0
    private var totalCompletedRequests = 0
    private var totalCancelledRequests = 0

    init(maxActiveRequests: Int) {
        precondition(maxActiveRequests > 0, "maxActiveRequests must be positive")
        self.maxActiveRequests = maxActiveRequests
    }

    func acquire() async throws -> RuntimeRequestAdmissionLease {
        if activeRequests < maxActiveRequests {
            return admit()
        }

        let waiterID = UUID()
        waiterStates[waiterID] = .pending
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                enqueueWaiter(id: waiterID, continuation: continuation)
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelWaiter(id: waiterID)
            }
        }
    }

    fileprivate func releaseLease() {
        activeRequests = max(0, activeRequests - 1)
        totalCompletedRequests += 1
        drain()
    }

    func snapshot() -> RuntimeRequestAdmissionSnapshot {
        RuntimeRequestAdmissionSnapshot(
            maxActiveRequests: maxActiveRequests,
            activeRequests: activeRequests,
            queuedRequests: waiters.count,
            totalAdmittedRequests: totalAdmittedRequests,
            totalCompletedRequests: totalCompletedRequests,
            totalCancelledRequests: totalCancelledRequests
        )
    }

    private func enqueueWaiter(
        id: UUID,
        continuation: CheckedContinuation<RuntimeRequestAdmissionLease, Error>
    ) {
        if waiterStates[id] == .cancelled {
            waiterStates.removeValue(forKey: id)
            continuation.resume(throwing: CancellationError())
            return
        }

        waiterStates[id] = .waiting
        waiters.append(Waiter(id: id, continuation: continuation))
    }

    private func cancelWaiter(id: UUID) {
        switch waiterStates[id] {
        case .pending:
            waiterStates[id] = .cancelled
            totalCancelledRequests += 1
        case .waiting:
            guard let index = waiters.firstIndex(where: { $0.id == id }) else {
                waiterStates.removeValue(forKey: id)
                return
            }
            let waiter = waiters.remove(at: index)
            waiterStates.removeValue(forKey: id)
            totalCancelledRequests += 1
            waiter.continuation.resume(throwing: CancellationError())
        case .cancelled, nil:
            return
        }
    }

    private func drain() {
        while activeRequests < maxActiveRequests, !waiters.isEmpty {
            let waiter = waiters.removeFirst()
            guard waiterStates[waiter.id] != .cancelled else {
                waiterStates.removeValue(forKey: waiter.id)
                waiter.continuation.resume(throwing: CancellationError())
                continue
            }
            waiterStates.removeValue(forKey: waiter.id)
            waiter.continuation.resume(returning: admit())
        }
    }

    private func admit() -> RuntimeRequestAdmissionLease {
        activeRequests += 1
        totalAdmittedRequests += 1
        return RuntimeRequestAdmissionLease(admission: self)
    }
}

final class RuntimeRequestAdmissionLease: @unchecked Sendable {
    private let admission: RuntimeRequestAdmission
    private let lock = NSLock()
    private var released = false

    init(admission: RuntimeRequestAdmission) {
        self.admission = admission
    }

    deinit {
        guard markReleased() else { return }
        let admission = admission
        Task {
            await admission.releaseLease()
        }
    }

    func release() async {
        guard markReleased() else { return }
        await admission.releaseLease()
    }

    private func markReleased() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !released else { return false }
        released = true
        return true
    }
}

final class RuntimeModelLease: @unchecked Sendable {
    let modelID: String
    let engine: RuntimeServingEngine

    private let loaded: RuntimeLoadedModel
    private let pool: RuntimeModelPool
    private let lock = NSLock()
    private var released = false

    init(
        modelID: String,
        engine: RuntimeServingEngine,
        loaded: RuntimeLoadedModel,
        pool: RuntimeModelPool
    ) {
        self.modelID = modelID
        self.engine = engine
        self.loaded = loaded
        self.pool = pool
    }

    deinit {
        guard markReleased() else { return }
        let pool = pool
        let modelID = modelID
        Task {
            await pool.releaseLease(modelID: modelID)
        }
    }

    func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        do {
            let response = try await loaded.chat(request, progressHandler: progressHandler)
            await pool.recordChatCompletion(modelID: modelID, response: response)
            return response
        } catch {
            await pool.recordChatFailure(modelID: modelID, error: error)
            throw error
        }
    }

    func deepseekChatCompletionsURL(
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> URL {
        try await loaded.deepseekChatCompletionsURL(progressHandler: progressHandler)
    }

    func release() async {
        guard markReleased() else { return }
        await pool.releaseLease(modelID: modelID)
    }

    private func markReleased() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !released else { return false }
        released = true
        return true
    }
}

enum RuntimeLoadedModel: Sendable {
    case textCode(CodeGenGenerator, modelPath: String?)
    case textChatKlein(Flux2KleinGenerator, modelPath: String?, useStandalone: Bool)
    case textChatGemma4(Gemma4Generator, modelPath: String?)
    case textChatQ35(Q35Generator, modelPath: String?)
    case textChatLFM2(LFM2Generator, modelPath: String?)
    case textChatDeepseekV4Flash(DeepseekV4FlashGenerator, modelPath: String?)

    func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        switch self {
        case .textCode(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatKlein(let generator, let modelPath, let useStandalone):
            guard let modelPath else {
                throw Flux2Error.modelNotFound(ModelResolver.ModelID.mebot.rawValue)
            }
            try await generator.prepareChat(
                modelPath: modelPath,
                standalone: useStandalone,
                progressHandler: progressHandler
            )
        case .textChatGemma4(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatQ35(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLFM2(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        }
    }

    func unload() async {
        switch self {
        case .textCode(let generator, _):
            await generator.unload()
        case .textChatKlein(let generator, _, _):
            await generator.unload()
        case .textChatGemma4(let generator, _):
            await generator.unload()
        case .textChatQ35(let generator, _):
            await generator.unload()
        case .textChatLFM2(let generator, _):
            await generator.unload()
        case .textChatDeepseekV4Flash(let generator, _):
            await generator.shutdown()
        }
    }

    func prefixKVCacheStats() async -> PrefixKVCacheStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textChatQ35(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textCode, .textChatKlein, .textChatLFM2, .textChatDeepseekV4Flash:
            return nil
        }
    }

    func continuousBatchingStats() async -> RuntimeDecodeBatchingStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatQ35(let generator, _):
            return await generator.continuousBatchingStats()
        case .textCode, .textChatKlein, .textChatLFM2, .textChatDeepseekV4Flash:
            return nil
        }
    }

    func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        switch self {
        case .textCode(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatKlein(let generator, let modelPath, let useStandalone):
            guard let modelPath else {
                throw Flux2Error.modelNotFound(ModelResolver.ModelID.mebot.rawValue)
            }
            if useStandalone {
                return try await generator.chatStandalone(
                    request,
                    modelPath: modelPath,
                    progressHandler: progressHandler
                )
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatGemma4(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatQ35(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLFM2(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        }
    }

    func deepseekChatCompletionsURL(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        switch self {
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            return try await generator.chatCompletionsURL(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
        case .textCode, .textChatKlein, .textChatGemma4, .textChatQ35, .textChatLFM2:
            throw RuntimeModelPoolError.rawProxyUnavailable("")
        }
    }
}

extension RuntimeServingEngine {
    var openAICompatibility: APIEngineCapabilities {
        switch self {
        case .textCode:
            return .localText
        case .textChatKlein:
            return .localTextWithStructuredJSON
        case .textChatGemma4:
            return .localTextWithTools
        case .textChatQ36, .textChatQ35:
            return .localTextWithToolsAndVision
        case .textChatLFM2:
            return .localTextWithTools
        case .textChatDeepseekV4Flash:
            return .rawProxy
        }
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
