import Foundation
import MLX
import MereRunCore
#if canImport(Darwin)
import Darwin
#endif

/// Tri-state environment flag: nil when unset, so callers can supply a
/// context-dependent default (e.g. continuous batching following
/// --max-active-requests).
func runtimeOptionalEnvironmentFlag(
    _ key: String,
    environment: [String: String] = ProcessInfo.processInfo.environment
) -> Bool? {
    guard let rawValue = environment[key] else {
        return nil
    }
    let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return !["0", "false", "no", "off"].contains(normalized)
}

struct RuntimeContinuousBatchingConfiguration: Equatable, Sendable {
    let gemma4: Bool
    let laguna: Bool
    let q35: Bool
    let lfm2: Bool

    init(
        maxActiveRequests: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        let defaultEnabled = maxActiveRequests > 1
        gemma4 = runtimeOptionalEnvironmentFlag(
            "MERERUN_GEMMA4_CONTINUOUS_BATCHING",
            environment: environment
        ) ?? defaultEnabled
        laguna = runtimeOptionalEnvironmentFlag(
            "MERERUN_LAGUNA_CONTINUOUS_BATCHING",
            environment: environment
        ) ?? defaultEnabled
        q35 = runtimeOptionalEnvironmentFlag(
            "MERERUN_Q35_CONTINUOUS_BATCHING",
            environment: environment
        ) ?? defaultEnabled
        lfm2 = runtimeOptionalEnvironmentFlag(
            "MERERUN_LFM2_CONTINUOUS_BATCHING",
            environment: environment
        ) ?? defaultEnabled
    }
}

private func runtimeDefaultOnEnvironmentFlag(_ key: String) -> Bool {
    guard let rawValue = ProcessInfo.processInfo.environment[key] else {
        return true
    }
    let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return !["0", "false", "no", "off"].contains(normalized)
}

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
    var sidecars: RuntimeSidecarPoolStatus? = nil
    /// Additive process/device telemetry. Older servers omit it and older clients ignore it.
    var process: RuntimeProcessTelemetry? = nil
}

enum RuntimeSidecarKind: String, Codable, Equatable, Sendable {
    case image
    case speech
    case transcription
    case embedding
}

enum RuntimeSidecarEvictionReason: String, Codable, Equatable, Sendable {
    case ttl
    case memoryPressure = "memory_pressure"
}

struct RuntimeSidecarResidentSnapshot: Codable, Equatable, Sendable {
    let kind: RuntimeSidecarKind
    let modelID: String?
    let modelPath: String?
    let variant: String?
    let loaded: Bool
    /// A resident generator can exist while its first model load is still in
    /// progress or after that load failed. Older payloads omit this field.
    var ready: Bool? = nil
    let activeRequests: Int
    let queuedRequests: Int
    let loadedAt: Date?
    let lastAccess: Date?
    let lastEvictedAt: Date?
    let lastEvictionReason: RuntimeSidecarEvictionReason?
    let pinned: Bool
    let ttlSeconds: Int
    let loadCount: Int
    let replacementCount: Int
    let evictionCount: Int
    let completedRequests: Int
    let failedRequests: Int
}

struct RuntimeSidecarPoolStatus: Codable, Equatable, Sendable {
    let defaultIdleTTLSeconds: Int
    let pressure: String
    let loadedCount: Int
    let activeRequests: Int
    let queuedRequests: Int
    let residents: [RuntimeSidecarResidentSnapshot]
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
        lagunaContinuousBatchingEnabled: Bool = false,
        q35ContinuousBatchingEnabled: Bool,
        q35PrefixKVCacheEnabled: Bool,
        lfm2ContinuousBatchingEnabled: Bool = false,
        lfm2PrefixKVCacheEnabled: Bool = false
    ) -> RuntimeControlPlaneCapabilities {
        let prefixKVCacheEnabled = gemma4PrefixKVCacheEnabled
            || q35PrefixKVCacheEnabled
            || lfm2PrefixKVCacheEnabled
        let continuousBatchingEnabled = gemma4ContinuousBatchingEnabled
            || lagunaContinuousBatchingEnabled
            || q35ContinuousBatchingEnabled
            || lfm2ContinuousBatchingEnabled
        return RuntimeControlPlaneCapabilities(
            requestAdmission: RuntimeCapabilityStatus(
                available: true,
                enabled: true,
                detail: "Fair FIFO request admission is active."
            ),
            chunkedPrefill: RuntimeCapabilityStatus(
                available: true,
                enabled: true,
                detail: "Gemma4, Laguna, Qwen-family, and LFM2 models prefill long prompts in cancellable chunks."
            ),
            continuousBatching: RuntimeCapabilityStatus(
                available: true,
                enabled: continuousBatchingEnabled,
                detail: continuousBatchingEnabled
                    ? "Gemma4, Laguna, Qwen-family, and LFM2 decode rows are packed when typed cache state is compatible; Laguna, Qwen-family, and LFM2 rows may batch across decode positions."
                    : "Decode batching engages automatically when --max-active-requests is above 1; the per-engine MERERUN_*_CONTINUOUS_BATCHING flags override it."
            ),
            prefixKVReuse: RuntimeCapabilityStatus(
                available: true,
                enabled: prefixKVCacheEnabled,
                detail: prefixKVCacheEnabled
                    ? "In-memory prefix KV reuse is enabled for matching Gemma4, Qwen-family, or LFM2 text token prefixes."
                    : "In-memory prefix KV reuse is disabled by " +
                        "the per-engine MERERUN_*_PREFIX_KV_CACHE flags."
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
    let admissionPaused: Bool?
    let pressure: String?
    var activeRequestDetails: [RuntimeActiveRequestSnapshot]? = nil
    var lastClientDisconnectAt: Date? = nil
    var lastCancellationAt: Date? = nil
    var lastSlotReleaseAt: Date? = nil

    init(
        maxActiveRequests: Int,
        activeRequests: Int,
        queuedRequests: Int,
        totalAdmittedRequests: Int,
        totalCompletedRequests: Int,
        totalCancelledRequests: Int,
        admissionPaused: Bool? = nil,
        pressure: String? = nil,
        activeRequestDetails: [RuntimeActiveRequestSnapshot]? = nil,
        lastClientDisconnectAt: Date? = nil,
        lastCancellationAt: Date? = nil,
        lastSlotReleaseAt: Date? = nil
    ) {
        self.maxActiveRequests = maxActiveRequests
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
        self.totalAdmittedRequests = totalAdmittedRequests
        self.totalCompletedRequests = totalCompletedRequests
        self.totalCancelledRequests = totalCancelledRequests
        self.admissionPaused = admissionPaused
        self.pressure = pressure
        self.activeRequestDetails = activeRequestDetails
        self.lastClientDisconnectAt = lastClientDisconnectAt
        self.lastCancellationAt = lastCancellationAt
        self.lastSlotReleaseAt = lastSlotReleaseAt
    }
}

struct RuntimeActiveRequestSnapshot: Codable, Equatable, Sendable {
    let id: UUID
    let admittedAt: Date
    let modelID: String?
    let streaming: Bool?
    let requestedMaxTokens: Int?
    let toolCount: Int?
    let phase: String
    let phaseDetail: String?
    let firstTokenAt: Date?
    let generatedTokenUpdates: Int
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
    /// Decode throughput over the most recent completions (rolling window of
    /// 10) — lifetime averages hide mid-flight regressions on long-running
    /// servers.
    let recentDecodeTokensPerSecond: Double?
    let lastCompletedAt: Date?
    var lastRequest: RuntimeRequestTimingSnapshot? = nil

    init(
        completedRequests: Int,
        failedRequests: Int,
        generatedTokens: Int,
        totalLoadSeconds: Double,
        totalPrefillSeconds: Double,
        totalDecodeSeconds: Double,
        recentDecodeTokens: Int = 0,
        recentDecodeSeconds: Double = 0,
        lastCompletedAt: Date?,
        lastRequest: RuntimeRequestTimingSnapshot? = nil
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
        self.recentDecodeTokensPerSecond = recentDecodeSeconds > 0
            ? Double(recentDecodeTokens) / recentDecodeSeconds
            : nil
        self.lastCompletedAt = lastCompletedAt
        self.lastRequest = lastRequest
    }

    private static func average(_ value: Double, count: Int) -> Double? {
        guard count > 0 else { return nil }
        return value / Double(count)
    }
}

struct RuntimeRequestTimingSnapshot: Codable, Equatable, Sendable {
    let promptTokens: Int
    let generatedTokens: Int
    let loadSeconds: Double
    let prefillSeconds: Double
    let cacheConversionSeconds: Double?
    let decodeSeconds: Double
    let timeToFirstTokenSeconds: Double?
    let prefillTokensPerSecond: Double?
    let decodeTokensPerSecond: Double?
    let prefillKVCache: String?
    let decodeKVCache: String?

    init(response: ChatResponse) {
        let timing = response.timing
        promptTokens = response.promptTokens ?? 0
        generatedTokens = response.tokensGenerated
        loadSeconds = timing?.loadSeconds ?? 0
        prefillSeconds = timing?.prefillSeconds ?? 0
        cacheConversionSeconds = timing?.cacheConversionSeconds
        decodeSeconds = timing?.decodeSeconds ?? 0
        if let timing, let firstTokenSeconds = timing.firstTokenSeconds {
            timeToFirstTokenSeconds = timing.loadSeconds
                + timing.prefillSeconds
                + (timing.cacheConversionSeconds ?? 0)
                + firstTokenSeconds
        } else {
            timeToFirstTokenSeconds = nil
        }
        prefillTokensPerSecond = timing?.prefillTokensPerSecond
        decodeTokensPerSecond = timing?.decodeTokensPerSecond
        prefillKVCache = timing?.prefillKVCache
        decodeKVCache = timing?.decodeKVCache
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
    let residentBytes: UInt64?
    let currentBytes: UInt64?
    let availableBytes: UInt64?
    let ceilingBytes: UInt64?
    let softLimitBytes: UInt64?
    let hardLimitBytes: UInt64?
    let activeRequests: Int
    let activeModelCount: Int
    let guardTier: RuntimeMemoryGuardTier
    let pressure: String

    init(
        physicalBytes: UInt64,
        residentBytes: UInt64? = nil,
        currentBytes: UInt64? = nil,
        availableBytes: UInt64? = nil,
        ceilingBytes: UInt64? = nil,
        softLimitBytes: UInt64? = nil,
        hardLimitBytes: UInt64? = nil,
        activeRequests: Int,
        activeModelCount: Int,
        guardTier: RuntimeMemoryGuardTier = .balanced,
        pressure: String
    ) {
        self.physicalBytes = physicalBytes
        self.residentBytes = residentBytes
        self.currentBytes = currentBytes
        self.availableBytes = availableBytes
        self.ceilingBytes = ceilingBytes
        self.softLimitBytes = softLimitBytes
        self.hardLimitBytes = hardLimitBytes
        self.activeRequests = activeRequests
        self.activeModelCount = activeModelCount
        self.guardTier = guardTier
        self.pressure = pressure
    }

    enum CodingKeys: String, CodingKey {
        case physicalBytes
        case residentBytes
        case currentBytes
        case availableBytes
        case ceilingBytes
        case softLimitBytes
        case hardLimitBytes
        case activeRequests
        case activeModelCount
        case guardTier
        case pressure
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        physicalBytes = try container.decode(UInt64.self, forKey: .physicalBytes)
        residentBytes = try container.decodeIfPresent(UInt64.self, forKey: .residentBytes)
        currentBytes = try container.decodeIfPresent(UInt64.self, forKey: .currentBytes)
        availableBytes = try container.decodeIfPresent(UInt64.self, forKey: .availableBytes)
        ceilingBytes = try container.decodeIfPresent(UInt64.self, forKey: .ceilingBytes)
        softLimitBytes = try container.decodeIfPresent(UInt64.self, forKey: .softLimitBytes)
        hardLimitBytes = try container.decodeIfPresent(UInt64.self, forKey: .hardLimitBytes)
        activeRequests = try container.decode(Int.self, forKey: .activeRequests)
        activeModelCount = try container.decode(Int.self, forKey: .activeModelCount)
        guardTier = try container.decodeIfPresent(RuntimeMemoryGuardTier.self, forKey: .guardTier) ?? .balanced
        pressure = try container.decode(String.self, forKey: .pressure)
    }
}

enum RuntimeMemoryPressureLevel: String, Codable, Equatable, Sendable {
    case disabled
    case unknown
    case nominal
    case elevated
    case critical
}

enum RuntimeMemoryGuardTier: String, Codable, CaseIterable, Equatable, Sendable {
    case off
    case safe
    case balanced
    case aggressive
    case custom

    static let `default` = RuntimeMemoryGuardTier.balanced
}

struct RuntimeMemorySample: Equatable, Sendable {
    let physicalBytes: UInt64
    let residentBytes: UInt64?
    let physicalFootprintBytes: UInt64?
    let availableBytes: UInt64?
    let freeBytes: UInt64?
    let activeBytes: UInt64?
    let inactiveBytes: UInt64?

    init(
        physicalBytes: UInt64,
        residentBytes: UInt64?,
        physicalFootprintBytes: UInt64? = nil,
        availableBytes: UInt64? = nil,
        freeBytes: UInt64? = nil,
        activeBytes: UInt64? = nil,
        inactiveBytes: UInt64? = nil
    ) {
        self.physicalBytes = physicalBytes
        self.residentBytes = residentBytes
        self.physicalFootprintBytes = physicalFootprintBytes
        self.availableBytes = availableBytes
        self.freeBytes = freeBytes
        self.activeBytes = activeBytes
        self.inactiveBytes = inactiveBytes
    }

    static func current() -> RuntimeMemorySample {
        RuntimeProcessMemory.currentSample()
    }
}

struct RuntimeMemoryPressurePolicy: Equatable, Sendable {
    let tier: RuntimeMemoryGuardTier
    let customCeilingBytes: UInt64?
    let softLimitFraction: Double
    let hardLimitFraction: Double

    static let `default` = RuntimeMemoryPressurePolicy(
        tier: .default
    )

    init(
        tier: RuntimeMemoryGuardTier = .default,
        customCeilingBytes: UInt64? = nil,
        softLimitFraction: Double = 0.90,
        hardLimitFraction: Double = 0.95
    ) {
        self.tier = tier
        self.customCeilingBytes = customCeilingBytes
        self.softLimitFraction = softLimitFraction
        self.hardLimitFraction = hardLimitFraction
    }

    func pressure(for sample: RuntimeMemorySample) -> RuntimeMemoryPressureLevel {
        guard tier != .off else {
            return .disabled
        }
        guard let currentBytes = currentBytes(for: sample),
              let limits = limits(for: sample) else {
            return .unknown
        }
        if currentBytes >= limits.hard {
            return .critical
        }
        if currentBytes >= limits.soft {
            return .elevated
        }
        return .nominal
    }

    func projectedPressure(
        for sample: RuntimeMemorySample,
        additionalBytes: UInt64
    ) -> RuntimeMemoryPressureLevel {
        guard tier != .off else {
            return .disabled
        }
        guard let currentBytes = currentBytes(for: sample),
              let limits = limits(for: sample) else {
            return .unknown
        }
        let (projected, overflow) = currentBytes.addingReportingOverflow(additionalBytes)
        guard !overflow else {
            return .critical
        }
        if projected >= limits.hard {
            return .critical
        }
        if projected >= limits.soft {
            return .elevated
        }
        return .nominal
    }

    func currentBytes(for sample: RuntimeMemorySample) -> UInt64? {
        sample.physicalFootprintBytes ?? sample.residentBytes
    }

    func limits(for sample: RuntimeMemorySample) -> (ceiling: UInt64, soft: UInt64, hard: UInt64)? {
        guard tier != .off, sample.physicalBytes > 0 else {
            return nil
        }
        let ceiling: UInt64
        switch tier {
        case .off:
            return nil
        case .custom:
            let custom = customCeilingBytes ?? 0
            guard custom > 0 else {
                return nil
            }
            ceiling = min(custom, staticCeiling(for: sample))
        case .safe, .balanced, .aggressive:
            ceiling = min(staticCeiling(for: sample), dynamicCeiling(for: sample))
        }
        guard ceiling > 0 else {
            return nil
        }
        let soft = UInt64((Double(ceiling) * softLimitFraction).rounded(.down))
        let hard = UInt64((Double(ceiling) * hardLimitFraction).rounded(.down))
        return (ceiling, soft, hard)
    }

    private func staticCeiling(for sample: RuntimeMemorySample) -> UInt64 {
        let reserve = staticReserveBytes(forPhysicalBytes: sample.physicalBytes)
        guard sample.physicalBytes > reserve else {
            return 0
        }
        return sample.physicalBytes - reserve
    }

    private func dynamicCeiling(for sample: RuntimeMemorySample) -> UInt64 {
        guard let currentBytes = currentBytes(for: sample) else {
            return staticCeiling(for: sample)
        }
        if let freeBytes = sample.freeBytes,
           let inactiveBytes = sample.inactiveBytes,
           let activeBytes = sample.activeBytes {
            return currentBytes
                + freeBytes
                + inactiveBytes
                + UInt64((Double(activeBytes) * activeReclaimRatio).rounded(.down))
        }
        if let availableBytes = sample.availableBytes {
            return currentBytes + availableBytes
        }
        return sample.physicalBytes
    }

    private var activeReclaimRatio: Double {
        switch tier {
        case .off, .custom:
            return 0
        case .safe:
            return 0.20
        case .balanced:
            return 0.50
        case .aggressive:
            return 0.80
        }
    }

    private func staticReserveBytes(forPhysicalBytes physicalBytes: UInt64) -> UInt64 {
        let gib = UInt64(1024 * 1024 * 1024)
        let smallSystemThreshold = 24 * gib
        if physicalBytes < smallSystemThreshold {
            return 4 * gib
        }
        switch tier {
        case .off:
            return physicalBytes
        case .safe:
            return 8 * gib
        case .balanced:
            return 6 * gib
        case .aggressive:
            return 4 * gib
        case .custom:
            return 2 * gib
        }
    }
}

private enum RuntimeProcessMemory {
    static func currentSample() -> RuntimeMemorySample {
        let physicalBytes = ProcessInfo.processInfo.physicalMemory
        let residentBytes = currentResidentBytes()
        let physicalFootprintBytes = currentPhysicalFootprintBytes()
        let host = currentHostMemory()
        return RuntimeMemorySample(
            physicalBytes: physicalBytes,
            residentBytes: residentBytes,
            physicalFootprintBytes: physicalFootprintBytes,
            availableBytes: host.availableBytes,
            freeBytes: host.freeBytes,
            activeBytes: host.activeBytes,
            inactiveBytes: host.inactiveBytes
        )
    }

    private static func currentPhysicalFootprintBytes() -> UInt64? {
        #if canImport(Darwin)
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            // The Darwin module imports `rusage_info_t *` as a pointer to an
            // optional raw pointer. Rebinding the struct storage mirrors the
            // C API's required `(rusage_info_t *)&info` cast; passing `&raw`
            // would instead let the kernel overwrite the pointer variable.
            pointer.withMemoryRebound(
                to: rusage_info_t?.self,
                capacity: 1
            ) { rebound in
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, rebound)
            }
        }
        guard result == 0 else {
            return nil
        }
        return info.ri_phys_footprint
        #else
        return nil
        #endif
    }

    private static func currentResidentBytes() -> UInt64? {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }
        return UInt64(info.resident_size)
        #elseif os(Linux)
        guard let status = try? String(contentsOfFile: "/proc/self/status") else {
            return nil
        }
        for line in status.split(separator: "\n") where line.hasPrefix("VmRSS:") {
            let parts = line.split(separator: " ").compactMap { UInt64($0) }
            guard let kilobytes = parts.first else {
                return nil
            }
            return kilobytes * 1024
        }
        return nil
        #else
        return nil
        #endif
    }

    private static func currentHostMemory() -> (
        availableBytes: UInt64?,
        freeBytes: UInt64?,
        activeBytes: UInt64?,
        inactiveBytes: UInt64?
    ) {
        #if canImport(Darwin)
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size) / 4
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return (nil, nil, nil, nil)
        }
        let pageSize = UInt64(getpagesize())
        let free = UInt64(stats.free_count) * pageSize
        let active = UInt64(stats.active_count) * pageSize
        let inactive = UInt64(stats.inactive_count) * pageSize
        return (free + inactive, free, active, inactive)
        #elseif os(Linux)
        guard let meminfo = try? String(contentsOfFile: "/proc/meminfo") else {
            return (nil, nil, nil, nil)
        }
        var values: [String: UInt64] = [:]
        for line in meminfo.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard let key = parts.first?.dropLast(),
                  let kilobytes = parts.dropFirst().compactMap({ UInt64($0) }).first else {
                continue
            }
            values[String(key)] = kilobytes * 1024
        }
        return (
            values["MemAvailable"],
            values["MemFree"],
            values["Active"],
            values["Inactive"]
        )
        #else
        return (nil, nil, nil, nil)
        #endif
    }
}

struct RuntimeModelPoolEntrySnapshot: Codable, Equatable, Sendable {
    let id: String
    let category: String
    let engine: RuntimeServingEngine
    let installPath: String?
    let loaded: Bool
    /// `loaded` remains the compatibility field for resident model objects.
    /// `ready` distinguishes a resident still preparing from one that can
    /// serve requests. Optional decoding preserves older status payloads.
    var ready: Bool? = nil
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
    let minP: Double?
    let engineOverride: RuntimeServingEngine?
    let kvCacheMode: RuntimeKVCacheMode?
    let prefixKVCache: PrefixKVCacheStats?
    let continuousBatching: RuntimeDecodeBatchingStats?
    let mtp: Gemma4MTPStats?
    let benchmarkStats: RuntimeModelBenchmarkStats?
    /// Startup load and optional graph-warmup timing. Older servers omit it.
    var startupTiming: RuntimeModelStartupTiming? = nil
}

struct RuntimeModelStartupTiming: Codable, Equatable, Sendable {
    let loadSeconds: Double
    let warmupSeconds: Double?
    let warmupPrefillSeconds: Double?
    let warmupDecodeSeconds: Double?
    let warmupTimeToFirstTokenSeconds: Double?
    let graphCompilationAccounting: String?
    let completedAt: Date
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
            return "Model '\(id)' is not installed. Run `\(CLICommandDisplay.modelPullCommand(for: id))` first."
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
    private struct ModelPreparation {
        let token: UUID
        let task: Task<RuntimeLoadedModel, Error>
        var waiterIDs: Set<UUID> = []
    }

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
        var recentDecodeSamples: [(tokens: Int, seconds: Double)] = []
        var lastCompletedAt: Date?
        var startupTiming: RuntimeModelStartupTiming?
        var lastRequestTiming: RuntimeRequestTimingSnapshot?

        var benchmarkStats: RuntimeModelBenchmarkStats {
            RuntimeModelBenchmarkStats(
                completedRequests: completedRequests,
                failedRequests: failedRequests,
                generatedTokens: generatedTokens,
                totalLoadSeconds: totalLoadSeconds,
                totalPrefillSeconds: totalPrefillSeconds,
                totalDecodeSeconds: totalDecodeSeconds,
                recentDecodeTokens: recentDecodeSamples.reduce(0) { $0 + $1.tokens },
                recentDecodeSeconds: recentDecodeSamples.reduce(0) { $0 + $1.seconds },
                lastCompletedAt: lastCompletedAt,
                lastRequest: lastRequestTiming
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

        var apiProfile: ManagedModelAPIProfile {
            spec?.apiProfile ?? .runtimeFallback(for: engine)
        }

        var openAICompatibility: APIEngineCapabilities {
            .catalog(apiProfile)
        }
    }

    private struct RuntimeLRUEvictionCandidate {
        let id: String
        let lastAccess: Date
    }

    private let defaultModelID: String
    private let defaultEngine: RuntimeServingEngine
    private let startupModelPath: String?
    private let settingsStore: RuntimeModelSettingsStore
    private let gemma4KVCacheQuantization: Gemma4KVCacheQuantization
    private let gemma4PrefixKVCacheEnabled: Bool
    private let gemma4ContinuousBatchingEnabled: Bool
    private let lagunaContinuousBatchingEnabled: Bool
    private let q35PrefixKVCacheEnabled: Bool
    private let q35ContinuousBatchingEnabled: Bool
    private let lfm2PrefixKVCacheEnabled: Bool
    private let lfm2ContinuousBatchingEnabled: Bool
    private let currentDate: @Sendable () -> Date
    private let currentMemorySample: @Sendable () -> RuntimeMemorySample
    private let memoryPressurePolicy: RuntimeMemoryPressurePolicy
    private let ensureMLXAvailable: @Sendable () throws -> Void
    private let prepareLoadedModel: @Sendable (RuntimeLoadedModel) async throws -> Void
    private let unloadLoadedModel: @Sendable (RuntimeLoadedModel) async -> Void
    private let clearMLXCache: @Sendable () -> Void

    private var loadedModels: [String: RuntimeLoadedModel] = [:]
    private var loadedModelTokens: [String: UUID] = [:]
    private var modelPreparations: [String: ModelPreparation] = [:]
    private var preparationCleanupTasks: [UUID: Task<Void, Never>] = [:]
    private var preparingModelIDs: Set<String> = []
    private var states: [String: MutableState] = [:]

    init(
        defaultModelID: String,
        defaultEngine: RuntimeServingEngine,
        startupModelPath: String?,
        settingsStore: RuntimeModelSettingsStore = RuntimeModelSettingsStore(),
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        gemma4PrefixKVCacheEnabled: Bool = runtimeDefaultOnEnvironmentFlag("MERERUN_GEMMA4_PREFIX_KV_CACHE"),
        gemma4ContinuousBatchingEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_CONTINUOUS_BATCHING"] == "1",
        lagunaContinuousBatchingEnabled: Bool =
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_CONTINUOUS_BATCHING"] == "1",
        q35PrefixKVCacheEnabled: Bool = runtimeDefaultOnEnvironmentFlag("MERERUN_Q35_PREFIX_KV_CACHE"),
        q35ContinuousBatchingEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_Q35_CONTINUOUS_BATCHING"] == "1",
        lfm2PrefixKVCacheEnabled: Bool = runtimeDefaultOnEnvironmentFlag("MERERUN_LFM2_PREFIX_KV_CACHE"),
        lfm2ContinuousBatchingEnabled: Bool =
            ProcessInfo.processInfo.environment["MERERUN_LFM2_CONTINUOUS_BATCHING"] == "1",
        currentDate: @escaping @Sendable () -> Date = { Date() },
        currentMemorySample: @escaping @Sendable () -> RuntimeMemorySample = { RuntimeMemorySample.current() },
        memoryPressurePolicy: RuntimeMemoryPressurePolicy = .default,
        ensureMLXAvailable: @escaping @Sendable () throws -> Void = {
            try MLXBundleSupport.ensureAvailable(quiet: true)
        },
        prepareLoadedModel: @escaping @Sendable (RuntimeLoadedModel) async throws -> Void = { loaded in
            try await loaded.prepare { progress in
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
        },
        unloadLoadedModel: @escaping @Sendable (RuntimeLoadedModel) async -> Void = { loaded in
            await loaded.unload()
        },
        clearMLXCache: @escaping @Sendable () -> Void = {
            Memory.clearCache()
        }
    ) {
        self.defaultModelID = defaultModelID
        self.defaultEngine = defaultEngine
        self.startupModelPath = startupModelPath
        self.settingsStore = settingsStore
        self.gemma4KVCacheQuantization = gemma4KVCacheQuantization
        self.gemma4PrefixKVCacheEnabled = gemma4PrefixKVCacheEnabled
        self.gemma4ContinuousBatchingEnabled = gemma4ContinuousBatchingEnabled
        self.lagunaContinuousBatchingEnabled = lagunaContinuousBatchingEnabled
        self.q35PrefixKVCacheEnabled = q35PrefixKVCacheEnabled
        self.q35ContinuousBatchingEnabled = q35ContinuousBatchingEnabled
        self.lfm2PrefixKVCacheEnabled = lfm2PrefixKVCacheEnabled
        self.lfm2ContinuousBatchingEnabled = lfm2ContinuousBatchingEnabled
        self.currentDate = currentDate
        self.currentMemorySample = currentMemorySample
        self.memoryPressurePolicy = memoryPressurePolicy
        self.ensureMLXAvailable = ensureMLXAvailable
        self.prepareLoadedModel = prepareLoadedModel
        self.unloadLoadedModel = unloadLoadedModel
        self.clearMLXCache = clearMLXCache
    }

    func preloadDefault(warmup: Bool = false) async throws {
        let resolved = try resolveModel(defaultModelID)
        let loadStart = currentDate()
        let loaded = try await ensureLoaded(resolved)
        let loadSeconds = currentDate().timeIntervalSince(loadStart)
        var startupTiming = RuntimeModelStartupTiming(
            loadSeconds: loadSeconds,
            warmupSeconds: nil,
            warmupPrefillSeconds: nil,
            warmupDecodeSeconds: nil,
            warmupTimeToFirstTokenSeconds: nil,
            graphCompilationAccounting: nil,
            completedAt: currentDate()
        )
        if warmup,
           resolved.engine == .textChatGemma4,
           Gemma4Resources.usesTurboDefaults(modelSpec: resolved.id) {
            let warmupStart = currentDate()
            let response = try await loaded.chat(
                ChatRequest(
                    messages: [ChatMessage(role: .user, content: "Reply with ready.")],
                    maxTokens: 1,
                    temperature: 0,
                    topP: 1,
                    showThinking: false
                ),
                progressHandler: nil
            )
            let timing = response.timing
            startupTiming = RuntimeModelStartupTiming(
                loadSeconds: loadSeconds,
                warmupSeconds: currentDate().timeIntervalSince(warmupStart),
                warmupPrefillSeconds: timing?.prefillSeconds,
                warmupDecodeSeconds: timing?.decodeSeconds,
                warmupTimeToFirstTokenSeconds: timing.map(Self.timeToFirstToken),
                graphCompilationAccounting: "included_in_warmup_prefill",
                completedAt: currentDate()
            )
        }
        var state = state(for: resolved.id)
        state.startupTiming = startupTiming
        states[resolved.id] = state
    }

    private static func timeToFirstToken(_ timing: ChatTiming) -> Double {
        timing.loadSeconds
            + timing.prefillSeconds
            + (timing.cacheConversionSeconds ?? 0)
            + (timing.firstTokenSeconds ?? 0)
    }

    func modelsResponse(
        serverContextSize: Int = 32_768,
        createdAt: Date = Date()
    ) throws -> OpenAIModelsResponse {
        let models = try listedOpenAIModelIDs().map { listedID in
            let resolved = try resolveModel(listedID, requireInstalled: false)
            let profile = resolved.apiProfile
            let configuredContextWindow = resolved.settings.maxContextTokens ?? serverContextSize
            let contextWindow = min(
                profile.contextWindow ?? configuredContextWindow,
                configuredContextWindow
            )
            let catalogOutputLimit = profile.maximumOutputTokens ?? contextWindow
            let configuredOutputLimit = resolved.settings.maxTokens ?? catalogOutputLimit
            let maximumOutputTokens = min(
                min(catalogOutputLimit, configuredOutputLimit),
                contextWindow
            )
            let name = resolved.spec?.upstreamRepoId ?? resolved.id
            return APIServerContract.chatModel(
                id: listedID,
                name: name,
                profile: profile,
                contextWindow: contextWindow,
                maximumOutputTokens: maximumOutputTokens,
                createdAt: createdAt
            )
        }
        return OpenAIModelsResponse(object: "list", data: models)
    }

    func status() async -> RuntimeModelPoolStatus {
        await status(admission: nil, sidecars: nil)
    }

    func status(
        admission: RuntimeRequestAdmissionSnapshot?,
        sidecars: RuntimeSidecarPoolStatus? = nil
    ) async -> RuntimeModelPoolStatus {
        await evictIdleModels()
        let memorySample = currentMemorySample()
        let memoryLimits = memoryPressurePolicy.limits(for: memorySample)
        let settings = (try? settingsStore.load())?.models ?? [:]
        let installed = installedServableCatalogIDs()
        var ids = Set<String>(installed)
        ids.formUnion(loadedModels.keys)
        ids.formUnion(states.keys)
        ids.insert(defaultModelID)
        ids.formUnion(settings.keys)
        var prefixStats: [String: PrefixKVCacheStats] = [:]
        var batchingStats: [String: RuntimeDecodeBatchingStats] = [:]
        var mtpStats: [String: Gemma4MTPStats] = [:]
        let currentLoadedModels = loadedModels
        for (id, loaded) in currentLoadedModels {
            // Generator actors can spend tens of seconds inside one evaluated
            // prefill chunk. Keep the control plane responsive while a model
            // is active; its cached counters return on the next idle status poll.
            guard state(for: id).activeRequests == 0 else { continue }
            if let stats = await loaded.prefixKVCacheStats() {
                prefixStats[id] = stats
            }
            if let stats = await loaded.continuousBatchingStats() {
                batchingStats[id] = stats
            }
            if let stats = await loaded.mtpStats() {
                mtpStats[id] = stats
            }
        }
        let snapshots: [RuntimeModelPoolEntrySnapshot] = ids.sorted().compactMap { id in
            snapshot(
                for: id,
                settings: settings,
                prefixKVCache: prefixStats[id],
                continuousBatching: batchingStats[id],
                mtp: mtpStats[id]
            )
        }
        let activeRequests = snapshots.reduce(0) { $0 + $1.activeRequests }
            + (sidecars?.activeRequests ?? 0)
        let loadedCount = snapshots.filter { $0.loaded }.count
            + (sidecars?.loadedCount ?? 0)
        return RuntimeModelPoolStatus(
            object: "runtime.status",
            defaultModel: defaultModelID,
            settingsPath: settingsStore.url.path,
            activeRequests: activeRequests,
            admission: admission,
            capabilities: .current(
                gemma4PrefixKVCacheEnabled: gemma4PrefixKVCacheEnabled,
                gemma4ContinuousBatchingEnabled: gemma4ContinuousBatchingEnabled,
                lagunaContinuousBatchingEnabled: lagunaContinuousBatchingEnabled,
                q35ContinuousBatchingEnabled: q35ContinuousBatchingEnabled,
                q35PrefixKVCacheEnabled: q35PrefixKVCacheEnabled,
                lfm2ContinuousBatchingEnabled: lfm2ContinuousBatchingEnabled,
                lfm2PrefixKVCacheEnabled: lfm2PrefixKVCacheEnabled
            ),
            memory: RuntimeMemorySnapshot(
                physicalBytes: memorySample.physicalBytes,
                residentBytes: memorySample.residentBytes,
                currentBytes: memoryPressurePolicy.currentBytes(for: memorySample),
                availableBytes: memorySample.availableBytes,
                ceilingBytes: memoryLimits?.ceiling,
                softLimitBytes: memoryLimits?.soft,
                hardLimitBytes: memoryLimits?.hard,
                activeRequests: activeRequests,
                activeModelCount: loadedCount,
                guardTier: memoryPressurePolicy.tier,
                pressure: memoryPressurePolicy.pressure(for: memorySample).rawValue
            ),
            models: snapshots,
            cacheStats: RuntimeCacheStatsSummary(
                prefixKVCaches: Array(prefixStats.values),
                decodeBatchers: Array(batchingStats.values)
            ),
            benchmarkStats: RuntimeBenchmarkStatsSummary(
                stats: snapshots.compactMap { $0.benchmarkStats }.filter {
                    $0.completedRequests > 0 || $0.failedRequests > 0
                }
            ),
            sidecars: sidecars
        )
    }

    func loadModel(idOrAlias: String) async throws -> RuntimeModelPoolEntrySnapshot {
        let resolved = try resolveModel(idOrAlias)
        await evictIdleModels(excluding: [resolved.id])
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
        if let preparation = modelPreparations[resolved.id] {
            await invalidatePreparation(
                preparation,
                modelID: resolved.id,
                error: nil
            )
        } else if let loaded = removeLoadedModel(for: resolved.id) {
            await unloadLoadedModel(loaded)
        }
        touch(id: resolved.id, error: nil)
        return try snapshot(idOrAlias: resolved.id)
    }

    func settings(idOrAlias: String) async throws -> RuntimeModelSettings {
        await evictIdleModels()
        let resolved = try resolveModel(idOrAlias)
        return resolved.settings
    }

    func updateSettings(idOrAlias: String, settings: RuntimeModelSettings) async throws -> RuntimeModelSettings {
        let resolved = try resolveModel(idOrAlias, requireInstalled: false)
        guard let spec = resolved.spec else {
            throw RuntimeModelPoolError.invalidSettings("Runtime settings require a managed catalog model id.")
        }
        do {
            try settingsStore.writeSettings(settings, for: spec.id)
            await evictIdleModels()
            return try settingsStore.settings(for: spec.id)
        } catch {
            throw RuntimeModelPoolError.invalidSettings(error.localizedDescription)
        }
    }

    func currentMemoryPressure() -> RuntimeMemoryPressureLevel {
        memoryPressurePolicy.pressure(for: currentMemorySample())
    }

    /// Releases idle text runtimes one at a time, sampling pressure again
    /// after every unload. Sidecar admission uses `preserveDefault: false`
    /// before considering its own residents: an idle startup model is cheaper
    /// to reload than keeping its full footprint beside a large media model.
    @discardableResult
    func relieveMemoryPressure(
        excluding excludedIDs: Set<String> = [],
        preserveDefault: Bool = true
    ) async -> [String] {
        var protectedIDs = excludedIDs
        if preserveDefault {
            protectedIDs.insert(defaultModelID)
        }
        var evicted: [String] = []
        var clearedReusableBuffers = false
        while true {
            let pressure = memoryPressurePolicy.pressure(for: currentMemorySample())
            switch pressure {
            case .disabled, .unknown, .nominal:
                return evicted.sorted()
            case .elevated, .critical:
                break
            }
            if !clearedReusableBuffers {
                clearMLXCache()
                clearedReusableBuffers = true
                await Task.yield()
                continue
            }
            guard let id = await evictOldestIdleUnpinnedModel(excluding: protectedIDs) else {
                return evicted.sorted()
            }
            evicted.append(id)
            await Task.yield()
        }
    }

    /// Proactively releases one least-recently-used idle, unpinned text
    /// resident for a cold sidecar whose projected load exceeds the memory
    /// guard. The caller re-samples after each eviction, preserving every warm
    /// model that is not needed for headroom.
    @discardableResult
    func releaseOneIdleModelForSidecarLoad(
        excluding excludedIDs: Set<String> = []
    ) async -> String? {
        await evictOldestIdleUnpinnedModel(excluding: excludedIDs)
    }

    func makeChatPlan(
        for openAIRequest: OpenAIChatRequest,
        fallbackLoraPath: String?,
        serverContextSize: Int
    ) async throws -> RuntimeChatPlan {
        let resolved = try resolveModel(openAIRequest.model)
        await evictIdleModels(excluding: [resolved.id])
        var effectiveRequest = openAIRequest
        applyDefaults(from: resolved.settings, to: &effectiveRequest)
        let contextSize = resolved.settings.maxContextTokens ?? serverContextSize
        let capabilities = resolved.openAICompatibility
        var chatRequest = try APIServerContract.chatRequest(
            from: effectiveRequest,
            fallbackLoraPath: fallbackLoraPath,
            contextSize: contextSize,
            capabilities: capabilities,
            servedModelID: resolved.id,
            apiProfile: resolved.apiProfile
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
        state.lastAccess = currentDate()
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
            state.recentDecodeSamples.append((tokens: response.tokensGenerated, seconds: timing.decodeSeconds))
            if state.recentDecodeSamples.count > 10 {
                state.recentDecodeSamples.removeFirst(state.recentDecodeSamples.count - 10)
            }
        }
        state.lastCompletedAt = currentDate()
        state.lastRequestTiming = RuntimeRequestTimingSnapshot(response: response)
        state.lastError = nil
        states[modelID] = state
    }

    fileprivate func recordChatFailure(modelID: String, error: Error) {
        var state = state(for: modelID)
        state.failedRequests += 1
        state.lastError = error.localizedDescription
        states[modelID] = state
    }

    @discardableResult
    func evictIdleModels(
        now: Date? = nil,
        memorySample: RuntimeMemorySample? = nil,
        excluding excludedIDs: Set<String> = []
    ) async -> [String] {
        let expired = await evictExpiredIdleModels(now: now, excluding: excludedIDs)
        let lru = await evictIdleModelsForMemoryPressure(sample: memorySample, excluding: excludedIDs)
        return Array(Set(expired + lru)).sorted()
    }

    @discardableResult
    func evictExpiredIdleModels(
        now: Date? = nil,
        excluding excludedIDs: Set<String> = []
    ) async -> [String] {
        let referenceDate = now ?? currentDate()
        let settings = (try? settingsStore.load())?.models ?? [:]
        let loadedEntries = loadedModels
        var evicted: [String] = []

        for (id, loaded) in loadedEntries {
            let state = state(for: id)
            guard loadedModels[id] != nil,
                  !preparingModelIDs.contains(id),
                  !excludedIDs.contains(id),
                  let modelSettings = settings[id],
                  !modelSettings.pinned,
                  let ttlSeconds = modelSettings.ttlSeconds,
                  let lastAccess = state.lastAccess,
                  state.activeRequests == 0,
                  referenceDate.timeIntervalSince(lastAccess) >= Double(ttlSeconds) else {
                continue
            }

            _ = removeLoadedModel(for: id)
            await unloadLoadedModel(loaded)
            evicted.append(id)
        }

        return evicted.sorted()
    }

    @discardableResult
    func evictIdleModelsForMemoryPressure(
        sample: RuntimeMemorySample? = nil,
        excluding excludedIDs: Set<String> = []
    ) async -> [String] {
        var memorySample = sample ?? currentMemorySample()
        var pressure = memoryPressurePolicy.pressure(for: memorySample)
        if pressure == .elevated || pressure == .critical {
            clearMLXCache()
            if sample == nil {
                memorySample = currentMemorySample()
                pressure = memoryPressurePolicy.pressure(for: memorySample)
            }
        }
        let evictionLimit: Int?
        switch pressure {
        case .disabled, .unknown, .nominal:
            return []
        case .elevated:
            evictionLimit = 1
        case .critical:
            evictionLimit = nil
        }

        let settings = (try? settingsStore.load())?.models ?? [:]
        let candidates = loadedModels.compactMap { id, _ -> RuntimeLRUEvictionCandidate? in
            let state = state(for: id)
            let modelSettings = settings[id] ?? RuntimeModelSettings()
            guard !excludedIDs.contains(id),
                  id != defaultModelID,
                  !preparingModelIDs.contains(id),
                  state.activeRequests == 0,
                  !modelSettings.pinned else {
                return nil
            }
            return RuntimeLRUEvictionCandidate(
                id: id,
                lastAccess: state.lastAccess ?? .distantPast
            )
        }
        .sorted {
            if $0.lastAccess == $1.lastAccess {
                return $0.id < $1.id
            }
            return $0.lastAccess < $1.lastAccess
        }

        var evicted: [String] = []
        for candidate in candidates {
            guard evictionLimit.map({ evicted.count < $0 }) ?? true else {
                break
            }
            guard let loaded = removeLoadedModel(for: candidate.id) else {
                continue
            }
            await unloadLoadedModel(loaded)
            evicted.append(candidate.id)
        }

        return evicted.sorted()
    }

    private func evictOldestIdleUnpinnedModel(
        excluding excludedIDs: Set<String>
    ) async -> String? {
        let settings = (try? settingsStore.load())?.models ?? [:]
        let candidate = loadedModels.keys.compactMap { id -> RuntimeLRUEvictionCandidate? in
            let state = state(for: id)
            let modelSettings = settings[id] ?? RuntimeModelSettings()
            guard !excludedIDs.contains(id),
                  !preparingModelIDs.contains(id),
                  state.activeRequests == 0,
                  !modelSettings.pinned else {
                return nil
            }
            return RuntimeLRUEvictionCandidate(
                id: id,
                lastAccess: state.lastAccess ?? .distantPast
            )
        }
        .sorted {
            if $0.lastAccess == $1.lastAccess {
                return $0.id < $1.id
            }
            return $0.lastAccess < $1.lastAccess
        }
        .first

        guard let candidate,
              let loaded = removeLoadedModel(for: candidate.id) else {
            return nil
        }
        await unloadLoadedModel(loaded)
        return candidate.id
    }

    private func ensureLoaded(_ resolved: ResolvedModel) async throws -> RuntimeLoadedModel {
        if let loaded = loadedModels[resolved.id], !preparingModelIDs.contains(resolved.id) {
            return loaded
        }
        if let preparation = modelPreparations[resolved.id] {
            return try await awaitPreparation(preparation, modelID: resolved.id)
        }

        try ensureMLXAvailable()
        let loaded = makeLoadedModel(for: resolved)
        let token = UUID()
        loadedModels[resolved.id] = loaded
        loadedModelTokens[resolved.id] = token
        preparingModelIDs.insert(resolved.id)
        let prepareLoadedModel = self.prepareLoadedModel
        let task = Task<RuntimeLoadedModel, Error> {
            try await prepareLoadedModel(loaded)
            return loaded
        }
        let preparation = ModelPreparation(token: token, task: task)
        modelPreparations[resolved.id] = preparation
        return try await awaitPreparation(preparation, modelID: resolved.id)
    }

    private func awaitPreparation(
        _ preparation: ModelPreparation,
        modelID: String
    ) async throws -> RuntimeLoadedModel {
        let waiterID = UUID()
        guard var currentPreparation = modelPreparations[modelID],
              currentPreparation.token == preparation.token else {
            return try await finalizedLoadedModel(
                modelID: modelID,
                token: preparation.token
            )
        }
        currentPreparation.waiterIDs.insert(waiterID)
        modelPreparations[modelID] = currentPreparation

        return try await withTaskCancellationHandler {
            do {
                _ = try await preparation.task.value
                try Task.checkCancellation()
                return try await finalizedLoadedModel(
                    modelID: modelID,
                    token: preparation.token
                )
            } catch {
                if Task.isCancelled {
                    await cancelPreparationWaiter(
                        modelID: modelID,
                        token: preparation.token,
                        waiterID: waiterID
                    )
                } else {
                    await failPreparation(
                        preparation,
                        modelID: modelID,
                        error: error
                    )
                }
                throw error
            }
        } onCancel: {
            Task {
                await self.cancelPreparationWaiter(
                    modelID: modelID,
                    token: preparation.token,
                    waiterID: waiterID
                )
            }
        }
    }

    private func finalizedLoadedModel(
        modelID: String,
        token: UUID
    ) async throws -> RuntimeLoadedModel {
        if let cleanup = preparationCleanupTasks[token] {
            await cleanup.value
            preparationCleanupTasks.removeValue(forKey: token)
            throw CancellationError()
        }

        if let preparation = modelPreparations[modelID], preparation.token == token {
            guard loadedModelTokens[modelID] == token,
                  let loaded = loadedModels[modelID] else {
                await invalidatePreparation(
                    preparation,
                    modelID: modelID,
                    error: CancellationError().localizedDescription
                )
                throw CancellationError()
            }
            modelPreparations.removeValue(forKey: modelID)
            preparingModelIDs.remove(modelID)
            touch(id: modelID, error: nil)
            return loaded
        }

        // A different waiter may already have finalized this generation. An
        // unload followed by a reload has a different token and must never be
        // returned to a stale waiter from the prior generation.
        if loadedModelTokens[modelID] == token,
           !preparingModelIDs.contains(modelID),
           let loaded = loadedModels[modelID] {
            return loaded
        }
        throw CancellationError()
    }

    private func cancelPreparationWaiter(
        modelID: String,
        token: UUID,
        waiterID: UUID
    ) async {
        guard var preparation = modelPreparations[modelID],
              preparation.token == token,
              preparation.waiterIDs.remove(waiterID) != nil else {
            await awaitPreparationCleanup(token: token)
            return
        }
        guard preparation.waiterIDs.isEmpty else {
            modelPreparations[modelID] = preparation
            return
        }
        await invalidatePreparation(preparation, modelID: modelID, error: nil)
    }

    private func failPreparation(
        _ preparation: ModelPreparation,
        modelID: String,
        error: Error
    ) async {
        await invalidatePreparation(
            preparation,
            modelID: modelID,
            error: error.localizedDescription
        )
    }

    private func invalidatePreparation(
        _ preparation: ModelPreparation,
        modelID: String,
        error: String?
    ) async {
        if modelPreparations[modelID]?.token == preparation.token {
            modelPreparations.removeValue(forKey: modelID)
            preparingModelIDs.remove(modelID)
            preparation.task.cancel()
            if let error {
                touch(id: modelID, error: error)
            }
            if let loaded = removeLoadedModel(for: modelID, matching: preparation.token) {
                let unloadLoadedModel = self.unloadLoadedModel
                let preparationTask = preparation.task
                preparationCleanupTasks[preparation.token] = Task {
                    // Generator actors are reentrant across model resolution
                    // and loading awaits. Let cancellation settle preparation
                    // before unloading so a resumed prepare cannot repopulate
                    // the old generation after cleanup.
                    _ = await preparationTask.result
                    await unloadLoadedModel(loaded)
                }
            }
        }
        await awaitPreparationCleanup(token: preparation.token)
    }

    private func awaitPreparationCleanup(token: UUID) async {
        guard let cleanup = preparationCleanupTasks[token] else {
            return
        }
        await cleanup.value
        preparationCleanupTasks.removeValue(forKey: token)
    }

    private func removeLoadedModel(for modelID: String) -> RuntimeLoadedModel? {
        loadedModelTokens.removeValue(forKey: modelID)
        return loadedModels.removeValue(forKey: modelID)
    }

    private func removeLoadedModel(
        for modelID: String,
        matching token: UUID
    ) -> RuntimeLoadedModel? {
        guard loadedModelTokens[modelID] == token else {
            return nil
        }
        return removeLoadedModel(for: modelID)
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
        case .textChatLaguna:
            return .textChatLaguna(
                LagunaGenerator(
                    continuousBatchingEnabled: lagunaContinuousBatchingEnabled,
                    dflashModelPath: LagunaResources.installedDFlashPath(
                        for: resolved.id
                    )
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
                LFM2Generator(
                    modelId: resolved.id,
                    prefixKVCacheEnabled: lfm2PrefixKVCacheEnabled,
                    continuousBatchingEnabled: lfm2ContinuousBatchingEnabled
                ),
                modelPath: resolved.installPath
            )
        case .textChatDeepseekV4Flash:
            return .textChatDeepseekV4Flash(
                DeepseekV4FlashGenerator(modelId: resolved.id),
                modelPath: resolved.installPath
            )
        case .textChatMuseGlimmer:
            return .textChatMuseGlimmer(
                MuseGlimmerGenerator(modelID: resolved.id),
                modelPath: resolved.installPath
            )
        case .textChatNemotronH:
            return .textChatNemotronH(
                NemotronHGenerator(),
                modelPath: resolved.installPath
            )
        case .textChatNemotronOmni:
            return .textChatNemotronOmni(
                NemotronOmniGenerator(),
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
        continuousBatching: RuntimeDecodeBatchingStats? = nil,
        mtp: Gemma4MTPStats? = nil
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
        var snapshot = RuntimeModelPoolEntrySnapshot(
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
            minP: modelSettings.minP,
            engineOverride: modelSettings.engineOverride,
            kvCacheMode: modelSettings.kvCacheMode,
            prefixKVCache: prefixKVCache,
            continuousBatching: continuousBatching,
            mtp: mtp,
            benchmarkStats: state.benchmarkStats
        )
        snapshot.ready = loadedModels[id] != nil && !preparingModelIDs.contains(id)
        snapshot.startupTiming = state.startupTiming
        return snapshot
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
        if request.min_p == nil {
            request.min_p = settings.minP
        }
    }

    private func state(for id: String) -> MutableState {
        states[id] ?? MutableState()
    }

    private func touch(id: String, error: String?) {
        var state = state(for: id)
        state.lastAccess = currentDate()
        state.lastError = error
        states[id] = state
    }

    private func retainLease(id: String) {
        var state = state(for: id)
        state.activeRequests += 1
        state.lastAccess = currentDate()
        states[id] = state
    }

    private func category(for engine: RuntimeServingEngine) -> String {
        switch engine {
        case .textCode:
            return ManagedModelCategory.textCode.rawValue
        case .textChatKlein,
             .textChatGemma4,
             .textChatLaguna,
             .textChatQ36,
             .textChatQ35,
             .textChatLFM2,
             .textChatDeepseekV4Flash,
             .textChatNemotronH:
            return ManagedModelCategory.textChat.rawValue
        case .textChatMuseGlimmer:
            return ManagedModelCategory.visionChat.rawValue
        case .textChatNemotronOmni:
            return ManagedModelCategory.omniChat.rawValue
        }
    }

    #if DEBUG
    func seedLoadedModelForTesting(
        id: String,
        lastAccess: Date,
        activeRequests: Int = 0
    ) {
        loadedModels[id] = .textCode(CodeGenGenerator(modelId: id), modelPath: nil)
        loadedModelTokens[id] = UUID()
        preparingModelIDs.remove(id)
        var state = state(for: id)
        state.activeRequests = activeRequests
        state.lastAccess = lastAccess
        states[id] = state
    }

    func seedLoadedLFM2ForTesting(
        id: String,
        continuousBatchingEnabled: Bool
    ) {
        loadedModels[id] = .textChatLFM2(
            LFM2Generator(
                modelId: id,
                prefixKVCacheEnabled: lfm2PrefixKVCacheEnabled,
                continuousBatchingEnabled: continuousBatchingEnabled
            ),
            modelPath: nil
        )
        loadedModelTokens[id] = UUID()
        preparingModelIDs.remove(id)
        var state = state(for: id)
        state.lastAccess = currentDate()
        states[id] = state
    }
    #endif
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

    private struct ActiveRequestState {
        let id: UUID
        let admittedAt: Date
        var modelID: String?
        var streaming: Bool?
        var requestedMaxTokens: Int?
        var toolCount: Int?
        var phase = "admitted"
        var phaseDetail: String?
        var firstTokenAt: Date?
        var generatedTokenUpdates = 0
        var lastEventSequence: UInt64 = 0

        var snapshot: RuntimeActiveRequestSnapshot {
            RuntimeActiveRequestSnapshot(
                id: id,
                admittedAt: admittedAt,
                modelID: modelID,
                streaming: streaming,
                requestedMaxTokens: requestedMaxTokens,
                toolCount: toolCount,
                phase: phase,
                phaseDetail: phaseDetail,
                firstTokenAt: firstTokenAt,
                generatedTokenUpdates: generatedTokenUpdates
            )
        }
    }

    private let maxActiveRequests: Int
    private let pressureProvider: @Sendable () async -> RuntimeMemoryPressureLevel
    private var activeRequests = 0
    private var waiters: [Waiter] = []
    private var waiterStates: [UUID: WaiterState] = [:]
    private var totalAdmittedRequests = 0
    private var totalCompletedRequests = 0
    private var totalCancelledRequests = 0
    private var activeRequestStates: [UUID: ActiveRequestState] = [:]
    private var lastClientDisconnectAt: Date?
    private var lastCancellationAt: Date?
    private var lastSlotReleaseAt: Date?

    init(
        maxActiveRequests: Int,
        pressureProvider: @escaping @Sendable () async -> RuntimeMemoryPressureLevel = { .nominal }
    ) {
        precondition(maxActiveRequests > 0, "maxActiveRequests must be positive")
        self.maxActiveRequests = maxActiveRequests
        self.pressureProvider = pressureProvider
    }

    func acquire() async throws -> RuntimeRequestAdmissionLease {
        await drain()
        if await canAdmitNow(requireEmptyQueue: true) {
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

    /// Acquires immediately without joining the FIFO, or returns nil when an
    /// active/queued request already owns the capacity. Eviction uses this to
    /// avoid waiting behind and then unloading a runtime that was active when
    /// maintenance began.
    func tryAcquire() async -> RuntimeRequestAdmissionLease? {
        guard waiters.isEmpty, activeRequests < maxActiveRequests else {
            return nil
        }
        let pressure = await pressureProvider()
        guard waiters.isEmpty,
              activeRequests < maxActiveRequests,
              !admissionPaused(for: pressure) || activeRequests == 0 else {
            return nil
        }
        return admit()
    }

    fileprivate func configureLease(
        id: UUID,
        modelID: String,
        streaming: Bool,
        requestedMaxTokens: Int,
        toolCount: Int
    ) {
        guard var state = activeRequestStates[id] else { return }
        state.modelID = modelID
        state.streaming = streaming
        state.requestedMaxTokens = requestedMaxTokens
        state.toolCount = toolCount
        activeRequestStates[id] = state
    }

    func recordProgress(
        id: UUID,
        sequence: UInt64,
        progress: ChatProgress,
        observedAt: Date = Date()
    ) {
        guard var state = activeRequestStates[id] else { return }
        if progress.stage == .generating, progress.message?.isEmpty == false {
            if let firstTokenAt = state.firstTokenAt {
                state.firstTokenAt = min(firstTokenAt, observedAt)
            } else {
                state.firstTokenAt = observedAt
            }
            state.generatedTokenUpdates += 1
        }
        guard state.phase != "cancelling", sequence > state.lastEventSequence else {
            activeRequestStates[id] = state
            return
        }
        state.lastEventSequence = sequence
        switch progress.stage {
        case .loadingModel:
            state.phase = "loading_model"
            state.phaseDetail = progress.message
        case .encoding:
            state.phase = "prefill"
            state.phaseDetail = progress.message
        case .generating:
            state.phase = "decode"
            state.phaseDetail = nil
        }
        activeRequestStates[id] = state
    }

    func recordClientDisconnect(id: UUID, sequence: UInt64, observedAt: Date = Date()) {
        guard var state = activeRequestStates[id] else { return }
        state.phase = "cancelling"
        state.phaseDetail = "Client disconnected"
        state.lastEventSequence = max(state.lastEventSequence, sequence)
        activeRequestStates[id] = state
        lastClientDisconnectAt = observedAt
    }

    fileprivate func releaseLease(id: UUID, cancelled: Bool) async {
        activeRequests = max(0, activeRequests - 1)
        activeRequestStates.removeValue(forKey: id)
        lastSlotReleaseAt = Date()
        if cancelled {
            totalCancelledRequests += 1
            lastCancellationAt = Date()
        } else {
            totalCompletedRequests += 1
        }
        await drain()
    }

    func snapshot() async -> RuntimeRequestAdmissionSnapshot {
        let pressure = await pressureProvider()
        return RuntimeRequestAdmissionSnapshot(
            maxActiveRequests: maxActiveRequests,
            activeRequests: activeRequests,
            queuedRequests: waiters.count,
            totalAdmittedRequests: totalAdmittedRequests,
            totalCompletedRequests: totalCompletedRequests,
            totalCancelledRequests: totalCancelledRequests,
            admissionPaused: admissionPaused(for: pressure),
            pressure: pressure.rawValue,
            activeRequestDetails: activeRequestStates.values
                .map(\.snapshot)
                .sorted { $0.admittedAt < $1.admittedAt },
            lastClientDisconnectAt: lastClientDisconnectAt,
            lastCancellationAt: lastCancellationAt,
            lastSlotReleaseAt: lastSlotReleaseAt
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

    private func drain() async {
        while activeRequests < maxActiveRequests, !waiters.isEmpty {
            guard await canAdmitNow() else {
                return
            }
            // Pressure sampling yields the actor. A concurrent drain may have
            // consumed capacity or cancellation may have emptied the FIFO, so
            // revalidate both before removing its first waiter.
            guard activeRequests < maxActiveRequests, !waiters.isEmpty else {
                return
            }
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
        let id = UUID()
        activeRequests += 1
        totalAdmittedRequests += 1
        activeRequestStates[id] = ActiveRequestState(id: id, admittedAt: Date())
        return RuntimeRequestAdmissionLease(id: id, admission: self)
    }

    private func canAdmitNow(requireEmptyQueue: Bool = false) async -> Bool {
        guard activeRequests < maxActiveRequests,
              !requireEmptyQueue || waiters.isEmpty else {
            return false
        }
        let pressure = await pressureProvider()
        // `pressureProvider` is an actor reentrancy point. Never rely on the
        // capacity/FIFO snapshot taken before it suspended.
        guard activeRequests < maxActiveRequests,
              !requireEmptyQueue || waiters.isEmpty else {
            return false
        }
        guard admissionPaused(for: pressure) else {
            return true
        }
        return activeRequests == 0
    }

    private func admissionPaused(for pressure: RuntimeMemoryPressureLevel) -> Bool {
        switch pressure {
        case .elevated, .critical:
            return true
        case .disabled, .unknown, .nominal:
            return false
        }
    }
}

final class RuntimeRequestAdmissionLease: @unchecked Sendable {
    private let id: UUID
    private let admission: RuntimeRequestAdmission
    private let lock = NSLock()
    private var released = false
    private var clientDisconnected = false
    private var nextEventSequence: UInt64 = 0

    init(id: UUID, admission: RuntimeRequestAdmission) {
        self.id = id
        self.admission = admission
    }

    var requestID: UUID { id }

    deinit {
        guard markReleased() else { return }
        let admission = admission
        let id = id
        Task {
            await admission.releaseLease(id: id, cancelled: false)
        }
    }

    func configure(
        modelID: String,
        streaming: Bool,
        requestedMaxTokens: Int,
        toolCount: Int
    ) async {
        await admission.configureLease(
            id: id,
            modelID: modelID,
            streaming: streaming,
            requestedMaxTokens: requestedMaxTokens,
            toolCount: toolCount
        )
    }

    func observe(_ progress: ChatProgress) {
        guard let sequence = reserveEventSequence() else { return }
        let admission = admission
        let id = id
        let observedAt = Date()
        Task {
            await admission.recordProgress(
                id: id,
                sequence: sequence,
                progress: progress,
                observedAt: observedAt
            )
        }
    }

    func observeClientDisconnect() {
        guard let sequence = reserveDisconnectSequence() else { return }
        let admission = admission
        let id = id
        let observedAt = Date()
        Task {
            await admission.recordClientDisconnect(
                id: id,
                sequence: sequence,
                observedAt: observedAt
            )
        }
    }

    func release(cancelled: Bool = false) async {
        guard markReleased() else { return }
        await admission.releaseLease(id: id, cancelled: cancelled)
    }

    private func markReleased() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !released else { return false }
        released = true
        return true
    }

    private func reserveEventSequence() -> UInt64? {
        lock.lock()
        defer { lock.unlock() }
        guard !released, !clientDisconnected else { return nil }
        nextEventSequence += 1
        return nextEventSequence
    }

    private func reserveDisconnectSequence() -> UInt64? {
        lock.lock()
        defer { lock.unlock() }
        guard !released, !clientDisconnected else { return nil }
        clientDisconnected = true
        nextEventSequence += 1
        return nextEventSequence
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
            if !(error is CancellationError) {
                await pool.recordChatFailure(modelID: modelID, error: error)
            }
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
    case textChatLaguna(LagunaGenerator, modelPath: String?)
    case textChatQ35(Q35Generator, modelPath: String?)
    case textChatLFM2(LFM2Generator, modelPath: String?)
    case textChatDeepseekV4Flash(DeepseekV4FlashGenerator, modelPath: String?)
    case textChatMuseGlimmer(MuseGlimmerGenerator, modelPath: String?)
    case textChatNemotronH(NemotronHGenerator, modelPath: String?)
    case textChatNemotronOmni(NemotronOmniGenerator, modelPath: String?)

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
        case .textChatLaguna(let generator, let modelPath):
            guard let modelPath else {
                throw LagunaError.modelPathRequired
            }
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatQ35(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLFM2(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatMuseGlimmer(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronH(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronOmni(let generator, let modelPath):
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
        case .textChatLaguna(let generator, _):
            await generator.unload()
        case .textChatQ35(let generator, _):
            await generator.unload()
        case .textChatLFM2(let generator, _):
            await generator.unload()
        case .textChatDeepseekV4Flash(let generator, _):
            await generator.shutdown()
        case .textChatMuseGlimmer(let generator, _):
            await generator.unload()
        case .textChatNemotronH(let generator, _):
            await generator.unload()
        case .textChatNemotronOmni(let generator, _):
            await generator.unload()
        }
    }

    func prefixKVCacheStats() async -> PrefixKVCacheStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textChatQ35(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textChatLFM2(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textCode, .textChatKlein, .textChatLaguna, .textChatDeepseekV4Flash,
             .textChatMuseGlimmer, .textChatNemotronH, .textChatNemotronOmni:
            return nil
        }
    }

    func continuousBatchingStats() async -> RuntimeDecodeBatchingStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatLaguna(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatQ35(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatLFM2(let generator, _):
            return await generator.continuousBatchingStats()
        case .textCode, .textChatKlein, .textChatDeepseekV4Flash, .textChatMuseGlimmer,
             .textChatNemotronH, .textChatNemotronOmni:
            return nil
        }
    }

    func mtpStats() async -> Gemma4MTPStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.mtpStats()
        case .textCode, .textChatKlein, .textChatLaguna, .textChatQ35, .textChatLFM2,
             .textChatDeepseekV4Flash, .textChatMuseGlimmer, .textChatNemotronH,
             .textChatNemotronOmni:
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
        case .textChatLaguna(let generator, let modelPath):
            guard let modelPath else {
                throw LagunaError.modelPathRequired
            }
            return try await generator.chat(
                request,
                modelPath: modelPath,
                progressHandler: progressHandler
            )
        case .textChatQ35(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLFM2(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatMuseGlimmer(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronH(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronOmni(let generator, let modelPath):
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
        case .textCode, .textChatKlein, .textChatGemma4, .textChatLaguna, .textChatQ35,
             .textChatLFM2, .textChatMuseGlimmer, .textChatNemotronH,
             .textChatNemotronOmni:
            throw RuntimeModelPoolError.rawProxyUnavailable("")
        }
    }
}

extension RuntimeServingEngine {
    var openAICompatibility: APIEngineCapabilities {
        .catalog(.runtimeFallback(for: self))
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
