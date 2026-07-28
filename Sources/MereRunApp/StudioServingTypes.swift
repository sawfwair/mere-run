import Foundation

/// App-side projection of `/runtime/status`. The Studio intentionally does not import the CLI
/// executable target, and every additive field is optional so it can monitor older servers.
struct StudioRuntimeSnapshot: Decodable, Equatable {
    let object: String?
    let defaultModel: String?
    let settingsPath: String?
    let activeRequests: Int?
    let admission: StudioRuntimeAdmission?
    let capabilities: StudioRuntimeCapabilities?
    let memory: StudioRuntimeMemory?
    let models: [StudioRuntimeModel]?
    let cacheStats: StudioRuntimeCacheStats?
    let benchmarkStats: StudioRuntimeTraffic?
    let sidecars: StudioRuntimeSidecarPool?
    let process: StudioRuntimeProcess?

    var textModels: [StudioRuntimeModel] { models ?? [] }
    var loadedTextModels: [StudioRuntimeModel] { textModels.filter(\.loaded) }
    var loadedSidecars: [StudioRuntimeSidecar] { sidecars?.residents.filter(\.loaded) ?? [] }
    var totalFailures: Int {
        let textFailures = textModels.reduce(0) { $0 + ($1.benchmarkStats?.failedRequests ?? 0) }
        let sidecarFailures = (sidecars?.residents ?? []).reduce(0) { $0 + $1.failedRequests }
        return max(benchmarkStats?.failedRequests ?? 0, textFailures) + sidecarFailures
    }
}

struct StudioRuntimeAdmission: Decodable, Equatable {
    let maxActiveRequests: Int
    let activeRequests: Int
    let queuedRequests: Int
    let totalAdmittedRequests: Int
    let totalCompletedRequests: Int
    let totalCancelledRequests: Int
    let admissionPaused: Bool?
    let pressure: String?
}

struct StudioRuntimeCapabilities: Decodable, Equatable {
    let requestAdmission: StudioRuntimeCapability?
    let chunkedPrefill: StudioRuntimeCapability?
    let continuousBatching: StudioRuntimeCapability?
    let prefixKVReuse: StudioRuntimeCapability?
    let ssdKVCache: StudioRuntimeCapability?
}

struct StudioRuntimeCapability: Decodable, Equatable {
    let available: Bool
    let enabled: Bool
    let detail: String
}

struct StudioRuntimeMemory: Decodable, Equatable {
    let physicalBytes: UInt64?
    let residentBytes: UInt64?
    let currentBytes: UInt64?
    let availableBytes: UInt64?
    let ceilingBytes: UInt64?
    let softLimitBytes: UInt64?
    let hardLimitBytes: UInt64?
    let activeRequests: Int?
    let activeModelCount: Int?
    let guardTier: String?
    let pressure: String?
}

struct StudioRuntimeProcess: Decodable, Equatable {
    let processID: Int32?
    let startedAt: Date?
    let uptimeSeconds: Double?
    let cpuPercent: Double?
    let thermalState: String?
    let lowPowerModeEnabled: Bool?
    let metalDeviceName: String?
    let metalCurrentAllocatedBytes: UInt64?
    let metalRecommendedMaxWorkingSetBytes: UInt64?
    let metalHasUnifiedMemory: Bool?
}

struct StudioRuntimeModel: Decodable, Equatable, Identifiable {
    let id: String
    let category: String?
    let engine: String?
    let installPath: String?
    let loaded: Bool
    let ready: Bool?
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
    let engineOverride: String?
    let kvCacheMode: String?
    let prefixKVCache: StudioRuntimePrefixKV?
    let continuousBatching: StudioRuntimeBatching?
    let mtp: StudioRuntimeMTP?
    let benchmarkStats: StudioRuntimeModelTraffic?

    var state: String {
        guard loaded else { return "Unloaded" }
        if ready == false { return "Loading" }
        if activeRequests > 0 { return "Active" }
        return "Ready"
    }
}

struct StudioRuntimePrefixKV: Decodable, Equatable {
    let enabled: Bool
    let entries: Int
    let maxEntries: Int
    let hits: Int
    let misses: Int
    let storedPrefixes: Int
    let reusedTokens: Int
    let storedTokens: Int
}

struct StudioRuntimeBatching: Decodable, Equatable {
    let enabled: Bool
    let activeRows: Int
    let queuedRows: Int
    let batchedDecodeSteps: Int
    let samePositionBatchedSteps: Int?
    let variablePositionBatchedSteps: Int?
    let singleDecodeSteps: Int
    let totalBatchedRows: Int
    let maxBatchSize: Int
}

struct StudioRuntimeMTP: Decodable, Equatable {
    let available: Bool
    let enabled: Bool
    let active: Bool
    let reason: String?
    let blockSize: Int
    let threshold: Int
    let rounds: Int
    let draftedTokens: Int
    let acceptedTokens: Int
    let rejectedTokens: Int
}

struct StudioRuntimeModelTraffic: Decodable, Equatable {
    let completedRequests: Int
    let failedRequests: Int
    let generatedTokens: Int
    let averageLoadSeconds: Double?
    let averagePrefillSeconds: Double?
    let averageDecodeSeconds: Double?
    let averageTotalSeconds: Double?
    let decodeTokensPerSecond: Double?
    let recentDecodeTokensPerSecond: Double?
    let lastCompletedAt: Date?
}

struct StudioRuntimeSidecarPool: Decodable, Equatable {
    let defaultIdleTTLSeconds: Int
    let pressure: String
    let loadedCount: Int
    let activeRequests: Int
    let queuedRequests: Int
    let residents: [StudioRuntimeSidecar]
}

struct StudioRuntimeSidecar: Decodable, Equatable, Identifiable {
    let kind: String
    let modelID: String?
    let modelPath: String?
    let variant: String?
    let loaded: Bool
    let ready: Bool?
    let activeRequests: Int
    let queuedRequests: Int
    let loadedAt: Date?
    let lastAccess: Date?
    let lastEvictedAt: Date?
    let lastEvictionReason: String?
    let pinned: Bool
    let ttlSeconds: Int
    let loadCount: Int
    let replacementCount: Int
    let evictionCount: Int
    let completedRequests: Int
    let failedRequests: Int

    var id: String { kind }
    var displayModel: String { modelID ?? variant ?? modelPath ?? "Not selected" }
    var state: String {
        guard loaded else { return "Unloaded" }
        if ready == false { return "Loading" }
        if activeRequests > 0 { return "Active" }
        if queuedRequests > 0 { return "Queued" }
        return "Ready"
    }
}

struct StudioRuntimeCacheStats: Decodable, Equatable {
    let available: Bool
    let detail: String
    let prefixKVReuse: StudioRuntimePrefixKVSummary?
    let decodeBatching: StudioRuntimeBatchingSummary?
}

struct StudioRuntimePrefixKVSummary: Decodable, Equatable {
    let reportedModelCount: Int
    let enabledModelCount: Int
    let entries: Int
    let maxEntries: Int
    let hits: Int
    let misses: Int
    let storedPrefixes: Int
    let reusedTokens: Int
    let storedTokens: Int

    var hitRate: Double? {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : nil
    }
}

struct StudioRuntimeBatchingSummary: Decodable, Equatable {
    let reportedModelCount: Int
    let enabledModelCount: Int
    let activeRows: Int
    let queuedRows: Int
    let batchedDecodeSteps: Int
    let samePositionBatchedSteps: Int?
    let variablePositionBatchedSteps: Int?
    let singleDecodeSteps: Int
    let totalBatchedRows: Int
    let maxBatchSize: Int
}

/// Runtime names this payload `benchmarkStats` for compatibility. Studio presents it as live,
/// observed service traffic; it is not a synthetic benchmark.
struct StudioRuntimeTraffic: Decodable, Equatable {
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
}

struct StudioAgentStatus: Decodable, Equatable {
    let machine: StudioAgentMachine
    let pi: StudioAgentPi
    let provider: StudioAgentProvider
    let recommendedModelID: String?
    let models: [StudioAgentModel]
}

struct StudioAgentMachine: Decodable, Equatable {
    let processor: String
    let unifiedMemoryGB: Int
    let appleSiliconMac: Bool
    let linux: Bool
}

struct StudioAgentPi: Decodable, Equatable {
    let installed: Bool
    let managedInstall: Bool
    let autoInstallSupported: Bool
    let path: String?
    let version: String?
}

struct StudioAgentProvider: Decodable, Equatable {
    let configured: Bool
    let host: String?
    let port: Int?
    let modelID: String?
    let updatedAt: Date?
    let configurationPath: String
    let extensionPath: String
}

struct StudioAgentModel: Decodable, Equatable, Identifiable {
    let id: String
    let displayName: String
    let summary: String
    let minimumUnifiedMemoryGB: Int
    let recommendedUnifiedMemoryGB: Int
    let servingEngine: String
    let startableByMereRun: Bool
    let sourceConfigurationRequired: Bool
    let installed: Bool
    let reason: String?
}

enum StudioServingSafety: Equatable {
    case loopback
    case protectedLAN
    case exposedWithoutAuthentication

    static func evaluate(host: String, apiKey: String) -> StudioServingSafety {
        if isLoopback(host) { return .loopback }
        return apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? .exposedWithoutAuthentication
            : .protectedLAN
    }

    static func isLoopback(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "localhost"
            || normalized == "::1"
            || normalized == "[::1]"
            || normalized.hasPrefix("127.")
    }
}

struct StudioServiceActivity: Identifiable, Equatable {
    enum Level: String, Equatable {
        case info
        case success
        case warning
        case error
    }

    let id: UUID
    let date: Date
    let level: Level
    let title: String
    let detail: String?

    init(
        id: UUID = UUID(),
        date: Date = Date(),
        level: Level,
        title: String,
        detail: String? = nil
    ) {
        self.id = id
        self.date = date
        self.level = level
        self.title = title
        self.detail = detail.map(StudioActivitySanitizer.sanitize)
    }
}

enum StudioActivitySanitizer {
    static func sanitize(_ text: String) -> String {
        var sanitized = text
        let patterns = [
            #"(?i)bearer\s+[A-Za-z0-9._~+/\-=]+"#,
            #"(?i)(api[_ -]?key|authorization)\s*[:=]\s*[^\s,}]+"#,
            #"(?i)"(prompt|messages|input)"\s*:\s*"[^"]*""#,
        ]
        let replacements = [
            "Bearer [redacted]",
            "$1=[redacted]",
            "\"$1\":\"[redacted]\"",
        ]
        for (pattern, replacement) in zip(patterns, replacements) {
            guard let expression = try? NSRegularExpression(pattern: pattern) else { continue }
            sanitized = expression.stringByReplacingMatches(
                in: sanitized,
                range: NSRange(sanitized.startIndex..., in: sanitized),
                withTemplate: replacement
            )
        }
        return String(sanitized.prefix(1_000))
    }
}

enum StudioServiceActivityDiff {
    static func events(
        previous: StudioRuntimeSnapshot?,
        current: StudioRuntimeSnapshot
    ) -> [StudioServiceActivity] {
        guard let previous else {
            return [
                StudioServiceActivity(
                    level: .success,
                    title: "Runtime connected",
                    detail: "\(current.loadedTextModels.count) text and \(current.loadedSidecars.count) sidecar models resident"
                ),
            ]
        }

        var result: [StudioServiceActivity] = []
        let oldText = Dictionary(uniqueKeysWithValues: previous.textModels.map { ($0.id, $0) })
        for model in current.textModels {
            let old = oldText[model.id]
            if old?.loaded != model.loaded {
                result.append(.init(
                    level: model.loaded ? .success : .info,
                    title: model.loaded ? "Text model loaded" : "Text model unloaded",
                    detail: model.id
                ))
            } else if old?.ready != model.ready, model.loaded {
                result.append(.init(
                    level: model.ready == false ? .info : .success,
                    title: model.ready == false ? "Text model preparing" : "Text model ready",
                    detail: model.id
                ))
            }
            if old?.lastError != model.lastError, let error = model.lastError {
                result.append(.init(level: .error, title: "Model error", detail: "\(model.id): \(error)"))
            }
        }

        let oldSidecars = Dictionary(
            uniqueKeysWithValues: (previous.sidecars?.residents ?? []).map { ($0.kind, $0) }
        )
        for sidecar in current.sidecars?.residents ?? [] {
            let old = oldSidecars[sidecar.kind]
            if old?.loaded != sidecar.loaded {
                result.append(.init(
                    level: sidecar.loaded ? .success : .info,
                    title: sidecar.loaded ? "\(sidecar.kind.capitalized) service loaded" : "\(sidecar.kind.capitalized) service unloaded",
                    detail: sidecar.displayModel
                ))
            }
            if old?.lastEvictedAt != sidecar.lastEvictedAt, let reason = sidecar.lastEvictionReason {
                result.append(.init(
                    level: reason == "memory_pressure" ? .warning : .info,
                    title: "\(sidecar.kind.capitalized) service evicted",
                    detail: reason.replacingOccurrences(of: "_", with: " ")
                ))
            }
        }

        if previous.admission?.queuedRequests != current.admission?.queuedRequests,
           let queued = current.admission?.queuedRequests, queued > 0 {
            result.append(.init(level: .info, title: "Requests queued", detail: "\(queued) waiting"))
        }
        if previous.memory?.pressure != current.memory?.pressure,
           let pressure = current.memory?.pressure, pressure != "nominal" {
            result.append(.init(level: .warning, title: "Memory pressure changed", detail: pressure))
        }
        if current.totalFailures > previous.totalFailures {
            result.append(.init(
                level: .error,
                title: "Request failure recorded",
                detail: "\(current.totalFailures - previous.totalFailures) new failure(s)"
            ))
        }
        return result
    }
}

enum StudioServingFormat {
    static func bytes(_ value: UInt64?) -> String {
        guard let value else { return "Unavailable" }
        return ByteCountFormatter.string(fromByteCount: Int64(clamping: value), countStyle: .memory)
    }

    static func duration(_ seconds: Double?) -> String {
        guard let seconds else { return "Unavailable" }
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        if seconds < 3_600 { return String(format: "%.0fm", seconds / 60) }
        return String(format: "%.1fh", seconds / 3_600)
    }

    static func number(_ value: Double?, suffix: String = "") -> String {
        guard let value else { return "Unavailable" }
        return String(format: "%.1f%@", value, suffix)
    }
}
