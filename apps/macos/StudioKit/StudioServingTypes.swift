import Foundation

/// App-side projection of `/runtime/status`. The Studio intentionally does not import the CLI
/// executable target, and every additive field is optional so it can monitor older servers.
package struct StudioRuntimeSnapshot: Decodable, Equatable {
    package let object: String?
    package let defaultModel: String?
    package let settingsPath: String?
    package let activeRequests: Int?
    package let admission: StudioRuntimeAdmission?
    package let capabilities: StudioRuntimeCapabilities?
    package let memory: StudioRuntimeMemory?
    package let models: [StudioRuntimeModel]?
    package let cacheStats: StudioRuntimeCacheStats?
    package let benchmarkStats: StudioRuntimeTraffic?
    package let sidecars: StudioRuntimeSidecarPool?
    package let process: StudioRuntimeProcess?

    package var textModels: [StudioRuntimeModel] { models ?? [] }
    package var loadedTextModels: [StudioRuntimeModel] { textModels.filter(\.loaded) }
    package var loadedSidecars: [StudioRuntimeSidecar] { sidecars?.residents.filter(\.loaded) ?? [] }
    package var totalFailures: Int {
        let textFailures = textModels.reduce(0) { $0 + ($1.benchmarkStats?.failedRequests ?? 0) }
        let sidecarFailures = (sidecars?.residents ?? []).reduce(0) { $0 + $1.failedRequests }
        return max(benchmarkStats?.failedRequests ?? 0, textFailures) + sidecarFailures
    }
}

package struct StudioRuntimeAdmission: Decodable, Equatable {
    package let maxActiveRequests: Int
    package let activeRequests: Int
    package let queuedRequests: Int
    package let totalAdmittedRequests: Int
    package let totalCompletedRequests: Int
    package let totalCancelledRequests: Int
    package let admissionPaused: Bool?
    package let pressure: String?
}

package struct StudioRuntimeCapabilities: Decodable, Equatable {
    package let requestAdmission: StudioRuntimeCapability?
    package let chunkedPrefill: StudioRuntimeCapability?
    package let continuousBatching: StudioRuntimeCapability?
    package let prefixKVReuse: StudioRuntimeCapability?
    package let ssdKVCache: StudioRuntimeCapability?
}

package struct StudioRuntimeCapability: Decodable, Equatable {
    package let available: Bool
    package let enabled: Bool
    package let detail: String
}

package struct StudioRuntimeMemory: Decodable, Equatable {
    package let physicalBytes: UInt64?
    package let residentBytes: UInt64?
    package let currentBytes: UInt64?
    package let availableBytes: UInt64?
    package let ceilingBytes: UInt64?
    package let softLimitBytes: UInt64?
    package let hardLimitBytes: UInt64?
    package let activeRequests: Int?
    package let activeModelCount: Int?
    package let guardTier: String?
    package let pressure: String?
}

package struct StudioRuntimeProcess: Decodable, Equatable {
    package let processID: Int32?
    package let startedAt: Date?
    package let uptimeSeconds: Double?
    package let cpuPercent: Double?
    package let thermalState: String?
    package let lowPowerModeEnabled: Bool?
    package let metalDeviceName: String?
    package let metalCurrentAllocatedBytes: UInt64?
    package let metalRecommendedMaxWorkingSetBytes: UInt64?
    package let metalHasUnifiedMemory: Bool?
}

package struct StudioRuntimeModel: Decodable, Equatable, Identifiable {
    package let id: String
    package let category: String?
    package let engine: String?
    package let installPath: String?
    package let loaded: Bool
    package let ready: Bool?
    package let activeRequests: Int
    package let lastAccess: Date?
    package let lastError: String?
    package let pinned: Bool
    package let alias: String?
    package let ttlSeconds: Int?
    package let maxContextTokens: Int?
    package let maxTokens: Int?
    package let temperature: Double?
    package let topP: Double?
    package let minP: Double?
    package let engineOverride: String?
    package let kvCacheMode: String?
    package let prefixKVCache: StudioRuntimePrefixKV?
    package let continuousBatching: StudioRuntimeBatching?
    package let mtp: StudioRuntimeMTP?
    package let benchmarkStats: StudioRuntimeModelTraffic?

    package var state: String {
        guard loaded else { return "Unloaded" }
        if ready == false { return "Loading" }
        if activeRequests > 0 { return "Active" }
        return "Ready"
    }
}

package struct StudioRuntimePrefixKV: Decodable, Equatable {
    package let enabled: Bool
    package let entries: Int
    package let maxEntries: Int
    package let hits: Int
    package let misses: Int
    package let storedPrefixes: Int
    package let reusedTokens: Int
    package let storedTokens: Int
}

package struct StudioRuntimeBatching: Decodable, Equatable {
    package let enabled: Bool
    package let activeRows: Int
    package let queuedRows: Int
    package let batchedDecodeSteps: Int
    package let samePositionBatchedSteps: Int?
    package let variablePositionBatchedSteps: Int?
    package let singleDecodeSteps: Int
    package let totalBatchedRows: Int
    package let maxBatchSize: Int
}

package struct StudioRuntimeMTP: Decodable, Equatable {
    package let available: Bool
    package let enabled: Bool
    package let active: Bool
    package let reason: String?
    package let blockSize: Int
    package let threshold: Int
    package let rounds: Int
    package let draftedTokens: Int
    package let acceptedTokens: Int
    package let rejectedTokens: Int
}

package struct StudioRuntimeModelTraffic: Decodable, Equatable {
    package let completedRequests: Int
    package let failedRequests: Int
    package let generatedTokens: Int
    package let averageLoadSeconds: Double?
    package let averagePrefillSeconds: Double?
    package let averageDecodeSeconds: Double?
    package let averageTotalSeconds: Double?
    package let decodeTokensPerSecond: Double?
    package let recentDecodeTokensPerSecond: Double?
    package let lastCompletedAt: Date?
}

package struct StudioRuntimeSidecarPool: Decodable, Equatable {
    package let defaultIdleTTLSeconds: Int
    package let pressure: String
    package let loadedCount: Int
    package let activeRequests: Int
    package let queuedRequests: Int
    package let residents: [StudioRuntimeSidecar]
}

package struct StudioRuntimeSidecar: Decodable, Equatable, Identifiable {
    package let kind: String
    package let modelID: String?
    package let modelPath: String?
    package let variant: String?
    package let loaded: Bool
    package let ready: Bool?
    package let activeRequests: Int
    package let queuedRequests: Int
    package let loadedAt: Date?
    package let lastAccess: Date?
    package let lastEvictedAt: Date?
    package let lastEvictionReason: String?
    package let pinned: Bool
    package let ttlSeconds: Int
    package let loadCount: Int
    package let replacementCount: Int
    package let evictionCount: Int
    package let completedRequests: Int
    package let failedRequests: Int

    package var id: String { kind }
    package var displayModel: String { modelID ?? variant ?? modelPath ?? "Not selected" }
    package var state: String {
        guard loaded else { return "Unloaded" }
        if ready == false { return "Loading" }
        if activeRequests > 0 { return "Active" }
        if queuedRequests > 0 { return "Queued" }
        return "Ready"
    }
}

package struct StudioRuntimeCacheStats: Decodable, Equatable {
    package let available: Bool
    package let detail: String
    package let prefixKVReuse: StudioRuntimePrefixKVSummary?
    package let decodeBatching: StudioRuntimeBatchingSummary?
}

package struct StudioRuntimePrefixKVSummary: Decodable, Equatable {
    package let reportedModelCount: Int
    package let enabledModelCount: Int
    package let entries: Int
    package let maxEntries: Int
    package let hits: Int
    package let misses: Int
    package let storedPrefixes: Int
    package let reusedTokens: Int
    package let storedTokens: Int

    package var hitRate: Double? {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : nil
    }
}

package struct StudioRuntimeBatchingSummary: Decodable, Equatable {
    package let reportedModelCount: Int
    package let enabledModelCount: Int
    package let activeRows: Int
    package let queuedRows: Int
    package let batchedDecodeSteps: Int
    package let samePositionBatchedSteps: Int?
    package let variablePositionBatchedSteps: Int?
    package let singleDecodeSteps: Int
    package let totalBatchedRows: Int
    package let maxBatchSize: Int
}

/// Runtime names this payload `benchmarkStats` for compatibility. Studio presents it as live,
/// observed service traffic; it is not a synthetic benchmark.
package struct StudioRuntimeTraffic: Decodable, Equatable {
    package let available: Bool
    package let detail: String
    package let reportedModelCount: Int
    package let completedRequests: Int
    package let failedRequests: Int
    package let generatedTokens: Int
    package let averageLoadSeconds: Double?
    package let averagePrefillSeconds: Double?
    package let averageDecodeSeconds: Double?
    package let averageTotalSeconds: Double?
    package let decodeTokensPerSecond: Double?
}

package struct StudioAgentStatus: Decodable, Equatable {
    package let machine: StudioAgentMachine
    package let pi: StudioAgentPi
    package let provider: StudioAgentProvider
    package let recommendedModelID: String?
    package let models: [StudioAgentModel]
}

package struct StudioAgentMachine: Decodable, Equatable {
    package let processor: String
    package let unifiedMemoryGB: Int
    package let appleSiliconMac: Bool
    package let linux: Bool
}

package struct StudioAgentPi: Decodable, Equatable {
    package let installed: Bool
    package let managedInstall: Bool
    package let autoInstallSupported: Bool
    package let path: String?
    package let version: String?
}

package struct StudioAgentProvider: Decodable, Equatable {
    package let configured: Bool
    package let host: String?
    package let port: Int?
    package let modelID: String?
    package let updatedAt: Date?
    package let configurationPath: String
    package let extensionPath: String
}

package struct StudioAgentModel: Decodable, Equatable, Identifiable {
    package let id: String
    package let displayName: String
    package let summary: String
    package let minimumUnifiedMemoryGB: Int
    package let recommendedUnifiedMemoryGB: Int
    package let servingEngine: String
    package let startableByMereRun: Bool
    package let sourceConfigurationRequired: Bool
    package let installed: Bool
    package let reason: String?
}

package enum StudioServingSafety: Equatable {
    case loopback
    case protectedLAN
    case exposedWithoutAuthentication

    package static func evaluate(host: String, apiKey: String) -> StudioServingSafety {
        if isLoopback(host) { return .loopback }
        return apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? .exposedWithoutAuthentication
            : .protectedLAN
    }

    package static func isLoopback(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "localhost"
            || normalized == "::1"
            || normalized == "[::1]"
            || normalized.hasPrefix("127.")
    }
}

package struct StudioServiceActivity: Identifiable, Equatable {
    package enum Level: String, Equatable {
        case info
        case success
        case warning
        case error
    }

    package let id: UUID
    package let date: Date
    package let level: Level
    package let title: String
    package let detail: String?

    package init(
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

package enum StudioActivitySanitizer {
    package static func sanitize(_ text: String) -> String {
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

package enum StudioServiceActivityDiff {
    package static func events(
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

package enum StudioServingFormat {
    package static func bytes(_ value: UInt64?) -> String {
        guard let value else { return "Unavailable" }
        return ByteCountFormatter.string(fromByteCount: Int64(clamping: value), countStyle: .memory)
    }

    package static func duration(_ seconds: Double?) -> String {
        guard let seconds else { return "Unavailable" }
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        if seconds < 3_600 { return String(format: "%.0fm", seconds / 60) }
        return String(format: "%.1fh", seconds / 3_600)
    }

    package static func number(_ value: Double?, suffix: String = "") -> String {
        guard let value else { return "Unavailable" }
        return String(format: "%.1f%@", value, suffix)
    }
}
