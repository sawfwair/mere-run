import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import MereRunCore

struct Status: AsyncParsableCommand {
    private static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Show local server, loaded model, and installed model status.",
        discussion: """
        Prints a quick snapshot of the active local setup.

        It probes the configured API server's /health endpoint, reads /v1/models
        when available, and summarizes the active model store plus installed
        managed models.

        Examples:
          mere.run status
          mere.run status --host 127.0.0.1 --port 11434
          mere.run status --json
        """
    )

    @Option(name: [.long], help: "Local API host to check.")
    var host: String = "127.0.0.1"

    @Option(name: [.long], help: "Local API port to check.")
    var port: Int = 8080

    @Option(name: [.long], help: "Bearer token for /v1/models. Also read from MERERUN_API_KEY.")
    var apiKey: String?

    @Option(name: [.long], help: "Network probe timeout in seconds.")
    var timeoutSeconds: Double = 1.0

    @Flag(name: [.long], help: "Emit the snapshot as JSON.")
    var json: Bool = false

    func run() async throws {
        let snapshot = await StatusSnapshotBuilder(
            host: host,
            port: port,
            apiKey: resolvedAPIKey(),
            timeoutSeconds: timeoutSeconds
        ).snapshot()

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(snapshot)
            guard let output = String(data: data, encoding: .utf8) else {
                throw ValidationError("Could not encode status snapshot as UTF-8.")
            }
            print(output)
        } else {
            print(StatusFormatter.text(snapshot))
        }
    }

    private func resolvedAPIKey() -> String? {
        if let apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !apiKey.isEmpty {
            return apiKey
        }
        if let apiKey = ProcessInfo.processInfo.environment[Self.apiKeyEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        return nil
    }
}

struct StatusSnapshot: Codable, Equatable {
    let server: StatusServerSnapshot
    let modelStore: StatusModelStoreSnapshot
    let knownModelCount: Int
    let installedModels: [StatusInstalledModelSnapshot]
}

struct StatusServerSnapshot: Codable, Equatable {
    let url: String
    let health: String
    let detail: String?
    let loadedModels: [String]
    let modelsDetail: String?
    let runtime: RuntimeModelPoolStatus?
    let runtimeDetail: String?
}

struct StatusModelStoreSnapshot: Codable, Equatable {
    let path: String
    let source: String
    let configuredPath: String?
    let isFallbackToDefault: Bool
}

struct StatusInstalledModelSnapshot: Codable, Equatable {
    let id: String
    let category: String
    let size: String
}

struct StatusSnapshotBuilder {
    let host: String
    let port: Int
    let apiKey: String?
    let timeoutSeconds: Double

    func snapshot() async -> StatusSnapshot {
        let inventoryRows = ModelInventory.rows()
        let installedModels = inventoryRows
            .filter(\.isInstalled)
            .map {
                StatusInstalledModelSnapshot(
                    id: $0.id,
                    category: $0.category,
                    size: $0.size
                )
            }
        let modelStore = MereRunModelPaths.modelStoreResolution()
        let server = await serverSnapshot()

        return StatusSnapshot(
            server: server,
            modelStore: StatusModelStoreSnapshot(
                path: modelStore.activeModelsDir.path,
                source: modelStore.source.rawValue,
                configuredPath: modelStore.configuredModelsDir?.path,
                isFallbackToDefault: modelStore.isFallbackToDefault
            ),
            knownModelCount: inventoryRows.count,
            installedModels: installedModels
        )
    }

    private func serverSnapshot() async -> StatusServerSnapshot {
        let baseURL = serverBaseURLString(host: host, port: port)
        let health = await probeHealth(baseURL: baseURL)
        guard health.isUp else {
            return StatusServerSnapshot(
                url: baseURL,
                health: "down",
                detail: health.detail,
                loadedModels: [],
                modelsDetail: nil,
                runtime: nil,
                runtimeDetail: nil
            )
        }

        let models = await probeModels(baseURL: baseURL)
        let runtime = await probeRuntimeStatus(baseURL: baseURL)
        return StatusServerSnapshot(
            url: baseURL,
            health: "up",
            detail: health.detail,
            loadedModels: models.loadedModels,
            modelsDetail: models.detail,
            runtime: runtime.snapshot,
            runtimeDetail: runtime.detail
        )
    }

    private func probeHealth(baseURL: String) async -> (isUp: Bool, detail: String?) {
        guard let url = URL(string: "\(baseURL)/health") else {
            return (false, "invalid server URL")
        }

        do {
            let (data, response) = try await data(from: url, authorize: false)
            guard let http = response as? HTTPURLResponse else {
                return (false, "health returned a non-HTTP response")
            }
            guard http.statusCode == 200 else {
                return (false, "health returned HTTP \(http.statusCode)")
            }
            guard let health = try? JSONDecoder().decode(APIHealthStatus.self, from: data),
                  health.status == "ok" else {
                return (true, "health responded with an unexpected payload")
            }
            return (true, nil)
        } catch {
            return (false, StatusSnapshotBuilder.shortNetworkError(error))
        }
    }

    private func probeModels(baseURL: String) async -> (loadedModels: [String], detail: String?) {
        guard let url = URL(string: "\(baseURL)/v1/models") else {
            return ([], "invalid models URL")
        }

        do {
            let (data, response) = try await data(from: url, authorize: true)
            guard let http = response as? HTTPURLResponse else {
                return ([], "models returned a non-HTTP response")
            }
            switch http.statusCode {
            case 200:
                let models = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
                return (models.data.map(\.id), nil)
            case 401, 403:
                return ([], "requires API key")
            default:
                return ([], "models returned HTTP \(http.statusCode)")
            }
        } catch {
            return ([], StatusSnapshotBuilder.shortNetworkError(error))
        }
    }

    private func probeRuntimeStatus(baseURL: String) async -> (snapshot: RuntimeModelPoolStatus?, detail: String?) {
        guard let url = URL(string: "\(baseURL)/runtime/status") else {
            return (nil, "invalid runtime status URL")
        }

        do {
            let (data, response) = try await data(from: url, authorize: true)
            guard let http = response as? HTTPURLResponse else {
                return (nil, "runtime status returned a non-HTTP response")
            }
            switch http.statusCode {
            case 200:
                return (try JSONDecoder().decode(RuntimeModelPoolStatus.self, from: data), nil)
            case 401, 403:
                return (nil, "requires API key")
            case 404:
                return (nil, "runtime status endpoint is unavailable")
            default:
                return (nil, "runtime status returned HTTP \(http.statusCode)")
            }
        } catch {
            return (nil, StatusSnapshotBuilder.shortNetworkError(error))
        }
    }

    private func data(from url: URL, authorize: Bool) async throws -> (Data, URLResponse) {
        var request = URLRequest(url: url, timeoutInterval: max(0.1, timeoutSeconds))
        if authorize, let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        return try await URLSession.shared.data(for: request)
    }

    private func serverBaseURLString(host: String, port: Int) -> String {
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let urlHost: String
        if trimmedHost.contains(":") && !trimmedHost.hasPrefix("[") {
            urlHost = "[\(trimmedHost)]"
        } else {
            urlHost = trimmedHost.isEmpty ? "127.0.0.1" : trimmedHost
        }
        return "http://\(urlHost):\(port)"
    }

    private static func shortNetworkError(_ error: Error) -> String {
        guard let urlError = error as? URLError else {
            return error.localizedDescription
        }

        switch urlError.code {
        case .cannotConnectToHost, .networkConnectionLost, .notConnectedToInternet:
            return "not reachable"
        case .timedOut:
            return "timed out"
        default:
            return urlError.localizedDescription
        }
    }
}

enum StatusFormatter {
    static func text(_ snapshot: StatusSnapshot) -> String {
        var lines: [String] = []
        lines.append("mere.run status")

        var serverLine = "  server: \(snapshot.server.health) (\(snapshot.server.url))"
        if let detail = snapshot.server.detail {
            serverLine += " - \(detail)"
        }
        lines.append(serverLine)

        if snapshot.server.health == "up" {
            if let runtime = snapshot.server.runtime {
                let activeModels = runtime.models.filter(\.loaded).map(\.id)
                if activeModels.isEmpty {
                    lines.append("  loaded models: none reported")
                } else {
                    lines.append("  loaded models: \(activeModels.joined(separator: ", "))")
                }
                lines.append("  active requests: \(runtime.activeRequests)")
                if let admission = runtime.admission {
                    lines.append("  request admission: \(admission.activeRequests)/\(admission.maxActiveRequests) active, \(admission.queuedRequests) queued")
                }
                lines.append("  continuous batching: \(capabilityText(runtime.capabilities.continuousBatching))")
                lines.append("  prefix KV reuse: \(capabilityText(runtime.capabilities.prefixKVReuse))")
                if let cacheStats = cacheStatsText(runtime.cacheStats) {
                    lines.append("  cache stats: \(cacheStats)")
                }
                if let benchmarkStats = runtime.benchmarkStats.flatMap(benchmarkStatsText) {
                    lines.append("  benchmark stats: \(benchmarkStats)")
                }
                lines.append("  memory: \(memoryText(runtime.memory))")
                for model in runtime.models {
                    guard let prefixKVCache = model.prefixKVCache else { continue }
                    lines.append("    \(model.id) prefix KV: \(prefixKVCache.entries)/\(prefixKVCache.maxEntries) entries, \(prefixKVCache.hits) hits, \(prefixKVCache.reusedTokens) reused tokens")
                }
                for model in runtime.models {
                    guard let batching = model.continuousBatching else { continue }
                    lines.append("    \(model.id) batching: \(decodeBatchingText(batching))")
                }
                for model in runtime.models {
                    guard let mtp = model.mtp else { continue }
                    lines.append("    \(model.id) MTP: \(mtpText(mtp))")
                }
            } else if !snapshot.server.loadedModels.isEmpty {
                lines.append("  loaded models: \(snapshot.server.loadedModels.joined(separator: ", "))")
            } else if let modelsDetail = snapshot.server.modelsDetail {
                lines.append("  loaded models: unavailable (\(modelsDetail))")
            } else {
                lines.append("  loaded models: none reported")
            }
        } else {
            lines.append("  loaded models: unavailable")
        }

        lines.append("  model store: \(modelStoreText(snapshot.modelStore))")
        if let runtime = snapshot.server.runtime {
            lines.append("  runtime settings: \(runtime.settingsPath)")
        } else if let detail = snapshot.server.runtimeDetail, snapshot.server.health == "up" {
            lines.append("  runtime settings: unavailable (\(detail))")
        }
        lines.append("  installed models: \(snapshot.installedModels.count)/\(snapshot.knownModelCount)")

        if snapshot.installedModels.isEmpty {
            lines.append("    none")
        } else {
            for model in snapshot.installedModels {
                lines.append("    - \(model.id) (\(model.category), \(model.size))")
            }
        }

        return lines.joined(separator: "\n")
    }

    private static func modelStoreText(_ modelStore: StatusModelStoreSnapshot) -> String {
        let source = modelStore.source
        guard modelStore.isFallbackToDefault, let configuredPath = modelStore.configuredPath else {
            return "\(modelStore.path) (source: \(source))"
        }
        return "\(modelStore.path) (source: \(source), fallback from unreadable \(configuredPath))"
    }

    private static func memoryText(_ memory: RuntimeMemorySnapshot) -> String {
        let physical = ByteCountFormatter.string(fromByteCount: Int64(memory.physicalBytes), countStyle: .memory)
        return "\(memory.pressure) pressure, \(memory.activeModelCount) active model(s), \(physical) physical"
    }

    private static func capabilityText(_ capability: RuntimeCapabilityStatus) -> String {
        if capability.enabled {
            return "enabled"
        }
        return capability.available ? "available but disabled" : "unavailable"
    }

    private static func cacheStatsText(_ stats: RuntimeCacheStatsSummary) -> String? {
        let prefix = stats.prefixKVReuse
        let batching = stats.decodeBatching
        guard prefix.reportedModelCount > 0 || batching.reportedModelCount > 0 else {
            return nil
        }

        var parts: [String] = []
        if prefix.reportedModelCount > 0 {
            parts.append(
                "prefix \(prefix.entries)/\(prefix.maxEntries) entries, "
                    + "\(prefix.hits) hits, \(prefix.reusedTokens) reused tokens"
            )
        }
        if batching.reportedModelCount > 0 {
            parts.append(
                "decode batching \(batching.batchedDecodeSteps) batched steps, "
                    + "\(batching.variablePositionBatchedSteps) variable-position, "
                    + "max batch \(batching.maxBatchSize)"
            )
        }
        return parts.joined(separator: "; ")
    }

    private static func decodeBatchingText(_ stats: RuntimeDecodeBatchingStats) -> String {
        "\(stats.batchedDecodeSteps) batched steps, "
            + "\(stats.samePositionBatchedSteps) same-position, "
            + "\(stats.variablePositionBatchedSteps) variable-position, "
            + "max batch \(stats.maxBatchSize), \(stats.queuedRows) queued rows"
    }

    private static func mtpText(_ stats: Gemma4MTPStats) -> String {
        let state: String
        if stats.active {
            state = "active"
        } else if stats.available {
            state = "available"
        } else if stats.enabled {
            state = "enabled, assistant unavailable"
        } else {
            state = "disabled"
        }
        var parts = [
            state,
            "block \(stats.blockSize)",
            "\(stats.acceptedTokens)/\(stats.draftedTokens) accepted",
        ]
        if let reason = stats.reason, !reason.isEmpty {
            parts.append(reason)
        }
        return parts.joined(separator: ", ")
    }

    private static func benchmarkStatsText(_ stats: RuntimeBenchmarkStatsSummary) -> String? {
        guard stats.completedRequests > 0 || stats.failedRequests > 0 else {
            return nil
        }

        var parts = [
            "\(stats.completedRequests) completed",
            "\(stats.failedRequests) failed",
            "\(stats.generatedTokens) tokens",
        ]
        if let averagePrefillSeconds = stats.averagePrefillSeconds {
            parts.append(String(format: "prefill avg %.2fs", averagePrefillSeconds))
        }
        if let averageDecodeSeconds = stats.averageDecodeSeconds {
            parts.append(String(format: "decode avg %.2fs", averageDecodeSeconds))
        }
        if let decodeTokensPerSecond = stats.decodeTokensPerSecond {
            parts.append(String(format: "decode %.2f tok/s", decodeTokensPerSecond))
        }
        return parts.joined(separator: ", ")
    }
}
