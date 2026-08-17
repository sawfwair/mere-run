import Foundation
import MereRunRelayKit
import MereRunCore

struct APIServePreflightRequest: Codable, Equatable {
    let host: String
    let port: Int
    let engine: String
    let model: String?
    let lora: String?
    let apiKeyPresent: Bool
    let apiKeySource: String
    let rateLimitPerMinute: Int
    let maxActiveRequests: Int
    let memoryGuard: String
    let memoryGuardCustomCeilingGB: Double?
    let contextSize: Int
    let kvBits: Double?
    let kvQuantScheme: String?
    let kvGroupSize: Int?
    let quantizedKVStart: Int?

    enum CodingKeys: String, CodingKey {
        case host
        case port
        case engine
        case model
        case lora
        case apiKeyPresent = "api_key_present"
        case apiKeySource = "api_key_source"
        case rateLimitPerMinute = "rate_limit_per_minute"
        case maxActiveRequests = "max_active_requests"
        case memoryGuard = "memory_guard"
        case memoryGuardCustomCeilingGB = "memory_guard_custom_ceiling_gb"
        case contextSize = "context_size"
        case kvBits = "kv_bits"
        case kvQuantScheme = "kv_quant_scheme"
        case kvGroupSize = "kv_group_size"
        case quantizedKVStart = "quantized_kv_start"
    }
}

struct APIServePreflightResult: Codable, Equatable {
    let server: APIServeServerPreflightSummary
    let model: APIServeModelPreflightSummary
    let runtime: APIServeRuntimePreflightSummary
    let capabilities: APIServeCapabilitiesPreflightSummary
    let companionModelIDs: [String]
    let modelStore: APIServeModelStorePreflightSummary

    enum CodingKeys: String, CodingKey {
        case server
        case model
        case runtime
        case capabilities
        case companionModelIDs = "companion_model_ids"
        case modelStore = "model_store"
    }
}

struct APIServeServerPreflightSummary: Codable, Equatable {
    let host: String
    let port: Int
    let baseURL: String
    let loopback: Bool
    let requiresAPIKey: Bool
    let apiKeyPresent: Bool
    let apiKeySource: String
    let rateLimitPerMinute: Int
    let maxActiveRequests: Int

    enum CodingKeys: String, CodingKey {
        case host
        case port
        case baseURL = "base_url"
        case loopback
        case requiresAPIKey = "requires_api_key"
        case apiKeyPresent = "api_key_present"
        case apiKeySource = "api_key_source"
        case rateLimitPerMinute = "rate_limit_per_minute"
        case maxActiveRequests = "max_active_requests"
    }
}

struct APIServeModelPreflightSummary: Codable, Equatable {
    let requested: String
    let defaultModelID: String
    let kind: String
    let status: String
    let installed: Bool
    let path: String?
    let id: String?
    let category: String?
    let runtimeAutoDownloadAllowed: Bool
    let upstreamRepoID: String?
    let estimatedDownloadBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case requested
        case defaultModelID = "default_model_id"
        case kind
        case status
        case installed
        case path
        case id
        case category
        case runtimeAutoDownloadAllowed = "runtime_auto_download_allowed"
        case upstreamRepoID = "upstream_repo_id"
        case estimatedDownloadBytes = "estimated_download_bytes"
    }
}

struct APIServeRuntimePreflightSummary: Codable, Equatable {
    let engine: String
    let runtimeServingEngine: String
    let contextSize: Int
    let memoryGuard: String
    let memoryGuardCustomCeilingBytes: UInt64?
    let kvQuantization: APIServeKVQuantizationPreflightSummary

    enum CodingKeys: String, CodingKey {
        case engine
        case runtimeServingEngine = "runtime_serving_engine"
        case contextSize = "context_size"
        case memoryGuard = "memory_guard"
        case memoryGuardCustomCeilingBytes = "memory_guard_custom_ceiling_bytes"
        case kvQuantization = "kv_quantization"
    }
}

struct APIServeKVQuantizationPreflightSummary: Codable, Equatable {
    let enabled: Bool
    let bits: Double?
    let scheme: String
    let groupSize: Int
    let quantizedStart: Int

    enum CodingKeys: String, CodingKey {
        case enabled
        case bits
        case scheme
        case groupSize = "group_size"
        case quantizedStart = "quantized_start"
    }
}

struct APIServeCapabilitiesPreflightSummary: Codable, Equatable {
    let rawProxy: Bool
    let tools: Bool
    let toolChoice: Bool
    let developerRole: Bool
    let structuredOutputs: Bool
    let reasoningEffort: Bool
    let maxCompletionTokens: Bool
    let usageInStreaming: Bool
    let visionContentParts: Bool
    let strictMode: Bool
    let stopSequences: Bool
    let seed: Bool
    let penalties: Bool
    let logprobs: Bool
    let providerThinkingControls: Bool

    enum CodingKeys: String, CodingKey {
        case rawProxy = "raw_proxy"
        case tools
        case toolChoice = "tool_choice"
        case developerRole = "developer_role"
        case structuredOutputs = "structured_outputs"
        case reasoningEffort = "reasoning_effort"
        case maxCompletionTokens = "max_completion_tokens"
        case usageInStreaming = "usage_in_streaming"
        case visionContentParts = "vision_content_parts"
        case strictMode = "strict_mode"
        case stopSequences = "stop_sequences"
        case seed
        case penalties
        case logprobs
        case providerThinkingControls = "provider_thinking_controls"
    }
}

struct APIServeModelStorePreflightSummary: Codable, Equatable {
    let path: String
    let source: String
    let configuredPath: String?
    let fallbackToDefault: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case source
        case configuredPath = "configured_path"
        case fallbackToDefault = "fallback_to_default"
    }
}

typealias APIServePreflightEnvelope = StructuredRunEnvelope<
    APIServePreflightRequest,
    APIServePreflightResult
>

struct APIServePreflightAnalyzer {
    let command: APIServe
    let fileManager: FileManager
    let environment: [String: String]
    let now: () -> Date

    init(
        command: APIServe,
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        now: @escaping () -> Date = Date.init
    ) {
        self.command = command
        self.fileManager = fileManager
        self.environment = environment
        self.now = now
    }

    func envelope() -> APIServePreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        let apiKey = apiKeyPresence()
        let resolvedModelPath = resolvedModelPath(diagnostics: &diagnostics)
        let kvQuantization = kvQuantization(diagnostics: &diagnostics)
        let memoryPolicy = memoryPolicy(diagnostics: &diagnostics)
        appendStaticDiagnostics(
            apiKey: apiKey,
            resolvedModelPath: resolvedModelPath,
            diagnostics: &diagnostics
        )

        let server = serverSummary(apiKey: apiKey)
        let model = modelSummary(
            requested: command.model,
            resolvedModelPath: resolvedModelPath,
            diagnostics: &diagnostics
        )
        let runtime = runtimeSummary(
            kvQuantization: kvQuantization,
            memoryPolicy: memoryPolicy
        )
        let modelStore = MereRunModelPaths.modelStoreResolution()
        let result = APIServePreflightResult(
            server: server,
            model: model,
            runtime: runtime,
            capabilities: capabilitiesSummary(command.engine.openAICompatibility),
            companionModelIDs: APIServerContract.companionModelIDs(fileManager: fileManager),
            modelStore: APIServeModelStorePreflightSummary(
                path: modelStore.activeModelsDir.path,
                source: modelStore.source.rawValue,
                configuredPath: modelStore.configuredModelsDir?.path,
                fallbackToDefault: modelStore.isFallbackToDefault
            )
        )
        let status = StructuredRunOutput.status(for: diagnostics)

        return APIServePreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["api", "serve"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: fileManager.currentDirectoryPath,
            summary: summary(status: status, diagnostics: diagnostics, result: result),
            request: request(apiKey: apiKey),
            result: result,
            diagnostics: diagnostics,
            actions: actions(status: status, server: server)
        )
    }

    private func request(apiKey: APIKeyPresence) -> APIServePreflightRequest {
        APIServePreflightRequest(
            host: command.host,
            port: command.port,
            engine: command.engine.rawValue,
            model: command.model,
            lora: command.lora,
            apiKeyPresent: apiKey.present,
            apiKeySource: apiKey.source,
            rateLimitPerMinute: command.rateLimitPerMinute,
            maxActiveRequests: command.maxActiveRequests,
            memoryGuard: command.memoryGuard.rawValue,
            memoryGuardCustomCeilingGB: command.memoryGuardCustomCeilingGB,
            contextSize: command.contextSize,
            kvBits: command.kvBits,
            kvQuantScheme: command.kvQuantScheme,
            kvGroupSize: command.kvGroupSize,
            quantizedKVStart: command.quantizedKVStart
        )
    }

    private func serverSummary(apiKey: APIKeyPresence) -> APIServeServerPreflightSummary {
        let loopback = APIServe.isLoopbackHost(command.host)
        return APIServeServerPreflightSummary(
            host: command.host,
            port: command.port,
            baseURL: baseURLString(host: command.host, port: command.port),
            loopback: loopback,
            requiresAPIKey: !loopback,
            apiKeyPresent: apiKey.present,
            apiKeySource: apiKey.source,
            rateLimitPerMinute: command.rateLimitPerMinute,
            maxActiveRequests: command.maxActiveRequests
        )
    }

    private func modelSummary(
        requested rawRequested: String?,
        resolvedModelPath: String?,
        diagnostics: inout [PreflightDiagnostic]
    ) -> APIServeModelPreflightSummary {
        let defaultModelID = command.defaultRuntimeModelID(modelPath: resolvedModelPath)
        let requested = rawRequested?.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveRequested = (requested?.isEmpty == false ? requested : nil) ?? defaultModelID
        let localPath = localPathSummary(effectiveRequested, resolvedModelPath: resolvedModelPath)
        let spec = ManagedModelCatalog.spec(for: effectiveRequested)
        let runtimePath = spec?.managedRuntimeURL(fileManager: fileManager)
        let installed = localPath.exists || runtimePath != nil
        let status = modelStatus(
            requested: effectiveRequested,
            localPath: localPath,
            spec: spec,
            installed: installed
        )

        appendModelDiagnostics(
            requested: effectiveRequested,
            localPath: localPath,
            spec: spec,
            status: status,
            diagnostics: &diagnostics
        )

        return APIServeModelPreflightSummary(
            requested: effectiveRequested,
            defaultModelID: defaultModelID,
            kind: modelKind(localPath: localPath, spec: spec),
            status: status,
            installed: installed,
            path: localPath.exists ? localPath.path : runtimePath?.path,
            id: spec?.id,
            category: spec?.category.rawValue,
            runtimeAutoDownloadAllowed: spec?.runtimeAutoDownloadAllowed ?? false,
            upstreamRepoID: spec?.upstreamRepoId,
            estimatedDownloadBytes: spec?.estimatedDownloadBytes
        )
    }

    private func modelStatus(
        requested: String,
        localPath: LocalPathSummary,
        spec: ManagedModelSpec?,
        installed: Bool
    ) -> String {
        if installed { return "installed" }
        if localPath.looksLikePath { return "missing_local_path" }
        guard let spec else { return "runtime_identifier" }
        if spec.runtimeAutoDownloadAllowed { return "runtime_auto_download" }
        if spec.hasAnyManagedDownloadSource() { return "missing_pull_available" }
        if command.engine == .textChatKlein || requested == ModelResolver.ModelID.mebot.rawValue {
            return "missing_required"
        }
        return "missing"
    }

    private func modelKind(localPath: LocalPathSummary, spec: ManagedModelSpec?) -> String {
        if spec != nil { return "managed_model" }
        if localPath.looksLikePath { return "local_path" }
        return "runtime_identifier"
    }

    private func appendModelDiagnostics(
        requested: String,
        localPath: LocalPathSummary,
        spec: ManagedModelSpec?,
        status: String,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        if localPath.looksLikePath, !localPath.exists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_path_missing",
                    severity: .blocker,
                    title: "Model path missing",
                    message: "Model path not found: \(requested).",
                    locations: [.init(kind: "file", path: localPath.path)]
                )
            )
        }
        if status == "missing_required" {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_required_missing",
                    severity: .blocker,
                    title: "Required model missing",
                    message: "Engine \(command.engine.rawValue) requires an installed local model for \(requested)."
                )
            )
        } else if status == "missing_pull_available", let spec {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_missing_pull_available",
                    severity: .warning,
                    title: "Model is not installed",
                    message: "Pull \(spec.id) before serving if you do not want the runtime to fail later.",
                    suggestedActionIDs: ["pull-model"]
                )
            )
        } else if status == "runtime_auto_download" {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_runtime_auto_download",
                    severity: .note,
                    title: "Runtime may download model assets",
                    message: "The selected engine can resolve or download \(requested) when it first loads."
                )
            )
        }
    }

    private func runtimeSummary(
        kvQuantization: Gemma4KVCacheQuantization?,
        memoryPolicy: RuntimeMemoryPressurePolicy?
    ) -> APIServeRuntimePreflightSummary {
        let kv = kvQuantization ?? Gemma4KVCacheQuantization()
        return APIServeRuntimePreflightSummary(
            engine: command.engine.rawValue,
            runtimeServingEngine: command.engine.runtimeServingEngine.rawValue,
            contextSize: command.contextSize,
            memoryGuard: command.memoryGuard.rawValue,
            memoryGuardCustomCeilingBytes: memoryPolicy?.customCeilingBytes,
            kvQuantization: APIServeKVQuantizationPreflightSummary(
                enabled: kv.isEnabled,
                bits: kv.bits,
                scheme: kv.scheme.rawValue,
                groupSize: kv.groupSize,
                quantizedStart: kv.quantizedStart
            )
        )
    }

    private func capabilitiesSummary(_ capabilities: APIEngineCapabilities) -> APIServeCapabilitiesPreflightSummary {
        APIServeCapabilitiesPreflightSummary(
            rawProxy: capabilities.supportsRawProxy,
            tools: capabilities.supportsTools,
            toolChoice: capabilities.supportsToolChoice,
            developerRole: capabilities.supportsDeveloperRole,
            structuredOutputs: capabilities.supportsStructuredOutputs,
            reasoningEffort: capabilities.supportsReasoningEffort,
            maxCompletionTokens: capabilities.supportsMaxCompletionTokens,
            usageInStreaming: capabilities.supportsUsageInStreaming,
            visionContentParts: capabilities.supportsVisionContentParts,
            strictMode: capabilities.supportsStrictMode,
            stopSequences: capabilities.supportsStopSequences,
            seed: capabilities.supportsSeed,
            penalties: capabilities.supportsPenalties,
            logprobs: capabilities.supportsLogprobs,
            providerThinkingControls: capabilities.supportsProviderThinkingControls
        )
    }

    private func actions(status: StructuredRunStatus, server: APIServeServerPreflightSummary) -> [DeclarativeAction] {
        let blocked = status == .blocked
        var actions = [
            DeclarativeAction(
                id: "start-api-server",
                label: "Start API server",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: serveArgv(redactSecrets: true),
                    cwd: fileManager.currentDirectoryPath,
                    commandPath: ["api", "serve"]
                ),
                requires: ["preflight.passed"]
            ),
            DeclarativeAction(
                id: "check-status",
                label: "Check status",
                kind: .command,
                style: .secondary,
                command: DeclarativeCommand(
                    argv: ["mere.run", "status", "--host", command.host, "--port", String(command.port), "--json"],
                    cwd: fileManager.currentDirectoryPath,
                    commandPath: ["status"]
                )
            ),
            DeclarativeAction(
                id: "copy-base-url",
                label: "Copy base URL",
                kind: .copyText,
                style: .link,
                text: server.baseURL
            ),
            DeclarativeAction(
                id: "open-model-store",
                label: "Open model store",
                kind: .openDirectory,
                style: .link,
                enabled: fileManager.fileExists(atPath: MereRunModelPaths.modelsDir.path),
                path: MereRunModelPaths.modelsDir.path
            ),
        ]

        if let spec = ManagedModelCatalog.spec(for: command.model ?? command.defaultRuntimeModelID(modelPath: nil)),
           spec.hasAnyManagedDownloadSource() {
            actions.append(
                DeclarativeAction(
                    id: "pull-model",
                    label: "Pull model",
                    kind: .command,
                    style: .secondary,
                    command: DeclarativeCommand(
                        argv: ["mere.run", "model", "pull", spec.id],
                        cwd: fileManager.currentDirectoryPath,
                        commandPath: ["model", "pull"]
                    )
                )
            )
        }
        return actions
    }

    private func summary(
        status: StructuredRunStatus,
        diagnostics: [PreflightDiagnostic],
        result: APIServePreflightResult
    ) -> String {
        let blockerCount = diagnostics.filter { $0.severity == .blocker }.count
        let warningCount = diagnostics.filter { $0.severity == .warning }.count
        switch status {
        case .blocked:
            return "API server preflight blocked for \(result.server.baseURL): \(blockerCount) blocker(s), \(warningCount) warning(s)."
        case .warning:
            return "API server preflight ready with \(warningCount) warning(s) for \(result.server.baseURL)."
        default:
            return "API server preflight ready for \(result.server.baseURL)."
        }
    }

    private func appendStaticDiagnostics(
        apiKey: APIKeyPresence,
        resolvedModelPath: String?,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        if command.host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "host_empty",
                    severity: .blocker,
                    title: "Host is empty",
                    message: "Provide a host such as 127.0.0.1 or 0.0.0.0."
                )
            )
        }
        if !(1...65_535).contains(command.port) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "port_out_of_range",
                    severity: .blocker,
                    title: "Port is out of range",
                    message: "--port must be between 1 and 65535."
                )
            )
        }
        if command.rateLimitPerMinute <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "rate_limit_invalid",
                    severity: .blocker,
                    title: "Rate limit is invalid",
                    message: "--rate-limit-per-minute must be greater than zero."
                )
            )
        }
        if command.maxActiveRequests <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "max_active_requests_invalid",
                    severity: .blocker,
                    title: "Max active requests is invalid",
                    message: "--max-active-requests must be greater than zero."
                )
            )
        }
        if let ceiling = command.memoryGuardCustomCeilingGB {
            if command.memoryGuard != .custom {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "memory_guard_custom_ceiling_without_custom_guard",
                        severity: .blocker,
                        title: "Custom memory ceiling requires custom guard",
                        message: "--memory-guard-custom-ceiling-gb requires --memory-guard custom."
                    )
                )
            }
            if !ceiling.isFinite || ceiling <= 0 {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "memory_guard_custom_ceiling_invalid",
                        severity: .blocker,
                        title: "Custom memory ceiling is invalid",
                        message: "--memory-guard-custom-ceiling-gb must be greater than zero."
                    )
                )
            }
        } else if command.memoryGuard == .custom {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "memory_guard_custom_ceiling_required",
                    severity: .blocker,
                    title: "Custom memory ceiling required",
                    message: "--memory-guard custom requires --memory-guard-custom-ceiling-gb."
                )
            )
        }
        if !(1...Int(Int32.max)).contains(command.contextSize) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "context_size_invalid",
                    severity: .blocker,
                    title: "Context size is invalid",
                    message: "--context-size must be between 1 and \(Int(Int32.max))."
                )
            )
        }
        if !APIServe.isLoopbackHost(command.host), !apiKey.present {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "api_key_required_for_non_loopback",
                    severity: .blocker,
                    title: "API key required",
                    message: "Binding to non-loopback hosts requires --api-key or MERERUN_API_KEY.",
                    suggestedActionIDs: ["start-api-server"]
                )
            )
        }
        if command.lora != nil {
            do {
                let resolved = try command.resolveLoraPath(
                    modelPath: resolvedModelPath,
                    fileManager: fileManager
                )
                guard let resolved else { return }
                let url = URL(fileURLWithPath: resolved).standardizedFileURL
                if fileManager.fileExists(atPath: url.path) { return }
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "lora_missing",
                        severity: .blocker,
                        title: "LoRA file missing",
                        message: "LoRA file not found: \(url.path).",
                        locations: [.init(kind: "file", path: url.path)]
                    )
                )
            } catch {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "lora_missing",
                        severity: .blocker,
                        title: "LoRA adapter unavailable",
                        message: error.localizedDescription
                    )
                )
            }
        }
    }

    private func resolvedModelPath(diagnostics: inout [PreflightDiagnostic]) -> String? {
        do {
            return try command.resolveModelPath()
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_resolution_failed",
                    severity: .blocker,
                    title: "Model resolution failed",
                    message: error.localizedDescription
                )
            )
            return nil
        }
    }

    private func kvQuantization(diagnostics: inout [PreflightDiagnostic]) -> Gemma4KVCacheQuantization? {
        do {
            return try command.resolveGemma4KVCacheQuantization()
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "kv_quantization_invalid",
                    severity: .blocker,
                    title: "KV quantization is invalid",
                    message: error.localizedDescription
                )
            )
            return nil
        }
    }

    private func memoryPolicy(diagnostics: inout [PreflightDiagnostic]) -> RuntimeMemoryPressurePolicy? {
        do {
            return try command.resolveMemoryPressurePolicy()
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "memory_guard_invalid",
                    severity: .blocker,
                    title: "Memory guard is invalid",
                    message: error.localizedDescription
                )
            )
            return nil
        }
    }

    private func localPathSummary(
        _ requested: String,
        resolvedModelPath: String?
    ) -> LocalPathSummary {
        let candidate = resolvedModelPath ?? requested
        let looksLikePath = candidate.hasPrefix("/")
            || candidate.hasPrefix("./")
            || candidate.hasPrefix("../")
            || candidate.hasPrefix("~")
        let url = URL(fileURLWithPath: candidate).standardizedFileURL
        var isDirectory: ObjCBool = false
        let exists = looksLikePath && fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return LocalPathSummary(
            path: url.path,
            looksLikePath: looksLikePath,
            exists: exists,
            isDirectory: exists && isDirectory.boolValue
        )
    }

    private func apiKeyPresence() -> APIKeyPresence {
        if let apiKey = command.apiKey?.trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return APIKeyPresence(present: true, source: "argument")
        }
        if let apiKey = environment[APIServe.apiKeyEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return APIKeyPresence(present: true, source: "environment")
        }
        return APIKeyPresence(present: false, source: "none")
    }

    private func serveArgv(redactSecrets: Bool) -> [String] {
        var args = [
            "mere.run",
            "api",
            "serve",
            "--host",
            command.host,
            "--port",
            String(command.port),
            "--engine",
            command.engine.rawValue,
        ]
        if let model = command.model {
            args += ["--model", model]
        }
        if let lora = command.lora {
            args += ["--lora", lora]
        }
        if !redactSecrets, let apiKey = command.apiKey {
            args += ["--api-key", apiKey]
        }
        args += [
            "--rate-limit-per-minute",
            String(command.rateLimitPerMinute),
            "--max-active-requests",
            String(command.maxActiveRequests),
            "--memory-guard",
            command.memoryGuard.rawValue,
            "--context-size",
            String(command.contextSize),
        ]
        if let memoryGuardCustomCeilingGB = command.memoryGuardCustomCeilingGB {
            args += ["--memory-guard-custom-ceiling-gb", String(memoryGuardCustomCeilingGB)]
        }
        if let kvBits = command.kvBits {
            args += ["--kv-bits", String(kvBits)]
        }
        if let kvQuantScheme = command.kvQuantScheme {
            args += ["--kv-quant-scheme", kvQuantScheme]
        }
        if let kvGroupSize = command.kvGroupSize {
            args += ["--kv-group-size", String(kvGroupSize)]
        }
        if let quantizedKVStart = command.quantizedKVStart {
            args += ["--quantized-kv-start", String(quantizedKVStart)]
        }
        return args
    }

    private func baseURLString(host: String, port: Int) -> String {
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let urlHost: String
        if trimmedHost.contains(":") && !trimmedHost.hasPrefix("[") {
            urlHost = "[\(trimmedHost)]"
        } else {
            urlHost = trimmedHost.isEmpty ? "127.0.0.1" : trimmedHost
        }
        return "http://\(urlHost):\(port)"
    }

    private struct APIKeyPresence {
        let present: Bool
        let source: String
    }

    private struct LocalPathSummary {
        let path: String
        let looksLikePath: Bool
        let exists: Bool
        let isDirectory: Bool
    }
}
