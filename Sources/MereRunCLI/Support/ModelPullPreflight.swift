import Foundation
import MereRunRelayKit
import MereRunCore

struct ModelPullPreflightInput {
    let target: String?
    let all: Bool
    let force: Bool
    let allowUnsupported: Bool
    let acceptUsageTerms: Bool
    let cacheDirectory: String?
    let pullArgv: [String]
    let cwd: String
}

struct ModelPullPreflightRequest: Codable, Equatable {
    let target: String?
    let all: Bool
    let force: Bool
    let allowUnsupported: Bool
    let acceptUsageTerms: Bool
    let cacheDirectory: String?

    enum CodingKeys: String, CodingKey {
        case target
        case all
        case force
        case allowUnsupported = "allow_unsupported"
        case acceptUsageTerms = "accept_usage_terms"
        case cacheDirectory = "cache_directory"
    }
}

struct ModelPullPreflightResult: Codable, Equatable {
    let mode: String
    let modelStore: ModelPullStoragePreflightSummary
    let hubCache: ModelPullStoragePreflightSummary
    let models: [ModelPullModelPreflightSummary]
    let selectedModelCount: Int
    let willDownloadCount: Int
    let estimatedDownloadBytes: Int64?
    let estimatedRequiredBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case mode
        case modelStore = "model_store"
        case hubCache = "hub_cache"
        case models
        case selectedModelCount = "selected_model_count"
        case willDownloadCount = "will_download_count"
        case estimatedDownloadBytes = "estimated_download_bytes"
        case estimatedRequiredBytes = "estimated_required_bytes"
    }
}

struct ModelPullStoragePreflightSummary: Codable, Equatable {
    let path: String
    let availableBytes: Int64?
    let requiredBytes: Int64?
    let headroomAfterBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case path
        case availableBytes = "available_bytes"
        case requiredBytes = "required_bytes"
        case headroomAfterBytes = "headroom_after_bytes"
    }
}

struct ModelPullModelPreflightSummary: Codable, Equatable {
    let id: String
    let title: String?
    let category: String
    let status: String
    let selected: Bool
    let supported: Bool
    let supportReasons: [String]
    let hasDownloadSource: Bool
    let installed: Bool
    let runtimeReady: Bool
    let conversionRequired: Bool
    let willDownload: Bool
    let force: Bool
    let installPath: String
    let runtimePath: String?
    let upstreamRepoID: String?
    let hubRepoIDs: [String]
    let estimatedDownloadBytes: Int64?
    let estimatedRequiredBytes: Int64?
    let companionModelIDs: [String]
    let usageTerms: [ManagedModelUsageTerm]
    let usageTermsAcknowledged: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case category
        case status
        case selected
        case supported
        case supportReasons = "support_reasons"
        case hasDownloadSource = "has_download_source"
        case installed
        case runtimeReady = "runtime_ready"
        case conversionRequired = "conversion_required"
        case willDownload = "will_download"
        case force
        case installPath = "install_path"
        case runtimePath = "runtime_path"
        case upstreamRepoID = "upstream_repo_id"
        case hubRepoIDs = "hub_repo_ids"
        case estimatedDownloadBytes = "estimated_download_bytes"
        case estimatedRequiredBytes = "estimated_required_bytes"
        case companionModelIDs = "companion_model_ids"
        case usageTerms = "usage_terms"
        case usageTermsAcknowledged = "usage_terms_acknowledged"
    }
}

typealias ModelPullPreflightEnvelope = StructuredRunEnvelope<
    ModelPullPreflightRequest,
    ModelPullPreflightResult
>

struct ModelPullPreflightAnalyzer {
    let input: ModelPullPreflightInput
    let fileManager: FileManager
    let hubCacheURL: URL?
    let modelStoreURL: URL?
    let estimatedDownloadBytesOverrides: [String: Int64]
    let modelLocations: ModelLocationSnapshot?
    let diskAvailableBytes: (URL) -> Int64?
    let now: () -> Date

    init(
        input: ModelPullPreflightInput,
        fileManager: FileManager = .default,
        hubCacheURL: URL? = nil,
        modelStoreURL: URL? = nil,
        estimatedDownloadBytesOverrides: [String: Int64] = [:],
        modelLocations: ModelLocationSnapshot? = nil,
        diskAvailableBytes: ((URL) -> Int64?)? = nil,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.hubCacheURL = hubCacheURL
        self.modelStoreURL = modelStoreURL
        self.estimatedDownloadBytesOverrides = estimatedDownloadBytesOverrides
        self.modelLocations = modelLocations
        self.diskAvailableBytes = diskAvailableBytes ?? {
            ModelPullDiskPreflight.availableBytes(onFileSystemContaining: $0, fileManager: fileManager)
        }
        self.now = now
    }

    func envelope() -> ModelPullPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        let specs = selectedSpecs(diagnostics: &diagnostics)
        let hubCache = resolvedHubCache(diagnostics: &diagnostics)
        if input.cacheDirectory != nil {
            diagnostics.append(PreflightDiagnostic(
                id: "external_hub_cache_availability",
                severity: .note,
                title: "External model storage",
                message: "Model payloads will resolve through \(hubCache.path); disconnecting that volume makes them unavailable."
            ))
        }
        let modelStore = (modelStoreURL ?? MereRunModelPaths.modelsDir).standardizedFileURL
        let models = specs.map {
            modelSummary(
                for: $0,
                modelStore: modelStore,
                hubCache: hubCache,
                diagnostics: &diagnostics
            )
        }
        appendAggregateDiskDiagnostics(
            models: models,
            hubCache: hubCache,
            modelStore: modelStore,
            diagnostics: &diagnostics
        )
        let result = result(models: models, hubCache: hubCache, modelStore: modelStore)
        let status = StructuredRunOutput.status(for: diagnostics)

        return ModelPullPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["model", "pull"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: input.cwd,
            summary: summary(status: status, diagnostics: diagnostics, result: result),
            request: ModelPullPreflightRequest(
                target: input.target,
                all: input.all,
                force: input.force,
                allowUnsupported: input.allowUnsupported,
                acceptUsageTerms: input.acceptUsageTerms,
                cacheDirectory: input.cacheDirectory
            ),
            result: result,
            diagnostics: diagnostics,
            actions: actions(status: status, result: result)
        )
    }

    private func selectedSpecs(diagnostics: inout [PreflightDiagnostic]) -> [ManagedModelSpec] {
        if input.all {
            return ManagedModelCatalog.allSpecs
        }

        guard let target = input.target?.trimmingCharacters(in: .whitespacesAndNewlines),
              !target.isEmpty else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_id_required",
                    severity: .blocker,
                    title: "Model id required",
                    message: "Provide a model id or use --all."
                )
            )
            return []
        }

        guard let spec = ManagedModelCatalog.spec(for: target) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_unknown",
                    severity: .blocker,
                    title: "Unknown model",
                    message: "Unknown canonical model id: \(target)."
                )
            )
            return []
        }
        return [spec]
    }

    private func resolvedHubCache(diagnostics: inout [PreflightDiagnostic]) -> URL {
        do {
            return try HubSnapshot.resolvedDownloadBase(requested: hubCacheURL, fileManager: fileManager).standardizedFileURL
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "hub_cache_unavailable",
                    severity: .blocker,
                    title: "Hub cache unavailable",
                    message: error.localizedDescription
                )
            )
            return (hubCacheURL ?? URL(fileURLWithPath: "/")).standardizedFileURL
        }
    }

    private func modelSummary(
        for spec: ManagedModelSpec,
        modelStore: URL,
        hubCache: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> ModelPullModelPreflightSummary {
        let support = ManagedModelCapabilityCatalog.support(for: spec)
        let hasSource = spec.hasAnyManagedDownloadSource()
        let installPath = modelStore.appendingPathComponent(spec.id, isDirectory: true).standardizedFileURL
        let installedInPrimary = ManagedModelResolver.isManagedInstallComplete(
            spec: spec,
            at: installPath,
            fileManager: fileManager
        )
        let resolver: ModelResolver? = if let modelLocations {
            ModelResolver(fileManager: fileManager, locations: modelLocations)
        } else if modelStoreURL == nil {
            ModelResolver(fileManager: fileManager)
        } else {
            nil
        }
        let unifiedResolution = spec.modelID.flatMap { resolver?.resolveIfPresent($0) }
        let installed = installedInPrimary || unifiedResolution != nil
        let runtimeURL = unifiedResolution?.rootURL
            ?? runtimeURL(for: spec, installPath: installPath, installed: installedInPrimary)
        let runtimeReady = runtimeURL != nil
        let conversionRequired = spec.requiresManagedConversion && installedInPrimary && !runtimeReady
        let selected = input.all || spec.id == input.target
        let blockedBySupport = !input.allowUnsupported && !support.isSupported
        let blockedBySource = !hasSource
        let requiresUsageTermsAcknowledgement = spec.usageRestriction != nil && (input.force || !installed)
        let blockedByUsageTerms = requiresUsageTermsAcknowledgement && !input.acceptUsageTerms
        let installedUsageTermsAcknowledged = (try? MereRunModelManifest.loadIfPresent(
            from: runtimeURL ?? installPath,
            fileManager: fileManager
        ))?.usageTermsAcknowledged == true || bindingAcknowledgesUsageTerms(
            resolution: unifiedResolution,
            resolver: resolver
        )
        let estimatedDownloadBytes = estimatedDownloadBytesOverrides[spec.id]
            ?? ModelPullDiskPreflight.estimatedDownloadBytes(
                for: spec,
                modelDir: installPath,
                force: input.force,
                hubCacheURL: hubCache,
                fileManager: fileManager
            )
        let willDownload = selected && hasSource && !blockedBySupport && !blockedByUsageTerms && (input.force || !installed)
        let status = Self.modelStatus(
            selected: selected,
            installed: installed,
            conversionRequired: conversionRequired,
            willDownload: willDownload,
            blockedBySupport: blockedBySupport,
            blockedBySource: blockedBySource,
            all: input.all,
            blockedByUsageTerms: blockedByUsageTerms
        )

        if !input.all, blockedBySupport {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_unsupported",
                    severity: .blocker,
                    title: "Model unsupported on this machine",
                    message: support.reasons.joined(separator: " "),
                    suggestedActionIDs: ["pull-model-allow-unsupported"]
                )
            )
        }
        if !input.all, blockedBySource {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_no_download_source",
                    severity: .blocker,
                    title: "Model has no public download source",
                    message: ManagedModelCatalog.missingHubSourceMessage(for: spec.id)
                )
            )
        }
        if !input.all, blockedByUsageTerms, let restriction = spec.usageRestriction {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_usage_terms_unacknowledged",
                    severity: .blocker,
                    title: "Third-party model terms require acceptance",
                    message: restriction.summary
                        + " Pass --accept-model-license to confirm that you reviewed and accept the listed terms "
                        + "and agree to comply with them. Download begins only after this confirmation."
                )
            )
        }
        if !input.all, conversionRequired,
           let guidance = spec.managedConversionGuidance(at: installPath) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_conversion_required",
                    severity: .warning,
                    title: "Downloaded model requires conversion",
                    message: guidance,
                    locations: [.init(kind: "directory", path: installPath.path)]
                )
            )
        }

        return ModelPullModelPreflightSummary(
            id: spec.id,
            title: support.descriptor.title,
            category: spec.category.rawValue,
            status: status,
            selected: selected,
            supported: support.isSupported,
            supportReasons: support.reasons,
            hasDownloadSource: hasSource,
            installed: installed,
            runtimeReady: runtimeReady,
            conversionRequired: conversionRequired,
            willDownload: willDownload,
            force: input.force,
            installPath: installPath.path,
            runtimePath: runtimeURL?.path,
            upstreamRepoID: spec.upstreamRepoId,
            hubRepoIDs: hubRepoIDs(for: spec),
            estimatedDownloadBytes: estimatedDownloadBytes,
            estimatedRequiredBytes: ModelPullDiskPreflight.requiredBytes(
                estimatedDownloadBytes: estimatedDownloadBytes
            ),
            companionModelIDs: spec.companionModelIDs,
            usageTerms: spec.usageRestriction?.terms ?? [],
            usageTermsAcknowledged: spec.usageRestriction != nil
                && (input.acceptUsageTerms || installedUsageTermsAcknowledged)
        )
    }

    static func modelStatus(
        selected: Bool,
        installed: Bool,
        conversionRequired: Bool,
        willDownload: Bool,
        blockedBySupport: Bool,
        blockedBySource: Bool,
        all: Bool,
        blockedByUsageTerms: Bool = false
    ) -> String {
        if !selected { return "not_selected" }
        if blockedBySupport { return all ? "skipped_unsupported" : "blocked_unsupported" }
        if blockedBySource { return all ? "skipped_no_source" : "blocked_no_source" }
        if blockedByUsageTerms { return all ? "skipped_usage_terms" : "blocked_usage_terms" }
        if willDownload { return "will_download" }
        if conversionRequired { return "conversion_required" }
        if installed { return "installed" }
        return "ready"
    }

    private func runtimeURL(for spec: ManagedModelSpec, installPath: URL, installed: Bool) -> URL? {
        guard installed, spec.isManagedRuntimeReady(installPath, fileManager: fileManager) else { return nil }
        guard modelStoreURL == nil else { return installPath }
        return spec.managedRuntimeURL(fileManager: fileManager)
    }

    private func bindingAcknowledgesUsageTerms(
        resolution: ModelResolver.Resolution?,
        resolver: ModelResolver?
    ) -> Bool {
        guard let resolution,
              resolution.source == .registeredBinding,
              let resolver else {
            return false
        }
        return resolver.locationCandidates(for: resolution.modelID).contains {
            $0.kind == .registeredBinding
                && $0.rootURL == resolution.rootURL
                && $0.usageTermsAcknowledged
        }
    }

    private func appendAggregateDiskDiagnostics(
        models: [ModelPullModelPreflightSummary],
        hubCache: URL,
        modelStore: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        let hubCacheAvailable = diskAvailableBytes(hubCache)
        let modelStoreAvailable = diskAvailableBytes(modelStore)
        let requiredBytes = aggregateRequiredBytes(for: models)
        let estimatedDownloadBytes = aggregateDownloadBytes(for: models)

        if let modelStoreAvailable,
           modelStoreAvailable < ModelPullDiskPreflight.minimumModelStoreBytes {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_store_space_insufficient",
                    severity: .blocker,
                    title: "Model store space insufficient",
                    message: """
                    Model store has only \(ModelPullDiskPreflight.formatBytes(modelStoreAvailable)) free at \(modelStore.path).
                    Free space or set MERERUN_MODELS_DIR=/Volumes/Models/mere.run before retrying.
                    """,
                    locations: [.init(kind: "directory", path: modelStore.path)]
                )
            )
        }

        guard let hubCacheAvailable else {
            if models.contains(where: \.willDownload) {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "hub_cache_space_unknown",
                        severity: .warning,
                        title: "Hub cache space unknown",
                        message: "Could not read free disk space for Hugging Face cache at \(hubCache.path).",
                        locations: [.init(kind: "directory", path: hubCache.path)]
                    )
                )
            }
            return
        }

        if let requiredBytes, hubCacheAvailable < requiredBytes {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "hub_cache_space_insufficient",
                    severity: .blocker,
                    title: "Hub cache space insufficient",
                    message: """
                    Hugging Face cache has \(ModelPullDiskPreflight.formatBytes(hubCacheAvailable)) free at \(hubCache.path).
                    Estimated required: \(ModelPullDiskPreflight.formatBytes(requiredBytes)).
                    """,
                    locations: [.init(kind: "directory", path: hubCache.path)]
                )
            )
            return
        }

        if let estimatedDownloadBytes {
            let remaining = hubCacheAvailable - estimatedDownloadBytes
            if remaining < ModelPullDiskPreflight.lowHeadroomWarningBytes {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "hub_cache_low_headroom",
                        severity: .warning,
                        title: "Hub cache headroom may be low",
                        message: "After pulling, the cache volume may have only about \(ModelPullDiskPreflight.formatBytes(remaining)) free.",
                        locations: [.init(kind: "directory", path: hubCache.path)]
                    )
                )
            }
        } else if models.contains(where: { $0.willDownload && $0.estimatedDownloadBytes == nil }),
                  hubCacheAvailable < ModelPullDiskPreflight.lowHeadroomWarningBytes {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "hub_cache_low_headroom_unknown_size",
                    severity: .warning,
                    title: "Hub cache headroom may be low",
                    message: """
                    Hugging Face cache has only \(ModelPullDiskPreflight.formatBytes(hubCacheAvailable)) free at \(hubCache.path), and at least one selected model has no size estimate.
                    """,
                    locations: [.init(kind: "directory", path: hubCache.path)]
                )
            )
        }
    }

    private func result(
        models: [ModelPullModelPreflightSummary],
        hubCache: URL,
        modelStore: URL
    ) -> ModelPullPreflightResult {
        let estimatedDownloadBytes = aggregateDownloadBytes(for: models)
        let requiredBytes = aggregateRequiredBytes(for: models)
        let hubCacheAvailable = diskAvailableBytes(hubCache)
        let modelStoreAvailable = diskAvailableBytes(modelStore)

        return ModelPullPreflightResult(
            mode: input.all ? "all" : "single",
            modelStore: ModelPullStoragePreflightSummary(
                path: modelStore.path,
                availableBytes: modelStoreAvailable,
                requiredBytes: ModelPullDiskPreflight.minimumModelStoreBytes,
                headroomAfterBytes: modelStoreAvailable.map { $0 - ModelPullDiskPreflight.minimumModelStoreBytes }
            ),
            hubCache: ModelPullStoragePreflightSummary(
                path: hubCache.path,
                availableBytes: hubCacheAvailable,
                requiredBytes: requiredBytes,
                headroomAfterBytes: headroomAfter(available: hubCacheAvailable, used: estimatedDownloadBytes)
            ),
            models: models,
            selectedModelCount: models.filter(\.selected).count,
            willDownloadCount: models.filter(\.willDownload).count,
            estimatedDownloadBytes: estimatedDownloadBytes,
            estimatedRequiredBytes: requiredBytes
        )
    }

    private func actions(status: StructuredRunStatus, result: ModelPullPreflightResult) -> [DeclarativeAction] {
        let blocked = status == .blocked
        var actions: [DeclarativeAction] = [
            DeclarativeAction(
                id: input.all ? "pull-models" : "pull-model",
                label: input.all ? "Pull selected models" : "Pull model",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.pullArgv,
                    cwd: input.cwd,
                    commandPath: ["model", "pull"]
                ),
                requires: ["preflight.passed"]
            ),
            DeclarativeAction(
                id: "open-model-store",
                label: "Open model store",
                kind: .openDirectory,
                style: .link,
                enabled: fileManager.fileExists(atPath: result.modelStore.path),
                path: result.modelStore.path
            ),
            DeclarativeAction(
                id: "open-hub-cache",
                label: "Open hub cache",
                kind: .openDirectory,
                style: .link,
                enabled: fileManager.fileExists(atPath: result.hubCache.path),
                path: result.hubCache.path
            ),
        ]

        if !input.all,
           let model = result.models.first,
           let runtimePath = model.runtimePath {
            actions.append(
                DeclarativeAction(
                    id: "reveal-model",
                    label: "Reveal model",
                    kind: .revealFile,
                    style: .link,
                    enabled: model.installed,
                    path: runtimePath
                )
            )
        }
        if !input.allowUnsupported, !input.all {
            actions.append(
                DeclarativeAction(
                    id: "pull-model-allow-unsupported",
                    label: "Pull anyway",
                    kind: .command,
                    style: .secondary,
                    enabled: result.models.first?.hasDownloadSource == true,
                    command: DeclarativeCommand(
                        argv: input.pullArgv + ["--allow-unsupported"],
                        cwd: input.cwd,
                        commandPath: ["model", "pull"]
                    ),
                    confirmation: "This model may not run correctly on this machine."
                )
            )
        }
        return actions
    }

    private func summary(
        status: StructuredRunStatus,
        diagnostics: [PreflightDiagnostic],
        result: ModelPullPreflightResult
    ) -> String {
        let blockerCount = diagnostics.filter { $0.severity == .blocker }.count
        let warningCount = diagnostics.filter { $0.severity == .warning }.count
        switch status {
        case .blocked:
            return "\(result.willDownloadCount) model(s) selected for download, \(blockerCount) blocker(s), \(warningCount) warning(s)."
        case .warning:
            return "\(result.willDownloadCount) model(s) selected for download, \(warningCount) warning(s)."
        default:
            if result.willDownloadCount == 0 {
                return "No model downloads needed."
            }
            return "\(result.willDownloadCount) model(s) ready to pull."
        }
    }

    private func aggregateDownloadBytes(for models: [ModelPullModelPreflightSummary]) -> Int64? {
        var total: Int64 = 0
        for model in models where model.willDownload {
            guard let estimate = model.estimatedDownloadBytes else { return nil }
            total += estimate
        }
        return total
    }

    private func aggregateRequiredBytes(for models: [ModelPullModelPreflightSummary]) -> Int64? {
        var total: Int64 = 0
        for model in models where model.willDownload {
            guard let required = model.estimatedRequiredBytes else { return nil }
            total += required
        }
        return total
    }

    private func hubRepoIDs(for spec: ManagedModelSpec) -> [String] {
        var ids: [String] = []
        if let primary = spec.hubFallback?.repoId {
            ids.append(primary)
        }
        ids.append(contentsOf: spec.mountedHubFallbacks.map { $0.hubFallback.repoId })
        return ids
    }

    private func headroomAfter(available: Int64?, used: Int64?) -> Int64? {
        guard let available, let used else { return nil }
        return available - used
    }
}
