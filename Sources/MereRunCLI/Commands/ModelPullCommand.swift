import ArgumentParser
import Foundation
import MereRunCore

struct ModelPull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a managed model into the local model store."
    )

    @Argument(help: "Canonical model id (for example: image-zimage-nano, text-chat-gemma4-max, or text-chat-lfm25-a1b-8bit). Omit when using --all.")
    var target: String?

    @Flag(name: [.long], help: "Pull every model that has a Hugging Face source.")
    var all: Bool = false

    @Flag(name: [.long], help: "Re-download even if the model is already installed.")
    var force: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

    @Flag(name: [.long], help: "Bypass the Apple Silicon and unified-memory support check.")
    var allowUnsupported: Bool = false

    @Flag(
        name: [.customLong("accept-model-license")],
        help: "Acknowledge the upstream usage restriction before downloading a restricted model."
    )
    var acceptModelLicense: Bool = false

    @Flag(name: [.customLong("preflight")], help: "Inspect support, source, disk, and install state without downloading.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    func validate() throws {
        if target == nil && !all {
            throw ValidationError("Provide a model id or use --all.")
        }
        if target != nil && all {
            throw ValidationError("Specify either a model id or --all, not both.")
        }
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for model pull.")
        }
    }

    func run() async throws {
        if preflight {
            try runPreflight()
            return
        }

        if all {
            for spec in ManagedModelCatalog.allSpecs {
                let support = ManagedModelCapabilityCatalog.support(for: spec)
                if !allowUnsupported, !support.isSupported {
                    if !quiet {
                        stderr("[\(spec.id)] skipping (unsupported: \(support.reasons.joined(separator: " ")))")
                    }
                    continue
                }
                if !spec.hasAnyManagedDownloadSource() {
                    if !quiet {
                        stderr("[\(spec.id)] skipping (no Hugging Face source in this public build)")
                    }
                    continue
                }
                if spec.usageRestriction != nil, !acceptModelLicense {
                    if !quiet {
                        stderr("[\(spec.id)] skipping (pass --accept-model-license to acknowledge its upstream usage restriction)")
                    }
                    continue
                }
                try await pullWithCompanions(spec)
            }
            return
        }

        guard let target else { return }
        guard let spec = ManagedModelCatalog.spec(for: target) else {
            throw ValidationError("Unknown canonical model id: \(target)")
        }
        try validateHardwareSupport(for: spec)
        if !spec.hasAnyManagedDownloadSource() {
            throw CleanExit.message(
                ManagedModelCatalog.missingHubSourceMessage(for: spec.id)
            )
        }
        try await pullWithCompanions(spec)
    }

    private func validateHardwareSupport(for spec: ManagedModelSpec) throws {
        guard !allowUnsupported else { return }
        let support = ManagedModelCapabilityCatalog.support(for: spec)
        guard support.isSupported else {
            throw ValidationError(
                """
                Model \(spec.id) is not supported on this machine.
                \(support.reasons.map { "- \($0)" }.joined(separator: "\n"))

                Run `mere.run model capabilities --all` to see supported models, or pass --allow-unsupported if you are intentionally using external hardware or accepting the risk.
                """
            )
        }
    }

    private func pullWithCompanions(_ spec: ManagedModelSpec) async throws {
        try await pull(spec)
        for companionID in spec.companionModelIDs {
            guard let companion = ManagedModelCatalog.spec(for: companionID) else {
                throw ValidationError("Unknown companion model id for \(spec.id): \(companionID)")
            }
            if !quiet {
                stderr("[\(spec.id)] pulling companion \(companion.id)")
            }
            try await pull(companion)
        }
    }

    private func pull(_ spec: ManagedModelSpec) async throws {
        let modelDir = spec.managedInstallRootURL()
        if !force, ManagedModelResolver.isManagedInstallComplete(spec: spec, at: modelDir) {
            if !quiet {
                if !spec.isManagedRuntimeReady(modelDir),
                   let guidance = spec.managedConversionGuidance(at: modelDir) {
                    stderr("[\(spec.id)] source already downloaded, skipping (use --force to re-download)")
                    stderr("  \(guidance)")
                } else {
                    stderr("[\(spec.id)] already installed, skipping (use --force to re-download)")
                }
            }
            return
        }
        if let message = licenseAcceptanceMessage(for: spec) {
            throw ValidationError(message)
        }
        if let restriction = spec.usageRestriction, !quiet {
            stderr("[\(spec.id)] usage restriction: \(restriction.summary)")
            stderr("  license: \(restriction.licenseURL)")
        }
        try ModelPullDiskPreflight.check(spec: spec, modelDir: modelDir) { warning in
            guard !quiet else { return }
            stderr("  warning: \(warning)")
        }

        let result: ManagedModelResolver.InstallResult
        let progressPrinter = ModelPullProgressPrinter(modelID: spec.id)
        do {
            result = try await ManagedModelResolver.installManagedModel(
                id: spec.id,
                force: force,
                progress: { progress in
                    guard !quiet else { return }
                    stderrRaw("\r\(progressPrinter.render(progress))          ")
                }
            )
        } catch let error as ManagedModelResolver.ResolverError {
            if !quiet {
                stderrRaw("\n")
            }
            if case .invalidInstalledModel(let message) = error {
                throw ModelPullInstallError(
                    modelID: spec.id,
                    modelDir: modelDir,
                    hubRepoID: spec.hubFallback?.repoId,
                    details: [message]
                )
            }
            throw error
        }
        if !quiet {
            stderrRaw("\n")
        }

        if result.manifest != nil {
            let report = MereRunModelValidator.validate(modelRoot: modelDir, expectedModelID: spec.id)
            if !quiet {
                for warning in report.warnings {
                    stderr("  warning: \(warning)")
                }
                for error in report.errors {
                    stderr("  error: \(error)")
                }
            }
            var errors = report.errors
            if spec.managedRuntimeURL() == nil {
                if spec.requiresManagedConversion,
                   ManagedModelResolver.isManagedInstallComplete(spec: spec, at: modelDir),
                   let guidance = spec.managedConversionGuidance(at: modelDir) {
                    if !quiet { stderr("  \(guidance)") }
                } else {
                    errors.append("Model was pulled but is not runnable or discoverable by `mere.run model list`.")
                }
            }
            if !errors.isEmpty {
                throw ModelPullInstallError(
                    modelID: spec.id,
                    modelDir: modelDir,
                    hubRepoID: spec.hubFallback?.repoId,
                    details: errors
                )
            }
        }

        print(modelDir.path)
    }

    private func stderr(_ message: String) {
        CLIStderr.write(message + "\n")
    }

    private func stderrRaw(_ message: String) {
        CLIStderr.write(message)
    }

    func licenseAcceptanceMessage(for spec: ManagedModelSpec) -> String? {
        guard let restriction = spec.usageRestriction, !acceptModelLicense else {
            return nil
        }
        return """
        Model \(spec.id) has an upstream usage restriction: \(restriction.summary)
        Review: \(restriction.licenseURL)
        Re-run with --accept-model-license to acknowledge the restriction before download.
        """
    }

    func makePreflightEnvelope(
        fileManager: FileManager = .default,
        hubCacheURL: URL? = nil,
        modelStoreURL: URL? = nil,
        diskAvailableBytes: ((URL) -> Int64?)? = nil,
        now: @escaping () -> Date = Date.init
    ) -> ModelPullPreflightEnvelope {
        let input = ModelPullPreflightInput(
            target: target,
            all: all,
            force: force,
            allowUnsupported: allowUnsupported,
            pullArgv: pullActionArguments(),
            cwd: fileManager.currentDirectoryPath
        )
        return ModelPullPreflightAnalyzer(
            input: input,
            fileManager: fileManager,
            hubCacheURL: hubCacheURL,
            modelStoreURL: modelStoreURL,
            diskAvailableBytes: diskAvailableBytes,
            now: now
        ).envelope()
    }

    private func runPreflight() throws {
        let envelope = makePreflightEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            for diagnostic in envelope.diagnostics {
                print("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    private func pullActionArguments() -> [String] {
        var args = ["mere.run", "model", "pull"]
        if all {
            args.append("--all")
        } else if let target {
            args.append(target)
        }
        if force {
            args.append("--force")
        }
        if quiet {
            args.append("--quiet")
        }
        if allowUnsupported {
            args.append("--allow-unsupported")
        }
        if acceptModelLicense {
            args.append("--accept-model-license")
        }
        return args
    }
}

struct ModelPullInstallError: LocalizedError, Sendable {
    let modelID: String
    let modelDir: URL
    let hubRepoID: String?
    let details: [String]

    var errorDescription: String? {
        var lines: [String] = []
        lines.append("Model \(modelID) was not installed cleanly.")
        lines.append(contentsOf: details.map { "- \($0)" })
        lines.append("")
        lines.append("Model store: \(modelDir.path)")
        lines.append("Retry with: mere.run model pull \(modelID)")
        lines.append("Use --force only if you intentionally want to replace a complete install.")
        if let hubRepoID,
           let hubCache = try? HubSnapshot.resolvedDownloadBase() {
            lines.append(
                "If this repeats, remove the stale Hub cache for \(hubRepoID) under \(hubCache.path)/models/ and retry."
            )
        }
        return lines.joined(separator: "\n")
    }
}

struct ModelPullDiskPreflight {
    static let bytesPerGiB: Int64 = 1_073_741_824
    static let safetyMarginBytes: Int64 = 2 * bytesPerGiB
    static let lowHeadroomWarningBytes: Int64 = 10 * bytesPerGiB
    static let minimumModelStoreBytes: Int64 = 64 * 1_048_576

    static func requiredBytes(estimatedDownloadBytes: Int64?) -> Int64? {
        guard let estimatedDownloadBytes else { return nil }
        return estimatedDownloadBytes + max(safetyMarginBytes, estimatedDownloadBytes / 5)
    }

    static func check(
        spec: ManagedModelSpec,
        modelDir: URL,
        fileManager: FileManager = .default,
        warn: (String) -> Void
    ) throws {
        let hubCache = try HubSnapshot.resolvedDownloadBase(fileManager: fileManager)
        let modelStoreParent = modelDir.deletingLastPathComponent()
        try? fileManager.createDirectory(at: modelStoreParent, withIntermediateDirectories: true)

        let warnings = try evaluate(
            modelID: spec.id,
            estimatedDownloadBytes: spec.estimatedDownloadBytes,
            hubCacheURL: hubCache,
            hubCacheAvailableBytes: availableBytes(onFileSystemContaining: hubCache, fileManager: fileManager),
            modelStoreURL: modelStoreParent,
            modelStoreAvailableBytes: availableBytes(onFileSystemContaining: modelStoreParent, fileManager: fileManager)
        )
        warnings.forEach(warn)
    }

    static func evaluate(
        modelID: String,
        estimatedDownloadBytes: Int64?,
        hubCacheURL: URL,
        hubCacheAvailableBytes: Int64?,
        modelStoreURL: URL,
        modelStoreAvailableBytes: Int64?
    ) throws -> [String] {
        var warnings: [String] = []

        if let modelStoreAvailableBytes, modelStoreAvailableBytes < minimumModelStoreBytes {
            throw ValidationError(
                """
                Not enough free disk space to link \(modelID) into the mere.run model store.
                Model store: \(modelStoreURL.path)
                Available: \(formatBytes(modelStoreAvailableBytes))
                Free space or set MERERUN_MODELS_DIR=/Volumes/Models/mere.run before retrying.
                """
            )
        }

        guard let hubCacheAvailableBytes else {
            warnings.append("Could not read free disk space for Hugging Face cache at \(hubCacheURL.path).")
            return warnings
        }

        guard let estimatedDownloadBytes else {
            if hubCacheAvailableBytes < lowHeadroomWarningBytes {
                warnings.append(
                    "Hugging Face cache has only \(formatBytes(hubCacheAvailableBytes)) free at \(hubCacheURL.path); this model does not publish a size estimate yet."
                )
            }
            return warnings
        }

        let requiredBytes = requiredBytes(estimatedDownloadBytes: estimatedDownloadBytes) ?? estimatedDownloadBytes
        if hubCacheAvailableBytes < requiredBytes {
            throw ValidationError(
                """
                Not enough free disk space to pull \(modelID).
                Hugging Face cache: \(hubCacheURL.path)
                Available: \(formatBytes(hubCacheAvailableBytes))
                Estimated required: \(formatBytes(requiredBytes)) (\(formatBytes(estimatedDownloadBytes)) model plus safety margin)

                Free space, remove old models with `mere.run model remove`, or move the cache/model store before retrying:
                  export MERERUN_HUB_CACHE=/Volumes/Models/huggingface
                  export MERERUN_MODELS_DIR=/Volumes/Models/mere.run
                """
            )
        }

        let remaining = hubCacheAvailableBytes - estimatedDownloadBytes
        if remaining < lowHeadroomWarningBytes {
            warnings.append(
                "After pulling \(modelID), the Hugging Face cache volume may have only about \(formatBytes(remaining)) free. Cache: \(hubCacheURL.path)"
            )
        }

        return warnings
    }

    static func availableBytes(
        onFileSystemContaining url: URL,
        fileManager: FileManager = .default
    ) -> Int64? {
        guard let existing = existingPath(for: url, fileManager: fileManager),
              let attrs = try? fileManager.attributesOfFileSystem(forPath: existing.path),
              let free = attrs[.systemFreeSize] as? NSNumber else {
            return nil
        }
        return free.int64Value
    }

    private static func existingPath(for url: URL, fileManager: FileManager) -> URL? {
        var candidate = url.standardizedFileURL
        while true {
            if fileManager.fileExists(atPath: candidate.path) {
                return candidate
            }
            let parent = candidate.deletingLastPathComponent()
            if parent.path == candidate.path {
                return nil
            }
            candidate = parent
        }
    }

    static func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
