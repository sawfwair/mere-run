import ArgumentParser
import Foundation
import MereRunCore

struct ModelPull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a managed model into the local model store."
    )

    @Argument(help: "Canonical model id (for example: image-zimage-nano, text-chat-gemma4-max, or text-chat-q35). Omit when using --all.")
    var target: String?

    @Flag(name: [.long], help: "Pull every model that has a Hugging Face source.")
    var all: Bool = false

    @Flag(name: [.long], help: "Re-download even if the model is already installed.")
    var force: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

    @Flag(name: [.long], help: "Bypass the Apple Silicon and unified-memory support check.")
    var allowUnsupported: Bool = false

    func validate() throws {
        if target == nil && !all {
            throw ValidationError("Provide a model id or use --all.")
        }
        if target != nil && all {
            throw ValidationError("Specify either a model id or --all, not both.")
        }
    }

    func run() async throws {
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
                try await pull(spec)
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
        try await pull(spec)
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

    private func pull(_ spec: ManagedModelSpec) async throws {
        let modelDir = spec.managedInstallRootURL()
        if !force, spec.isManagedRootComplete(modelDir) {
            if !quiet { stderr("[\(spec.id)] already installed, skipping (use --force to re-download)") }
            return
        }
        try ModelPullDiskPreflight.check(spec: spec, modelDir: modelDir) { warning in
            guard !quiet else { return }
            stderr("  warning: \(warning)")
        }

        let result = try await ManagedModelResolver.installManagedModel(
            id: spec.id,
            force: force,
            progress: { progress in
                guard !quiet else { return }
                switch progress {
                case .downloadingBytes(let completed, let total):
                    let done = ByteCountFormatter.string(fromByteCount: completed, countStyle: .file)
                    if let total, total > 0 {
                        let totalText = ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
                        let pct = min(100, max(0, Int(Double(completed) / Double(total) * 100)))
                        stderrRaw("\r[\(spec.id)] \(pct)%  \(done) / \(totalText)          ")
                    } else {
                        stderrRaw("\r[\(spec.id)] \(done)          ")
                    }
                case .downloadingPercent(let percent, let speed):
                    if let speed, speed > 0 {
                        let speedText = ByteCountFormatter.string(fromByteCount: Int64(speed), countStyle: .file)
                        stderrRaw("\r[\(spec.id)] \(percent)%  (\(speedText)/s)          ")
                    } else {
                        stderrRaw("\r[\(spec.id)] \(percent)%          ")
                    }
                case .extracting:
                    stderrRaw("\r[\(spec.id)] extracting…          ")
                }
            }
        )
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
                errors.append("Model was pulled but is not discoverable by `mere.run model list`.")
            }
            if !errors.isEmpty {
                throw ValidationError(
                    """
                    Model \(spec.id) was not installed cleanly.
                    \(errors.map { "- \($0)" }.joined(separator: "\n"))
                    """
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
}

struct ModelPullDiskPreflight {
    static let bytesPerGiB: Int64 = 1_073_741_824
    static let safetyMarginBytes: Int64 = 2 * bytesPerGiB
    static let lowHeadroomWarningBytes: Int64 = 10 * bytesPerGiB
    static let minimumModelStoreBytes: Int64 = 64 * 1_048_576

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

        let requiredBytes = estimatedDownloadBytes + max(safetyMarginBytes, estimatedDownloadBytes / 5)
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
