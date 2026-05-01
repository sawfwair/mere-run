import ArgumentParser
import Foundation
import MereRunCore

struct ModelPull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a managed model into the local model store."
    )

    @Argument(help: "Canonical model id (for example: image-klein-nano, text-chat-gemma4-max, or text-chat-q35). Omit when using --all.")
    var target: String?

    @Flag(name: [.long], help: "Pull every model that has a managed download source.")
    var all: Bool = false

    @Flag(name: [.long], help: "Re-download even if the model is already installed.")
    var force: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

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
                if !spec.hasAnyManagedDownloadSource() {
                    if !quiet {
                        stderr("[\(spec.id)] skipping (no managed download source available in current configuration)")
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
        if !spec.hasAnyManagedDownloadSource() {
            throw CleanExit.message(
                MereRunModelSourceConfiguration.missingConfigurationMessage(
                    purpose: "Managed model downloads for \(spec.id)"
                )
            )
        }
        try await pull(spec)
    }

    private func pull(_ spec: ManagedModelSpec) async throws {
        let modelDir = spec.managedInstallRootURL()
        if !force, spec.isManagedRootComplete(modelDir) {
            if !quiet { stderr("[\(spec.id)] already installed, skipping (use --force to re-download)") }
            return
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
