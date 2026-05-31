import ArgumentParser
import Foundation
import MereRunCore

struct Agent: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "agent",
        abstract: "Install and start the optional guided local setup agent.",
        subcommands: [
            AgentOnboard.self,
            AgentInstallPi.self,
            AgentStart.self,
        ],
        defaultSubcommand: AgentOnboard.self
    )
}

struct AgentOnboard: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "onboard",
        abstract: "Summarize this machine's model capabilities and prepare the optional Pi agent."
    )

    @Flag(name: [.long], help: "Pull the supported first-setup model package.")
    var pullRecommended: Bool = false

    @Flag(name: [.long], help: "Install the latest Pi coding-agent release from GitHub.")
    var installPi: Bool = false

    @Flag(name: [.long], help: "Write a Pi extension that registers the local mere.run API provider.")
    var configurePi: Bool = false

    @Option(name: [.long], help: "Local API host used in the Pi provider extension.")
    var host: String = "127.0.0.1"

    @Option(name: [.long], help: "Local API port used in the Pi provider extension.")
    var port: Int = 8080

    @Option(name: [.long], help: "Model id to expose in the Pi provider extension. Defaults to this machine's recommended tier (DS4 on 96 GB+).")
    var model: String?

    @Flag(name: [.short, .long], help: "Suppress progress output for installs.")
    var quiet: Bool = false

    func run() async throws {
        let machine = MereRunMachineProfile.current
        let reports = ManagedModelCapabilityCatalog.supportReports(on: machine)
        let supported = reports.filter(\.isSupported)
        let unsupported = reports.filter { !$0.isSupported }
        let recommended = ManagedModelCapabilityCatalog.recommendedSetupReports(on: machine)

        print("mere.run onboarding")
        print("  machine: \(machine.processorName), \(machine.unifiedMemoryGB) GB unified memory")
        print("  appleSiliconMac: \(machine.isAppleSiliconMac)")
        print("  linux: \(machine.isLinux)")
        print("  supportedModels: \(supported.count)")
        print("  unsupportedModels: \(unsupported.count)")

        print("\nRecommended setup coverage (downloadable from Hugging Face)")
        print("  Cross-modality starter set; lower-memory agent alternatives are not ranked upgrades.")
        if recommended.isEmpty {
            print("  No recommended managed models are supported on this machine.")
        } else {
            for report in recommended {
                print("  - \(report.spec.id): \(report.descriptor.summary)")
            }
        }
        let unavailableRecommended = reports.filter {
            $0.isSupported
                && $0.descriptor.isRecommendedForSetup
                && !$0.spec.hasAnyManagedDownloadSource()
        }
        if !unavailableRecommended.isEmpty {
            let ids = unavailableRecommended.map(\.spec.id).joined(separator: ", ")
            print("  Additional supported recommendations need local model paths: \(ids)")
        }

        let startable = MereRunAgentModelCatalog.fallbackStartableRecommendation(on: machine)
        let selectedAgentID = startable?.id ?? AgentModelResources.qwen35NineBModelId
        print("\nAgent readiness")
        if let startable {
            print("  Recommended setup agent: \(startable.id) (\(startable.displayName)).")
            print("  Start a guided session with: \(CLICommandDisplay.command("agent start --model \(selectedAgentID)"))")
            if startable.id == DeepseekV4FlashResources.defaultModelId {
                print("  DeepSeek V4 Flash is the preferred 96 GB+ setup-agent tier; smaller Qwen agents are alternatives, not upgrades.")
            }
        }
        if let codeReport = reports.first(where: { $0.spec.id == CodeGenResources.defaultModelId }) {
            if codeReport.isSupported {
                print("  Qwen3-Coder Next is also supported for comparison or coding-specific sessions.")
                print("  Pull it with: \(CLICommandDisplay.command("model pull \(CodeGenResources.defaultModelId)"))")
            } else {
                print("  Qwen3-Coder Next is not supported here: \(codeReport.reasons.joined(separator: " "))")
            }
        }
        print("  Install Pi with: \(CLICommandDisplay.command("agent install-pi"))")
        print("  Configure Pi provider with: \(CLICommandDisplay.command("agent onboard --configure-pi --model \(selectedAgentID)"))")

        if pullRecommended {
            try await pullRecommendedModels(recommended)
        }

        if installPi {
            let result = try await PiAgentIntegration.installLatest(force: false) { message in
                guard !quiet else { return }
                CLIStderr.write("[pi] \(message)\n")
            }
            print("\nPi installed")
            print("  version: \(result.version)")
            print("  path: \(result.binaryURL.path)")
            print("  release: \(result.releaseURL.absoluteString)")
        }

        if configurePi {
            let resolvedModelID = model ?? selectedAgentID
            let providerModel = try providerModel(for: resolvedModelID)
            let extensionURL = try PiAgentIntegration.writeLocalProviderExtension(
                host: host,
                port: port,
                model: providerModel,
                homeDirectory: PiAgentIntegration.mereRunPiHomeDirectory()
            )
            print("\nPi provider configured")
            print("  extension: \(extensionURL.path)")
            print("  provider: mere-run")
            print("  model: \(providerModel.id)")
        }
    }

    private func pullRecommendedModels(_ reports: [ManagedModelSupportReport]) async throws {
        for report in reports {
            guard report.spec.hasAnyManagedDownloadSource() else { continue }
            if !quiet {
                CLIStderr.write("[\(report.spec.id)] installing recommended model\n")
            }
            _ = try await ManagedModelResolver.installManagedModel(
                id: report.spec.id,
                force: false,
                progress: { progress in
                    guard !quiet else { return }
                    switch progress {
                    case .downloadingBytes(let completed, let total):
                        let done = ByteCountFormatter.string(fromByteCount: completed, countStyle: .file)
                        if let total, total > 0 {
                            let totalText = ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
                            CLIStderr.write("\r[\(report.spec.id)] \(done) / \(totalText)          ")
                        } else {
                            CLIStderr.write("\r[\(report.spec.id)] \(done)          ")
                        }
                    case .downloadingPercent(let percent, let speed):
                        if let speed, speed > 0 {
                            let speedText = ByteCountFormatter.string(
                                fromByteCount: Int64(speed),
                                countStyle: .file
                            )
                            CLIStderr.write("\r[\(report.spec.id)] \(percent)% (\(speedText)/s)          ")
                        } else {
                            CLIStderr.write("\r[\(report.spec.id)] \(percent)%          ")
                        }
                    case .extracting:
                        CLIStderr.write("\r[\(report.spec.id)] extracting          ")
                    }
                }
            )
            if !quiet {
                CLIStderr.write("\n")
            }
        }
    }

    private func providerModel(for modelID: String) throws -> PiProviderModel {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let recommendation = SetupAgentRuntime.recommendation(forManagedModelID: normalized) else {
            throw ValidationError("Unsupported Pi provider model: \(modelID)")
        }
        return SetupAgentRuntime.providerModel(for: recommendation)
    }
}

struct AgentInstallPi: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "install-pi",
        abstract: "Install the latest Pi coding-agent release for use with mere.run."
    )

    @Flag(name: [.long], help: "Re-download and replace the current installed Pi release.")
    var force: Bool = false

    func run() async throws {
        let result = try await PiAgentIntegration.installLatest(force: force) { message in
            CLIStderr.write("[pi] \(message)\n")
        }
        print(result.binaryURL.path)
        CLIStderr.write("[pi] Installed \(result.version) from \(result.releaseURL.absoluteString)\n")
    }
}

struct AgentStart: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "start",
        abstract: "Start Pi against a local mere.run setup-agent API server."
    )

    @Option(name: [.long], help: "Local API host.")
    var host: String = "127.0.0.1"

    @Option(name: [.long], help: "Local API port.")
    var port: Int = 8080

    @Option(name: [.long], help: "Path to the pi executable. Defaults to the managed install or PATH.")
    var piPath: String?

    @Option(name: [.long], help: "Initial prompt sent to Pi.")
    var prompt: String = SetupAgentPrompt.defaultUserRequest

    @Option(name: [.long], help: "Managed agent model id to serve through Pi.")
    var model: String?

    @Flag(name: [.long], help: "Use an already-running mere.run API server instead of starting one.")
    var skipServer: Bool = false

    @Flag(name: [.long], help: "Start even if the local hardware support catalog marks the selected model unsupported.")
    var allowUnsupported: Bool = false

    @Flag(name: [.long], help: "Refuse to download a missing GGUF or install Pi. By default, agent start auto-fetches both.")
    var noBootstrap: Bool = false

    @Flag(name: [.short, .long], help: "Suppress bootstrap (model pull / Pi install) progress output.")
    var quiet: Bool = false

    func run() async throws {
        let modelID = try resolvedModelID()
        let runtime = try SetupAgentRuntime.runtime(forManagedModelID: modelID)
        let spec = runtime.spec
        let support = ManagedModelCapabilityCatalog.support(for: spec)
        if !allowUnsupported, !support.isSupported {
            throw ValidationError(
                """
                \(spec.id) is not supported on this machine.
                \(support.reasons.map { "- \($0)" }.joined(separator: "\n"))
                """
            )
        }

        let modelURL: URL
        if let existing = spec.managedRuntimeURL() {
            modelURL = existing
        } else if noBootstrap {
            throw ValidationError(
                "Model \(spec.id) is not installed. Run `mere.run model pull \(spec.id)` "
                + "or drop --no-bootstrap to auto-pull."
            )
        } else {
            modelURL = try await autoPullManagedModel(spec: spec)
        }

        let piURL: URL
        if let existing = PiAgentIntegration.findPiExecutable(explicitPath: piPath) {
            piURL = existing
        } else if noBootstrap {
            throw PiAgentIntegration.IntegrationError.piBinaryNotFound
        } else {
            piURL = try await autoInstallPi()
        }

        let extensionURL = try PiAgentIntegration.writeLocalProviderExtension(
            host: host,
            port: port,
            model: runtime.providerModel,
            homeDirectory: PiAgentIntegration.mereRunPiHomeDirectory()
        )
        CLIStderr.write("[agent] Pi provider extension: \(extensionURL.path)\n")

        let serverProcess: Process?
        if skipServer {
            serverProcess = nil
        } else {
            let startedServer = try startAPIServer(modelURL: modelURL, engine: runtime.engine)
            serverProcess = startedServer.process
            CLIStderr.write("[agent] Loading \(runtime.providerModel.id). Server log: \(startedServer.logURL.path)\n")
            try await PiAgentIntegration.waitForHealth(host: host, port: port, timeoutSeconds: runtime.healthTimeoutSeconds)
            CLIStderr.write("[agent] Local API is ready. \(PiTerminalLauncher.launchProgressMessage)\n")
        }
        defer {
            if let serverProcess, serverProcess.isRunning {
                serverProcess.terminate()
            }
        }

        try runPi(
            piURL: piURL,
            modelID: runtime.providerModel.id,
            engine: runtime.engine,
            modelURL: modelURL
        )
        if serverProcess != nil {
            CLIStderr.write("[agent] \(PiTerminalLauncher.runningProgressMessage)\n")
            if PiTerminalLauncher.launchesDetached {
                waitForServerProcess()
            }
        }
    }

    private func autoPullManagedModel(spec: ManagedModelSpec) async throws -> URL {
        guard spec.hasAnyManagedDownloadSource() else {
            throw ValidationError(
                "Model \(spec.id) is not installed and has no managed Hugging Face source."
            )
        }
        if !quiet {
            CLIStderr.write(
                "[agent] \(spec.id) not installed. Downloading from Hugging Face "
                + "(this is a one-time fetch — Ctrl+C to cancel).\n"
            )
        }
        _ = try await ManagedModelResolver.installManagedModel(
            id: spec.id,
            force: false,
            progress: { progress in
                guard !self.quiet else { return }
                switch progress {
                case .downloadingBytes(let completed, let total):
                    let done = ByteCountFormatter.string(fromByteCount: completed, countStyle: .file)
                    if let total, total > 0 {
                        let totalText = ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
                        CLIStderr.write("\r[\(spec.id)] \(done) / \(totalText)          ")
                    } else {
                        CLIStderr.write("\r[\(spec.id)] \(done)          ")
                    }
                case .downloadingPercent(let percent, let speed):
                    if let speed, speed > 0 {
                        let speedText = ByteCountFormatter.string(
                            fromByteCount: Int64(speed),
                            countStyle: .file
                        )
                        CLIStderr.write("\r[\(spec.id)] \(percent)% (\(speedText)/s)          ")
                    } else {
                        CLIStderr.write("\r[\(spec.id)] \(percent)%          ")
                    }
                case .extracting:
                    CLIStderr.write("\r[\(spec.id)] extracting          ")
                }
            }
        )
        if !quiet {
            CLIStderr.write("\n")
        }
        guard let url = spec.managedRuntimeURL() else {
            throw ValidationError(
                "Model \(spec.id) appears installed but the runtime resolver could not locate it."
            )
        }
        return url
    }

    private func autoInstallPi() async throws -> URL {
        guard PiAgentIntegration.canInstallLatestRelease else {
            throw PiAgentIntegration.IntegrationError.unsupportedPlatform
        }
        if !quiet {
            CLIStderr.write("[agent] Pi is not installed. Installing the latest release.\n")
        }
        let result = try await PiAgentIntegration.installLatest(force: false) { message in
            guard !self.quiet else { return }
            CLIStderr.write("[pi] \(message)\n")
        }
        if !quiet {
            CLIStderr.write("[agent] Installed Pi \(result.version) at \(result.binaryURL.path)\n")
        }
        return result.binaryURL
    }

    private func resolvedModelID() throws -> String {
        if let model, !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return model
        }
        // Prefer the highest-tier model that's actually installed — that
        // reflects what the user has chosen to pull onto this machine. The
        // persisted provider.json is only used as a tiebreaker / hint when no
        // installed model is found, because it can go stale across upgrades
        // and pin you to a lower tier than your current hardware supports.
        if let installed = Self.bestInstalledStartableAgentModel(on: MereRunMachineProfile.current) {
            return installed.id
        }
        if let configured = PiAgentIntegration.readProviderConfiguration()?.modelID,
           Self.isInstalledStartableAgentModel(configured, on: MereRunMachineProfile.current) {
            return configured
        }
        guard let recommendation = MereRunAgentModelCatalog.fallbackStartableRecommendation(
            on: MereRunMachineProfile.current
        ) else {
            throw ValidationError(
                "No local setup-agent model is supported on this machine. Run `\(CLICommandDisplay.command("setup --mode byoa"))`."
            )
        }
        return recommendation.id
    }

    static func bestInstalledStartableAgentModel(
        on machine: MereRunMachineProfile,
        fileManager: FileManager = .default
    ) -> MereRunAgentModelRecommendation? {
        MereRunAgentModelCatalog.allTierRecommendations(on: machine)
            .filter(\.isStartableByMereRun)
            .filter { recommendation in
                guard let managedModelID = recommendation.managedModelID,
                      let spec = ManagedModelCatalog.spec(for: managedModelID) else {
                    return false
                }
                return spec.managedRuntimeURL(fileManager: fileManager) != nil
            }
            .max { lhs, rhs in
                lhs.recommendedUnifiedMemoryGB < rhs.recommendedUnifiedMemoryGB
            }
    }

    static func isInstalledStartableAgentModel(
        _ modelID: String,
        on machine: MereRunMachineProfile,
        fileManager: FileManager = .default
    ) -> Bool {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return MereRunAgentModelCatalog.allTierRecommendations(on: machine)
            .contains { recommendation in
                guard let managedModelID = recommendation.managedModelID,
                      let spec = ManagedModelCatalog.spec(for: managedModelID),
                      spec.managedRuntimeURL(fileManager: fileManager) != nil else {
                    return false
                }
                return recommendation.isStartableByMereRun
                    && (recommendation.id == normalized || recommendation.managedModelID == normalized)
            }
    }

    private func startAPIServer(modelURL: URL, engine: APIEngine) throws -> (process: Process, logURL: URL) {
        let log = try AgentServerLog.makeLogHandle(prefix: "api-server")
        let process = Process()
        process.executableURL = Self.currentExecutableURL()
        process.arguments = [
            "api",
            "serve",
            "--engine",
            engine.rawValue,
            "--model",
            modelURL.path,
            "--host",
            host,
            "--port",
            String(port),
        ]
        process.standardInput = AgentServerLog.nullInputHandle()
        process.standardOutput = log.handle
        process.standardError = log.handle
        try process.run()
        CLIStderr.write("[agent] Started mere.run API server on \(host):\(port)\n")
        return (process, log.url)
    }

    private static func currentExecutableURL() -> URL {
        CurrentExecutable.url()
    }

    private func runPi(piURL: URL, modelID: String, engine: APIEngine, modelURL: URL?) throws {
        try PiTerminalLauncher.launch(
            piURL: piURL,
            modelID: modelID,
            prompt: SetupAgentPrompt.render(
                userRequest: prompt,
                selectedModelID: modelID,
                engine: engine,
                modelURL: modelURL,
                host: host,
                port: port
            ),
            homeDirectory: PiAgentIntegration.mereRunPiHomeDirectory()
        )
    }

    private func waitForServerProcess() -> Never {
        while true {
            Thread.sleep(forTimeInterval: 3600)
        }
    }
}
