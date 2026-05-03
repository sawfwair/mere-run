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

    @Option(name: [.long], help: "Model id to expose in the Pi provider extension.")
    var model: String = CodeGenResources.defaultModelId

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
        print("  supportedModels: \(supported.count)")
        print("  unsupportedModels: \(unsupported.count)")

        print("\nRecommended first setup (downloadable with current configuration)")
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
            print("  Additional supported recommendations need model-source configuration: \(ids)")
        }

        print("\nAgent readiness")
        if let codeReport = reports.first(where: { $0.spec.id == CodeGenResources.defaultModelId }) {
            if codeReport.isSupported {
                print("  Qwen3-Coder Next is supported for `\(CLICommandDisplay.command("agent start --model \(CodeGenResources.defaultModelId)"))`.")
                print("  Pull it with: \(CLICommandDisplay.command("model pull \(CodeGenResources.defaultModelId)"))")
            } else {
                print("  Qwen3-Coder Next is not supported here: \(codeReport.reasons.joined(separator: " "))")
            }
        }
        let startable = MereRunAgentModelCatalog.fallbackStartableRecommendation(on: machine)
        let selectedAgentID = startable?.id ?? AgentModelResources.qwen35NineBModelId
        print("  Install Pi with: \(CLICommandDisplay.command("agent install-pi"))")
        print("  Configure Pi provider with: \(CLICommandDisplay.command("agent onboard --configure-pi --model \(selectedAgentID)"))")
        print("  Start a guided session with: \(CLICommandDisplay.command("agent start --model \(selectedAgentID)"))")

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
            let providerModel = try providerModel(for: model)
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
    var prompt: String = """
    Guide me through setting up mere.run on this machine. Start by summarizing what this Mac can run, then help me install only supported models.
    """

    @Option(name: [.long], help: "Managed agent model id to serve through Pi.")
    var model: String?

    @Flag(name: [.long], help: "Use an already-running mere.run API server instead of starting one.")
    var skipServer: Bool = false

    @Flag(name: [.long], help: "Start even if the local hardware support catalog marks the selected model unsupported.")
    var allowUnsupported: Bool = false

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

        guard let modelURL = spec.managedRuntimeURL() else {
            throw ValidationError(
                "Model \(spec.id) is not installed. Run `mere.run model pull \(spec.id)` first."
            )
        }

        guard let piURL = PiAgentIntegration.findPiExecutable(explicitPath: piPath) else {
            throw PiAgentIntegration.IntegrationError.piBinaryNotFound
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
            try await PiAgentIntegration.waitForHealth(host: host, port: port, timeoutSeconds: 60)
            CLIStderr.write("[agent] Local API is ready. Opening Pi in Terminal.app.\n")
        }
        defer {
            if let serverProcess, serverProcess.isRunning {
                serverProcess.terminate()
            }
        }

        try runPi(piURL: piURL, modelID: runtime.providerModel.id)
        if serverProcess != nil {
            CLIStderr.write("[agent] Pi is running in Terminal.app. Keep this command running; press Ctrl+C here to stop the API server.\n")
            waitForServerProcess()
        }
    }

    private func resolvedModelID() throws -> String {
        if let model, !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return model
        }
        if let configured = PiAgentIntegration.readProviderConfiguration()?.modelID,
           Self.isInstalledStartableAgentModel(configured, on: MereRunMachineProfile.current) {
            return configured
        }
        if let installed = Self.bestInstalledStartableAgentModel(on: MereRunMachineProfile.current) {
            return installed.id
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
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
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

    private func runPi(piURL: URL, modelID: String) throws {
        try PiTerminalLauncher.launch(
            piURL: piURL,
            modelID: modelID,
            prompt: prompt,
            homeDirectory: PiAgentIntegration.mereRunPiHomeDirectory()
        )
    }

    private func waitForServerProcess() -> Never {
        while true {
            Thread.sleep(forTimeInterval: 3600)
        }
    }
}
