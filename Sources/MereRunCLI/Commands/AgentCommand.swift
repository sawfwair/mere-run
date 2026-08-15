import ArgumentParser
import Foundation
import MereRunCore

struct Agent: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "agent",
        abstract: "Install and start the optional guided local setup agent.",
        subcommands: [
            AgentOnboard.self,
            AgentStatus.self,
            AgentInstallPi.self,
            AgentStart.self,
        ],
        defaultSubcommand: AgentOnboard.self
    )
}

struct AgentStatus: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Inspect local agent, Pi, provider, and model readiness."
    )

    @Option(name: [.long], help: "Optional Pi executable override to inspect.")
    var piPath: String?

    @Flag(name: [.long], help: "Emit a machine-readable readiness snapshot.")
    var json: Bool = false

    func run() throws {
        let snapshot = AgentStatusSnapshot.current(piPath: piPath)
        if json {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(snapshot), as: UTF8.self))
            return
        }

        print("mere.run agent status")
        print("  machine: \(snapshot.machine.processor), \(snapshot.machine.unifiedMemoryGB) GB")
        if snapshot.pi.installed {
            let version = snapshot.pi.version.map { " \($0)" } ?? ""
            print("  Pi: installed\(version) at \(snapshot.pi.path ?? "unknown path")")
        } else {
            print("  Pi: not installed")
        }
        if snapshot.provider.configured {
            print(
                "  provider: \(snapshot.provider.modelID ?? "unknown") at "
                    + "\(snapshot.provider.host ?? "127.0.0.1"):\(snapshot.provider.port ?? 8080)"
            )
        } else {
            print("  provider: not configured")
        }
        if let recommendedModelID = snapshot.recommendedModelID {
            print("  recommended agent: \(recommendedModelID)")
        } else {
            print("  recommended agent: unavailable")
        }
        for model in snapshot.models {
            let install = model.installed ? "installed" : "not installed"
            print("    \(model.id): \(install), \(model.servingEngine)")
        }
    }
}

struct AgentStatusSnapshot: Codable, Equatable {
    let machine: AgentStatusMachine
    let pi: AgentStatusPi
    let provider: AgentStatusProvider
    let recommendedModelID: String?
    let models: [AgentStatusModel]

    static func current(
        piPath: String?,
        fileManager: FileManager = .default
    ) -> AgentStatusSnapshot {
        let machine = MereRunMachineProfile.current
        let piURL = PiAgentIntegration.findPiExecutable(explicitPath: piPath)
        let managedPiURL = PiAgentIntegration.installedPiBinaryURL()?.standardizedFileURL
        let provider = PiAgentIntegration.readProviderConfiguration(fileManager: fileManager)
        let extensionURL = PiAgentIntegration.localProviderExtensionURL(
            homeDirectory: PiAgentIntegration.mereRunPiHomeDirectory(),
            fileManager: fileManager
        )
        let models = MereRunAgentModelCatalog.allTierRecommendations(on: machine).map {
            AgentStatusModel(
                id: $0.id,
                displayName: $0.displayName,
                summary: $0.summary,
                minimumUnifiedMemoryGB: $0.minimumUnifiedMemoryGB,
                recommendedUnifiedMemoryGB: $0.recommendedUnifiedMemoryGB,
                servingEngine: $0.servingEngine.rawValue,
                startableByMereRun: $0.isStartableByMereRun,
                sourceConfigurationRequired: $0.sourceConfigurationRequired,
                installed: $0.managedModelID
                    .flatMap { ManagedModelCatalog.spec(for: $0) }?
                    .managedRuntimeURL(fileManager: fileManager) != nil,
                reason: $0.reason
            )
        }
        return AgentStatusSnapshot(
            machine: AgentStatusMachine(
                processor: machine.processorName,
                unifiedMemoryGB: machine.unifiedMemoryGB,
                appleSiliconMac: machine.isAppleSiliconMac,
                linux: machine.isLinux
            ),
            pi: AgentStatusPi(
                installed: piURL != nil,
                managedInstall: piURL?.standardizedFileURL == managedPiURL,
                autoInstallSupported: PiAgentIntegration.canInstallLatestRelease,
                path: piURL?.path,
                version: piURL?.standardizedFileURL == managedPiURL
                    ? PiAgentIntegration.installedPiVersion()
                    : nil
            ),
            provider: AgentStatusProvider(
                configured: provider != nil
                    && fileManager.fileExists(atPath: extensionURL.path),
                host: provider?.host,
                port: provider?.port,
                modelID: provider?.modelID,
                updatedAt: provider?.updatedAt,
                configurationPath: PiAgentIntegration.providerConfigurationURL().path,
                extensionPath: extensionURL.path
            ),
            recommendedModelID: MereRunAgentModelCatalog
                .fallbackStartableRecommendation(on: machine)?
                .id,
            models: models
        )
    }
}

struct AgentStatusMachine: Codable, Equatable {
    let processor: String
    let unifiedMemoryGB: Int
    let appleSiliconMac: Bool
    let linux: Bool
}

struct AgentStatusPi: Codable, Equatable {
    let installed: Bool
    let managedInstall: Bool
    let autoInstallSupported: Bool
    let path: String?
    let version: String?
}

struct AgentStatusProvider: Codable, Equatable {
    let configured: Bool
    let host: String?
    let port: Int?
    let modelID: String?
    let updatedAt: Date?
    let configurationPath: String
    let extensionPath: String
}

struct AgentStatusModel: Codable, Equatable {
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

struct AgentOnboard: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "onboard",
        abstract: "Summarize this machine's model capabilities and prepare the optional Pi agent."
    )

    @Flag(name: [.long], help: "Pull the supported first-setup model package.")
    var pullRecommended: Bool = false

    @Flag(
        name: [.customLong("accept-model-license")],
        help: "Confirm that you reviewed and accept listed third-party model/component terms before downloading restricted recommended models."
    )
    var acceptModelLicense: Bool = false

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
        let selectedAgentID = startable?.id ?? Q35Resources.ornith9BModelId
        print("\nAgent readiness")
        if let startable {
            print("  Recommended setup agent: \(startable.id) (\(startable.displayName)).")
            print("  Start a guided session with: \(CLICommandDisplay.command("agent start --model \(selectedAgentID)"))")
            if startable.id == DeepseekV4FlashResources.defaultModelId {
                print("  DeepSeek V4 Flash is the preferred 96 GB+ setup-agent tier; smaller tool-capable native agents are alternatives, not upgrades.")
            }
        }
        if let codeReport = reports.first(where: { $0.spec.id == CodeGenResources.defaultModelId }) {
            if codeReport.isSupported {
                print("  Qwen3-Coder Next is also supported for comparison or coding-specific sessions.")
                print("  Pull it with: \(CLICommandDisplay.modelPullCommand(for: CodeGenResources.defaultModelId))")
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
            let isInstalled = report.spec.managedRuntimeURL() != nil
            if let restriction = report.spec.usageRestriction,
               !isInstalled,
               !acceptModelLicense {
                CLIStderr.write(
                    "[\(report.spec.id)] skipping restricted model: \(restriction.summary) "
                    + "Review \(restriction.terms.map(\.licenseURL).joined(separator: ", ")) and pass "
                    + "--accept-model-license to accept the terms and agree to comply with them.\n"
                )
                continue
            }
            if let restriction = report.spec.usageRestriction, !isInstalled, !quiet {
                CLIStderr.write("[\(report.spec.id)] third-party usage terms: \(restriction.summary)\n")
                for term in restriction.terms {
                    CLIStderr.write("  \(term.component): \(term.license) \(term.licenseURL)\n")
                }
                CLIStderr.write("  You are responsible for determining whether your use complies.\n")
            }
            if isInstalled {
                if !quiet {
                    CLIStderr.write("[\(report.spec.id)] already available in the unified model catalog\n")
                }
                continue
            }
            if !quiet {
                CLIStderr.write("[\(report.spec.id)] installing recommended model\n")
            }
            _ = try await ManagedModelResolver.installManagedModel(
                id: report.spec.id,
                force: false,
                usageTermsAcknowledged: acceptModelLicense,
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
        guard recommendation.isStartableByMereRun else {
            throw ValidationError(
                "Pi requires a tool-capable chat model; \(modelID) is only available through the text-code API lane."
            )
        }
        return try SetupAgentRuntime.providerModel(for: recommendation)
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

    @Option(
        name: [.customLong("pi-argument")],
        parsing: .unconditionalSingleValue,
        help: "Additional argument forwarded to Pi. Repeat for each argument."
    )
    var piArguments: [String] = []

    @Option(name: [.long], help: "Working directory for the Pi process.")
    var workingDirectory: String?

    @Flag(name: [.long], help: "Run Pi in the current terminal instead of opening Terminal.app.")
    var inline: Bool = false

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
                "Model \(spec.id) is not installed. Run `\(CLICommandDisplay.modelPullCommand(for: spec.id))` "
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

        var serverProcess: Process?
        defer {
            if let serverProcess, serverProcess.isRunning {
                serverProcess.terminate()
            }
        }
        if skipServer {
            serverProcess = nil
        } else {
            let startedServer = try startAPIServer(modelURL: modelURL, engine: runtime.engine)
            serverProcess = startedServer.process
            CLIStderr.write("[agent] Loading \(runtime.providerModel.id). Server log: \(startedServer.logURL.path)\n")
            try await PiAgentIntegration.waitForHealth(
                host: host,
                port: port,
                timeoutSeconds: runtime.healthTimeoutSeconds,
                serverProcess: startedServer.process,
                serverLogURL: startedServer.logURL
            )
            CLIStderr.write("[agent] Local API is ready. \(PiTerminalLauncher.launchProgressMessage(inline: inline))\n")
        }

        try runPi(
            piURL: piURL,
            modelID: runtime.providerModel.id,
            engine: runtime.engine,
            modelURL: modelURL
        )
        if serverProcess != nil {
            CLIStderr.write("[agent] \(PiTerminalLauncher.runningProgressMessage(inline: inline))\n")
            if PiTerminalLauncher.launchesDetached(inline: inline) {
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
        process.environment = Self.apiServerEnvironment()
        process.standardInput = AgentServerLog.nullInputHandle()
        process.standardOutput = log.handle
        process.standardError = log.handle
        try process.run()
        CLIStderr.write("[agent] Started mere.run API server on \(host):\(port)\n")
        return (process, log.url)
    }

    static func apiServerEnvironment(
        inheriting environment: [String: String] = ProcessInfo.processInfo.environment,
        processID: Int32 = ProcessInfo.processInfo.processIdentifier
    ) -> [String: String] {
        var environment = environment
        environment[PiAgentIntegration.agentParentProcessEnvironment] = String(processID)
        // Agent tool turns repeatedly extend one conversation. Gemma4's
        // cross-request prefix cache can currently retain an incompatible
        // partial KV shape across those turns, which terminates the server.
        // Keep the stable path as the agent default while preserving an
        // explicit opt-in for controlled runtime experiments.
        if environment["MERERUN_GEMMA4_PREFIX_KV_CACHE"] == nil {
            environment["MERERUN_GEMMA4_PREFIX_KV_CACHE"] = "0"
        }
        return environment
    }

    private static func currentExecutableURL() -> URL {
        CurrentExecutable.url()
    }

    private func runPi(piURL: URL, modelID: String, engine: APIEngine, modelURL: URL?) throws {
        let directory = workingDirectory.map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ValidationError("Pi working directory does not exist: \(directory.path)")
        }
        let initialPrompt = piArguments.isEmpty
            ? SetupAgentPrompt.render(
                userRequest: prompt,
                selectedModelID: modelID,
                engine: engine,
                modelURL: modelURL,
                host: host,
                port: port
            )
            : prompt
        try PiTerminalLauncher.launch(
            piURL: piURL,
            modelID: modelID,
            prompt: initialPrompt,
            homeDirectory: PiAgentIntegration.mereRunPiHomeDirectory(),
            workingDirectory: directory,
            additionalArguments: piArguments,
            inline: inline
        )
    }

    private func waitForServerProcess() -> Never {
        while true {
            Thread.sleep(forTimeInterval: 3600)
        }
    }
}
