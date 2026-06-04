import ArgumentParser
import Foundation
import MereRunCore

enum SetupMode: String, ExpressibleByArgument, CaseIterable {
    case agent
    case byoa
    case manual
}

enum SetupAgentModelChoice: String, ExpressibleByArgument, CaseIterable {
    case small
    case tier
    case premier

    var coreChoice: MereRunAgentModelChoice {
        switch self {
        case .small: return .small
        case .tier: return .tier
        case .premier: return .premier
        }
    }
}

struct Setup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "setup",
        abstract: "Choose a guided, BYOA, or manual mere.run setup path."
    )

    @Option(name: [.long], help: "Setup mode: agent, byoa, or manual.")
    var mode: SetupMode?

    @Option(name: [.long], help: "Agent model choice: small, tier, or premier.")
    var agentModel: SetupAgentModelChoice = .tier

    @Flag(name: [.long], help: "Install selected setup dependencies and models.")
    var install: Bool = false

    @Flag(name: [.long], help: "Start the selected setup path after installing prerequisites.")
    var start: Bool = false

    @Flag(name: [.long], help: "Print the planned setup steps without installing or starting anything.")
    var dryRun: Bool = false

    @Option(name: [.long], help: "Local API host for the Pi-backed setup agent.")
    var host: String = "127.0.0.1"

    @Option(name: [.long], help: "Local API port for the Pi-backed setup agent.")
    var port: Int = 8080

    @Option(name: [.long], help: "Path to the pi executable. Defaults to the managed install or PATH.")
    var piPath: String?

    @Flag(name: [.short, .long], help: "Suppress install progress output.")
    var quiet: Bool = false

    func run() async throws {
        let resolvedMode = try mode ?? promptForMode()
        switch resolvedMode {
        case .agent:
            try await runAgentSetup()
        case .byoa:
            printBYOAHandoff()
        case .manual:
            printManualSetup()
        }
    }

    private func promptForMode() throws -> SetupMode {
        print("mere.run setup")
        print("  1. Mere agent (powered by Pi)")
        print("  2. Bring your own agent (Claude, Codex)")
        print("  3. Manual setup")
        print("Choose setup path [1]: ", terminator: "")
        guard let line = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines),
              !line.isEmpty else {
            return .agent
        }
        switch line {
        case "1", "agent":
            return .agent
        case "2", "byoa":
            return .byoa
        case "3", "manual":
            return .manual
        default:
            throw ValidationError("Unknown setup path: \(line)")
        }
    }

    private func runAgentSetup() async throws {
        let machine = MereRunMachineProfile.current
        guard machine.isSupportedRuntime else {
            throw ValidationError("The local Mere agent requires Apple Silicon macOS or Linux. Use `mere.run setup --mode byoa` or `--mode manual`.")
        }
        guard let recommendation = MereRunAgentModelCatalog.recommendation(
            for: agentModel.coreChoice,
            on: machine
        ) else {
            if dryRun {
                printUnsupportedAgentPlan(machine: machine)
                return
            }
            throw ValidationError("No local Mere agent model is supported on this machine. Use BYOA or manual setup.")
        }

        printAgentPlan(recommendation: recommendation, machine: machine)
        if dryRun {
            return
        }

        guard recommendation.isStartableByMereRun else {
            let fallback = MereRunAgentModelCatalog.fallbackStartableRecommendation(on: machine)
            let fallbackText = fallback.map { " Use `--agent-model small` or model \($0.id) to start today." } ?? ""
            throw ValidationError("\(recommendation.displayName) requires an external local model before mere.run can start it.\(fallbackText)")
        }

        let runtime = try SetupAgentRuntime.runtime(for: recommendation)
        if install || start {
            try await installAgentRuntime(runtime)
        }
        if start {
            try await startAgentRuntime(runtime)
        }
    }

    private func printAgentPlan(
        recommendation: MereRunAgentModelRecommendation,
        machine: MereRunMachineProfile
    ) {
        printSetupTitle("Mere agent")
        print("")
        print("This machine")
        print("  \(machine.processorName)")
        print("  \(machine.unifiedMemoryGB) GB unified memory")
        print("")
        print("Recommended agent")
        print("  \(recommendation.displayName)")
        print("  \(recommendation.id)")
        print("  \(recommendation.summary)")
        print("")
        print("Fit")
        print("  Minimum memory: \(recommendation.minimumUnifiedMemoryGB) GB")
        print("  Recommended memory: \(recommendation.recommendedUnifiedMemoryGB) GB")
        print("  Serving engine: \(recommendation.servingEngine.rawValue)")
        if recommendation.sourceConfigurationRequired {
            print("")
            print("Status")
            print("  Premier available after external local model setup.")
            if let reason = recommendation.reason {
                print("  \(reason)")
            }
            print("")
            print("Start today")
            printCommand(CLICommandDisplay.command("setup --mode agent --agent-model small --dry-run"))
            printCommand(CLICommandDisplay.command("setup --mode byoa"))
        } else {
            print("")
            print("Next commands")
            printNumberedCommand(
                index: 1,
                title: "Pull the model",
                command: CLICommandDisplay.command("model pull \(recommendation.id)"),
                trailingBlankLine: true
            )
            printNumberedCommand(
                index: 2,
                title: "Configure Pi",
                command: CLICommandDisplay.command("agent onboard --configure-pi --model \(recommendation.id)"),
                trailingBlankLine: true
            )
            printNumberedCommand(
                index: 3,
                title: "Start the guided setup agent",
                command: CLICommandDisplay.command("agent start --model \(recommendation.id)")
            )
        }
        print("")
        print("Guidance")
        print("  Pi acts as a guided executor for local mere.run setup.")
        print("  It should summarize changes before running setup commands.")
    }

    private func printUnsupportedAgentPlan(machine: MereRunMachineProfile) {
        printSetupTitle("Mere agent")
        print("")
        print("This machine")
        print("  \(machine.processorName)")
        print("  \(machine.unifiedMemoryGB) GB unified memory")
        print("")
        print("Requested agent")
        print("  \(agentModel.rawValue)")
        print("")
        print("Status")
        switch agentModel {
        case .small:
            print("  Unavailable.")
            print("  Qwen3.5 9B setup agent requires at least 16 GB unified memory.")
        case .tier:
            print("  Unavailable.")
            print("  No local agent tier is supported on this machine.")
        case .premier:
            print("  Unavailable on this machine.")
            print("  DeepSeek V4 Flash requires at least 96 GB unified memory and a supported runtime.")
            print("  It is the preferred managed setup-agent tier when available.")
        }
        print("")
        print("Available paths")
        printCommand(CLICommandDisplay.command("setup --mode byoa"))
        printCommand(CLICommandDisplay.command("setup --mode manual"))
    }

    private func installAgentRuntime(_ runtime: SetupAgentRuntime) async throws {
        if !quiet {
            CLIStderr.write("[setup] installing \(runtime.spec.id)\n")
        }
        _ = try await ManagedModelResolver.installManagedModel(
            id: runtime.spec.id,
            force: false,
            progress: { progress in
                guard !quiet else { return }
                switch progress {
                case .downloadingBytes(let completed, let total):
                    let done = ByteCountFormatter.string(fromByteCount: completed, countStyle: .file)
                    if let total, total > 0 {
                        let totalText = ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
                        CLIStderr.write("\r[setup] \(runtime.spec.id) \(done) / \(totalText)          ")
                    } else {
                        CLIStderr.write("\r[setup] \(runtime.spec.id) \(done)          ")
                    }
                case .downloadingPercent(let percent, _):
                    CLIStderr.write("\r[setup] \(runtime.spec.id) \(percent)%          ")
                case .extracting:
                    CLIStderr.write("\r[setup] \(runtime.spec.id) extracting          ")
                }
            }
        )
        if PiAgentIntegration.canInstallLatestRelease {
            if !quiet {
                CLIStderr.write("\n[setup] installing Pi\n")
            }
            let piResult = try await PiAgentIntegration.installLatest(force: false) { message in
                guard !quiet else { return }
                CLIStderr.write("[pi] \(message)\n")
            }
            if !quiet {
                CLIStderr.write("[setup] Pi installed at \(piResult.binaryURL.path)\n")
            }
        } else if let piURL = PiAgentIntegration.findPiExecutable(explicitPath: piPath) {
            if !quiet {
                CLIStderr.write("\n[setup] using Pi at \(piURL.path)\n")
            }
        } else if !quiet {
            CLIStderr.write("\n[setup] Pi auto-install unavailable\n")
            CLIStderr.write("[setup] Pi auto-install is not available on this platform. Install Pi separately and pass --pi-path or put pi on PATH.\n")
        }
        let extensionURL = try PiAgentIntegration.writeLocalProviderExtension(
            host: host,
            port: port,
            model: runtime.providerModel,
            homeDirectory: PiAgentIntegration.mereRunPiHomeDirectory()
        )
        print("Pi provider configured: \(extensionURL.path)")
    }

    private func startAgentRuntime(_ runtime: SetupAgentRuntime) async throws {
        guard let modelURL = runtime.spec.managedRuntimeURL() else {
            throw ValidationError("Model \(runtime.spec.id) is not installed. Run `mere.run setup --mode agent --agent-model \(agentModel.rawValue) --install` first.")
        }
        guard let piURL = PiAgentIntegration.findPiExecutable(explicitPath: piPath) else {
            throw PiAgentIntegration.IntegrationError.piBinaryNotFound
        }
        let startedServer = try startAPIServer(modelURL: modelURL, engine: runtime.engine)
        let server = startedServer.process
        defer {
            if server.isRunning {
                server.terminate()
            }
        }
        CLIStderr.write("[setup] Loading \(runtime.providerModel.id). Server log: \(startedServer.logURL.path)\n")
        try await PiAgentIntegration.waitForHealth(host: host, port: port, timeoutSeconds: runtime.healthTimeoutSeconds)
        CLIStderr.write("[setup] Local API is ready. \(PiTerminalLauncher.launchProgressMessage)\n")
        try runPi(
            piURL: piURL,
            modelID: runtime.providerModel.id,
            engine: runtime.engine,
            modelURL: modelURL
        )
        CLIStderr.write("[setup] \(PiTerminalLauncher.runningProgressMessage)\n")
        if PiTerminalLauncher.launchesDetached {
            waitForServerProcess()
        }
    }

    private func startAPIServer(modelURL: URL, engine: APIEngine) throws -> (process: Process, logURL: URL) {
        let log = try AgentServerLog.makeLogHandle(prefix: "setup-api-server")
        let process = Process()
        process.executableURL = CurrentExecutable.url()
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
        CLIStderr.write("[setup] Started mere.run API server on \(host):\(port)\n")
        return (process, log.url)
    }

    private func runPi(piURL: URL, modelID: String, engine: APIEngine, modelURL: URL?) throws {
        try PiTerminalLauncher.launch(
            piURL: piURL,
            modelID: modelID,
            prompt: SetupAgentPrompt.render(
                userRequest: SetupAgentPrompt.defaultUserRequest,
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

    private func printBYOAHandoff() {
        let machine = MereRunMachineProfile.current
        let supported = ManagedModelCapabilityCatalog.supportReports(on: machine)
            .filter(\.isSupported)
            .map(\.spec.id)
            .joined(separator: ", ")
        printSetupTitle("Bring your own agent")
        print("")
        print("Copy this prompt into Claude, Codex, or another local/remote agent:")
        print("")
        print("""
        You are helping me set up mere.run on this machine.
        Machine: \(machine.processorName), \(machine.unifiedMemoryGB) GB memory, Apple Silicon: \(machine.isAppleSiliconMac), Linux: \(machine.isLinux).
        Supported managed models: \(supported.isEmpty ? "none detected" : supported).
        First run `\(CLICommandDisplay.command("model capabilities --all"))`.
        Only pull models that mere.run reports as supported on this machine.
        Prefer `\(CLICommandDisplay.command("setup --mode manual"))` commands when explaining manual steps.
        Summarize every proposed change before running commands.
        """)
    }

    private func printManualSetup() {
        let machine = MereRunMachineProfile.current
        let recommendation = MereRunAgentModelCatalog.fallbackStartableRecommendation(on: machine)
        printSetupTitle("Manual setup")
        print("")
        print("Commands")
        printNumberedCommand(
            index: 1,
            title: "Inspect this machine",
            command: CLICommandDisplay.command("model capabilities --all"),
            trailingBlankLine: true
        )
        if let recommendation {
            printNumberedCommand(
                index: 2,
                title: "Pull the setup agent model",
                command: CLICommandDisplay.command("model pull \(recommendation.id)"),
                trailingBlankLine: true
            )
            printNumberedCommand(
                index: 3,
                title: "Serve locally",
                command: """
                \(CLICommandDisplay.command("api serve --engine \(recommendation.servingEngine.rawValue) --model <installed-model-path>"))
                """,
                trailingBlankLine: true
            )
        } else {
            print("")
            print("Local agent")
            print("  This machine is not eligible for a local Mere agent.")
            print("  Choose supported task models from capabilities instead.")
        }
        printNumberedCommand(
            index: 4,
            title: "Optionally install Pi",
            command: CLICommandDisplay.command("agent install-pi")
        )
        print("")
        print("Docs")
        print("  README.md")
        print("  docs/cli.md")
    }

    private func printSetupTitle(_ subtitle: String) {
        print("mere.run setup")
        print(subtitle)
    }

    private func printNumberedCommand(
        index: Int,
        title: String,
        command: String,
        trailingBlankLine: Bool = false
    ) {
        print("  \(index). \(title)")
        printCommand(command)
        if trailingBlankLine {
            print("")
        }
    }

    private func printCommand(_ command: String) {
        let normalized = command
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        print("     \(normalized)")
    }

}

struct SetupAgentRuntime {
    let recommendation: MereRunAgentModelRecommendation
    let spec: ManagedModelSpec
    let engine: APIEngine
    let providerModel: PiProviderModel

    var healthTimeoutSeconds: TimeInterval {
        switch engine {
        case .textChatDeepseekV4Flash:
            return DeepseekV4FlashResources.serverStartupTimeoutSeconds + 30
        default:
            return 60
        }
    }

    static func recommendation(forManagedModelID modelID: String) -> MereRunAgentModelRecommendation? {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return MereRunAgentModelCatalog.allTierRecommendations(
            on: MereRunMachineProfile(
                physicalMemoryBytes: 512 * 1_073_741_824,
                processorName: "Apple Silicon",
                isAppleSiliconMac: true
            )
        )
        .first { $0.managedModelID == normalized || $0.id == normalized }
    }

    static func runtime(for recommendation: MereRunAgentModelRecommendation) throws -> SetupAgentRuntime {
        guard let modelID = recommendation.managedModelID,
              let spec = ManagedModelCatalog.spec(for: modelID) else {
            throw ValidationError("\(recommendation.displayName) is not managed by mere.run yet.")
        }
        let engine: APIEngine
        switch recommendation.servingEngine {
        case .textCode:
            engine = .textCode
        case .textChatQ36:
            engine = .textChatQ36
        case .textChatQ35:
            engine = .textChatQ36
        case .deepseekV4Flash:
            engine = .textChatDeepseekV4Flash
        case .sourceConfigured:
            throw ValidationError("\(recommendation.displayName) requires an external local model before it can be started.")
        }
        return SetupAgentRuntime(
            recommendation: recommendation,
            spec: spec,
            engine: engine,
            providerModel: providerModel(for: recommendation)
        )
    }

    static func runtime(forManagedModelID modelID: String) throws -> SetupAgentRuntime {
        guard let recommendation = recommendation(forManagedModelID: modelID) else {
            throw ValidationError("Unsupported managed agent model: \(modelID)")
        }
        return try runtime(for: recommendation)
    }

    static func providerModel(for recommendation: MereRunAgentModelRecommendation) -> PiProviderModel {
        // DeepSeek V4 Flash has a specific Pi compat profile (DSML thinking
        // format, reasoning effort, etc.) documented in the ds4 README.
        if recommendation.servingEngine == .deepseekV4Flash {
            return .deepseekV4Flash
        }
        return PiProviderModel(
            id: recommendation.id,
            name: "\(recommendation.displayName) (mere.run)",
            contextWindow: contextWindow(for: recommendation),
            maxTokens: 4096
        )
    }

    private static func contextWindow(for recommendation: MereRunAgentModelRecommendation) -> Int {
        switch recommendation.servingEngine {
        case .textChatQ36, .textChatQ35:
            return Q35Resources.defaultContextLength
        case .deepseekV4Flash:
            return DeepseekV4FlashResources.defaultContextLength
        case .textCode, .sourceConfigured:
            return 32768
        }
    }
}
