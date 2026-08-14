import Foundation
import MereRunCore

enum SetupAgentPrompt {
    static let defaultUserRequest = """
    Guide me through setting up mere.run on this machine. Start by summarizing what this Mac can run, then help me install only supported models.
    """

    static func render(
        userRequest: String,
        selectedModelID: String,
        engine: APIEngine,
        modelURL: URL?,
        host: String,
        port: Int,
        machine: MereRunMachineProfile = .current
    ) -> String {
        let supportedSetup = ManagedModelCapabilityCatalog.recommendedSetupReports(on: machine)
            .map { "\($0.spec.id) (\($0.descriptor.title), recommended \($0.descriptor.recommendedUnifiedMemoryGB) GB)" }
            .joined(separator: ", ")
        let startableAgents = MereRunAgentModelCatalog.allTierRecommendations(on: machine)
            .filter(\.isStartableByMereRun)
            .map { "\($0.id) (\($0.displayName))" }
            .joined(separator: ", ")
        let recommendedAgent = MereRunAgentModelCatalog.recommendation(for: .tier, on: machine)
        let recommendedAgentLine = recommendedAgent.map {
            "\($0.id) (\($0.displayName)): \($0.summary)"
        } ?? "none detected"
        let selectedIsRecommended = recommendedAgent?.id == selectedModelID
        let modelPath = modelURL?.path ?? "already-running server"

        return """
        \(userRequest.trimmingCharacters(in: .whitespacesAndNewlines))

        Current mere.run setup context:
        - Machine: \(machine.processorName), \(machine.unifiedMemoryGB) GB unified memory, Apple Silicon: \(machine.isAppleSiliconMac).
        - The local API server is already configured at http://\(host):\(port)/v1.
        - Pi is already using provider `mere-run` with model `\(selectedModelID)`.
        - Served engine: \(engine.rawValue).
        - Served model path: \(modelPath).
        - Recommended setup-agent tier for this Mac: \(recommendedAgentLine).
        - Selected setup-agent is recommended: \(selectedIsRecommended).
        - Supported recommended managed models: \(supportedSetup.isEmpty ? "none detected" : supportedSetup).
        - Startable setup-agent models on this Mac: \(startableAgents.isEmpty ? "none detected" : startableAgents).

        Setup workflow rules:
        - If the selected setup-agent is recommended, do not suggest another setup/chat agent as an upgrade.
        - On 96 GB+ Apple Silicon Macs, DeepSeek V4 Flash is the preferred premier setup-agent tier; smaller tool-capable native agents are alternatives for lower-memory workflows.
        - Treat the recommended setup-agent line as authoritative for agent/chat setup recommendations; the broader supported managed-model list is cross-modality coverage, not a ranked upgrade list.
        - Do not explore the repository to discover setup facts unless a listed command fails.
        - Do not run demo scripts, sample scripts, `demo.sh`, `scripts/check.sh`, `swift build`, or `swift test` for onboarding.
        - Use `mere.run status` for a quick server, served-model, and installed-model snapshot.
        - First use `mere.run model capabilities --recommended` for the concise supported setup list.
        - Use `mere.run model list` to check what is already installed before suggesting downloads.
        - Use `mere.run model capabilities --all` only when explaining why a model is hidden or unsupported.
        - Only suggest or run `mere.run model pull <id>` for models reported as supported on this Mac.
        - Never pass `--allow-unsupported` unless the user explicitly asks to override the support gate.
        - Ask before starting long downloads or deleting/removing any model.
        - Summarize the next command and why it is needed before running it.
        """
    }
}
