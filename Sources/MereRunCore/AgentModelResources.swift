import Foundation

public struct AgentModelResources: Sendable, Hashable {
    public static let qwen35NineBModelId = "text-agent-qwen35-9b"
    public static let qwen35NineBRepoId = "unsloth/Qwen3.5-9B-GGUF"
    public static let qwen35NineBRevision = "main"
    public static let qwen35NineBGGUFFile = "Qwen3.5-9B-Q4_K_M.gguf"
    public static let qwen35NineBRelativePath = "\(qwen35NineBModelId).gguf"

    public static let qwen35NineBHubFallbackConfig = HubFallbackConfig(
        repoId: qwen35NineBRepoId,
        revision: qwen35NineBRevision,
        patterns: [qwen35NineBGGUFFile],
        filePath: qwen35NineBGGUFFile
    )
}

public enum MereRunAgentServingEngine: String, Hashable, Sendable {
    case textCode = "text-code"
    case textChatQ36 = "text-chat-q36"
    case textChatQ35 = "text-chat-q35"
    case deepseekV4Flash = "text-chat-deepseek-v4-flash"
    case sourceConfigured = "external"
}

public enum MereRunAgentModelChoice: String, CaseIterable, Hashable, Sendable {
    case small
    case tier
    case premier
}

public struct MereRunAgentModelRecommendation: Hashable, Sendable {
    public let id: String
    public let displayName: String
    public let summary: String
    public let minimumUnifiedMemoryGB: Int
    public let recommendedUnifiedMemoryGB: Int
    public let servingEngine: MereRunAgentServingEngine
    public let managedModelID: String?
    public let sourceConfigurationRequired: Bool
    public let reason: String?

    public var isStartableByMereRun: Bool {
        managedModelID != nil && !sourceConfigurationRequired
    }

    public init(
        id: String,
        displayName: String,
        summary: String,
        minimumUnifiedMemoryGB: Int,
        recommendedUnifiedMemoryGB: Int,
        servingEngine: MereRunAgentServingEngine,
        managedModelID: String?,
        sourceConfigurationRequired: Bool = false,
        reason: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.summary = summary
        self.minimumUnifiedMemoryGB = minimumUnifiedMemoryGB
        self.recommendedUnifiedMemoryGB = recommendedUnifiedMemoryGB
        self.servingEngine = servingEngine
        self.managedModelID = managedModelID
        self.sourceConfigurationRequired = sourceConfigurationRequired
        self.reason = reason
    }
}

public enum MereRunAgentModelCatalog {
    public static func recommendation(
        for choice: MereRunAgentModelChoice,
        on machine: MereRunMachineProfile = .current
    ) -> MereRunAgentModelRecommendation? {
        guard machine.isSupportedRuntime else {
            return nil
        }
        switch choice {
        case .small:
            return smallRecommendation(on: machine)
        case .tier:
            return tierRecommendation(on: machine)
        case .premier:
            return premierRecommendation(on: machine)
        }
    }

    public static func smallRecommendation(
        on machine: MereRunMachineProfile = .current
    ) -> MereRunAgentModelRecommendation? {
        guard machine.unifiedMemoryGB >= 16 else { return nil }
        return qwen35NineB()
    }

    public static func tierRecommendation(
        on machine: MereRunMachineProfile = .current
    ) -> MereRunAgentModelRecommendation? {
        guard machine.isSupportedRuntime else { return nil }
        if machine.unifiedMemoryGB >= 96 {
            return deepseekV4Flash()
        }
        if machine.unifiedMemoryGB >= 64 {
            return qwen3CoderNext()
        }
        if machine.unifiedMemoryGB >= 24 {
            return q36Nano()
        }
        return smallRecommendation(on: machine)
    }

    public static func premierRecommendation(
        on machine: MereRunMachineProfile = .current
    ) -> MereRunAgentModelRecommendation? {
        guard machine.isSupportedRuntime, machine.unifiedMemoryGB >= 96 else {
            return nil
        }
        return deepseekV4Flash()
    }

    public static func fallbackStartableRecommendation(
        on machine: MereRunMachineProfile = .current
    ) -> MereRunAgentModelRecommendation? {
        guard machine.isSupportedRuntime else { return nil }
        if machine.unifiedMemoryGB >= 96 {
            return deepseekV4Flash()
        }
        if machine.unifiedMemoryGB >= 64 {
            return qwen3CoderNext()
        }
        if machine.unifiedMemoryGB >= 24 {
            return q36Nano()
        }
        return smallRecommendation(on: machine)
    }

    public static func allTierRecommendations(
        on machine: MereRunMachineProfile = .current
    ) -> [MereRunAgentModelRecommendation] {
        [
            qwen35NineB(),
            q36Nano(),
            qwen3CoderNext(),
            deepseekV4Flash(),
        ].filter { machine.isSupportedRuntime && machine.unifiedMemoryGB >= $0.minimumUnifiedMemoryGB }
    }

    private static func qwen35NineB() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: AgentModelResources.qwen35NineBModelId,
            displayName: "Qwen3.5 9B Q4",
            summary: "Small GGUF agent model for setup, tool guidance, and lightweight local tasks.",
            minimumUnifiedMemoryGB: 16,
            recommendedUnifiedMemoryGB: 16,
            servingEngine: .textCode,
            managedModelID: AgentModelResources.qwen35NineBModelId
        )
    }

    private static func q36Nano() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: Q35Resources.q36NanoModelId,
            displayName: "Qwen3.6 35B-A3B OptiQ 4-bit",
            summary: "Higher-quality Qwen3.6 MLX agent tier for Macs with enough memory.",
            minimumUnifiedMemoryGB: 24,
            recommendedUnifiedMemoryGB: 32,
            servingEngine: .textChatQ36,
            managedModelID: Q35Resources.q36NanoModelId
        )
    }

    private static func qwen3CoderNext() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: CodeGenResources.defaultModelId,
            displayName: "Qwen3-Coder Next",
            summary: "Large GGUF coding-agent tier for 64 GB and larger Macs.",
            minimumUnifiedMemoryGB: 64,
            recommendedUnifiedMemoryGB: 96,
            servingEngine: .textCode,
            managedModelID: CodeGenResources.defaultModelId
        )
    }

    private static func deepseekV4Flash() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: DeepseekV4FlashResources.defaultModelId,
            displayName: "DeepSeek V4 Flash IQ2 imatrix",
            summary: "Premier 284B MoE agent tier (~81 GB imatrix-tuned GGUF, the upstream "
                + "README's preferred quant) for 96 GB and larger Macs. Runs on the bundled "
                + "ds4-server engine with a 65,536-token configured context.",
            minimumUnifiedMemoryGB: 96,
            recommendedUnifiedMemoryGB: 128,
            servingEngine: .deepseekV4Flash,
            managedModelID: DeepseekV4FlashResources.defaultModelId
        )
    }
}
