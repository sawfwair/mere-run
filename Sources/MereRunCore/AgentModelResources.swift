import Foundation

public struct AgentModelResources: Sendable, Hashable {
    public static let qwen35NineBModelId = "text-agent-qwen35-9b"
    public static let qwen35NineBRepoId = "unsloth/Qwen3.5-9B-GGUF"
    public static let qwen35NineBRevision = "main"
    public static let qwen35NineBGGUFFile = "Qwen3.5-9B-Q4_K_M.gguf"
    public static let qwen35NineBArchiveSize: Int64 = 0
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
    case textChatQ35 = "text-chat-q35"
    case sourceConfigured = "source-configured"
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
        guard machine.isAppleSiliconMac else {
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
        guard machine.isAppleSiliconMac else { return nil }
        if machine.unifiedMemoryGB >= 96 {
            return qwen35122B4Bit()
        }
        if machine.unifiedMemoryGB >= 64 {
            return qwen3CoderNext()
        }
        if machine.unifiedMemoryGB >= 24 {
            return q35Nano()
        }
        return smallRecommendation(on: machine)
    }

    public static func premierRecommendation(
        on machine: MereRunMachineProfile = .current
    ) -> MereRunAgentModelRecommendation? {
        guard machine.isAppleSiliconMac, machine.unifiedMemoryGB >= 96 else {
            return nil
        }
        return machine.unifiedMemoryGB >= 128 ? qwen35122B8Bit() : qwen35122BMXFP4()
    }

    public static func fallbackStartableRecommendation(
        on machine: MereRunMachineProfile = .current
    ) -> MereRunAgentModelRecommendation? {
        guard machine.isAppleSiliconMac else { return nil }
        if machine.unifiedMemoryGB >= 96 {
            return qwen35122B4Bit()
        }
        if machine.unifiedMemoryGB >= 64 {
            return qwen3CoderNext()
        }
        if machine.unifiedMemoryGB >= 24 {
            return q35Nano()
        }
        return smallRecommendation(on: machine)
    }

    public static func allTierRecommendations(
        on machine: MereRunMachineProfile = .current
    ) -> [MereRunAgentModelRecommendation] {
        [
            qwen35NineB(),
            q35Nano(),
            qwen3CoderNext(),
            qwen35122B4Bit(),
            qwen35122BMXFP4(),
            qwen35122B8Bit(),
        ].filter { machine.isAppleSiliconMac && machine.unifiedMemoryGB >= $0.minimumUnifiedMemoryGB }
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

    private static func q35Nano() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: Q35Resources.nanoModelId,
            displayName: "Qwen3.5 35B-A3B 4-bit",
            summary: "Higher-quality Q35 MLX agent tier for Macs with enough memory.",
            minimumUnifiedMemoryGB: 24,
            recommendedUnifiedMemoryGB: 32,
            servingEngine: .textChatQ35,
            managedModelID: Q35Resources.nanoModelId
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

    private static func qwen35122B4Bit() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: Q35Resources.defaultModelId,
            displayName: "Qwen3.5-122B-A10B 4-bit",
            summary: "Premier 122B MoE 4-bit agent tier for 96 GB and larger Macs.",
            minimumUnifiedMemoryGB: 96,
            recommendedUnifiedMemoryGB: 128,
            servingEngine: .textChatQ35,
            managedModelID: Q35Resources.defaultModelId
        )
    }

    private static func qwen35122BMXFP4() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: "text-agent-qwen35-122b-a10b-mxfp4",
            displayName: "Qwen3.5-122B-A10B mxfp4",
            summary: "Premier 122B MoE agent tier for 96 GB and larger Macs.",
            minimumUnifiedMemoryGB: 96,
            recommendedUnifiedMemoryGB: 128,
            servingEngine: .sourceConfigured,
            managedModelID: nil,
            sourceConfigurationRequired: true,
            reason: "Premier source configuration is not managed by mere.run yet."
        )
    }

    private static func qwen35122B8Bit() -> MereRunAgentModelRecommendation {
        MereRunAgentModelRecommendation(
            id: "text-agent-qwen35-122b-a10b-8bit",
            displayName: "Qwen3.5-122B-A10B 8-bit",
            summary: "Premier 122B 8-bit agent tier for 128 GB and larger Macs.",
            minimumUnifiedMemoryGB: 128,
            recommendedUnifiedMemoryGB: 160,
            servingEngine: .sourceConfigured,
            managedModelID: nil,
            sourceConfigurationRequired: true,
            reason: "Premier source configuration is not managed by mere.run yet."
        )
    }
}
