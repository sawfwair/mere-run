import Foundation
#if canImport(Darwin)
import Darwin
#endif

public struct MereRunMachineProfile: Hashable, Sendable {
    public let physicalMemoryBytes: UInt64
    public let processorName: String
    public let isAppleSiliconMac: Bool
    public let isLinux: Bool

    public init(
        physicalMemoryBytes: UInt64,
        processorName: String,
        isAppleSiliconMac: Bool,
        isLinux: Bool = false
    ) {
        self.physicalMemoryBytes = physicalMemoryBytes
        self.processorName = processorName
        self.isAppleSiliconMac = isAppleSiliconMac
        self.isLinux = isLinux
    }

    public var unifiedMemoryGB: Int {
        max(1, Int(physicalMemoryBytes / 1_073_741_824))
    }

    public var isSupportedRuntime: Bool {
        isAppleSiliconMac || isLinux
    }

    public static var current: MereRunMachineProfile {
        MereRunMachineProfile(
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            processorName: currentProcessorName(),
            isAppleSiliconMac: currentIsAppleSiliconMac(),
            isLinux: currentIsLinux()
        )
    }

    private static func currentIsAppleSiliconMac() -> Bool {
        #if os(macOS) && arch(arm64)
        return true
        #else
        return false
        #endif
    }

    private static func currentIsLinux() -> Bool {
        #if os(Linux)
        return true
        #else
        return false
        #endif
    }

    private static func currentProcessorName() -> String {
        #if canImport(Darwin)
        if let brand = sysctlString("machdep.cpu.brand_string"), !brand.isEmpty {
            if brand.hasPrefix("Apple ") {
                return String(brand.dropFirst("Apple ".count))
            }
            return brand
        }
        #endif
        #if os(Linux)
        return "Linux"
        #else
        return currentIsAppleSiliconMac() ? "Apple Silicon" : "Unknown processor"
        #endif
    }

    #if canImport(Darwin)
    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
            return nil
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
            return nil
        }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
    #endif
}

public struct ManagedModelCapabilityDescriptor: Hashable, Sendable {
    public let modelID: String
    public let title: String
    public let summary: String
    public let minimumUnifiedMemoryGB: Int
    public let recommendedUnifiedMemoryGB: Int
    public let isRecommendedForSetup: Bool

    public init(
        modelID: String,
        title: String,
        summary: String,
        minimumUnifiedMemoryGB: Int,
        recommendedUnifiedMemoryGB: Int,
        isRecommendedForSetup: Bool
    ) {
        self.modelID = modelID
        self.title = title
        self.summary = summary
        self.minimumUnifiedMemoryGB = minimumUnifiedMemoryGB
        self.recommendedUnifiedMemoryGB = recommendedUnifiedMemoryGB
        self.isRecommendedForSetup = isRecommendedForSetup
    }
}

public enum ManagedModelSupportStatus: String, Hashable, Sendable {
    case supported
    case unsupported
}

public struct ManagedModelSupportReport: Hashable, Sendable {
    public let spec: ManagedModelSpec
    public let descriptor: ManagedModelCapabilityDescriptor
    public let machine: MereRunMachineProfile
    public let status: ManagedModelSupportStatus
    public let reasons: [String]

    public var isSupported: Bool {
        status == .supported
    }

    public var memoryHeadroomGB: Int {
        machine.unifiedMemoryGB - descriptor.minimumUnifiedMemoryGB
    }

    public var meetsRecommendedMemory: Bool {
        machine.unifiedMemoryGB >= descriptor.recommendedUnifiedMemoryGB
    }
}

public struct ManagedChatModelBandRecommendation: Hashable, Sendable {
    public let minimumUnifiedMemoryGB: Int
    public let maximumUnifiedMemoryGB: Int?
    public let modelID: String
    public let title: String
    public let summary: String
    public let alternateModelIDs: [String]

    public init(
        minimumUnifiedMemoryGB: Int,
        maximumUnifiedMemoryGB: Int?,
        modelID: String,
        title: String,
        summary: String,
        alternateModelIDs: [String] = []
    ) {
        self.minimumUnifiedMemoryGB = minimumUnifiedMemoryGB
        self.maximumUnifiedMemoryGB = maximumUnifiedMemoryGB
        self.modelID = modelID
        self.title = title
        self.summary = summary
        self.alternateModelIDs = alternateModelIDs
    }

    public var bandLabel: String {
        if let maximumUnifiedMemoryGB {
            return "\(minimumUnifiedMemoryGB)-\(maximumUnifiedMemoryGB) GB"
        }
        return "\(minimumUnifiedMemoryGB)+ GB"
    }

    public func contains(unifiedMemoryGB: Int) -> Bool {
        guard unifiedMemoryGB >= minimumUnifiedMemoryGB else { return false }
        if let maximumUnifiedMemoryGB {
            return unifiedMemoryGB <= maximumUnifiedMemoryGB
        }
        return true
    }
}

public enum Ornith35BMLXPerformanceProfile: String, CaseIterable, Hashable, Sendable {
    case speed
    case balanced
    case quality
}

public struct ManagedOrnith35BMLXProfileRecommendation: Hashable, Sendable {
    public let profile: Ornith35BMLXPerformanceProfile
    public let modelID: String
    public let summary: String

    public init(
        profile: Ornith35BMLXPerformanceProfile,
        modelID: String,
        summary: String
    ) {
        self.profile = profile
        self.modelID = modelID
        self.summary = summary
    }
}

public enum ManagedModelCapabilityCatalog {
    private static let descriptorsByID: [String: ManagedModelCapabilityDescriptor] = {
        Dictionary(
            uniqueKeysWithValues: ManagedModelCatalog.allKnownSpecs.compactMap { spec in
                spec.capabilityDescriptor.map { (spec.id, $0) }
            }
        )
    }()

    public static var allDescriptors: [ManagedModelCapabilityDescriptor] {
        ManagedModelCatalog.allSpecs.compactMap { descriptorsByID[$0.id] }
    }

    public static func descriptor(for spec: ManagedModelSpec) -> ManagedModelCapabilityDescriptor {
        if let descriptor = descriptorsByID[spec.id] {
            return descriptor
        }
        preconditionFailure("Missing managed model capability descriptor for \(spec.id)")
    }

    public static func descriptor(for modelID: String) -> ManagedModelCapabilityDescriptor? {
        descriptorsByID[modelID]
    }

    public static func support(
        for spec: ManagedModelSpec,
        on machine: MereRunMachineProfile = .current
    ) -> ManagedModelSupportReport {
        let descriptor = descriptor(for: spec)
        var reasons: [String] = []

        if !machine.isSupportedRuntime {
            reasons.append("Apple Silicon macOS or Linux is required.")
        }

        if spec.validationKind == .magentaRT2 && !machine.isAppleSiliconMac {
            reasons.append("Magenta RT2 requires Apple Silicon macOS.")
        }

        if spec.validationKind == .insightFaceBuffaloL && !machine.isAppleSiliconMac {
            reasons.append("InsightFace Buffalo-L requires the macOS ONNX Runtime build.")
        }

        if spec.id == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue && !machine.isAppleSiliconMac {
            reasons.append("MiniMax-H3 FastH3 VSA requires the Metal sparse-attention kernels on Apple Silicon macOS.")
        }

        if spec.validationKind == .terramindFlood && !machine.isAppleSiliconMac {
            reasons.append("TerraMind Flood requires Apple Silicon macOS.")
        }

        if spec.validationKind == .terramindFire && !machine.isAppleSiliconMac {
            reasons.append("TerraMind Fire requires Apple Silicon macOS.")
        }

        if machine.unifiedMemoryGB < descriptor.minimumUnifiedMemoryGB {
            reasons.append(
                "Requires at least \(descriptor.minimumUnifiedMemoryGB) GB unified memory; detected \(machine.unifiedMemoryGB) GB."
            )
        }

        return ManagedModelSupportReport(
            spec: spec,
            descriptor: descriptor,
            machine: machine,
            status: reasons.isEmpty ? .supported : .unsupported,
            reasons: reasons
        )
    }

    public static func supportReports(
        on machine: MereRunMachineProfile = .current
    ) -> [ManagedModelSupportReport] {
        ManagedModelCatalog.allSpecs.map { support(for: $0, on: machine) }
    }

    public static func recommendedSetupReports(
        on machine: MereRunMachineProfile = .current
    ) -> [ManagedModelSupportReport] {
        supportReports(on: machine).filter {
            $0.isSupported
                && $0.descriptor.isRecommendedForSetup
                && $0.spec.hasAnyManagedDownloadSource()
        }
    }

    public static func recommendedChatBandReports(
        on machine: MereRunMachineProfile = .current
    ) -> [ManagedChatModelBandRecommendation] {
        let q36ModelID = machine.isLinux ? ModelResolver.ModelID.q36NanoGGUF.rawValue : Q35Resources.q36NanoModelId
        let defaultAssistantModelID = machine.isLinux ? q36ModelID : Gemma4Resources.twelveB4BitModelId
        let defaultAssistantSummary: String
        let headroomAssistantSummary: String
        if machine.isLinux {
            defaultAssistantSummary = "Q36 GGUF stays the Linux CUDA chat default; Gemma 12B 4-bit is the Apple Silicon local-assistant default."
            headroomAssistantSummary = "Keep the optimized Q36 GGUF Linux lane; use headroom for context, concurrency, or larger alternates."
        } else {
            defaultAssistantSummary = "Gemma 12B 4-bit is the conservative default for grounded local chat; use Turbo when you want to spend more memory."
            headroomAssistantSummary = "Keep the proven Gemma local-assistant lane as default; use the headroom for context, concurrency, or larger Gemma alternates."
        }
        return [
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 8,
                maximumUnifiedMemoryGB: 15,
                modelID: LFM2Resources.smallQADModelId,
                title: "Memory-first native chat",
                summary: "QAD-trained 1.2B 4-bit is the compact chat pick; step up to the optimized 2.6B MLX lane when its larger resident footprint fits.",
                alternateModelIDs: [
                    LFM2Resources.denseModelId,
                    LFM2Resources.denseQADModelId,
                    LFM2Resources.smallModelId,
                ]
            ),
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 16,
                maximumUnifiedMemoryGB: 23,
                modelID: Gemma4Resources.twelveB4BitModelId,
                title: "Compact native chat",
                summary: "Best first chat pick for the smallest supported RAM band; use nano only when the 12B 4-bit lane is too tight.",
                alternateModelIDs: [
                    Gemma4Resources.nanoModelId,
                    LFM2Resources.denseModelId,
                    LFM2Resources.defaultModelId,
                ]
            ),
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 24,
                maximumUnifiedMemoryGB: 63,
                modelID: defaultAssistantModelID,
                title: "Default local assistant",
                summary: defaultAssistantSummary,
                alternateModelIDs: [Gemma4Resources.turboModelId, q36ModelID]
            ),
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 64,
                maximumUnifiedMemoryGB: 95,
                modelID: defaultAssistantModelID,
                title: "Gemma chat with headroom",
                summary: headroomAssistantSummary,
                alternateModelIDs: [Gemma4Resources.maxModelId, Gemma4Resources.turboModelId, q36ModelID]
            ),
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 96,
                maximumUnifiedMemoryGB: nil,
                modelID: DeepseekV4FlashResources.defaultModelId,
                title: "Premier agent/API chat",
                summary: "Highest-tier agent/API chat model; keep Gemma 12B 4-bit as the normal interactive local-assistant fallback.",
                alternateModelIDs: [Gemma4Resources.twelveB4BitModelId, Gemma4Resources.maxModelId, q36ModelID]
            ),
        ]
    }

    public static func recommendedCodeModelReport(
        on machine: MereRunMachineProfile = .current
    ) -> ManagedModelSupportReport? {
        let candidateIDs: [String]
        if machine.unifiedMemoryGB >= 64 {
            candidateIDs = [
                CodeGenResources.defaultModelId,
                NorthMiniCodeResources.modelId,
                AgentModelResources.qwen35NineBModelId,
            ]
        } else if machine.unifiedMemoryGB >= 24 {
            candidateIDs = [
                NorthMiniCodeResources.modelId,
                AgentModelResources.qwen35NineBModelId,
            ]
        } else {
            candidateIDs = [
                AgentModelResources.qwen35NineBModelId,
            ]
        }

        return candidateIDs.compactMap { modelID -> ManagedModelSupportReport? in
            guard let spec = ManagedModelCatalog.spec(for: modelID) else { return nil }
            let report = support(for: spec, on: machine)
            return report.isSupported ? report : nil
        }.first
    }

    /// Select explicit, reproducible Ornith MLX ids for the current RAM tier.
    /// The speed and balanced lanes stay quantized even on large Macs; quality
    /// is the only profile that advances to BF16 when sufficient headroom is
    /// available. The shared BF16 MTP companion is included by `model pull`.
    public static func recommendedOrnith35BMLXProfiles(
        on machine: MereRunMachineProfile = .current
    ) -> [ManagedOrnith35BMLXProfileRecommendation] {
        guard machine.isAppleSiliconMac, machine.unifiedMemoryGB >= 32 else {
            return []
        }

        let balancedModelID = machine.unifiedMemoryGB >= 48
            ? Q35Resources.ornith35BMLX6BitModelId
            : Q35Resources.ornith35BMLX4BitModelId
        let qualityModelID: String
        if machine.unifiedMemoryGB >= 96 {
            qualityModelID = Q35Resources.ornith35BMLXModelId
        } else if machine.unifiedMemoryGB >= 64 {
            qualityModelID = Q35Resources.ornith35BMLX8BitModelId
        } else if machine.unifiedMemoryGB >= 48 {
            qualityModelID = Q35Resources.ornith35BMLX6BitModelId
        } else {
            qualityModelID = Q35Resources.ornith35BMLX4BitModelId
        }

        return [
            ManagedOrnith35BMLXProfileRecommendation(
                profile: .speed,
                modelID: Q35Resources.ornith35BMLX4BitModelId,
                summary: "Highest decode speed and smallest residency; use for interactive agent loops."
            ),
            ManagedOrnith35BMLXProfileRecommendation(
                profile: .balanced,
                modelID: balancedModelID,
                summary: machine.unifiedMemoryGB >= 48
                    ? "Q6 is the default speed/quality compromise with useful context headroom."
                    : "Q4 is the only 35B MLX tier with conservative headroom in this RAM band."
            ),
            ManagedOrnith35BMLXProfileRecommendation(
                profile: .quality,
                modelID: qualityModelID,
                summary: qualityModelID == Q35Resources.ornith35BMLXModelId
                    ? "BF16 preserves the official unquantized target when this machine has enough headroom."
                    : "Uses the highest-precision quantized tier with conservative headroom on this machine."
            ),
        ]
    }

    public static func supportedCodeBenchmarkModelIDs(
        on machine: MereRunMachineProfile = .current
    ) -> [String] {
        [
            Q35Resources.ornith9BModelId,
            Q35Resources.ornith35BMLX4BitModelId,
            Q35Resources.ornith35BMLX6BitModelId,
            Q35Resources.ornith35BMLX8BitModelId,
            Q35Resources.ornith35BMLXModelId,
            NorthMiniCodeResources.modelId,
            CodeGenResources.defaultModelId,
        ].filter { modelID in
            guard let spec = ManagedModelCatalog.spec(for: modelID),
                  spec.category == .textCode
                    || spec.defaultCLICommands.contains("model benchmark code") else {
                return false
            }
            return support(for: spec, on: machine).isSupported
        }
    }

}
