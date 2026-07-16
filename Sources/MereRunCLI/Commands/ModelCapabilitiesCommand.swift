import ArgumentParser
import Foundation
import MereRunCore

struct ModelCapabilities: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "capabilities",
        abstract: "Show which managed models this machine can run."
    )

    @Flag(name: [.long], help: "Include unsupported models and their reasons.")
    var all: Bool = false

    @Flag(name: [.long], help: "Show only models recommended for first setup.")
    var recommended: Bool = false

    @Flag(name: [.long], help: "Emit a machine-readable capability report.")
    var json: Bool = false

    func run() throws {
        let machine = MereRunMachineProfile.current
        let reports = ManagedModelCapabilityCatalog.supportReports(on: machine)
        let recommendedCodeReport = ManagedModelCapabilityCatalog.recommendedCodeModelReport(on: machine)
        let visibleReports = reports.filter { report in
            if recommended {
                return report.isSupported
                    && report.descriptor.isRecommendedForSetup
                    && report.spec.hasAnyManagedDownloadSource()
            }
            return all || report.isSupported
        }
        let chatBands = ManagedModelCapabilityCatalog.recommendedChatBandReports(on: machine)
        let recommendedReports = ManagedModelCapabilityCatalog.recommendedSetupReports(on: machine)
        let unavailableRecommended = reports.filter {
            $0.isSupported
                && $0.descriptor.isRecommendedForSetup
                && !$0.spec.hasAnyManagedDownloadSource()
        }

        if json {
            let payload = ModelCapabilitiesOutput(
                machine: .init(machine),
                chatBands: chatBands.map { .init($0, machine: machine) },
                recommendedChatModel: chatBands
                    .first { $0.contains(unifiedMemoryGB: machine.unifiedMemoryGB) }
                    .map { .init($0, machine: machine) },
                recommendedCodeModel: recommendedCodeReport.map(ModelCapabilitiesModel.init),
                setupAgent: MereRunAgentModelCatalog
                    .recommendation(for: .tier, on: machine)
                    .map(ModelCapabilitiesSetupAgent.init),
                recommendedSetupModels: recommendedReports.map(ModelCapabilitiesModel.init),
                unavailableRecommendedModelIDs: unavailableRecommended.map { $0.spec.id },
                models: visibleReports.map(ModelCapabilitiesModel.init)
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(payload), as: UTF8.self))
            return
        }

        print("Machine")
        print("  processor: \(machine.processorName)")
        print("  unifiedMemory: \(machine.unifiedMemoryGB) GB")
        print("  appleSiliconMac: \(machine.isAppleSiliconMac)")

        if !chatBands.isEmpty {
            print("\nRecommended chat winners by RAM band")
            for band in chatBands {
                let marker = band.contains(unifiedMemoryGB: machine.unifiedMemoryGB) ? " (this machine)" : ""
                print("  \(band.bandLabel): \(band.modelID)\(marker)")
                print("    \(band.title): \(band.summary)")
                if !band.alternateModelIDs.isEmpty {
                    print("    alternatives: \(band.alternateModelIDs.joined(separator: ", "))")
                }
            }
        }

        if let recommendedCodeReport {
            print("\nRecommended code model")
            print("  \(CLICommandDisplay.modelPullCommand(for: recommendedCodeReport.spec.id))")
            print("  \(recommendedCodeReport.descriptor.title): \(recommendedCodeReport.descriptor.summary)")
        }

        if let agent = MereRunAgentModelCatalog.recommendation(for: .tier, on: machine) {
            print("\nRecommended setup agent")
            print("  \(CLICommandDisplay.command("agent start --model \(agent.id)"))")
            print("  \(agent.displayName): \(agent.summary)")
            if agent.id == DeepseekV4FlashResources.defaultModelId {
                print("  note: DeepSeek V4 Flash is the preferred 96 GB+ setup-agent tier; smaller Qwen agents are lower-memory alternatives, not upgrades.")
            }
        }

        if !recommendedReports.isEmpty {
            print("\nRecommended setup coverage (downloadable from Hugging Face)")
            print("  note: cross-modality starter set; lower-memory agent alternatives are not ranked upgrades.")
            for report in recommendedReports {
                print("  \(CLICommandDisplay.modelPullCommand(for: report.spec.id))")
            }
        }
        if !unavailableRecommended.isEmpty {
            let ids = unavailableRecommended.map(\.spec.id).joined(separator: ", ")
            print("  additional supported recommendations need local model paths: \(ids)")
        }

        print("\nModel capabilities")
        for report in visibleReports {
            printCapability(report)
        }

        if !all, reports.contains(where: { !$0.isSupported }) {
            print("\nRun `mere.run model capabilities --all` to include unsupported models and reasons.")
        }
    }

    private func printCapability(_ report: ManagedModelSupportReport) {
        let status = report.isSupported ? "supported" : "unsupported"
        print("- \(report.spec.id) [\(status)]")
        print("  title: \(report.descriptor.title)")
        print("  category: \(report.spec.category.rawValue)")
        print("  summary: \(report.descriptor.summary)")
        print("  memory: minimum \(report.descriptor.minimumUnifiedMemoryGB) GB, recommended \(report.descriptor.recommendedUnifiedMemoryGB) GB")
        print("  download: \(downloadLabel(for: report.spec))")
        if !report.spec.defaultCLICommands.isEmpty {
            print("  commands: \(report.spec.defaultCLICommands.joined(separator: ", "))")
        }
        if !report.reasons.isEmpty {
            print("  reason: \(report.reasons.joined(separator: " "))")
        }
    }

    private func downloadLabel(for spec: ManagedModelSpec) -> String {
        spec.hubFallback == nil ? "local path only" : "Hugging Face snapshot"
    }
}

struct ModelCapabilitiesOutput: Codable, Equatable {
    let machine: ModelCapabilitiesMachine
    let chatBands: [ModelCapabilitiesChatBand]
    let recommendedChatModel: ModelCapabilitiesChatBand?
    let recommendedCodeModel: ModelCapabilitiesModel?
    let setupAgent: ModelCapabilitiesSetupAgent?
    let recommendedSetupModels: [ModelCapabilitiesModel]
    let unavailableRecommendedModelIDs: [String]
    let models: [ModelCapabilitiesModel]
}

struct ModelCapabilitiesMachine: Codable, Equatable {
    let processor: String
    let unifiedMemoryGB: Int
    let appleSiliconMac: Bool
    let linux: Bool

    init(_ machine: MereRunMachineProfile) {
        processor = machine.processorName
        unifiedMemoryGB = machine.unifiedMemoryGB
        appleSiliconMac = machine.isAppleSiliconMac
        linux = machine.isLinux
    }
}

struct ModelCapabilitiesChatBand: Codable, Equatable {
    let bandLabel: String
    let minimumUnifiedMemoryGB: Int
    let maximumUnifiedMemoryGB: Int?
    let modelID: String
    let title: String
    let summary: String
    let alternateModelIDs: [String]
    let currentMachine: Bool

    init(_ band: ManagedChatModelBandRecommendation, machine: MereRunMachineProfile) {
        bandLabel = band.bandLabel
        minimumUnifiedMemoryGB = band.minimumUnifiedMemoryGB
        maximumUnifiedMemoryGB = band.maximumUnifiedMemoryGB
        modelID = band.modelID
        title = band.title
        summary = band.summary
        alternateModelIDs = band.alternateModelIDs
        currentMachine = band.contains(unifiedMemoryGB: machine.unifiedMemoryGB)
    }
}

struct ModelCapabilitiesSetupAgent: Codable, Equatable {
    let id: String
    let displayName: String
    let summary: String
    let minimumUnifiedMemoryGB: Int
    let recommendedUnifiedMemoryGB: Int
    let servingEngine: String
    let startableByMereRun: Bool
    let sourceConfigurationRequired: Bool
    let reason: String?

    init(_ recommendation: MereRunAgentModelRecommendation) {
        id = recommendation.id
        displayName = recommendation.displayName
        summary = recommendation.summary
        minimumUnifiedMemoryGB = recommendation.minimumUnifiedMemoryGB
        recommendedUnifiedMemoryGB = recommendation.recommendedUnifiedMemoryGB
        servingEngine = recommendation.servingEngine.rawValue
        startableByMereRun = recommendation.isStartableByMereRun
        sourceConfigurationRequired = recommendation.sourceConfigurationRequired
        reason = recommendation.reason
    }
}

struct ModelCapabilitiesModel: Codable, Equatable {
    let id: String
    let title: String
    let category: String
    let summary: String
    let minimumUnifiedMemoryGB: Int
    let recommendedUnifiedMemoryGB: Int
    let supported: Bool
    let reasons: [String]
    let recommendedForSetup: Bool
    let managedDownloadAvailable: Bool
    let download: String
    let commands: [String]

    init(_ report: ManagedModelSupportReport) {
        id = report.spec.id
        title = report.descriptor.title
        category = report.spec.category.rawValue
        summary = report.descriptor.summary
        minimumUnifiedMemoryGB = report.descriptor.minimumUnifiedMemoryGB
        recommendedUnifiedMemoryGB = report.descriptor.recommendedUnifiedMemoryGB
        supported = report.isSupported
        reasons = report.reasons
        recommendedForSetup = report.descriptor.isRecommendedForSetup
        managedDownloadAvailable = report.spec.hasAnyManagedDownloadSource()
        download = report.spec.hubFallback == nil ? "local-path" : "hugging-face"
        commands = report.spec.defaultCLICommands
    }
}
