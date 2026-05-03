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

    func run() throws {
        let machine = MereRunMachineProfile.current
        let reports = ManagedModelCapabilityCatalog.supportReports(on: machine)
        let visibleReports = reports.filter { report in
            if recommended {
                return report.isSupported
                    && report.descriptor.isRecommendedForSetup
                    && report.spec.hasAnyManagedDownloadSource()
            }
            return all || report.isSupported
        }

        print("Machine")
        print("  processor: \(machine.processorName)")
        print("  unifiedMemory: \(machine.unifiedMemoryGB) GB")
        print("  appleSiliconMac: \(machine.isAppleSiliconMac)")

        let recommendedReports = ManagedModelCapabilityCatalog.recommendedSetupReports(on: machine)
        if !recommendedReports.isEmpty {
            print("\nRecommended setup (downloadable with current configuration)")
            for report in recommendedReports {
                print("  mere.run model pull \(report.spec.id)")
            }
        }
        let unavailableRecommended = reports.filter {
            $0.isSupported
                && $0.descriptor.isRecommendedForSetup
                && !$0.spec.hasAnyManagedDownloadSource()
        }
        if !unavailableRecommended.isEmpty {
            let ids = unavailableRecommended.map(\.spec.id).joined(separator: ", ")
            print("  additional supported recommendations need model-source configuration: \(ids)")
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
        let archiveSize = archiveSizeLabel(for: report.spec)
        print("- \(report.spec.id) [\(status)]")
        print("  title: \(report.descriptor.title)")
        print("  category: \(report.spec.category.rawValue)")
        print("  summary: \(report.descriptor.summary)")
        print("  memory: minimum \(report.descriptor.minimumUnifiedMemoryGB) GB, recommended \(report.descriptor.recommendedUnifiedMemoryGB) GB")
        print("  download: \(archiveSize)")
        if !report.spec.defaultCLICommands.isEmpty {
            print("  commands: \(report.spec.defaultCLICommands.joined(separator: ", "))")
        }
        if !report.reasons.isEmpty {
            print("  reason: \(report.reasons.joined(separator: " "))")
        }
    }

    private func archiveSizeLabel(for spec: ManagedModelSpec) -> String {
        guard let archive = spec.archiveSource else {
            return spec.hubFallback == nil ? "no managed source" : "Hugging Face snapshot"
        }
        guard archive.size > 0 else {
            if spec.hubFallback != nil {
                return "configured source or Hugging Face snapshot"
            }
            return "configured source"
        }
        return ByteCountFormatter.string(fromByteCount: archive.size, countStyle: .file)
    }
}
