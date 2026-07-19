import ArgumentParser
import Foundation
import MereRunCore

struct ModelGarbageCollect: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gc",
        abstract: "Find or delete unreferenced model payloads and partial downloads.",
        discussion: """
        The default is a read-only dry run. Pass --force to delete exactly the
        items in a freshly computed plan. Payloads referenced by any current or
        legacy MereRun install are preserved.
        """
    )

    @Flag(name: [.long], help: "Delete the planned garbage. Without this flag, only show the plan.")
    var force: Bool = false

    @Flag(name: [.long], help: "Emit a structured JSON plan and result.")
    var json: Bool = false

    func run() throws {
        let manager = try ModelStorageManager()
        let plan = try manager.garbageCollectionPlan()
        let result = force ? try manager.execute(plan) : nil
        let output = ModelGarbageCollectOutput(
            mode: force ? "executed" : "dry-run",
            plan: plan,
            result: result
        )

        if json {
            print(try ModelStorageCommandOutput.encode(output))
        } else {
            print(Self.text(output))
        }
    }

    static func text(_ output: ModelGarbageCollectOutput) -> String {
        var lines = [
            output.mode == "executed" ? "Model storage cleanup complete" : "Model storage cleanup dry run",
            "  Reclaimable: \(ModelStorageCommandOutput.bytes(output.plan.reclaimableBytes))",
            "  Partial downloads: \(ModelStorageCommandOutput.bytes(output.plan.incompleteDownloadBytes))",
            "  Items: \(output.plan.items.count)",
        ]
        for item in output.plan.items {
            lines.append(
                "  - \(item.kind.rawValue): \(ModelStorageCommandOutput.bytes(item.logicalBytes))  \(item.path)"
            )
        }
        if let result = output.result {
            lines.append("  Reclaimed: \(ModelStorageCommandOutput.bytes(result.reclaimedBytes))")
        } else if !output.plan.items.isEmpty {
            lines.append("")
            lines.append("Run `mere.run model gc --force` to delete this plan after it is recomputed.")
        }
        return lines.joined(separator: "\n")
    }
}

struct ModelGarbageCollectOutput: Codable, Equatable {
    let mode: String
    let plan: ModelStorageGarbagePlan
    let result: ModelStorageGarbageResult?
}
