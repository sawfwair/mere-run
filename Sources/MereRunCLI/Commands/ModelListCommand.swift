import ArgumentParser
import Foundation
import MereRunCore

struct ModelList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List all known models with install status."
    )

    @Flag(
        name: [.long],
        help: "Measure referenced payload sizes (slower; follows model-store symlinks)."
    )
    var measureSizes = false

    @Flag(name: .long, help: "Print typed inventory, declared context windows, and usage terms as JSON.")
    var json = false

    private struct Document: Encodable {
        let inventory: ModelInventorySnapshot
        let usageTerms: [String]
    }

    func run() throws {
        let mode: ModelInventoryMode = measureSizes ? .measured : .fast
        let snapshot = ModelInventory.snapshot(mode: mode)
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let document = Document(inventory: snapshot, usageTerms: Self.usageRestrictionLines())
            print(String(decoding: try encoder.encode(document), as: UTF8.self))
            return
        }
        let rows = snapshot.rows

        let widths = ModelListColumnWidths(rows: rows)
        printRow("ID", "Category", "Status", "Referenced", widths: widths)
        print(String(repeating: "-", count: widths.totalWidth))
        for row in rows {
            let size = row.size ?? (row.isInstalled ? "not measured" : "—")
            printRow(row.id, row.category, row.status, size, widths: widths)
        }
        for line in Self.usageRestrictionLines() {
            print("\n\(line)")
        }
    }

    private func printRow(_ id: String, _ category: String, _ status: String, _ size: String, widths: ModelListColumnWidths) {
        let row = [
            id.padding(toLength: widths.id, withPad: " ", startingAt: 0),
            category.padding(toLength: widths.category, withPad: " ", startingAt: 0),
            status.padding(toLength: widths.status, withPad: " ", startingAt: 0),
            size
        ].joined(separator: "  ")
        print(row)
    }

    static func usageRestrictionLines(
        specs: [ManagedModelSpec] = ManagedModelCatalog.allSpecs
    ) -> [String] {
        specs.compactMap { spec in
            guard let restriction = spec.usageRestriction else { return nil }
            let terms = restriction.terms
                .map { "\($0.component): \($0.license) \($0.licenseURL)" }
                .joined(separator: "; ")
            return "Usage terms: \(spec.id) - \(restriction.summary) [\(terms)]"
        }
    }

}

private struct ModelListColumnWidths {
    let id: Int
    let category: Int
    let status: Int

    init(rows: [ModelInventoryRow]) {
        self.id = max("ID".count, rows.map(\.id.count).max() ?? 0)
        self.category = max("Category".count, rows.map(\.category.count).max() ?? 0)
        self.status = max("Status".count, rows.map(\.status.count).max() ?? 0)
    }

    var totalWidth: Int {
        id + category + status + "Referenced".count + 6
    }
}
