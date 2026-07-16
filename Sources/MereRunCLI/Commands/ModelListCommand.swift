import ArgumentParser
import Foundation
import MereRunCore

struct ModelList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List all known models with install status."
    )

    func run() throws {
        let rows = ModelInventory.rows()

        let widths = ModelListColumnWidths(rows: rows)
        printRow("ID", "Category", "Status", "Size", widths: widths)
        print(String(repeating: "-", count: widths.totalWidth))
        for row in rows {
            printRow(row.id, row.category, row.status, row.size, widths: widths)
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
            return "Usage restriction: \(spec.id) - \(restriction.summary) \(restriction.licenseURL)"
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
        id + category + status + "Size".count + 6
    }
}
