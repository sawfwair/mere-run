import ArgumentParser
import Foundation
import MereRunCore

struct ModelStorage: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "storage",
        abstract: "Inspect physical model storage, sharing, and reclaimable space."
    )

    @Flag(name: [.long], help: "Emit a structured JSON storage report.")
    var json: Bool = false

    func run() throws {
        let report = try ModelStorageManager().report()
        if json {
            print(try ModelStorageCommandOutput.encode(report))
        } else {
            print(ModelStorageCommandOutput.text(report))
        }
    }
}

enum ModelStorageCommandOutput {
    static func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        guard let output = String(data: data, encoding: .utf8) else {
            throw ValidationError("Could not encode model storage output as UTF-8.")
        }
        return output
    }

    static func text(_ report: ModelStorageReport) -> String {
        var lines = [
            "Model storage",
            "  Application support: \(bytes(report.applicationSupportBytes))  \(report.applicationSupportPath)",
            "  Hub payloads:       \(bytes(report.hubBytes))  \(report.hubPath)",
            "  Model links/local:  \(bytes(report.modelStoreBytes))  \(report.modelStorePath)",
            "  Other app data:     \(bytes(report.otherApplicationSupportBytes))",
            "  Safe to collect:    \(bytes(report.garbageCollectableBytes))",
            "  Partial downloads:  \(bytes(report.incompleteDownloadBytes))",
        ]
        let installed = report.models.filter(\.installed)
        if !installed.isEmpty {
            lines.append("")
            lines.append("Installed model references (do not add these values):")
            for model in installed.sorted(by: { $0.id < $1.id }) {
                var detail = "  \(model.id): \(bytes(model.referencedBytes)) referenced"
                detail += " · \(bytes(model.reclaimableBytes)) reclaimable on removal"
                if model.sharedBytes > 0 {
                    detail += " · \(bytes(model.sharedBytes)) shared"
                }
                if model.externalBytes > 0 {
                    detail += " · \(bytes(model.externalBytes)) external (not managed here)"
                }
                lines.append(detail)
            }
        }
        return lines.joined(separator: "\n")
    }

    static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }
}
