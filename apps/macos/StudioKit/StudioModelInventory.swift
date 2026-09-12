import Foundation

package struct StudioModelCatalogMetadata: Decodable, Equatable {
    package let id: String
    package let title: String
    package let summary: String
    package let minimumUnifiedMemoryGB: Int
    package let recommendedUnifiedMemoryGB: Int
    package let supported: Bool
    package let reasons: [String]
    package let estimatedDownloadBytes: Int64?
    package let sourceRepository: String?
    package let publisher: String?
}

private struct StudioModelCapabilitiesOutput: Decodable {
    let models: [StudioModelCatalogMetadata]
}

package enum StudioModelCatalogParser {
    package static func metadataByID(from output: String) -> [String: StudioModelCatalogMetadata] {
        guard let data = output.data(using: .utf8),
              let payload = try? JSONDecoder().decode(StudioModelCapabilitiesOutput.self, from: data) else {
            return [:]
        }
        return Dictionary(uniqueKeysWithValues: payload.models.map { ($0.id, $0) })
    }

    package static func applying(
        _ metadataByID: [String: StudioModelCatalogMetadata],
        to rows: [StudioModelInventoryRow]
    ) -> [StudioModelInventoryRow] {
        rows.map { row in
            guard let metadata = metadataByID[row.id] else { return row }
            return StudioModelInventoryRow(
                id: row.id,
                category: row.category,
                status: row.status,
                size: row.size,
                usageTerms: row.usageTerms,
                title: metadata.title,
                summary: metadata.summary,
                estimatedDownloadBytes: metadata.estimatedDownloadBytes,
                minimumUnifiedMemoryGB: metadata.minimumUnifiedMemoryGB,
                recommendedUnifiedMemoryGB: metadata.recommendedUnifiedMemoryGB,
                supported: metadata.supported,
                supportReasons: metadata.reasons,
                sourceRepository: metadata.sourceRepository,
                publisher: metadata.publisher,
                referencedBytes: row.referencedBytes,
                reclaimableBytes: row.reclaimableBytes,
                sharedBytes: row.sharedBytes,
                externalBytes: row.externalBytes,
                contextWindow: row.contextWindow
            )
        }
    }
}

package enum StudioModelInventoryParser {
    private struct Document: Decodable {
        struct Inventory: Decodable { let rows: [Row] }
        struct Row: Decodable {
            let id: String
            let category: String
            let status: String
            let size: String?
            let contextWindow: Int?
        }
        let inventory: Inventory
        let usageTerms: [String]
    }

    package static func rows(from output: String) -> [StudioModelInventoryRow] {
        if output.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("{") {
            return (try? decodeRows(from: output)) ?? []
        }
        return legacyRows(from: output)
    }

    package static func decodeRows(from output: String) throws -> [StudioModelInventoryRow] {
        let document = try JSONDecoder().decode(Document.self, from: Data(output.utf8))
        let terms = usageTermsByID(from: document.usageTerms.joined(separator: "\n"))
        return document.inventory.rows.map {
            StudioModelInventoryRow(id: $0.id, category: $0.category, status: $0.status,
                size: $0.size ?? ($0.status == "installed" ? "not measured" : "—"),
                usageTerms: terms[$0.id], contextWindow: $0.contextWindow)
        }
    }

    private static func legacyRows(from output: String) -> [StudioModelInventoryRow] {
        let usageTerms = usageTermsByID(from: output)
        return output
            .components(separatedBy: .newlines)
            .compactMap { line -> StudioModelInventoryRow? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty,
                      !trimmed.hasPrefix("-"),
                      !trimmed.hasPrefix("ID "),
                      !trimmed.hasPrefix("Usage restriction:"),
                      !trimmed.hasPrefix("Usage terms:") else {
                    return nil
                }

                let fields = trimmed.split(whereSeparator: \.isWhitespace).map(String.init)
                guard fields.count >= 4 else { return nil }
                return StudioModelInventoryRow(
                    id: fields[0],
                    category: fields[1],
                    status: fields[2],
                    size: fields.dropFirst(3).joined(separator: " "),
                    usageTerms: usageTerms[fields[0]]
                )
            }
    }

    package static func usageTermsByID(from output: String) -> [String: StudioModelUsageTerms] {
        var result: [String: StudioModelUsageTerms] = [:]
        for line in output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            let marker = "Usage terms: "
            guard trimmed.hasPrefix(marker) else { continue }
            let payload = String(trimmed.dropFirst(marker.count))
            guard let separator = payload.range(of: " - ") else { continue }
            let id = String(payload[..<separator.lowerBound])
            let summary = String(payload[separator.upperBound...])
            let links = summary
                .split(whereSeparator: \.isWhitespace)
                .compactMap { token -> URL? in
                    let candidate = token.trimmingCharacters(in: CharacterSet(charactersIn: "[];"))
                    guard candidate.hasPrefix("https://") else { return nil }
                    return URL(string: candidate)
                }
            result[id] = StudioModelUsageTerms(summary: summary, links: links)
        }
        return result
    }

    package static func modelRoot(from output: String) -> URL? {
        for line in output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            let marker = "Model Root:"
            guard trimmed.hasPrefix(marker) else { continue }
            let path = trimmed.dropFirst(marker.count).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else { return nil }
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return nil
    }
}


package struct StudioModelStorageReport: Decodable {
    package let applicationSupportBytes: Int64
    package let garbageCollectableBytes: Int64
    package let models: [StudioModelStorageUsage]
}

package struct StudioModelStorageUsage: Decodable {
    package let id: String
    package let installed: Bool
    package let referencedBytes: Int64
    package let reclaimableBytes: Int64
    package let sharedBytes: Int64
    package let externalBytes: Int64
}
