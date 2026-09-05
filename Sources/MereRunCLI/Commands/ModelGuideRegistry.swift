import ArgumentParser
import Foundation

/// Model handbook metadata and prose ship in the same resource bundle as command cookbooks.
struct ModelGuide: Codable, Equatable {
    let topic: String
    let title: String
    let category: String
    let models: [String]
    let resourceName: String

    var entry: GuideTopic {
        GuideTopic(
            topic: topic, title: title, commandPaths: [], models: models,
            resourceName: resourceName
        )
    }
}

enum ModelGuideRegistry {
    static func all() throws -> [ModelGuide] {
        let content = try GuideRegistry.resourceContent(named: "model-guides.json")
        return try JSONDecoder().decode([ModelGuide].self, from: Data(content.utf8))
    }

    static func guide(for model: String) throws -> ModelGuide {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let guide = try all().first(where: { $0.models.contains(normalized) }) else {
            throw ValidationError("No model guide for \(model). Run `mere.run guide --list-models`.")
        }
        return guide
    }

    static func renderList(json: Bool, markdown: Bool) throws -> String {
        let guides = try all()
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            return String(decoding: try encoder.encode(guides), as: UTF8.self)
        }
        if markdown {
            var lines = ["| Family | Category | Managed models |", "| --- | --- | --- |"]
            lines += guides.map {
                "| \($0.title) | \($0.category) | \($0.models.joined(separator: "<br>")) |"
            }
            return lines.joined(separator: "\n")
        }
        return guides.map { "\($0.title) [\($0.category)]\n  \($0.models.joined(separator: ", "))" }
            .joined(separator: "\n\n") + "\n\nRead a guide with: mere.run guide --model image-klein-9b"
    }
}
