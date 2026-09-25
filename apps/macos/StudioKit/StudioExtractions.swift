import Foundation

/// Editable GLiNER entity, relation, and structure request for Text > Extract.
package struct StudioExtractionDocument: Codable, Equatable {
    package struct Term: Codable, Equatable, Identifiable {
        package var id = UUID()
        package var name = ""
        package var detail = ""

        package init(name: String = "", detail: String = "") {
            self.name = name
            self.detail = detail
        }
    }

    package struct Field: Codable, Equatable, Identifiable {
        package var id = UUID()
        package var name = ""
        package var detail = ""
        package var multiple = true

        package init(name: String = "", detail: String = "", multiple: Bool = true) {
            self.name = name
            self.detail = detail
            self.multiple = multiple
        }
    }

    package struct Structure: Codable, Equatable, Identifiable {
        package var id = UUID()
        package var name = ""
        package var fields: [Field] = [.init()]

        package init(name: String = "", fields: [Field] = [.init()]) {
            self.name = name
            self.fields = fields
        }
    }

    package var text = ""
    package var entities: [Term] = []
    package var relations: [Term] = []
    package var structures: [Structure] = []
    package var classifications: [StudioClassificationDocument.Task] = []
    package var threshold = 0.5

    package init() {}

    package static let example: StudioExtractionDocument = {
        var value = StudioExtractionDocument()
        value.text = "Alice Smith joined Acme in Paris in 2024."
        value.entities = [.init(name: "person"), .init(name: "organization"), .init(name: "location")]
        value.relations = [.init(name: "works_for"), .init(name: "located_in")]
        value.structures = [Structure(name: "employment", fields: [
            Field(name: "person"), Field(name: "organization"), Field(name: "location")
        ])]
        return value
    }()

    package var problems: [String] {
        var issues: [String] = []
        if text.isEmpty { issues.append("Add text to extract from.") }
        if entities.isEmpty && relations.isEmpty && structures.isEmpty && classifications.isEmpty {
            issues.append("Add a schema field.")
        }
        if !(0...1).contains(threshold) { issues.append("Threshold must be between 0 and 1.") }
        let names = entities.map(\.name) + relations.map(\.name) + structures.map(\.name)
            + classifications.map(\.name)
        if Set(names).count != names.count { issues.append("Schema names must be unique.") }
        if (relations.map(\.name) + structures.map(\.name) + classifications.map(\.name)).contains("entities") {
            issues.append("entities is a reserved schema name.")
        }
        if !classifications.isEmpty {
            issues += StudioClassificationDocument(text: text, tasks: classifications).problems
        }
        for term in entities + relations {
            if !Self.valid(term.name) || (!term.detail.isEmpty && !Self.valid(term.detail)) {
                issues.append("Entity and relation names or descriptions contain invalid terms.")
            }
        }
        for structure in structures {
            if !Self.valid(structure.name) || structure.fields.isEmpty {
                issues.append("Each structure needs a name and fields.")
            }
            if Set(structure.fields.map(\.name)).count != structure.fields.count {
                issues.append("Structure fields must be unique.")
            }
            for field in structure.fields {
                if !Self.valid(field.name) || (!field.detail.isEmpty && !Self.valid(field.detail)) {
                    issues.append("Structure field names or descriptions contain invalid terms.")
                }
            }
        }
        return Array(Set(issues)).sorted()
    }

    private static func valid(_ value: String) -> Bool {
        let markers = ["[P]", "[L]", "[C]", "[E]", "[R]", "[SEP_STRUCT]", "[SEP_TEXT]",
                       "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]"]
        return !value.isEmpty && value.count <= 512 && !markers.contains(where: value.contains)
    }

    package func requestJSON() throws -> Data {
        let request = Request(text: text,
                              entities: entities.map { .init(name: $0.name, description: $0.detail.isEmpty ? nil : $0.detail) },
                              relations: relations.map { .init(name: $0.name, description: $0.detail.isEmpty ? nil : $0.detail) },
                              structures: structures.map { structure in
                                  .init(name: structure.name, fields: structure.fields.map { field in
                                      .init(name: field.name, description: field.detail.isEmpty ? nil : field.detail,
                                            multiple: field.multiple)
                                  })
                              }, classifications: classifications.isEmpty ? nil : classifications.map { task in
                                  let descriptions = Dictionary(uniqueKeysWithValues: task.labels.compactMap { label -> (String, String)? in
                                      label.detail.isEmpty ? nil : (label.name, label.detail)
                                  })
                                  return .init(name: task.name, labels: task.labels.map(\.name),
                                               descriptions: descriptions.isEmpty ? nil : descriptions,
                                               prompt: task.prompt.isEmpty ? nil : task.prompt,
                                               multiLabel: task.multiLabel, threshold: task.threshold)
                              }, threshold: threshold)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(request)
    }

    package static func importing(_ data: Data) throws -> StudioExtractionDocument {
        let request = try JSONDecoder().decode(Request.self, from: data)
        var value = StudioExtractionDocument()
        value.text = request.text
        value.entities = request.entities.map { .init(name: $0.name, detail: $0.description ?? "") }
        value.relations = request.relations.map { .init(name: $0.name, detail: $0.description ?? "") }
        value.structures = request.structures.map { structure in
            Structure(name: structure.name, fields: structure.fields.map { field in
                Field(name: field.name, detail: field.description ?? "", multiple: field.multiple)
            })
        }
        value.classifications = (request.classifications ?? []).map { task in
            StudioClassificationDocument.Task(name: task.name, labels: task.labels.map { label in
                .init(name: label, detail: task.descriptions?[label] ?? "")
            }, prompt: task.prompt ?? "", multiLabel: task.multiLabel, threshold: task.threshold)
        }
        value.threshold = request.threshold
        return value
    }

    package static func draftRequestURL(fileManager: FileManager = .default) -> URL {
        StudioOutputLocation.supportRoot(fileManager: fileManager)
            .appendingPathComponent("Extractions", isDirectory: true)
            .appendingPathComponent("request.json")
    }

    private struct Request: Codable {
        struct Term: Codable {
            let name: String
            let description: String?
        }
        struct Field: Codable {
            let name: String
            let description: String?
            let multiple: Bool

            enum CodingKeys: String, CodingKey { case name, description, multiple }

            init(name: String, description: String?, multiple: Bool) {
                self.name = name
                self.description = description
                self.multiple = multiple
            }

            init(from decoder: any Decoder) throws {
                let values = try decoder.container(keyedBy: CodingKeys.self)
                name = try values.decode(String.self, forKey: .name)
                description = try values.decodeIfPresent(String.self, forKey: .description)
                multiple = try values.decodeIfPresent(Bool.self, forKey: .multiple) ?? true
            }
        }
        struct Structure: Codable {
            let name: String
            let fields: [Field]
        }
        struct Classification: Codable {
            let name: String
            let labels: [String]
            let descriptions: [String: String]?
            let prompt: String?
            let multiLabel: Bool
            let threshold: Double

            enum CodingKeys: String, CodingKey {
                case name, labels, descriptions, prompt, threshold
                case multiLabel = "multi_label"
            }

            init(name: String, labels: [String], descriptions: [String: String]?, prompt: String?,
                 multiLabel: Bool, threshold: Double) {
                self.name = name
                self.labels = labels
                self.descriptions = descriptions
                self.prompt = prompt
                self.multiLabel = multiLabel
                self.threshold = threshold
            }

            init(from decoder: any Decoder) throws {
                let values = try decoder.container(keyedBy: CodingKeys.self)
                name = try values.decode(String.self, forKey: .name)
                labels = try values.decode([String].self, forKey: .labels)
                descriptions = try values.decodeIfPresent([String: String].self, forKey: .descriptions)
                prompt = try values.decodeIfPresent(String.self, forKey: .prompt)
                multiLabel = try values.decodeIfPresent(Bool.self, forKey: .multiLabel) ?? false
                threshold = try values.decodeIfPresent(Double.self, forKey: .threshold) ?? 0.5
            }
        }
        let text: String
        let entities: [Term]
        let relations: [Term]
        let structures: [Structure]
        let classifications: [Classification]?
        let threshold: Double

        enum CodingKeys: String, CodingKey {
            case text, entities, relations, structures, classifications, threshold
        }

        init(text: String, entities: [Term], relations: [Term], structures: [Structure],
             classifications: [Classification]?, threshold: Double) {
            self.text = text
            self.entities = entities
            self.relations = relations
            self.structures = structures
            self.classifications = classifications
            self.threshold = threshold
        }

        init(from decoder: any Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            text = try values.decode(String.self, forKey: .text)
            entities = try values.decodeIfPresent([Term].self, forKey: .entities) ?? []
            relations = try values.decodeIfPresent([Term].self, forKey: .relations) ?? []
            structures = try values.decodeIfPresent([Structure].self, forKey: .structures) ?? []
            classifications = try values.decodeIfPresent([Classification].self, forKey: .classifications)
            threshold = try values.decodeIfPresent(Double.self, forKey: .threshold) ?? 0.5
        }
    }
}

package struct StudioExtractionPlan: Decodable, Equatable {
    package let model: String
    package let inputTokens: Int
    package let wordCount: Int
    package let schemaNames: [String]
}

package struct StudioExtractionResult: Decodable, Equatable {
    package struct Span: Decodable, Equatable {
        package let text: String
        package let confidence: Double
        package let start: Int
        package let end: Int
    }
    package struct Relation: Decodable, Equatable {
        package let head: Span
        package let tail: Span
    }
    package struct Classification: Decodable, Equatable {
        package let labels: [String]
        package let probabilities: [String: Double]
    }
    package let model: String
    package let runtime: String
    package let entities: [String: [Span]]
    package let relations: [String: [Relation]]
    package let structures: [String: [[String: [Span]]]]
    package let classifications: [String: Classification]
    package let inputTokens: Int
}
