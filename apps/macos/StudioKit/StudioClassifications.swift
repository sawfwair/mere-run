import Foundation

/// The Studio's editable form of a GLiNER classification request.
package struct StudioClassificationDocument: Codable, Equatable {
    package struct Label: Codable, Equatable, Identifiable {
        package var id = UUID()
        package var name: String
        package var detail: String

        package init(name: String, detail: String = "") {
            self.name = name
            self.detail = detail
        }
    }

    package struct Task: Codable, Equatable, Identifiable {
        package var id = UUID()
        package var name: String
        package var labels: [Label]
        package var prompt: String
        package var multiLabel: Bool
        package var threshold: Double

        package init(name: String = "", labels: [Label] = [.init(name: ""), .init(name: "")],
                     prompt: String = "", multiLabel: Bool = false, threshold: Double = 0.5) {
            self.name = name
            self.labels = labels
            self.prompt = prompt
            self.multiLabel = multiLabel
            self.threshold = threshold
        }
    }

    package var text: String
    package var tasks: [Task]

    package init(text: String = "", tasks: [Task] = []) {
        self.text = text
        self.tasks = tasks
    }

    package static let example = StudioClassificationDocument(
        text: "I was charged twice for my subscription. Please refund the duplicate charge.",
        tasks: [
            Task(name: "department", labels: [
                Label(name: "billing", detail: "Payments, charges, and refunds"),
                Label(name: "technical support", detail: "Product problems and troubleshooting"),
                Label(name: "sales", detail: "Purchases and plans"),
            ]),
            Task(name: "intent", labels: [Label(name: "refund"), Label(name: "complaint"), Label(name: "question")],
                 multiLabel: true),
        ]
    )

    package static func draftRequestURL(fileManager: FileManager = .default) -> URL {
        StudioOutputLocation.supportRoot(fileManager: fileManager)
            .appendingPathComponent("Classifications", isDirectory: true)
            .appendingPathComponent("request.json")
    }

    package var problems: [String] {
        var issues: [String] = []
        if text.isEmpty { issues.append("Add the text to classify.") }
        if tasks.isEmpty { issues.append("Add a task.") }
        if tasks.count > 16 { issues.append("Use at most 16 tasks.") }
        let names = tasks.map(\.name)
        if Set(names).count != names.count { issues.append("Task names must be unique.") }
        for (index, task) in tasks.enumerated() {
            let title = "Task \(index + 1)"
            if let problem = Self.termProblem(task.name) { issues.append("\(title) name \(problem)") }
            if task.labels.isEmpty || task.labels.count > 64 {
                issues.append("\(title) needs 1–64 labels.")
            }
            let labels = task.labels.map(\.name)
            if Set(labels).count != labels.count { issues.append("\(title) labels must be unique.") }
            for label in task.labels {
                if let problem = Self.termProblem(label.name) { issues.append("\(title) label \(problem)") }
                if !label.detail.isEmpty, let problem = Self.termProblem(label.detail) {
                    issues.append("\(title) description \(problem)")
                }
            }
            if !task.prompt.isEmpty, let problem = Self.termProblem(task.prompt) {
                issues.append("\(title) prompt \(problem)")
            }
            if !(0...1).contains(task.threshold) { issues.append("\(title) threshold must be between 0 and 1.") }
        }
        return issues
    }

    private static func termProblem(_ value: String) -> String? {
        if value.isEmpty { return "cannot be empty." }
        if value.count > 512 { return "must be at most 512 characters." }
        let markers = ["[P]", "[L]", "[C]", "[E]", "[R]", "[SEP_STRUCT]", "[SEP_TEXT]",
                       "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]"]
        if markers.contains(where: value.contains) { return "contains a reserved marker." }
        return nil
    }

    package func requestJSON() throws -> Data {
        let request = Request(text: text, tasks: tasks.map { task in
            let descriptions = Dictionary(uniqueKeysWithValues: task.labels.compactMap { label -> (String, String)? in
                label.detail.isEmpty ? nil : (label.name, label.detail)
            })
            return Request.Task(name: task.name, labels: task.labels.map(\.name),
                                descriptions: descriptions.isEmpty ? nil : descriptions,
                                prompt: task.prompt.isEmpty ? nil : task.prompt,
                                multiLabel: task.multiLabel, threshold: task.threshold)
        })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(request)
    }

    package static func importing(_ data: Data) throws -> StudioClassificationDocument {
        let request = try JSONDecoder().decode(Request.self, from: data)
        return StudioClassificationDocument(text: request.text, tasks: request.tasks.map { task in
            Task(name: task.name, labels: task.labels.map {
                Label(name: $0, detail: task.descriptions?[$0] ?? "")
            }, prompt: task.prompt ?? "", multiLabel: task.multiLabel, threshold: task.threshold)
        })
    }

    private struct Request: Codable {
        struct Task: Codable {
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

            init(from decoder: Decoder) throws {
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
        let tasks: [Task]
    }
}

package struct StudioClassificationPlan: Decodable, Equatable {
    package let model: String
    package let inputTokens: Int
    package let labelCount: Int
    package let taskNames: [String]
}

package struct StudioClassificationResult: Decodable, Equatable {
    package struct Head: Decodable, Equatable {
        package let labels: [String]
        package let probabilities: [String: Double]
    }

    package let model: String
    package let runtime: String
    package let heads: [String: Head]
    package let inputTokens: Int
}

package enum StudioClassificationOutput: Equatable {
    case result(StudioClassificationResult)
    case fit(StudioClassificationPlan)
    case fits([StudioClassificationPlan])

    package init?(data: Data) {
        if let result = try? JSONDecoder().decode(StudioClassificationResult.self, from: data) {
            self = .result(result)
        } else if let plan = try? JSONDecoder().decode(StudioClassificationPlan.self, from: data) {
            self = .fit(plan)
        } else if let plans = try? JSONDecoder().decode([StudioClassificationPlan].self, from: data) {
            self = .fits(plans)
        } else {
            return nil
        }
    }

    package init?(outputText: String) {
        let stdout = outputText.components(separatedBy: "\n\nSTDERR\n").first ?? outputText
        self.init(data: Data(stdout.utf8))
    }
}
