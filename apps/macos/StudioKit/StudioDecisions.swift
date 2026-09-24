import Foundation

// Text ▸ Decisions builds the request `mere.run text decide` reads, instead of asking for a
// hand-written JSON file, and reads back its result document. These mirror `LayaDecisionRequest`
// and `LayaDecisionResponse` in `MereRunCore` (which the Studio does not import): the request
// encodes to the documented shape, and a decode failure on the result means the CLI changed.

/// One question the decision model answers about the text.
package struct StudioDecisionQuestion: Codable, Equatable, Identifiable {
    package enum Kind: String, Codable, CaseIterable, Identifiable {
        /// Pick one of the options.
        case choice
        /// Place the text on an ordered scale, lowest level first.
        case score
        /// Whether a statement holds. The CLI calls this type `noul`.
        case yesNo = "noul"

        package var id: String { rawValue }

        package var title: String {
            switch self {
            case .choice: return "Choice"
            case .score: return "Score"
            case .yesNo: return "Yes or no"
            }
        }
    }

    /// An option (choice), a level (score), or the wording of "no" and "yes" (yes or no).
    package struct Option: Codable, Equatable, Identifiable {
        package var id = UUID()
        package var label: String
        package var detail: String

        package init(label: String, detail: String = "") {
            self.label = label
            self.detail = detail
        }
    }

    package var id = UUID()
    package var kind: Kind
    /// What is asked. For yes or no, the statement that holds or does not.
    package var prompt: String
    package var options: [Option]
    /// The question's id in the request and the result. Blank derives one from the prompt.
    package var key: String

    package init(kind: Kind, prompt: String, options: [Option] = [], key: String = "") {
        self.kind = kind
        self.prompt = prompt
        self.options = options
        self.key = key
    }

    /// A new, empty question of `kind`, with the slots its kind needs.
    package static func blank(_ kind: Kind) -> StudioDecisionQuestion {
        switch kind {
        case .choice: return .init(kind: .choice, prompt: "", options: [.init(label: ""), .init(label: "")])
        case .score: return .init(kind: .score, prompt: "", options: [.init(label: ""), .init(label: ""), .init(label: "")])
        case .yesNo: return .init(kind: .yesNo, prompt: "")
        }
    }
}

/// The request the Decisions page edits: the text to judge and its ordered questions.
package struct StudioDecisionDocument: Codable, Equatable {
    package var text: String
    package var questions: [StudioDecisionQuestion]
    /// Token budgets an imported request set. The page does not edit them; they carry through
    /// to every run and export so a request file keeps its meaning.
    package var maxTokens: Int?
    package var headMaxTokens: Int?

    package init(text: String = "", questions: [StudioDecisionQuestion] = [], maxTokens: Int? = nil, headMaxTokens: Int? = nil) {
        self.text = text
        self.questions = questions
        self.maxTokens = maxTokens
        self.headMaxTokens = headMaxTokens
    }

    /// Where the page keeps the request it is editing, so the Command view's Run has a real file
    /// to pass as `--input`. Each Decide or Check fit writes its own copy beside its output.
    package static func draftRequestURL(fileManager: FileManager = .default) -> URL {
        StudioOutputLocation.supportRoot(fileManager: fileManager)
            .appendingPathComponent("Decisions", isDirectory: true)
            .appendingPathComponent("request.json")
    }

    /// The Laya handbook's example: a support message, which department, how urgent, and
    /// whether it asks for a refund.
    package static let example = StudioDecisionDocument(
        text: "I was charged twice for my subscription. Please refund the duplicate charge.",
        questions: [
            .init(
                kind: .choice,
                prompt: "Which department should handle this request?",
                options: [.init(label: "billing"), .init(label: "technical support"), .init(label: "sales")],
                key: "department"
            ),
            .init(
                kind: .score,
                prompt: "Rate how urgently this request needs attention.",
                options: [.init(label: "routine"), .init(label: "soon"), .init(label: "immediate")],
                key: "urgency"
            ),
            .init(kind: .yesNo, prompt: "The customer requests a refund.", key: "refund"),
        ]
    )

    /// Each question's id in the request: its own key, or one derived from its prompt, made
    /// unique in order ("urgency", "urgency-2").
    package var resolvedKeys: [String] {
        var used: Set<String> = []
        return questions.enumerated().map { index, question in
            let base = question.key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? Self.slug(question.prompt, fallback: "question-\(index + 1)")
                : question.key.trimmingCharacters(in: .whitespacesAndNewlines)
            var key = base
            var suffix = 2
            while used.contains(key) {
                key = "\(base)-\(suffix)"
                suffix += 1
            }
            used.insert(key)
            return key
        }
    }

    /// What stops the request from running, in the order the page shows them; empty when it is
    /// ready. These are the CLI's own checks, stated in the page's words.
    package var problems: [String] {
        var problems: [String] = []
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            problems.append("Add the text to judge.")
        }
        if text.utf8.count > 1_048_576 { problems.append("The text is longer than 1 MB.") }
        if questions.isEmpty { problems.append("Add a question.") }
        if questions.count > 256 { problems.append("Ask at most 256 questions.") }
        let keys = resolvedKeys
        for (index, question) in questions.enumerated() {
            let name = "Question \(index + 1)"
            if question.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                problems.append(question.kind == .yesNo ? "\(name) needs a statement." : "\(name) needs a question.")
            }
            if question.prompt.utf8.count > 65_536 { problems.append("\(name) is longer than 64 KB.") }
            if keys[index].utf8.count > 256 { problems.append("\(name)'s ID is longer than 256 bytes.") }
            let labels = question.options.map { $0.label.trimmingCharacters(in: .whitespacesAndNewlines) }
            let named = labels.filter { !$0.isEmpty }
            switch question.kind {
            case .choice:
                if named.count < 2 {
                    problems.append("\(name) needs at least two options.")
                } else if named.count < labels.count {
                    problems.append("\(name) has an empty option.")
                }
                if Set(named).count != named.count { problems.append("\(name) repeats an option.") }
            case .score:
                if named.count < 2 {
                    problems.append("\(name) needs at least two levels.")
                } else if named.count < labels.count {
                    problems.append("\(name) has an empty level.")
                }
            case .yesNo:
                break
            }
            if question.options.count > 255 { problems.append("\(name) has more than 255 options.") }
            if question.options.contains(where: { $0.label.utf8.count > 16_384 || $0.detail.utf8.count > 65_536 }) {
                problems.append("\(name) has an option too long for the model.")
            }
        }
        return problems
    }

    /// The JSON `text decide --input` reads.
    package func requestJSON() throws -> Data {
        let keys = resolvedKeys
        let request = Request(
            state: text,
            questions: questions.enumerated().map { index, question in
                Request.Question(
                    id: keys[index],
                    type: question.kind.rawValue,
                    instructions: question.prompt.trimmingCharacters(in: .whitespacesAndNewlines),
                    criteria: Self.criteria(for: question)
                )
            },
            maxTokens: maxTokens,
            headMaxTokens: headMaxTokens
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(request)
    }

    /// Reads a request file written by hand or by this page. Options given as objects keep their
    /// descriptions; ids become the questions' keys.
    package static func importing(_ data: Data) throws -> StudioDecisionDocument {
        let request = try JSONDecoder().decode(Request.self, from: data)
        return StudioDecisionDocument(
            text: request.state,
            questions: try request.questions.map { question in
                guard let kind = StudioDecisionQuestion.Kind(rawValue: question.type) else {
                    throw StudioDecisionImportError.unknownType(question.type)
                }
                let options = (question.criteria ?? []).map {
                    StudioDecisionQuestion.Option(label: $0.label, detail: $0.description ?? "")
                }
                return StudioDecisionQuestion(kind: kind, prompt: question.instructions, options: options, key: question.id)
            },
            maxTokens: request.maxTokens,
            headMaxTokens: request.headMaxTokens
        )
    }

    /// Choice options and score levels in order; for yes or no, only the answers given wording.
    private static func criteria(for question: StudioDecisionQuestion) -> [Request.Criterion]? {
        switch question.kind {
        case .choice, .score:
            return question.options.map {
                let detail = $0.detail.trimmingCharacters(in: .whitespacesAndNewlines)
                return Request.Criterion(label: $0.label.trimmingCharacters(in: .whitespacesAndNewlines), description: detail.isEmpty ? nil : detail)
            }
        case .yesNo:
            let worded = question.options.filter { ["false", "true"].contains($0.label) && !$0.detail.isBlank }
            return worded.isEmpty ? nil : worded.map {
                Request.Criterion(label: $0.label, description: $0.detail.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
    }

    /// "Which department should handle this?" → "which-department-should-handle-this".
    static func slug(_ text: String, fallback: String) -> String {
        let words = text.lowercased().split { !$0.isLetter && !$0.isNumber }.prefix(6)
        let slug = words.joined(separator: "-")
        return slug.isEmpty ? fallback : String(slug.prefix(48))
    }

    /// The wire shape. A criterion encodes as a bare label unless it has a description.
    private struct Request: Codable {
        struct Criterion: Codable {
            let label: String
            let description: String?

            init(label: String, description: String?) {
                self.label = label
                self.description = description
            }

            init(from decoder: Decoder) throws {
                let single = try decoder.singleValueContainer()
                if let label = try? single.decode(String.self) {
                    self.init(label: label, description: nil)
                } else {
                    let object = try decoder.container(keyedBy: CodingKeys.self)
                    self.init(
                        label: try object.decode(String.self, forKey: .label),
                        description: try object.decodeIfPresent(String.self, forKey: .description)
                    )
                }
            }

            func encode(to encoder: Encoder) throws {
                if let description {
                    var object = encoder.container(keyedBy: CodingKeys.self)
                    try object.encode(label, forKey: .label)
                    try object.encode(description, forKey: .description)
                } else {
                    var single = encoder.singleValueContainer()
                    try single.encode(label)
                }
            }

            enum CodingKeys: String, CodingKey { case label, description }
        }

        struct Question: Codable {
            let id: String
            let type: String
            let instructions: String
            let criteria: [Criterion]?
        }

        let state: String
        let questions: [Question]
        let maxTokens: Int?
        let headMaxTokens: Int?

        enum CodingKeys: String, CodingKey {
            case state, questions, maxTokens = "max_tokens", headMaxTokens = "head_max_tokens"
        }
    }
}

package enum StudioDecisionImportError: LocalizedError, Equatable {
    case unknownType(String)

    package var errorDescription: String? {
        switch self {
        case .unknownType(let type):
            return "This request has a question of type \"\(type)\"; Decisions supports choice, score, and noul."
        }
    }
}

// MARK: - Result

/// What `text decide` writes: one answer per question id, and the token plan it ran with.
/// `--preflight` writes only the plan.
package struct StudioDecisionResult: Decodable, Equatable {
    package struct Answer: Decodable, Equatable {
        /// "choice", "score", or "noul".
        package let type: String
        /// The most likely option, for choice.
        package let choice: String?
        /// The probability-weighted level, for score (0 is the first level).
        package let score: Double?
        /// The probability that the statement holds, for yes or no.
        package let noul: Double?
        /// Keyed by option label (choice), level index as a string (score), or "false"/"true".
        package let probabilities: [String: Double]
        /// 0…1: one minus normalized entropy (choice, score), or the larger of no/yes.
        package let confidence: Double
        package let temperatureClamped: Bool
    }

    package let model: String
    package let answers: [String: Answer]
    package let plan: StudioDecisionPlan
}

/// How each question fit the model's token budget.
package struct StudioDecisionPlan: Decodable, Equatable {
    package struct Question: Decodable, Equatable {
        package let id: String
        package let inputTokens: Int
        package let stateTokens: Int
        package let stateTokensDropped: Int
        package let instructionTokensDropped: Int
        package let optionTokensDropped: [Int]
        package let optionCount: Int

        /// Whether anything was cut to fit.
        package var wasTruncated: Bool {
            stateTokensDropped > 0 || instructionTokensDropped > 0 || optionTokensDropped.contains { $0 > 0 }
        }

        /// "Cut 120 tokens from the end of the text to fit.", or nil when everything fit.
        package var truncationNote: String? {
            var parts: [String] = []
            if stateTokensDropped > 0 { parts.append("\(stateTokensDropped) tokens from the end of the text") }
            if instructionTokensDropped > 0 { parts.append("\(instructionTokensDropped) from the question") }
            let optionDrops = optionTokensDropped.reduce(0, +)
            if optionDrops > 0 { parts.append("\(optionDrops) from the options") }
            return parts.isEmpty ? nil : "Cut \(parts.joined(separator: ", ")) to fit."
        }
    }

    package let model: String
    package let maxTokens: Int
    package let headMaxTokens: Int
    package let questions: [Question]

    package func question(_ id: String) -> Question? {
        questions.first { $0.id == id }
    }
}

/// What a finished run left: answers, or for `--preflight` only the plan.
package enum StudioDecisionOutput: Equatable {
    case answers(StudioDecisionResult)
    case fit(StudioDecisionPlan)

    package init?(data: Data) {
        let decoder = JSONDecoder()
        if let result = try? decoder.decode(StudioDecisionResult.self, from: data) {
            self = .answers(result)
        } else if let plan = try? decoder.decode(StudioDecisionPlan.self, from: data) {
            self = .fit(plan)
        } else {
            return nil
        }
    }

    /// Reads a run's captured output, where stdout comes first and any stderr follows a
    /// `STDERR` line.
    package init?(outputText: String) {
        let stdout = outputText.components(separatedBy: "\n\nSTDERR\n").first ?? outputText
        self.init(data: Data(stdout.utf8))
    }

    package var plan: StudioDecisionPlan {
        switch self {
        case .answers(let result): return result.plan
        case .fit(let plan): return plan
        }
    }

    package var result: StudioDecisionResult? {
        if case .answers(let result) = self { return result }
        return nil
    }
}
