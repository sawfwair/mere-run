import Foundation
import MereRunLayaModel

public enum LayaQuestionType: String, Codable, Sendable {
    case choice, score, noul
    public var index: Int { switch self { case .choice: 0; case .score: 1; case .noul: 2 } }
}

/// Arrays make label order explicit; JSON object key order is not a portable contract.
public struct LayaCriterion: Codable, Sendable, Equatable {
    public let label: String
    public let description: String?

    public init(label: String, description: String? = nil) {
        self.label = label
        self.description = description
    }

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let label = try? value.decode(String.self) {
            self.init(label: label)
        } else {
            let object = try decoder.container(keyedBy: CodingKeys.self)
            self.init(label: try object.decode(String.self, forKey: .label),
                      description: try object.decodeIfPresent(String.self, forKey: .description))
        }
    }
}

public struct LayaQuestion: Codable, Sendable {
    public let id: String
    public let type: LayaQuestionType
    public let instructions: String
    public let criteria: [LayaCriterion]?

    public init(id: String, type: LayaQuestionType, instructions: String, criteria: [LayaCriterion]? = nil) {
        self.id = id
        self.type = type
        self.instructions = instructions
        self.criteria = criteria
    }

    public var labels: [String] {
        switch type {
        case .choice: (criteria ?? []).map(\.label)
        case .score: (criteria ?? []).indices.map(String.init)
        case .noul: ["false", "true"]
        }
    }

    public var renderedOptions: [String] {
        switch type {
        case .choice:
            return (criteria ?? []).map { option in
                guard let detail = option.description, !detail.isEmpty else { return option.label }
                return "\(option.label): \(detail)"
            }
        case .score:
            return (criteria ?? []).enumerated().map { "level \($0.offset): \($0.element.description ?? $0.element.label)" }
        case .noul:
            let options = criteria ?? []
            return labels.map { label in
                let defaultText = label == "false" ? "no, the statement does not hold" : "yes, the statement holds"
                let detail = options.first { $0.label == label }?.description
                return "\(label): \(detail?.isEmpty == false ? detail ?? defaultText : defaultText)"
            }
        }
    }

    public func validate() throws {
        guard !id.isEmpty, id.utf8.count <= 256,
              !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              instructions.utf8.count <= 65_536 else {
            throw LayaModelError.invalidInput("Each question needs an id (1–256 bytes) and instructions (1–65536 bytes).")
        }
        let options = criteria ?? []
        guard options.count <= 255, options.allSatisfy({
            !$0.label.isEmpty && $0.label.utf8.count <= 16_384 && ($0.description?.utf8.count ?? 0) <= 65_536
        }), type == .score || Set(options.map(\.label)).count == options.count else {
            throw LayaModelError.invalidInput("Question \(id) needs unique nonempty criteria; at most 255 are supported.")
        }
        switch type {
        case .choice, .score:
            guard !options.isEmpty else { throw LayaModelError.invalidInput("Question \(id) needs criteria.") }
        case .noul:
            guard options.allSatisfy({ ["false", "true"].contains($0.label) }) else {
                throw LayaModelError.invalidInput("Noul criteria labels must be false or true.")
            }
        }
    }
}

public struct LayaDecisionRequest: Codable, Sendable {
    public let state: String
    public let questions: [LayaQuestion]
    public let maxTokens: Int?
    public let headMaxTokens: Int?

    enum CodingKeys: String, CodingKey {
        case state, questions, maxTokens = "max_tokens", headMaxTokens = "head_max_tokens"
    }

    public init(state: String, questions: [LayaQuestion], maxTokens: Int? = nil, headMaxTokens: Int? = nil) {
        self.state = state
        self.questions = questions
        self.maxTokens = maxTokens
        self.headMaxTokens = headMaxTokens
    }

    public func validate() throws {
        guard state.utf8.count <= 1_048_576, (1...256).contains(questions.count),
              Set(questions.map(\.id)).count == questions.count else {
            throw LayaModelError.invalidInput("Provide 1–256 questions with unique ids and at most 1 MiB of state text.")
        }
        for question in questions { try question.validate() }
        if let maxTokens, !(16...8_192).contains(maxTokens) {
            throw LayaModelError.invalidInput("max_tokens must be between 16 and 8192.")
        }
        if let headMaxTokens, !(8...8_191).contains(headMaxTokens) {
            throw LayaModelError.invalidInput("head_max_tokens must be between 8 and 8191.")
        }
    }
}

public struct LayaPreparedQuestion: Codable, Sendable {
    public let id: String
    public let inputTokens: Int
    public let stateTokens: Int
    public let stateTokensDropped: Int
    public let instructionTokensDropped: Int
    public let optionTokensDropped: [Int]
    public let optionCount: Int
}

public struct LayaDecisionPlan: Codable, Sendable {
    public let model: String
    public let maxTokens: Int
    public let headMaxTokens: Int
    public let questions: [LayaPreparedQuestion]
}

public struct LayaDecisionAnswer: Codable, Sendable {
    public let type: LayaQuestionType
    public let choice: String?
    public let score: Double?
    public let noul: Double?
    public let probabilities: [String: Double]
    public let confidence: Double
    public let actProbability: Double
    public let rawTemperature: Double
    public let appliedTemperature: Double
    public let temperatureClamped: Bool
}

public struct LayaDecisionResponse: Codable, Sendable {
    public let model: String
    public let runtime: String
    public let answers: [String: LayaDecisionAnswer]
    public let plan: LayaDecisionPlan
    public let inputTokens: Int
    public let outputTokens: Int
}
