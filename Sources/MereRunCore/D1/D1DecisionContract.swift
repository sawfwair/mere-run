// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MereRunD1Model

public struct D1Question: Sendable {
    public enum Kind: String, Sendable { case choice, score, noul }
    public let id: String
    public let type: Kind
    public let instructions: String
    public let options: [(id: String, description: ClefJSON)]
    let criteria: ClefJSON?
    init(id: String, value: ClefJSON) throws {
        guard !id.isEmpty, let raw = value["type"]?.string, let type = Kind(rawValue: raw),
              let instructions = value["instructions"]?.string, !instructions.isEmpty else {
            throw D1Error.invalid("D1 questions require an id, type, and nonempty instructions string.")
        }
        self.id = id; self.type = type; self.instructions = instructions; criteria = value["criteria"]
        switch type {
        case .choice:
            guard let fields = criteria?.fields, (2...256).contains(fields.count) else {
                throw D1Error.invalid("D1 choice criteria require 2–256 named options.")
            }
            options = fields.map { ($0.key, $0.value) }
        case .score:
            guard let levels = criteria?.array, (2...10).contains(levels.count) else {
                throw D1Error.invalid("D1 score criteria require 2–10 levels.")
            }
            options = levels.enumerated().map { (String($0.offset), $0.element) }
        case .noul:
            let nullCriteria: Bool
            if case .null? = criteria { nullCriteria = true } else { nullCriteria = false }
            guard criteria == nil || criteria?.fields != nil || nullCriteria else {
                throw D1Error.invalid("D1 noul criteria must be an object or null.")
            }
            options = [("true", criteria?["true"] ?? criteria?["yes"] ?? .null), ("false", criteria?["false"] ?? criteria?["no"] ?? .null)]
        }
    }
}

public struct D1DecisionRequest: Sendable {
    public let state: ClefJSON
    public let questions: [D1Question]
    public let images: [String]
    public let audio: String?
    public let maxTokens: Int?
    public static func decode(_ data: Data) throws -> D1DecisionRequest {
        guard data.count <= 2 * 1_024 * 1_024 else { throw D1Error.invalid("D1 requests must not exceed 2 MiB.") }
        let json = try ClefJSON.parse(data)
        guard let state = json["state"], let fields = json["questions"]?.fields, (1...128).contains(fields.count) else {
            throw D1Error.invalid("D1 requires state and a questions object with 1–128 named fields.")
        }
        let questions = try fields.map { try D1Question(id: $0.key, value: $0.value) }
        var images: [String] = []
        if let value = json["images"] {
            guard let paths = value.array, paths.count <= 16 else { throw D1Error.invalid("D1 images must be an array of up to 16 local paths.") }
            images = try paths.map {
                guard let path = $0.string, !path.isEmpty else { throw D1Error.invalid("D1 image paths must be nonempty strings.") }
                return path
            }
        }
        var audio: String?
        if let value = json["audio"] {
            guard let path = value.string, !path.isEmpty else { throw D1Error.invalid("D1 audio must be a local file path.") }
            audio = path
        }
        guard images.isEmpty || audio == nil else { throw D1Error.invalid("D1 requests carry images or audio, never both.") }
        guard json["videos"] == nil, json["media_kwargs"] == nil, json["max_state_tokens"] == nil else {
            throw D1Error.invalid("D1 does not accept videos, media_kwargs, or max_state_tokens.")
        }
        var maxTokens: Int?
        if let value = json["max_tokens"] {
            guard case .number(let raw) = value, let count = Int(raw), (64...32_768).contains(count) else {
                throw D1Error.invalid("D1 max_tokens must be an integer from 64 to 32768.")
            }
            maxTokens = count
        }
        return D1DecisionRequest(state: state, questions: questions, images: images, audio: audio, maxTokens: maxTokens)
    }
}

public struct D1DecisionPlan: Encodable, Sendable {
    public let model: String
    public let runtime: String
    public let mediaTokens: Int
    public let inputTokens: Int
    public let questions: [Field]
    public struct Field: Encodable, Sendable {
        public let id: String
        public let inputTokens: Int
        public let stateTokensDropped: Int
        public let optionIDs: [String]
        public let markerPositions: [Int]
    }
}

public struct D1DecisionResponse: Encodable, Sendable {
    public let model: String
    public let answers: [String: Answer]
    public let usage: Usage
    public struct Usage: Encodable, Sendable { public let input_tokens: Int; public let output_tokens: Int }
    public struct Answer: Encodable, Sendable {
        public let type: String
        public var noul: Double?
        public var choice: String?
        public var score: Double?
        public var confidence: Double?
        public var probabilities: [String: Double]?
        public var legend: [String: ClefJSON]?
    }
    static func answer(_ question: D1Question, probabilities: [Float], stringLegend: Bool = false) throws -> Answer {
        guard probabilities.count == question.options.count, probabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
              abs(probabilities.reduce(0, +) - 1) < 0.0001 else { throw D1Error.invalid("D1 produced invalid probabilities.") }
        var answer = Answer(type: question.type.rawValue)
        if question.type == .noul { answer.noul = Double(probabilities[0]); return answer }
        var best = 0
        for index in probabilities.indices.dropFirst() where probabilities[index] > probabilities[best] { best = index }
        answer.confidence = Double(probabilities[best])
        answer.probabilities = Dictionary(uniqueKeysWithValues: zip(question.options, probabilities).map { ($0.id, Double($1)) })
        if question.type == .choice { answer.choice = question.options[best].id }
        else {
            answer.score = probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * Double($1.element) }
            answer.legend = try Dictionary(uniqueKeysWithValues: question.options.map {
                ($0.id, stringLegend ? .string(try D1JSON.render($0.description, stringAsText: true)) : $0.description)
            })
        }
        return answer
    }
}
