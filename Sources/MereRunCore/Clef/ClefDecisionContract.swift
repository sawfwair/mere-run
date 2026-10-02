import Foundation
import MereRunQwenModel

public struct ClefQuestion: Sendable {
    public enum Kind: String, Sendable { case noul, choice, score }
    public let id: String
    public let type: Kind
    public let instructions: ClefJSON
    public let options: [Option]
    let responseOptionIDs: [String]
    public struct Option: Sendable {
        public let id: String
        public let description: ClefJSON
    }

    init(id: String, value: ClefJSON) throws {
        guard value.fields != nil, let rawType = value["type"]?.string, let type = Kind(rawValue: rawType) else {
            throw ClefError.invalidInput("\(id): type must be noul, choice, or score.")
        }
        self.id = id
        self.type = type
        if let instructions = value["instructions"], instructions.isTruthy { self.instructions = instructions }
        else { self.instructions = .string(id) }
        let criteria = value["criteria"]
        switch type {
        case .noul:
            if let criteria, case .null = criteria {} else if let criteria, criteria.fields == nil {
                throw ClefError.invalidInput("\(id): noul criteria must be an object.")
            }
            options = [Option(id: "true", description: criteria?["true"] ?? .string("The proposition is true or the answer is yes.")),
                       Option(id: "false", description: criteria?["false"] ?? .string("The proposition is false or the answer is no."))]
            responseOptionIDs = options.map(\.id)
        case .choice:
            guard let fields = criteria?.fields, !fields.isEmpty else {
                throw ClefError.invalidInput("\(id): choice criteria must be a nonempty object.")
            }
            options = fields.sorted { ClefJSON.keyOrder($0.key, $1.key) }.map { Option(id: $0.key, description: $0.value) }
            responseOptionIDs = fields.map(\.key)
        case .score:
            guard let levels = criteria?.array, !levels.isEmpty else {
                throw ClefError.invalidInput("\(id): score criteria must be a nonempty array.")
            }
            options = levels.enumerated().map { Option(id: String($0.offset), description: $0.element) }
            responseOptionIDs = options.map(\.id)
        }
        guard !id.isEmpty, options.count <= 256 else { throw ClefError.invalidInput("Clef fields require an id and at most 256 options.") }
    }
}

public struct ClefDecisionRequest: Sendable {
    public let state: ClefJSON
    public let questions: [ClefQuestion]
    public let images: [String]
    /// Each video is an ordered array of local frame paths, matching the reference's frame-array input.
    public let videos: [[String]]
    public let maxTokens: Int
    public let maxStateTokens: Int?

    public static func decode(_ data: Data) throws -> ClefDecisionRequest {
        guard data.count <= 2 * 1_024 * 1_024 else { throw ClefError.invalidInput("The Clef request must not exceed 2 MiB.") }
        return try ClefDecisionRequest(json: ClefJSON.parse(data))
    }

    init(json: ClefJSON) throws {
        guard let state = json["state"], let fields = json["questions"]?.fields, !fields.isEmpty, fields.count <= 128 else {
            throw ClefError.invalidInput("Clef requires state and a questions object with 1–128 fields.")
        }
        self.state = state
        questions = try fields.map { try ClefQuestion(id: $0.key, value: $0.value) }
        func paths(_ value: ClefJSON?) throws -> [String] {
            guard let value else { return [] }
            guard let entries = value.array else { throw ClefError.invalidInput("Clef media must be arrays of local paths.") }
            return try entries.map {
                guard let path = $0.string, !path.isEmpty else { throw ClefError.invalidInput("Clef media paths must be nonempty strings.") }
                return path
            }
        }
        images = try paths(json["images"])
        if let value = json["videos"] {
            guard let entries = value.array else { throw ClefError.invalidInput("Clef videos must be arrays of frame-path arrays.") }
            videos = try entries.map { try paths($0) }
        } else { videos = [] }
        guard images.isEmpty || videos.isEmpty, images.count <= 16, videos.count <= 4,
              videos.allSatisfy({ !$0.isEmpty && $0.count <= 768 }) else {
            throw ClefError.invalidInput("Use up to 16 images or 4 videos with 1–768 frames each; mixed images and videos are unsupported.")
        }
        guard json["media_kwargs"] == nil else {
            throw ClefError.invalidInput("Clef uses checkpoint processor settings; media_kwargs overrides are unsupported.")
        }
        func integer(_ key: String) throws -> Int? {
            guard let value = json[key] else { return nil }
            guard case .number(let raw) = value, let number = Int(raw) else {
                throw ClefError.invalidInput("\(key) must be an integer.")
            }
            return number
        }
        maxTokens = try integer("max_tokens") ?? 16_384
        maxStateTokens = try integer("max_state_tokens")
        guard (1...16_384).contains(maxTokens), maxStateTokens.map({ $0 >= 0 }) ?? true else {
            throw ClefError.invalidInput("Clef max_tokens must be 1–16384; max_state_tokens must be nonnegative.")
        }
    }
}

public struct ClefDecisionPlan: Encodable, Sendable {
    public let model: String
    public let inputTokens: Int
    public let stateTokens: Int
    public let stateTokensDropped: Int
    public let questions: [Field]
    public struct Field: Encodable, Sendable {
        public let id: String
        public let optionIDs: [String]
        public let questionSpan: [Int]
        public let optionSpans: [[Int]]
    }
}

public struct ClefDecisionResponse: Encodable, Sendable {
    public let model: String
    public let answers: [String: Answer]
    public let usage: Usage
    public struct Usage: Encodable, Sendable {
        public let input_tokens: Int
        public let output_tokens: Int
    }
    public struct Answer: Encodable, Sendable {
        public let type: String
        public var noul: Double?
        public var choice: String?
        public var score: Double?
        public var confidence: Double?
        public var probabilities: [String: Double]?
        public var legend: [String: ClefJSON]?
    }

    static func answer(question: ClefQuestion, probabilities: [Float]) throws -> Answer {
        guard probabilities.count == question.options.count,
              probabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw ClefError.invalidInput("Clef produced invalid option probabilities.")
        }
        func rounded(_ value: Double) -> Double { (value * 10_000).rounded(.toNearestOrEven) / 10_000 }
        let pairs = zip(question.options, probabilities)
        var answer = Answer(type: question.type.rawValue)
        if question.type == .noul {
            answer.noul = rounded(Double(probabilities[0]))
        } else {
            answer.probabilities = Dictionary(uniqueKeysWithValues: pairs.map { ($0.id, rounded(Double($1))) })
            let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
            answer.confidence = rounded(Double(probabilities[best]))
            if question.type == .choice {
                let values = Dictionary(uniqueKeysWithValues: zip(question.options.map(\.id), probabilities))
                var selected = question.responseOptionIDs[0]
                for option in question.responseOptionIDs.dropFirst() where (values[option] ?? 0) > (values[selected] ?? 0) {
                    selected = option
                }
                answer.choice = selected
            }
            else {
                answer.score = rounded(probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * Double($1.element) })
                answer.legend = Dictionary(uniqueKeysWithValues: question.options.map { ($0.id, $0.description) })
            }
        }
        return answer
    }
}
