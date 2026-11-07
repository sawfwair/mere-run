import Foundation

public enum YuE2PlanningMode: String, Codable, CaseIterable, Sendable {
    case full, melody, off

    var instruction: String {
        switch self {
        case .full:
            "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
        case .melody:
            "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
        case .off:
            "Generate music with codec tokens from the given conditions."
        }
    }
}

public struct YuE2Sampling: Codable, Equatable, Sendable {
    public var temperature: Float = 1
    public var topP: Float = 0.95
    public var topK: Int = 100
    public var repetitionPenalty: Float = 1.2
    public var penaltyWindow: Int = 50
    public var minimumTokens: Int = 200
    public var maximumTokens: Int = 9000

    public init() {}

    public static var score: Self {
        var value = Self()
        value.temperature = 0.7
        value.topP = 0.9
        value.topK = 30
        value.repetitionPenalty = 1.005
        value.penaltyWindow = 100
        value.minimumTokens = 32
        value.maximumTokens = 4096
        return value
    }

    func validate() throws {
        guard temperature.isFinite, (0...5).contains(temperature), topP.isFinite, topP > 0, topP <= 1,
              (1...YuE2Protocol.vocabularySize).contains(topK), repetitionPenalty.isFinite, repetitionPenalty > 0,
              (1...100).contains(penaltyWindow), minimumTokens >= 0, minimumTokens <= maximumTokens,
              (1..<YuE2Protocol.context).contains(maximumTokens) else {
            throw YuE2Error.invalidRequest("Invalid sampling limits, temperature, top-p, top-k, or repetition penalty.")
        }
    }
}

/// Validated, immutable effective inputs shared by native execution and CLI receipts.
public struct YuE2GenerationPlan: Codable, Equatable, Sendable {
    public let style: String
    public let lyrics: String
    public let planning: YuE2PlanningMode
    public let abc: String?
    public let seed: UInt64
    public let guidanceScale: Float
    public let scoreSampling: YuE2Sampling
    public let semanticSampling: YuE2Sampling
    public let steps: Int
    public let instrumental: Bool?

    public init(
        style: String, lyrics: String, planning: YuE2PlanningMode = .full, abc: String? = nil,
        seed: UInt64 = 831001, guidanceScale: Float? = nil, scoreSampling: YuE2Sampling = .score,
        semanticSampling: YuE2Sampling = YuE2Sampling(), steps: Int = 32, instrumental: Bool = false
    ) throws {
        self.style = style
        self.lyrics = lyrics
        self.planning = planning
        self.abc = abc
        self.seed = seed
        self.guidanceScale = guidanceScale ?? (planning == .off ? 1.01 : 1)
        self.scoreSampling = scoreSampling
        self.semanticSampling = semanticSampling
        self.steps = steps
        self.instrumental = instrumental
        try validate()
    }

    public func validate() throws {
        if instrumental == true, planning == .off || !lyrics.isEmpty {
            throw YuE2Error.invalidRequest("Instrumental score conversion requires full or melody planning and empty lyrics.")
        }
        guard !style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw YuE2Error.invalidRequest("Provide a music style in the caption.")
        }
        guard seed < 1 << 63, guidanceScale.isFinite, (0...20).contains(guidanceScale),
              (1...1000).contains(steps), semanticSampling.maximumTokens <= 9000 else {
            throw YuE2Error.invalidRequest("Require seed < 2^63, guidance in 0...20, steps in 1...1000, and at most 9000 music frames.")
        }
        if let abc, planning == .off || abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw YuE2Error.invalidRequest("An external ABC score requires nonempty text and full or melody planning.")
        }
        try scoreSampling.validate()
        try semanticSampling.validate()
    }

    enum CodingKeys: String, CodingKey {
        case style, lyrics, planning, abc, seed, guidanceScale, scoreSampling, semanticSampling, steps, instrumental
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(style: values.decode(String.self, forKey: .style),
            lyrics: values.decodeIfPresent(String.self, forKey: .lyrics) ?? "",
            planning: values.decodeIfPresent(YuE2PlanningMode.self, forKey: .planning) ?? .full,
            abc: values.decodeIfPresent(String.self, forKey: .abc), seed: values.decodeIfPresent(UInt64.self, forKey: .seed) ?? 831001,
            guidanceScale: values.decodeIfPresent(Float.self, forKey: .guidanceScale),
            scoreSampling: values.decodeIfPresent(YuE2Sampling.self, forKey: .scoreSampling) ?? .score,
            semanticSampling: values.decodeIfPresent(YuE2Sampling.self, forKey: .semanticSampling) ?? YuE2Sampling(),
            steps: values.decodeIfPresent(Int.self, forKey: .steps) ?? 32,
            instrumental: values.decodeIfPresent(Bool.self, forKey: .instrumental) ?? false)
    }

    var prompt: String { prompt(lyrics: lyrics) }

    var planningPrompt: String {
        prompt(lyrics: instrumental == true ? "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n" : lyrics)
    }

    private func prompt(lyrics text: String) -> String {
        "\(planning.instruction)\n[Tags]\n\(style)\n[Lyrics]\n\(text)\n"
    }
}

enum YuE2Protocol {
    static let version = "yue2-native-v1"
    static let endOfText = 151643
    static let abcStart = 151847
    static let abcEnd = 151848
    static let musicStart = 151851
    static let musicEnd = 151852
    static let codecOffset = 151853
    static let codecSize = 32768
    static let vocabularySize = 184704
    static let context = 24576

    static func prefix(plan: YuE2GenerationPlan, tokenizer: YuE2Tokenizer, score: [Int]?) throws -> [Int] {
        let text = score == nil && plan.planning != .off ? plan.planningPrompt : plan.prompt
        let base = try [endOfText] + tokenizer.encode(text)
        if plan.planning == .off { return base + [abcStart, abcEnd, musicStart] }
        guard let score else { return base + [abcStart] }
        try validateScore(score)
        return base + [abcStart] + score + [abcEnd, musicStart]
    }

    static func negative(plan: YuE2GenerationPlan, tokenizer: YuE2Tokenizer, score: [Int]) throws -> [Int] {
        let base = try [endOfText] + tokenizer.encode(plan.planning.instruction)
        if plan.planning == .off { return base + [musicStart] }
        try validateScore(score)
        return base + [abcStart] + score + [abcEnd, musicStart]
    }

    static func validateScore(_ ids: [Int]) throws {
        guard ids.allSatisfy({ (0..<endOfText).contains($0) }) else {
            throw YuE2Error.invalidRequest("ABC IDs must stay inside the ordinary text vocabulary.")
        }
    }

    static func chunkRanges(frames: Int, prefixTokens: Int, context: Int = context) throws -> [Range<Int>] {
        guard (1...Self.context).contains(context), prefixTokens > 0, prefixTokens < context,
              frames > 0, frames <= 9000 else {
            throw YuE2Error.invalidRequest("Invalid acoustic frame count or prefix length.")
        }
        let size = (context - prefixTokens - 3) / 2
        guard size > 0 else { throw YuE2Error.invalidRequest("The score leaves no acoustic context.") }
        return stride(from: 0, to: frames, by: size).map { $0..<min($0 + size, frames) }
    }
}
