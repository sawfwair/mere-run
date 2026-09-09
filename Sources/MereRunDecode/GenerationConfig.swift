public struct GenerationConfig: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    /// Minimum token probability relative to the most likely token. Zero disables it.
    public var minP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    public var presencePenalty: Float
    public var frequencyPenalty: Float
    /// Prompt tokens excluded from presence and frequency penalties.
    public var penaltyPromptTokenCount: Int

    var penaltyHistorySize: Int {
        presencePenalty != 0 || frequencyPenalty != 0 ? Int.max : repetitionContextSize
    }

    package var hasActivePenalties: Bool {
        (repetitionPenalty != nil && repetitionPenalty != 1) || presencePenalty != 0 || frequencyPenalty != 0
    }

    var needsPenaltyHistory: Bool {
        repetitionPenalty != nil || presencePenalty != 0 || frequencyPenalty != 0
    }
    /// Token ids that must never be sampled. Applied as a -inf logit mask.
    public var bannedTokens: [Int]
    /// Per-request top-p candidate limit. Nil uses the process policy; zero
    /// requests exact full-vocabulary top-p sampling.
    public var topPPrefilter: Int?

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topK: Int = 0,
        topP: Float = 0.9,
        minP: Float = 0,
        repetitionPenalty: Float? = 1.05,
        repetitionContextSize: Int = 20,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        penaltyPromptTokenCount: Int = 0,
        bannedTokens: [Int] = [],
        topPPrefilter: Int? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.penaltyPromptTokenCount = penaltyPromptTokenCount
        self.bannedTokens = bannedTokens
        self.topPPrefilter = topPPrefilter
    }
}
