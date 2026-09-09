import MLX

/// Input to a shared autoregressive decode: everything the loop needs that
/// is not the model itself.
public struct AutoregressiveDecodeRequest {
    public let initialLogits: MLXArray
    public let generationConfig: GenerationConfig
    public let eosTokens: Set<Int>
    public let tokenBudget: Int
    /// Seeds the on-GPU repetition window (typically the prompt tokens).
    public let historySeedTokens: [Int]
    /// Optional additive logit bias applied every step (e.g. token bans).
    public let banMask: MLXArray?
    public let logprobCapture: ChatLogprobCapture
    public let logprobRegion: ChatLogprobRegion

    public init(
        initialLogits: MLXArray,
        generationConfig: GenerationConfig,
        eosTokens: Set<Int>,
        tokenBudget: Int,
        historySeedTokens: [Int] = [],
        banMask: MLXArray? = nil,
        logprobCapture: ChatLogprobCapture = .none,
        logprobRegion: ChatLogprobRegion = .visible
    ) {
        self.initialLogits = initialLogits
        self.generationConfig = generationConfig
        self.eosTokens = eosTokens
        self.tokenBudget = tokenBudget
        self.historySeedTokens = historySeedTokens
        self.banMask = banMask
        self.logprobCapture = logprobCapture
        self.logprobRegion = logprobRegion
    }
}

public struct AutoregressiveDecodeResult {
    public let generatedTokens: [Int]
    public let decodeSeconds: Double
    /// Seconds from decode start to the first confirmed token (time to
    /// first token, excluding prefill). Nil when nothing was generated.
    public let firstTokenSeconds: Double?
    /// Host time spent building/scheduling each step's graph (sample +
    /// forward + asyncEval). With the wait time, callers can emit the same
    /// build/wait decode-trace lines the hand-rolled loops printed.
    public let buildSeconds: Double
    /// Host time spent blocked on step confirmation readbacks.
    public let waitSeconds: Double
    public let logprobs: ChatLogprobDiagnostics?

    public init(
        generatedTokens: [Int],
        decodeSeconds: Double,
        firstTokenSeconds: Double? = nil,
        buildSeconds: Double = 0,
        waitSeconds: Double = 0,
        logprobs: ChatLogprobDiagnostics? = nil
    ) {
        self.generatedTokens = generatedTokens
        self.decodeSeconds = decodeSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.buildSeconds = buildSeconds
        self.waitSeconds = waitSeconds
        self.logprobs = logprobs
    }
}
