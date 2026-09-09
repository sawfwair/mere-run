public struct ChatLogprobCapture: Codable, Sendable, Hashable {
    public enum Mode: String, Codable, Sendable, Hashable {
        case none
        case summary
        case tokens
        case top
    }

    public let mode: Mode
    public let topLogprobs: Int

    public static let none = ChatLogprobCapture(mode: .none)
    public static let summary = ChatLogprobCapture(mode: .summary)
    public static let tokens = ChatLogprobCapture(mode: .tokens)

    public static func top(_ count: Int) -> ChatLogprobCapture {
        ChatLogprobCapture(mode: .top, topLogprobs: count)
    }

    public init(mode: Mode, topLogprobs: Int = 0) {
        self.mode = mode
        self.topLogprobs = mode == .top ? min(max(topLogprobs, 1), 20) : 0
    }

    public var isEnabled: Bool { mode != .none }
    public var includesTokens: Bool { mode == .tokens || mode == .top }
}

public enum ChatLogprobRegion: String, Codable, Sendable, Hashable {
    case reasoning
    case visible
    case code
    case toolName = "tool_name"
    case toolArgument = "tool_argument"
    case markup
    case unknown
}

public struct ChatTopLogprob: Codable, Sendable, Hashable {
    public var tokenID: Int
    public var token: String?
    public var rawLogprob: Double
    public var policyLogprob: Double

    public init(
        tokenID: Int,
        token: String? = nil,
        rawLogprob: Double,
        policyLogprob: Double
    ) {
        self.tokenID = tokenID
        self.token = token
        self.rawLogprob = rawLogprob
        self.policyLogprob = policyLogprob
    }
}

public struct ChatTokenLogprob: Codable, Sendable, Hashable {
    public var tokenID: Int
    public var token: String?
    public var region: ChatLogprobRegion
    public var rawLogprob: Double
    public var policyLogprob: Double
    public var rawEntropy: Double
    public var policyEntropy: Double
    public var rawTop1Top2Margin: Double
    public var policyTop1Top2Margin: Double
    public var topLogprobs: [ChatTopLogprob]

    public init(
        tokenID: Int,
        token: String? = nil,
        region: ChatLogprobRegion = .unknown,
        rawLogprob: Double,
        policyLogprob: Double,
        rawEntropy: Double,
        policyEntropy: Double,
        rawTop1Top2Margin: Double,
        policyTop1Top2Margin: Double,
        topLogprobs: [ChatTopLogprob] = []
    ) {
        self.tokenID = tokenID
        self.token = token
        self.region = region
        self.rawLogprob = rawLogprob
        self.policyLogprob = policyLogprob
        self.rawEntropy = rawEntropy
        self.policyEntropy = policyEntropy
        self.rawTop1Top2Margin = rawTop1Top2Margin
        self.policyTop1Top2Margin = policyTop1Top2Margin
        self.topLogprobs = topLogprobs
    }
}

public struct ChatLogprobSummary: Codable, Sendable, Hashable {
    public let tokenCount: Int
    public let meanRawLogprob: Double
    public let minimumRawLogprob: Double
    public let meanPolicyLogprob: Double
    public let minimumPolicyLogprob: Double
    public let meanRawEntropy: Double
    public let meanPolicyEntropy: Double
    public let meanRawTop1Top2Margin: Double
    public let meanPolicyTop1Top2Margin: Double

    public init(tokens: [ChatTokenLogprob]) {
        tokenCount = tokens.count
        meanRawLogprob = Self.mean(tokens.map(\.rawLogprob))
        minimumRawLogprob = tokens.map(\.rawLogprob).min() ?? 0
        meanPolicyLogprob = Self.mean(tokens.map(\.policyLogprob))
        minimumPolicyLogprob = tokens.map(\.policyLogprob).min() ?? 0
        meanRawEntropy = Self.mean(tokens.map(\.rawEntropy))
        meanPolicyEntropy = Self.mean(tokens.map(\.policyEntropy))
        meanRawTop1Top2Margin = Self.mean(tokens.map(\.rawTop1Top2Margin))
        meanPolicyTop1Top2Margin = Self.mean(tokens.map(\.policyTop1Top2Margin))
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }
}

public struct ChatLogprobDiagnostics: Codable, Sendable, Hashable {
    public enum Source: String, Codable, Sendable, Hashable {
        case finalTarget = "final_target"
    }

    public let capture: ChatLogprobCapture
    public let source: Source
    public let summary: ChatLogprobSummary
    public let tokens: [ChatTokenLogprob]?
    public let captureSeconds: Double

    public init(
        capture: ChatLogprobCapture,
        source: Source = .finalTarget,
        measuredTokens: [ChatTokenLogprob],
        captureSeconds: Double
    ) {
        self.capture = capture
        self.source = source
        self.summary = ChatLogprobSummary(tokens: measuredTokens)
        self.tokens = capture.includesTokens ? measuredTokens : nil
        self.captureSeconds = captureSeconds
    }
}
