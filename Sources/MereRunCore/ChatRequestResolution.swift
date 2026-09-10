import Foundation

/// Defaults are an entry-point compatibility policy. Explicit values take
/// precedence independently; JSON output always suppresses thinking output.
public enum ChatDefaultPolicy: Sendable {
    case command
    case openAI
}

public struct ChatSamplingOptions: Sendable, Hashable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var thinking: Bool?

    public init(
        temperature: Double? = nil, topP: Double? = nil, topK: Int? = nil,
        minP: Double? = nil, thinking: Bool? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.thinking = thinking
    }
}

public struct ChatRequestIssue: Error, LocalizedError, Equatable, Sendable {
    public let field: String
    public let message: String
    public var errorDescription: String? { "\(field) \(message)" }

    public init(_ field: String, _ message: String) {
        self.field = field
        self.message = message
    }
}

public enum ChatRequestResolver {
    /// Preserves all non-sampling request fields, including tools, media,
    /// cache policy, stop sequences, reasoning budgets, and diagnostics.
    public static func resolve(
        _ request: ChatRequest, modelID: String, sampling: ChatSamplingOptions,
        policy: ChatDefaultPolicy, apiProfile: ManagedModelAPIProfile? = nil
    ) throws -> ChatRequest {
        let modelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let recommended = Q35Resources.recommendedSampling(forModelId: modelID)
        var temperature = recommended?.temperature ?? (policy == .command ? 0.7 : 1.0)
        var topP = recommended?.topP ?? (policy == .command ? 0.9 : 0.95)
        var topK = recommended?.topK
        var minP = 0.0
        var thinking = Q35Resources.thinkingDefault(forModelId: modelID)

        if LagunaResources.handles(modelSpec: modelID) {
            temperature = LagunaResources.recommendedTemperature
            topP = LagunaResources.recommendedTopP
            topK = LagunaResources.recommendedTopK
            minP = LagunaResources.recommendedMinP
        }
        switch policy {
        case .command:
            if modelID == LFM2Resources.visionModelId {
                temperature = 0.2
                topK = 50
            } else if NemotronOmniResources.handles(modelSpec: modelID) {
                temperature = NemotronOmniResources.thinkingTemperature
                topP = NemotronOmniResources.thinkingTopP
                thinking = true
            } else if NemotronHResources.handles(modelSpec: modelID) {
                temperature = NemotronHResources.recommendedTemperature
                topP = NemotronHResources.recommendedTopP
            } else if MuseGlimmerResources.handles(modelSpec: modelID) {
                temperature = MuseGlimmerResources.recommendedTemperature
                topP = MuseGlimmerResources.recommendedTopP
                topK = MuseGlimmerResources.recommendedTopK
                thinking = false
            }
        case .openAI:
            let profile = apiProfile ?? ManagedModelCatalog.apiProfile(for: modelID)
            thinking = thinking || profile?.thinkingLevels == [.high]
        }

        var effective = request
        effective.temperature = sampling.temperature ?? temperature
        effective.topP = sampling.topP ?? topP
        effective.topK = sampling.topK ?? topK
        effective.minP = sampling.minP ?? minP
        effective.showThinking = !request.requiresJSON && (sampling.thinking ?? thinking)
        try validate(effective)
        return effective
    }

    /// Shared numerical invariants, checked before native execution. Transport
    /// capabilities and model-specific options remain adapter constraints.
    public static func validate(_ request: ChatRequest) throws {
        guard !request.messages.isEmpty else {
            throw ChatRequestIssue("messages", "must contain at least one message")
        }
        let upperBound = min(request.maxContextTokens ?? Int(Int32.max), Int(Int32.max))
        guard upperBound > 0 else {
            throw ChatRequestIssue("context_size", "must be greater than zero")
        }
        guard (1...upperBound).contains(request.maxTokens) else {
            throw ChatRequestIssue("max_tokens", "must be between 1 and \(upperBound)")
        }
        for (field, value, range) in [
            ("temperature", request.temperature, 0.0...2.0),
            ("top_p", request.topP, 0.0...1.0),
            ("min_p", request.minP, 0.0...1.0),
            ("presence_penalty", request.presencePenalty, -2.0...2.0),
            ("frequency_penalty", request.frequencyPenalty, -2.0...2.0)
        ] {
            guard value.isFinite, range.contains(value) else {
                throw ChatRequestIssue(field, "must be between \(Int(range.lowerBound)) and \(Int(range.upperBound))")
            }
        }
        if let topK = request.topK, topK < 0 {
            throw ChatRequestIssue("top_k", "must be zero or greater")
        }
        guard request.repetitionPenalty.isFinite,
              Float(request.repetitionPenalty).isFinite, Float(request.repetitionPenalty) > 0 else {
            throw ChatRequestIssue("repetition_penalty", "must be a finite positive sampler value")
        }
    }
}
