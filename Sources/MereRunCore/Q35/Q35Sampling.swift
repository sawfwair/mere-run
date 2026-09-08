import MLX

enum Q35Sampling {
    static func generationConfig(for request: ChatRequest, promptTokenCount: Int) -> GenerationConfig {
        GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topK: request.topK ?? 0,
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: request.repetitionPenalty == 1 ? nil : Float(request.repetitionPenalty),
            repetitionContextSize: Int.max,
            presencePenalty: Float(request.presencePenalty),
            frequencyPenalty: Float(request.frequencyPenalty),
            penaltyPromptTokenCount: promptTokenCount
        )
    }

    static func acceptsDraft(probability: Float) -> Bool {
        MLXRandom.uniform().item(Float.self) < probability
    }

    /// Create keys on the request's stream instead of extending a global lazy
    /// random-state graph that can belong to another thread's GPU stream.
    static func withRequestState<Result: Sendable>(
        seed: UInt64?,
        _ operation: @Sendable () async throws -> Result
    ) async rethrows -> Result {
        let state = seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
        return try await withRandomState(state, body: operation)
    }
}
