import MLX

enum Q35Sampling {
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
