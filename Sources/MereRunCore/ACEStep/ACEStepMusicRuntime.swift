import MLX

/// Serializes the complete lifetime of lazy music tensors on retained CPU and GPU streams.
public actor ACEStepMusicRuntime {
    private let resources: ACEStepModelResources
    private let variant: ACEStepCheckpointVariant
    private let streams = Stream.Context()
    private var operation: ACEStepGenerationOperation?

    public init(resources: ACEStepModelResources, variant: ACEStepCheckpointVariant) {
        self.resources = resources
        self.variant = variant
    }

    /// The synchronous body must evaluate or export tensors before returning a Sendable result.
    public func perform<Result: Sendable>(
        _ body: (ACEStepGenerationOperation) throws -> Result
    ) async throws -> Result {
        try Task.checkCancellation()
        return try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            let active = try operation ?? ACEStepGenerationOperation(resources: resources, variant: variant)
            operation = active
            try Task.checkCancellation()
            return try body(active)
        }
    }
}
