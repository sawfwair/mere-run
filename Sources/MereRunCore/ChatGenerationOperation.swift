import Foundation

/// One native response, including each response in a tool conversation.
/// The owner retains its runtime or resident lease for the conversation scope.
public enum ChatGenerationOperation {
    public static func run(
        _ request: ChatRequest,
        generate: () async throws -> ChatResponse
    ) async throws -> ChatResponse {
        try Task.checkCancellation()
        try ChatRequestResolver.validate(request)
        let response = try await generate()
        try Task.checkCancellation()
        return response
    }

    /// Awaits cleanup on success, failure, and cancellation. This also keeps a
    /// command's runtime alive across all iterations of its tool conversation.
    public static func withCleanup<Result>(
        _ operation: () async throws -> Result,
        cleanup: () async -> Void
    ) async rethrows -> Result {
        do {
            let result = try await operation()
            await cleanup()
            return result
        } catch {
            await cleanup()
            throw error
        }
    }
}
