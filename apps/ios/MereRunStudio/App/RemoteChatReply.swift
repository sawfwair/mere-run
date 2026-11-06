import Foundation

/// Stops a submitted relay job when cancellation ends observation of its reply.
@MainActor
enum RemoteChatReply {
    static func observe<Value>(
        read: @MainActor () async throws -> Value,
        cancelJob: @escaping @MainActor () async -> Void
    ) async throws -> Value {
        do {
            return try await read()
        } catch {
            guard Task.isCancelled || error is CancellationError else { throw error }
            // URLSession can throw URLError.cancelled rather than CancellationError.
            // Cleanup needs its own uncancelled task to send the remote cancellation.
            Task { await cancelJob() }
            throw CancellationError()
        }
    }
}
