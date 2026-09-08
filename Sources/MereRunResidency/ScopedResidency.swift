import Foundation

/// Holds an acquired model lease until the operation succeeds, fails, or cancels.
/// Streaming producers that outlive this call must retain an explicit lease.
public func withResidentRuntimeLease<Key: Hashable & Sendable, Value: Sendable, Result: Sendable>(
    using lease: ResidentRuntimeLease<Key, Value>,
    isolation _: isolated (any Actor)? = #isolation,
    operation: (Value) async throws -> Result
) async throws -> Result {
    do {
        try Task.checkCancellation()
        let result = try await operation(lease.value)
        await lease.release()
        return result
    } catch {
        await lease.release()
        throw error
    }
}
