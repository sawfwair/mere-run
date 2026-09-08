import Foundation

/// Holds one machine reservation until the operation finishes, fails, or cancels.
/// A resident server should use this scope for its full lifetime, not per request.
public func withMachineInferenceAdmission<Result: Sendable>(
    using coordinator: MachineInferenceCoordinator,
    request: MachineInferenceRequest,
    onWait: (@Sendable (MachineInferenceAdmissionSnapshot) -> Void)? = nil,
    isolation _: isolated (any Actor)? = #isolation,
    operation: () async throws -> Result
) async throws -> Result {
    let lease = try await coordinator.acquire(request, onWait: onWait)
    defer { lease.release() }
    try Task.checkCancellation()
    return try await operation()
}

/// Owns a request slot until this operation completes. Streaming producers that
/// outlive their handler must retain an explicit lease until stream termination.
public func withRuntimeRequestAdmission<Result: Sendable>(
    using admission: RuntimeRequestAdmission,
    isolation _: isolated (any Actor)? = #isolation,
    operation: () async throws -> Result
) async throws -> Result {
    let lease = try await admission.acquire()
    do {
        try Task.checkCancellation()
        let result = try await operation()
        await lease.release(cancelled: Task.isCancelled)
        return result
    } catch {
        await lease.release(cancelled: error is CancellationError || Task.isCancelled)
        throw error
    }
}
