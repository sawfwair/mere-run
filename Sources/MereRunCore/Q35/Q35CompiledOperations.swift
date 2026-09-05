import Foundation
import MLX
import MLXNN

/// Owns compiled graphs for one request's MLX stream. The C++ compile cache
/// keys on its thread-local default stream, while Swift async inference uses
/// task-local streams. Sharing a compiled function across requests can reuse
/// a graph traced on another stream and insert cross-stream synchronization.
final class Q35CompiledOperations: @unchecked Sendable {
    @TaskLocal private static var scoped: Q35CompiledOperations?
    private static let shared = Q35CompiledOperations()

    static var current: Q35CompiledOperations { scoped ?? shared }

    let stream = StreamOrDevice.default.stream
    let swiglu = compile(shapeless: true) { gate, up in
        MLXNN.silu(gate) * up
    }
    let preciseSwiglu = compile(shapeless: true) { hiddenStates, gate, normalized in
        let gate = MLXNN.silu(gate.asType(.float32))
        return (gate * normalized.asType(.float32)).asType(hiddenStates.dtype)
    }
    let computeG = compile(shapeless: true) { aLog, activation, dtBias in
        let delta = softplus(activation.asType(.float32) + dtBias.asType(.float32).expandedDimensions(axes: [0, 1]))
        let decayBase = MLX.exp(aLog.asType(.float32)).expandedDimensions(axes: [0, 1])
        return MLX.exp(-decayBase * delta)
    }

    static func withNewDefaultStream<Result>(
        scoped enabled: Bool = ProcessInfo.processInfo.environment["MERERUN_Q35_SCOPED_COMPILE"] == "1",
        isolation _: isolated (any Actor)? = #isolation,
        _ operation: () async throws -> Result
    ) async rethrows -> Result {
        try await Stream.withNewDefaultStream {
            guard enabled else { return try await operation() }
            return try await $scoped.withValue(Q35CompiledOperations(), operation: operation)
        }
    }
}
