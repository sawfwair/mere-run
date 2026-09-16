import Foundation
import MLX

/// Exclusive CPU/GPU stream leases that survive runtime eviction.
///
/// MLX retains backend streams until process exit. Pool completed contexts across
/// generator instances, growing only with simultaneous leases for each selected
/// device. A nested or suspended operation keeps its own context until it ends.
public enum MLXRequestStreams {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var available: [DeviceType?: [MLX.Stream.Context]] = [:]

    /// Run graph construction and evaluation on reusable task-local streams.
    /// Evaluate lazy results before returning, and serialize access to mutable
    /// model state separately. Both devices finish submitted work before reuse.
    public static func withStream<Result>(
        isolation: isolated (any Actor)? = #isolation,
        _ operation: () async throws -> Result
    ) async rethrows -> Result {
        let device = Device.defaultDevice()
        let key = device.deviceType
        let context = lock.withLock { available[key, default: []].popLast() }
            ?? MLX.Stream.Context(device: device)
        defer {
            context.synchronize()
            lock.withLock { available[key, default: []].append(context) }
        }
        return try await Stream.withDefaultStream(context, isolation: isolation, operation)
    }
}
