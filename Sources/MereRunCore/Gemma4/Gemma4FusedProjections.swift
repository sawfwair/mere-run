import Foundation
import MLX
import MLXNN

enum Gemma4FusedProjectionPolicy {
    /// Kill switch: MERERUN_GEMMA4_FUSED_PROJ=0 disables fused projections.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_FUSED_PROJ"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()

    /// Opt-in: MERERUN_GEMMA4_COMPILED_SEGMENTS=1 enables MLX-compiled per-layer
    /// decode segments. Off by default: mlx-swift 0.31.4 routes every compiled
    /// call through the global evalLock plus a fresh closure trampoline, which
    /// measured 2.6x SLOWER than the interpreted path at 96 calls/token
    /// (70.8ms vs 27.1ms per token). Revisit if CompiledFunction.call gets
    /// cheaper in a future mlx-swift.
    static let compiledSegmentsEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_COMPILED_SEGMENTS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw == "1" || raw == "true" || raw == "on"
    }()

    /// Opt-in: MERERUN_GEMMA4_FUSED_DECODE_KERNELS=1 enables the custom fused
    /// Metal kernels on the seq==1 decode path. Off by default: they hold
    /// single-stream decode neutral (~±2%) on an idle GPU, but their
    /// single-rounding float32 numerics diverge from the multi-token verify
    /// forward's per-op rounding, which degrades Gemma MTP speculative
    /// acceptance at long context (measured 43.7 -> 28.9 tok/s at 7.4k).
    /// They reduce per-token dispatches ~45%, which still pays off when the
    /// GPU is shared with training — enable explicitly for that.
    static let fusedDecodeKernelsEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_FUSED_DECODE_KERNELS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw == "1" || raw == "true" || raw == "on"
    }()
}

/// An MLX-compiled decode segment plus the identity fingerprint of every module
/// and parameter the trace baked in. Compiled graphs freeze whatever the closure
/// captured, so any module replacement (LoRA injection, requantization) must
/// invalidate the segment — callers compare `fingerprint` before each use.
struct Gemma4CompiledSegment {
    let function: ([MLXArray]) -> [MLXArray]
    let fingerprint: [ObjectIdentifier]

    func matches(_ current: [ObjectIdentifier]) -> Bool {
        fingerprint == current
    }
}

/// Small cached MLXArray scalars (norm epsilons) so decode doesn't allocate a
/// fresh host array per layer per token.
enum Gemma4DecodeScalarCache {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var arrays: [UInt32: MLXArray] = [:]

    static func epsilon(_ value: Float) -> MLXArray {
        let key = value.bitPattern
        lock.lock()
        defer { lock.unlock() }
        if let existing = arrays[key] {
            return existing
        }
        let array = MLXArray([value])
        arrays[key] = array
        return array
    }
}
