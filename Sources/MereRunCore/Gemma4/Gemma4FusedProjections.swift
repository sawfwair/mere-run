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

/// A fused view over several `QuantizedLinear` projections that share one input,
/// so decode issues a single quantized matmul instead of N and splits the output.
///
/// Quantized weights pack each output row independently (both affine and nvfp4),
/// so row-wise concatenation of `weight`/`scales`/`biases` is exact — the fused
/// matmul produces bit-identical results to the separate matmuls.
///
/// The fusion lives outside the module tree: LoRA injection, weight export, and
/// re-quantization keep operating on the original modules. `matches(_:)` compares
/// source-module identity so any module replacement (e.g. LoRA wrapping) makes
/// the caller drop the fusion and fall back to the unfused path.
final class Gemma4FusedQuantizedProjection {
    private let weight: MLXArray
    private let scales: MLXArray
    private let biases: MLXArray?
    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode
    private let splitIndices: [Int]
    private let sourceIDs: [ObjectIdentifier]

    private init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        splitIndices: [Int],
        sourceIDs: [ObjectIdentifier]
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.splitIndices = splitIndices
        self.sourceIDs = sourceIDs
    }

    /// Only these exact classes carry plain `quantizedMM` semantics on Metal.
    /// Subclasses with extra math (ResidualQuantizedLinear's correction term)
    /// or wrapped behavior (LoRA layers)
    /// must never fuse — the fused matmul would silently drop their deltas.
    private static func isFusableClass(_ module: Linear) -> Bool {
        type(of: module) == QuantizedLinear.self
            || type(of: module) == PortableQuantizedLinear.self
    }

    /// Builds a fusion over the given projections, or nil when any projection is
    /// missing, biased, not an exactly-fusable quantized class, or when the
    /// quantization layouts disagree.
    static func fuse(_ projections: [Linear?]) -> Gemma4FusedQuantizedProjection? {
        var quantized: [QuantizedLinear] = []
        quantized.reserveCapacity(projections.count)
        for module in projections {
            guard let module,
                  isFusableClass(module),
                  let projection = module as? QuantizedLinear,
                  projection.bias == nil else {
                return nil
            }
            quantized.append(projection)
        }
        guard quantized.count >= 2, let first = quantized.first else { return nil }

        let packedWidth = first.weight.dim(-1)
        let scaleWidth = first.scales.dim(-1)
        let layoutMatches = quantized.allSatisfy {
            $0.groupSize == first.groupSize
                && $0.bits == first.bits
                && $0.mode == first.mode
                && $0.weight.ndim == 2
                && $0.weight.dim(-1) == packedWidth
                && $0.weight.dtype == first.weight.dtype
                && $0.scales.dim(-1) == scaleWidth
        }
        guard layoutMatches else { return nil }

        let biasesPresent = quantized.allSatisfy { $0.biases != nil }
        let biasesAbsent = quantized.allSatisfy { $0.biases == nil }
        guard biasesPresent || biasesAbsent else { return nil }

        let outputDims = quantized.map { $0.weight.dim(0) }
        var splitIndices: [Int] = []
        var runningTotal = 0
        for dim in outputDims.dropLast() {
            runningTotal += dim
            splitIndices.append(runningTotal)
        }

        let weight = concatenated(quantized.map(\.weight), axis: 0)
        let scales = concatenated(quantized.map(\.scales), axis: 0)
        let biases = biasesPresent
            ? concatenated(quantized.compactMap(\.biases), axis: 0)
            : nil
        var toEvaluate = [weight, scales]
        if let biases {
            toEvaluate.append(biases)
        }
        MLX.eval(toEvaluate)

        return Gemma4FusedQuantizedProjection(
            weight: weight,
            scales: scales,
            biases: biases,
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode,
            splitIndices: splitIndices,
            sourceIDs: quantized.map(ObjectIdentifier.init)
        )
    }

    /// True while the fusion still mirrors exactly these module instances.
    func matches(_ projections: [Linear?]) -> Bool {
        guard projections.count == sourceIDs.count else { return false }
        for (module, id) in zip(projections, sourceIDs) {
            guard let module, ObjectIdentifier(module) == id else { return false }
        }
        return true
    }

    /// One fused quantized matmul, split back into per-projection outputs.
    func callSplit(_ x: MLXArray) -> [MLXArray] {
        split(callFused(x), indices: splitIndices, axis: -1)
    }

    /// The fused quantized matmul without the split — for consumers (fused
    /// decode kernels) that index the concatenated row directly.
    func callFused(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
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
