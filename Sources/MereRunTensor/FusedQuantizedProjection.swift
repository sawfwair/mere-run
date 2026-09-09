import MLX
import MLXNN

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
package final class FusedQuantizedProjection {
    private let projection: PortableQuantizedLinear
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
        self.projection = PortableQuantizedLinear(
            weight: weight,
            bias: nil,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
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
    package static func fuse(_ projections: [Linear?]) -> FusedQuantizedProjection? {
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

        return FusedQuantizedProjection(
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
    package func matches(_ projections: [Linear?]) -> Bool {
        guard projections.count == sourceIDs.count else { return false }
        for (module, id) in zip(projections, sourceIDs) {
            guard let module, ObjectIdentifier(module) == id else { return false }
        }
        return true
    }

    /// One fused quantized matmul, split back into per-projection outputs.
    package func callSplit(_ x: MLXArray) -> [MLXArray] {
        split(callFused(x), indices: splitIndices, axis: -1)
    }

    /// The fused quantized matmul without the split — for consumers (fused
    /// decode kernels) that index the concatenated row directly.
    package func callFused(_ x: MLXArray) -> MLXArray {
        projection(x)
    }
}
