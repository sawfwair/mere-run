import Foundation
import MLX
import MLXNN

/// Controls whether the Linux/CUDA quantized paths dequantize to dense (the safe
/// fallback) or call MLX's native quantized kernels.
///
/// The dense fallback exists because early MLX CUDA builds lacked
/// `quantized_matmul`/`GatherQMM`, but dequantizing 4-bit weights to bf16 forfeits
/// the quantization speedup (3-4x slower decode on GB10 vs native quantized kernels).
/// Newer MLX may support them; set `MERERUN_MLX_CUDA_NATIVE_QUANT=1` to use native
/// quantized ops on CUDA. Default keeps the safe dense fallback.
public enum MLXCUDAQuant {
    public static let preferNativeQuant: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_MLX_CUDA_NATIVE_QUANT"]?
            .lowercased() else { return false }
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }()
}

public final class PortableQuantizedLinear: QuantizedLinear {
    private var cachedDequantizedWeight: MLXArray?
    private var cachedDequantizedWeightDType: DType?

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        #if os(Linux)
        if Device.defaultDevice().deviceType == .gpu, mode == .affine, !MLXCUDAQuant.preferNativeQuant {
            var output = MLX.matmul(x, dequantizedWeight(dtype: x.dtype).T)
            if let bias {
                output = output + bias
            }
            return output
        }
        #endif

        return super.callAsFunction(x)
    }

    private func dequantizedWeight(dtype: DType) -> MLXArray {
        if let cachedDequantizedWeight, cachedDequantizedWeightDType == dtype {
            return cachedDequantizedWeight
        }

        let fullWeight = MLX.dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            dtype: dtype
        )
        eval(fullWeight)
        cachedDequantizedWeight = fullWeight
        cachedDequantizedWeightDType = dtype
        return fullWeight
    }
}

func portableGatherQuantizedMM(
    _ x: MLXArray,
    _ weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?,
    rhsIndices: MLXArray,
    transpose: Bool,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode,
    sortedIndices: Bool,
    forceDequantizedFallback: Bool = false
) -> MLXArray {
    #if os(Linux)
    let shouldUseFallback =
        (Device.defaultDevice().deviceType == .gpu && !MLXCUDAQuant.preferNativeQuant)
        || forceDequantizedFallback
    #else
    let shouldUseFallback = forceDequantizedFallback
    #endif

    if shouldUseFallback {
        let selectedWeight = MLX.take(weight, rhsIndices, axis: 0)
        let selectedScales = MLX.take(scales, rhsIndices, axis: 0)
        let selectedBiases = biases.map { MLX.take($0, rhsIndices, axis: 0) }
        let denseWeight = MLX.dequantized(
            selectedWeight,
            scales: selectedScales,
            biases: selectedBiases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            dtype: x.dtype
        )
        let rhs = transpose ? denseWeight.swappedAxes(-1, -2) : denseWeight
        return MLX.matmul(x, rhs)
    }

    return MLX.gatherQuantizedMM(
        x,
        weight,
        scales: scales,
        biases: biases,
        rhsIndices: rhsIndices,
        transpose: transpose,
        groupSize: groupSize,
        bits: bits,
        mode: mode,
        sortedIndices: sortedIndices
    )
}
