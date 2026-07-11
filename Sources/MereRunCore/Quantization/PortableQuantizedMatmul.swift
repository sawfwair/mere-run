import Foundation
import MLX
import MLXNN

/// Selects Linux/CUDA quantized matrix multiplication. Automatic mode probes
/// each native MLX operation once, promotes it when evaluation succeeds, and
/// permanently falls back to dense compatibility when that operation is not
/// present in the linked CUDA runtime.
public enum MLXCUDAQuant {
    public enum Mode: String, Sendable {
        case automatic
        case native
        case dense
    }

    enum Operation: String, Hashable {
        case quantizedMM = "quantized_mm"
        case gatherQMM = "gather_qmm"
    }

    enum NativeDecision {
        case native
        case probe
        case dense
    }

    private enum Capability {
        case unknown
        case probing
        case supported
        case unsupported
    }

    private static let stateLock = NSLock()
    private nonisolated(unsafe) static var capabilities: [Operation: Capability] = [:]
    private nonisolated(unsafe) static var loggedSelections: Set<String> = []

    public static let mode = parseMode(
        ProcessInfo.processInfo.environment["MERERUN_MLX_CUDA_NATIVE_QUANT"]
    )

    public static func parseMode(_ raw: String?) -> Mode {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case nil, "", "auto", "automatic":
            return .automatic
        case "1", "true", "yes", "on", "native":
            return .native
        case "0", "false", "no", "off", "dense":
            return .dense
        default:
            return .automatic
        }
    }

    static func nativeDecision(for operation: Operation) -> NativeDecision {
        switch mode {
        case .native:
            logSelection(operation: operation, mode: .native, backend: "native")
            return .native
        case .dense:
            logSelection(operation: operation, mode: .dense, backend: "dense")
            return .dense
        case .automatic:
            return stateLock.withLock {
                switch capabilities[operation] ?? .unknown {
                case .unknown:
                    capabilities[operation] = .probing
                    return .probe
                case .probing:
                    return .dense
                case .supported:
                    return .native
                case .unsupported:
                    return .dense
                }
            }
        }
    }

    static func completeProbe(operation: Operation, supported: Bool) {
        stateLock.withLock {
            capabilities[operation] = supported ? .supported : .unsupported
        }
        logSelection(
            operation: operation,
            mode: .automatic,
            backend: supported ? "native" : "dense"
        )
    }

    private static func logSelection(operation: Operation, mode: Mode, backend: String) {
        let key = "\(operation.rawValue):\(mode.rawValue):\(backend)"
        let shouldLog = stateLock.withLock { loggedSelections.insert(key).inserted }
        guard shouldLog else { return }
        let message = "[mere.run] mlx_cuda_quant operation=\(operation.rawValue) mode=\(mode.rawValue) backend=\(backend)\n"
        FileHandle.standardError.write(Data(message.utf8))
    }
}

public final class PortableQuantizedLinear: QuantizedLinear {
    private var cachedDequantizedWeight: MLXArray?
    private var cachedDequantizedWeightDType: DType?

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        #if os(Linux)
        if Device.defaultDevice().deviceType == .gpu, mode == .affine {
            switch MLXCUDAQuant.nativeDecision(for: .quantizedMM) {
            case .native:
                return super.callAsFunction(x)
            case .dense:
                return denseOutput(x)
            case .probe:
                let output = super.callAsFunction(x)
                do {
                    try MLX.checkedEval(output)
                    MLXCUDAQuant.completeProbe(operation: .quantizedMM, supported: true)
                    return output
                } catch {
                    MLXCUDAQuant.completeProbe(operation: .quantizedMM, supported: false)
                    return denseOutput(x)
                }
            }
        }
        #endif

        return super.callAsFunction(x)
    }

    private func denseOutput(_ x: MLXArray) -> MLXArray {
        var output = MLX.matmul(x, dequantizedWeight(dtype: x.dtype).T)
        if let bias {
            output = output + bias
        }
        return output
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
    if forceDequantizedFallback {
        return dequantizedGatherQuantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: rhsIndices,
            transpose: transpose,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    #if os(Linux)
    if Device.defaultDevice().deviceType == .gpu {
        switch MLXCUDAQuant.nativeDecision(for: .gatherQMM) {
        case .native:
            break
        case .dense:
            return dequantizedGatherQuantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: rhsIndices,
                transpose: transpose,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        case .probe:
            let output = MLX.gatherQuantizedMM(
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
            do {
                try MLX.checkedEval(output)
                MLXCUDAQuant.completeProbe(operation: .gatherQMM, supported: true)
                return output
            } catch {
                MLXCUDAQuant.completeProbe(operation: .gatherQMM, supported: false)
                return dequantizedGatherQuantizedMM(
                    x,
                    weight,
                    scales: scales,
                    biases: biases,
                    rhsIndices: rhsIndices,
                    transpose: transpose,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }
        }
    }
    #endif

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

private func dequantizedGatherQuantizedMM(
    _ x: MLXArray,
    _ weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?,
    rhsIndices: MLXArray,
    transpose: Bool,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> MLXArray {
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
