import Foundation
import MLX
import MLXNN

/// Selects Linux/CUDA quantized matrix multiplication. Automatic mode probes
/// each native MLX operation and packed layout once, promotes it when
/// evaluation succeeds, and permanently falls back to dense compatibility
/// when that combination is not present in the linked CUDA runtime.
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

    struct CapabilityKey: Hashable {
        let operation: Operation
        let bits: Int
        let groupSize: Int
        let quantizationMode: String

        init(
            operation: Operation,
            bits: Int,
            groupSize: Int,
            quantizationMode: QuantizationMode
        ) {
            self.operation = operation
            self.bits = bits
            self.groupSize = groupSize
            self.quantizationMode = quantizationMode.rawValue
        }
    }

    private enum Capability {
        case unknown
        case probing
        case supported
        case unsupported
    }

    private static let stateLock = NSLock()
    private nonisolated(unsafe) static var capabilities: [CapabilityKey: Capability] = [:]
    private nonisolated(unsafe) static var loggedSelections: Set<String> = []

    public static let mode = parseMode(
        ProcessInfo.processInfo.environment["MERERUN_MLX_CUDA_NATIVE_QUANT"]
    )

    /// Compatibility view of the pre-automatic policy. Automatic probing is
    /// intentionally not reported as a forced native preference because each
    /// operation may independently fall back to dense execution.
    @available(*, deprecated, message: "Use MLXCUDAQuant.mode; automatic mode probes each operation independently.")
    public static let preferNativeQuant = mode == .native

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

    static func nativeDecision(
        for operation: Operation,
        bits: Int,
        groupSize: Int,
        quantizationMode: QuantizationMode
    ) -> NativeDecision {
        let key = CapabilityKey(
            operation: operation,
            bits: bits,
            groupSize: groupSize,
            quantizationMode: quantizationMode
        )
        switch mode {
        case .native:
            logSelection(key: key, mode: .native, backend: "native")
            return .native
        case .dense:
            logSelection(key: key, mode: .dense, backend: "dense")
            return .dense
        case .automatic:
            return stateLock.withLock {
                switch capabilities[key] ?? .unknown {
                case .unknown:
                    capabilities[key] = .probing
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

    static func completeProbe(
        operation: Operation,
        bits: Int,
        groupSize: Int,
        quantizationMode: QuantizationMode,
        supported: Bool
    ) {
        let key = CapabilityKey(
            operation: operation,
            bits: bits,
            groupSize: groupSize,
            quantizationMode: quantizationMode
        )
        stateLock.withLock {
            capabilities[key] = supported ? .supported : .unsupported
        }
        logSelection(
            key: key,
            mode: .automatic,
            backend: supported ? "native" : "dense"
        )
    }

    private static func logSelection(key: CapabilityKey, mode: Mode, backend: String) {
        let selectionKey = [
            key.operation.rawValue,
            String(key.bits),
            String(key.groupSize),
            key.quantizationMode,
            mode.rawValue,
            backend,
        ].joined(separator: ":")
        let shouldLog = stateLock.withLock { loggedSelections.insert(selectionKey).inserted }
        guard shouldLog else { return }
        let message = """
        [mere.run] mlx_cuda_quant operation=\(key.operation.rawValue) \
        mode=\(mode.rawValue) backend=\(backend) bits=\(key.bits) \
        group_size=\(key.groupSize) quantization_mode=\(key.quantizationMode)\n
        """
        FileHandle.standardError.write(Data(message.utf8))
    }
}

public final class PortableQuantizedLinear: QuantizedLinear {
    private var cachedDequantizedWeight: MLXArray?
    private var cachedDequantizedWeightDType: DType?
    /// Retain the dequantized CUDA fallback weight between calls. Large dense
    /// transformers can disable this and evaluate layerwise to bound residency
    /// while preserving native quantized kernels when they are available.
    public var cacheDenseFallbackWeight = true
    /// Training can prefer bounded residency over the native quantized kernel.
    /// The dequantized weight remains part of the current lazy graph only and is
    /// not retained by the module between calls.
    public var useUncachedDenseFallback = false

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        #if os(macOS)
        if bias == nil,
           let biases,
           let output = SmallBatchAffineQMV.matmul(
               x,
               weight: weight,
               scales: scales,
               biases: biases,
               groupSize: groupSize,
               bits: bits,
               mode: mode
           ) {
            return output
        }
        #endif

        #if os(Linux)
        if Device.defaultDevice().deviceType == .gpu, mode == .affine {
            if useUncachedDenseFallback {
                return denseOutput(x, cacheWeight: false)
            }
            switch MLXCUDAQuant.nativeDecision(
                for: .quantizedMM,
                bits: bits,
                groupSize: groupSize,
                quantizationMode: mode
            ) {
            case .native:
                return super.callAsFunction(x)
            case .dense:
                return denseOutput(x, cacheWeight: cacheDenseFallbackWeight)
            case .probe:
                let output = super.callAsFunction(x)
                do {
                    try MLX.checkedEval(output)
                    MLXCUDAQuant.completeProbe(
                        operation: .quantizedMM,
                        bits: bits,
                        groupSize: groupSize,
                        quantizationMode: mode,
                        supported: true
                    )
                    return output
                } catch {
                    MLXCUDAQuant.completeProbe(
                        operation: .quantizedMM,
                        bits: bits,
                        groupSize: groupSize,
                        quantizationMode: mode,
                        supported: false
                    )
                    return denseOutput(x, cacheWeight: cacheDenseFallbackWeight)
                }
            }
        }
        #endif

        return super.callAsFunction(x)
    }

    private func denseOutput(_ x: MLXArray, cacheWeight: Bool) -> MLXArray {
        let fullWeight = cacheWeight
            ? dequantizedWeight(dtype: x.dtype)
            : MLX.dequantized(
                weight,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                dtype: x.dtype
            )
        var output = MLX.matmul(x, fullWeight.T)
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
        switch MLXCUDAQuant.nativeDecision(
            for: .gatherQMM,
            bits: bits,
            groupSize: groupSize,
            quantizationMode: mode
        ) {
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
                MLXCUDAQuant.completeProbe(
                    operation: .gatherQMM,
                    bits: bits,
                    groupSize: groupSize,
                    quantizationMode: mode,
                    supported: true
                )
                return output
            } catch {
                MLXCUDAQuant.completeProbe(
                    operation: .gatherQMM,
                    bits: bits,
                    groupSize: groupSize,
                    quantizationMode: mode,
                    supported: false
                )
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
