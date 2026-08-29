import Foundation
import MLX

enum DiffusionCFGExecutionMode: String, Sendable {
    case automatic
    case batched
    case serial

    static func parse(_ raw: String?) -> DiffusionCFGExecutionMode {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on", "batch", "batched":
            return .batched
        case "0", "false", "no", "off", "serial":
            return .serial
        default:
            return .automatic
        }
    }

    static func current(modelEnvironmentKey: String) -> DiffusionCFGExecutionMode {
        let environment = ProcessInfo.processInfo.environment
        return parse(environment[modelEnvironmentKey] ?? environment["MERERUN_IMAGE_BATCHED_CFG"])
    }
}

enum DiffusionCFGExecution {
    static let gibibyte = UInt64(1_073_741_824)

    static func shouldBatch(
        mode: DiffusionCFGExecutionMode,
        width: Int,
        height: Int,
        physicalMemoryBytes: UInt64,
        activeMemoryBytes: Int,
        cacheMemoryBytes: Int,
        isUnifiedMemory: Bool,
        baseReserveBytes: UInt64 = 4 * gibibyte,
        activationBytesPerPixel: Int = 2_048
    ) -> Bool {
        switch mode {
        case .batched:
            return true
        case .serial:
            return false
        case .automatic:
            guard isUnifiedMemory, physicalMemoryBytes >= 24 * gibibyte else {
                return false
            }
            let allocated = UInt64(max(0, activeMemoryBytes)) + UInt64(max(0, cacheMemoryBytes))
            let headroom = physicalMemoryBytes > allocated ? physicalMemoryBytes - allocated : 0
            let pixelCount = UInt64(max(1, width)) * UInt64(max(1, height))
            let pixelReserve = pixelCount.multipliedReportingOverflow(
                by: UInt64(max(1, activationBytesPerPixel))
            )
            guard !pixelReserve.overflow else {
                return false
            }
            let requiredHeadroom = baseReserveBytes.addingReportingOverflow(pixelReserve.partialValue)
            guard !requiredHeadroom.overflow else {
                return false
            }
            return headroom >= requiredHeadroom.partialValue
        }
    }

    static func canPair(_ first: MLXArray, _ second: MLXArray) -> Bool {
        guard first.ndim > 0, second.ndim == first.ndim,
              first.dim(0) == 1, second.dim(0) == 1 else {
            return false
        }
        return Array(first.shape.dropFirst()) == Array(second.shape.dropFirst())
    }

    static func paired(_ unconditional: MLXArray, _ conditional: MLXArray) -> MLXArray {
        precondition(canPair(unconditional, conditional), "CFG conditions must have matching non-batch dimensions.")
        return MLX.concatenated([unconditional, conditional], axis: 0)
    }

    static func duplicateBatch(_ array: MLXArray) -> MLXArray {
        precondition(array.ndim > 0 && array.dim(0) == 1, "CFG batching expects a single source sample.")
        var shape = array.shape
        shape[0] = 2
        return MLX.broadcast(array, to: shape)
    }

    static func combinePredictions(_ predictions: MLXArray, guidanceScale: Float) -> MLXArray {
        precondition(predictions.dim(0) == 2, "CFG predictions must be ordered [unconditional, conditional].")
        let pair = MLX.split(predictions, parts: 2, axis: 0)
        let unconditional = pair[0]
        let conditional = pair[1]
        return unconditional + (conditional - unconditional) * MLXArray(guidanceScale)
    }

    /// Qwen Image true CFG with the reference pipeline's positive-norm
    /// rescaling. Predictions must be ordered [negative, positive].
    static func combineQwenImagePredictions(
        _ predictions: MLXArray,
        guidanceScale: Float
    ) -> MLXArray {
        precondition(predictions.dim(0) == 2, "CFG predictions must be ordered [negative, positive].")
        let pair = MLX.split(predictions, parts: 2, axis: 0)
        let negative = pair[0]
        let positive = pair[1]
        let combined = negative + (positive - negative) * MLXArray(guidanceScale)
        let positiveNorm = MLX.sqrt(MLX.sum(positive * positive, axis: -1, keepDims: true))
        let combinedNorm = MLX.sqrt(MLX.sum(combined * combined, axis: -1, keepDims: true))
        return combined * (positiveNorm / combinedNorm)
    }

    static func combinePositiveAnchoredPredictions(
        _ predictions: MLXArray,
        guidanceScale: Float
    ) -> MLXArray {
        precondition(predictions.dim(0) == 2, "CFG predictions must be ordered [unconditional, conditional].")
        let pair = MLX.split(predictions, parts: 2, axis: 0)
        let unconditional = pair[0]
        let conditional = pair[1]
        return conditional + (conditional - unconditional) * MLXArray(guidanceScale)
    }
}

typealias QwenImageEditCFGExecutionMode = DiffusionCFGExecutionMode
typealias QwenImageEditCFGExecution = DiffusionCFGExecution

extension DiffusionCFGExecutionMode {
    static var current: DiffusionCFGExecutionMode {
        current(modelEnvironmentKey: "MERERUN_QWEN_IMAGE_BATCHED_CFG")
    }
}
