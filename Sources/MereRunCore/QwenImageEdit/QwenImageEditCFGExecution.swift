import Foundation
import MLX

enum QwenImageEditCFGExecutionMode: String, Sendable {
    case automatic
    case batched
    case serial

    static func parse(_ raw: String?) -> QwenImageEditCFGExecutionMode {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on", "batch", "batched":
            return .batched
        case "0", "false", "no", "off", "serial":
            return .serial
        default:
            return .automatic
        }
    }

    static var current: QwenImageEditCFGExecutionMode {
        parse(ProcessInfo.processInfo.environment["MERERUN_QWEN_IMAGE_BATCHED_CFG"])
    }
}

enum QwenImageEditCFGExecution {
    private static let gibibyte = 1_073_741_824.0

    static func shouldBatch(
        mode: QwenImageEditCFGExecutionMode,
        width: Int,
        height: Int,
        physicalMemoryBytes: UInt64,
        activeMemoryBytes: Int,
        cacheMemoryBytes: Int,
        isUnifiedMemory: Bool
    ) -> Bool {
        switch mode {
        case .batched:
            return true
        case .serial:
            return false
        case .automatic:
            guard isUnifiedMemory, Double(physicalMemoryBytes) >= 24 * gibibyte else {
                return false
            }
            let allocated = Double(max(0, activeMemoryBytes) + max(0, cacheMemoryBytes))
            let headroom = max(0, Double(physicalMemoryBytes) - allocated)
            let pixelReserve = Double(max(1, width)) * Double(max(1, height)) * 2_048
            let requiredHeadroom = (4 * gibibyte) + pixelReserve
            return headroom >= requiredHeadroom
        }
    }

    static func duplicateBatch(_ array: MLXArray) -> MLXArray {
        precondition(array.dim(0) == 1, "CFG batching expects a single source sample.")
        var shape = array.shape
        shape[0] = 2
        return MLX.broadcast(array, to: shape)
    }

    static func combinePredictions(_ predictions: MLXArray, guidanceScale: Float) -> MLXArray {
        precondition(predictions.dim(0) == 2, "CFG predictions must be ordered [unconditional, conditional].")
        let unconditional = predictions[0..<1, 0..., 0..., 0...]
        let conditional = predictions[1..<2, 0..., 0..., 0...]
        return unconditional + (conditional - unconditional) * MLXArray(guidanceScale)
    }
}
