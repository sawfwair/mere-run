import Foundation

/// Selects the scheduling defaults qualified with the managed Q4 checkpoints.
enum Q35RuntimeTuning {
    enum Feature: String, CaseIterable {
        case scopedCompilation = "MERERUN_Q35_SCOPED_COMPILE"
        case asynchronousDecode = "MERERUN_Q35_ASYNC_DECODE_BLOCKS"
        case pipelinedFallback = "MERERUN_Q35_MTP_PIPELINED_FALLBACK"
    }

    static func isEnabled(
        _ feature: Feature,
        modelID: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        if let override = environment[feature.rawValue] {
            return override == "1"
        }
        return modelID == Q35Resources.q38TwentySevenB4BitModelId
            || modelID == Q35Resources.ornith35BMLX4BitModelId
    }
}
