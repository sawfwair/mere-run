import Foundation
import MereRunTensor

/// Experimental prefill execution. The task-local override also permits
/// checkpoint comparisons without changing another request's policy.
package enum Q35PrefillProjectionPolicy {
    package typealias Mode = QuantizedPrefillMode
    @TaskLocal package static var modeOverride: Mode?
    package static var mode: Mode { modeOverride ?? configuredMode ?? (QwenDFlashDefaults.active ? .gdnRepaired : .reference) }

    private static let configuredMode: Mode? = {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_Q38_PREFILL_PACKED"] != nil || env["MERERUN_Q38_PREFILL_MATERIALIZE"] != nil else { return nil }
        switch env["MERERUN_Q38_PREFILL_PACKED"] {
        case "simd": return .packedSIMD
        case "tensor": return .packedTensor
        case "gdn-tensor": return .gdnTensor
        case "gdn-repaired": return .gdnRepaired
        case "gdn-late-tensor": return .gdnLateTensor
        default:
            switch env["MERERUN_Q38_PREFILL_MATERIALIZE"] {
            case "1", "all": return .all
            case "gate-up": return .gateUp
            default: return .reference
            }
        }
    }()

    package static func eligible(_ config: Q35Config) -> Bool {
        let text = config.textConfig
        return mode != .reference && !text.usesMoE && text.hiddenSize == 5_120
            && text.intermediateSize == 17_408 && text.numHiddenLayers == 64
            && config.quantization?.bits == 4 && config.quantization?.groupSize == 64
    }
}
