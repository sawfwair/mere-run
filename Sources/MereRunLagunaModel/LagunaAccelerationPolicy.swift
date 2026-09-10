import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package enum LagunaMoEAccelerationPolicy {
    package static func supportsActive64ByDefault(architecture: String?) -> Bool {
        architecture == "applegpu_g16s" || architecture == "applegpu_g17s"
    }

    static let m5MaxDefaultsEnabled: Bool = {
        #if os(macOS)
        Device.defaultDevice().deviceType == .gpu
            && GPU.deviceInfo().architecture == "applegpu_g17s"
        #else
        false
        #endif
    }()

    static let active64DefaultsEnabled: Bool = {
        #if os(macOS)
        Device.defaultDevice().deviceType == .gpu
            && supportsActive64ByDefault(architecture: GPU.deviceInfo().architecture)
        #else
        false
        #endif
    }()

    static let defaultDecodeNVFP4RowsPerSIMDGroup: Int = {
        #if os(macOS)
        defaultDecodeRowsPerSIMDGroup(
            architecture: Device.defaultDevice().deviceType == .gpu
                ? GPU.deviceInfo().architecture
                : nil
        )
        #else
        4
        #endif
    }()

    package static let sortedRoutingEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_SORTED_MOE",
        default: true
    )
    package static let fusedNVFP4MoEEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_NVFP4_MOE",
        default: true
    )
    package static let fastSortedInverseEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FAST_SORTED_INVERSE",
        default: true
    )
    package static let rankedPrefillRouteStagingEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_RANKED_PREFILL_ROUTE_STAGING",
        default: true
    )
    package static let fusedSortedNVFP4MoEEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_SORTED_NVFP4_MOE",
        default: true
    )
    package static let fusedSortedNVFP4DownEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_SORTED_NVFP4_DOWN",
        default: true
    )
    package static let fusedRoutedSharedDownResidualEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_ROUTED_SHARED_DOWN_RESIDUAL",
        default: m5MaxDefaultsEnabled
    )
    package static let prefillExpertPairwiseScaleReuseEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_PREFILL_EXPERT_PAIRWISE_SCALES",
        default: m5MaxDefaultsEnabled
    )
    package static let active64RouterTournamentEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_ACTIVE64_ROUTER",
        default: active64DefaultsEnabled
    )
    package static let decodeNVFP4RowsPerSIMDGroup = decodeRowsPerSIMDGroup(
        ProcessInfo.processInfo.environment[
            "MERERUN_LAGUNA_DECODE_NVFP4_ROWS_PER_SIMDGROUP"
        ],
        default: defaultDecodeNVFP4RowsPerSIMDGroup
    )
    package static let fusedSortedMinimumSequenceLength = 64

    package static func parseBoolean(_ raw: String?, default defaultValue: Bool) -> Bool {
        guard let raw else { return defaultValue }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return defaultValue
        }
    }

    static func booleanEnvironment(
        _ name: String,
        default defaultValue: Bool
    ) -> Bool {
        parseBoolean(ProcessInfo.processInfo.environment[name], default: defaultValue)
    }

    package static func decodeRowsPerSIMDGroup(
        _ raw: String?,
        default defaultValue: Int
    ) -> Int {
        guard let raw,
              let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              value == 1 || value == 2 || value == 4 else {
            return defaultValue
        }
        return value
    }

    package static func defaultDecodeRowsPerSIMDGroup(architecture: String?) -> Int {
        architecture == "applegpu_g17s" ? 2 : 4
    }

    package static func decodeRowsPerSIMDGroup(
        hiddenSize: Int,
        intermediateSize: Int,
        topK: Int,
        xsCandidate: Int
    ) -> Int {
        guard hiddenSize == 2_048,
              intermediateSize == 512,
              topK == 8 else {
            return 4
        }
        return xsCandidate
    }
}

/// Proves that every adjacent pair in an NVFP4 group-16 scale plane is
/// byte-identical except for explicitly preserved pairs. The Laguna XS expert
/// tensors shipped by Poolside were produced by a simdgroup-wide quantizer:
/// the two group-16 halves of each 32-weight span therefore share one scale,
/// while the first pair may carry the quantizer's duplicated-write exception.
///
/// This is a fail-closed, input-independent certificate over the weights that
/// were actually loaded. A false result keeps the stock prefill loader. A true
/// result permits the M5 kernel to load the even scale byte once and broadcast
/// it to the adjacent SIMD lane without changing a single dequantized value.
package func lagunaNVFP4AdjacentScalePairsCertified(
    _ scales: MLXArray,
    allowedFlatPairs: Set<Int> = [0]
) -> Bool {
    let pairCount = scales.size / 2
    guard scales.dtype == .uint8,
          scales.ndim >= 1,
          scales.size.isMultiple(of: 2),
          scales.dim(-1).isMultiple(of: 2),
          allowedFlatPairs.allSatisfy({ $0 >= 0 && $0 < pairCount }) else {
        return false
    }

    let pairs = contiguous(scales).reshaped([pairCount, 2])
    let mismatch = (pairs[0..., 0] .!= pairs[0..., 1]).asType(.int32)
    var violations = mismatch.sum()
    for index in allowedFlatPairs {
        violations = violations - mismatch[index]
    }
    return violations.item(Int32.self) == 0
}

package enum LagunaGraphAccelerationPolicy {
    package static func supportsNativeAffineByDefault(architecture: String?) -> Bool {
        architecture == "applegpu_g16s" || architecture == "applegpu_g17s"
    }

    static let m5MaxDefaultsEnabled: Bool = {
        #if os(macOS)
        Device.defaultDevice().deviceType == .gpu
            && GPU.deviceInfo().architecture == "applegpu_g17s"
        #else
        false
        #endif
    }()

    static let nativeAffineQKVDefaultsEnabled: Bool = {
        #if os(macOS)
        Device.defaultDevice().deviceType == .gpu
            && supportsNativeAffineByDefault(architecture: GPU.deviceInfo().architecture)
        #else
        false
        #endif
    }()

    package static let sharedAttentionMasksEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_SHARED_ATTENTION_MASKS"],
        default: true
    )
    package static let prefillAsyncLadderStride = parseLadderStride(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_PREFILL_ASYNC_LADDER"],
        default: 8
    )
    package static let prefillFusedResidualRMSNormEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_PREFILL_FUSED_RESIDUAL_RMSNORM"],
        default: m5MaxDefaultsEnabled
    )
    package static let prefillQKNormRoPEEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_PREFILL_QK_NORM_ROPE"],
        default: m5MaxDefaultsEnabled
    )
    package static let terminalPrefillRowEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_TERMINAL_PREFILL_ROW"],
        default: m5MaxDefaultsEnabled
    )
    package static let terminalPrefillProjectionBanksEnabled = parseBoolean(
        ProcessInfo.processInfo.environment[
            "MERERUN_LAGUNA_TERMINAL_PREFILL_PROJECTION_BANKS"
        ],
        default: m5MaxDefaultsEnabled
    )
    package static let nativeAffineQKVEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_QKV"],
        default: nativeAffineQKVDefaultsEnabled
    )
    package static let nativeAffineQKVLayerCount = nativeAffineQKVEnabled
        ? parseLayerCount(
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_QKV_LAYERS"],
            default: 40
        )
        : 0
    package static let nativeAffineOProjEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_OPROJ"],
        default: nativeAffineQKVDefaultsEnabled
    )
    package static let nativeAffineOProjLayerCount = nativeAffineOProjEnabled
        ? parseLayerCount(
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_OPROJ_LAYERS"],
            default: 40
        )
        : 0
    package static let nativeAffineGProjEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_GPROJ"],
        default: nativeAffineQKVDefaultsEnabled
    )
    package static let nativeAffineGProjLayerCount = nativeAffineGProjEnabled
        ? parseLayerCount(
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_GPROJ_LAYERS"],
            default: 40
        )
        : 0
    package static let nativeAffineGProjFoldEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_GPROJ_FOLD"],
        default: m5MaxDefaultsEnabled
    )
    package static let fusedNormAffineQKVEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_FUSED_NORM_AFFINE_QKV"],
        default: m5MaxDefaultsEnabled
    )
    package static let fusedGatedAffineOProjEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_FUSED_GATED_AFFINE_OPROJ"],
        default: true
    )
    package static let decodeAsyncStageEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_DECODE_ASYNC_STAGE"],
        default: nativeAffineQKVDefaultsEnabled
    )
    package static let decodeAsyncLayerIndices: Set<Int> = [0, 1, 7, 15, 23, 31, 39]

    package static func parseBoolean(_ raw: String?, default defaultValue: Bool) -> Bool {
        LagunaMoEAccelerationPolicy.parseBoolean(raw, default: defaultValue)
    }

    package static func parseLadderStride(_ raw: String?, default defaultValue: Int) -> Int {
        guard let raw else { return defaultValue }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.isEmpty || normalized == "off" || normalized == "0" {
            return 0
        }
        guard let value = Int(normalized), (1...40).contains(value) else {
            return defaultValue
        }
        return value
    }

    package static func parseLayerCount(_ raw: String?, default defaultValue: Int) -> Int {
        guard let raw,
              let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return defaultValue
        }
        return min(max(value, 0), 40)
    }

    package static func usesNativeAffineQKV(layerIndex: Int) -> Bool {
        layerIndex >= 0 && layerIndex < nativeAffineQKVLayerCount
    }

    package static func usesNativeAffineOProj(layerIndex: Int) -> Bool {
        layerIndex >= 0 && layerIndex < nativeAffineOProjLayerCount
    }

    package static func usesNativeAffineGProj(layerIndex: Int) -> Bool {
        layerIndex >= 0 && layerIndex < nativeAffineGProjLayerCount
    }
}
