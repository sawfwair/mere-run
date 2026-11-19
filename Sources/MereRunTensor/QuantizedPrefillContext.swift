import MLX

/// Projection experiments apply only within the owning model's prefill scope.
/// Decode and independent draft models keep their existing kernels.
package enum QuantizedPrefillMode: String, Sendable, Hashable {
    case reference
    case all
    case gateUp = "gate-up"
    case packedSIMD = "packed-simd"
    case packedTensor = "packed-tensor"
    case gdnTensor = "gdn-tensor"
    case gdnLateTensor = "gdn-late-tensor"

    case gdnRepaired = "gdn-repaired"

    package var usesLateGDNTensor: Bool { self == .gdnLateTensor || self == .gdnRepaired }

    package var materializesWeights: Bool { self == .all || self == .gateUp }
}

package enum QuantizedPrefillContext {
    @TaskLocal package static var mode: QuantizedPrefillMode = .reference

    package static func projection(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) -> MLXArray? {
        guard Self.mode != .reference, Device.defaultDevice().deviceType == .gpu,
              x.dtype == .bfloat16, x.size / x.dim(-1) >= 512,
              groupSize == 64, bits == 4, mode == .affine,
              let biases else { return nil }
        if Self.mode == .gateUp, x.dim(-1) != 5_120 || weight.dim(0) != 17_408 { return nil }
        if Self.mode == .gdnTensor || Self.mode.usesLateGDNTensor {
            #if os(macOS)
            // Fused QKV and output gate of the dense 27B GDN. Stage one
            // complete affine group for each pair of threadgroup barriers.
            guard x.dim(-1) == 5_120, weight.dim(0) == 16_384 else { return nil }
            let engine: PackedAffineQ4Prefill.Engine = switch Self.mode {
            case .gdnTensor: .gdnTensor
            case .gdnRepaired where M5AffineQ4Verification.isSupported: .gdnRepaired
            default: .simd
            }
            return PackedAffineQ4Prefill.matmul(x, weight: weight, scales: scales, biases: biases, engine: engine)
            #else
            return nil
            #endif
        }
        if Self.mode == .packedSIMD || Self.mode == .packedTensor {
            // Keep small projections on native QMM, including its split-K
            // reduction. At least 512 rows and 1024 columns avoid that split.
            guard weight.dim(0) >= 1024 else { return nil }
            #if os(macOS)
            return PackedAffineQ4Prefill.matmul(x, weight: weight, scales: scales, biases: biases,
                engine: Self.mode == .packedSIMD ? .simd : .tensor)
            #else
            return nil
            #endif
        }
        // Native QMM's block loader dequantizes in FP32 before its BF16
        // staging cast. A direct BF16 dequantization adds an intermediate
        // BF16 rounding and is not the same weight representation.
        let decoded = MLX.dequantized(
            weight, scales: scales.asType(.float32), biases: biases.asType(.float32),
            groupSize: groupSize, bits: bits
        ).asType(.bfloat16)
        return MLX.matmul(x, decoded.T)
    }
}
