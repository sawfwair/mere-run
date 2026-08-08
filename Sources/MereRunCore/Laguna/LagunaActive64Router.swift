import MLX
import MLXFast

/// Exact Laguna XS 256-expert Top-8 routing on Metal.
///
/// Each SIMD group sorts its 32 experts and contributes its local Top-8. The
/// final 64 candidates are sorted by only the first two SIMD groups instead of
/// redundantly running four copies of the same second-stage network. The
/// ordinal mapping gives every Float32 key (including NaNs, infinities, and
/// signed zero) a total order, with the original expert index as the stable
/// tie-breaker. The implementation is the mechanism officially promoted by
/// MLX Fast submission `cc6ddc12` after exact 1,344-step M5 Max validation.
enum LagunaActive64Router {
    private static let ordinalHeader = """
    METAL_FUNC uint mererun_laguna_router_key_ordinal(float key) {
        uint bits = as_type<uint>(key);
        uint magnitude = bits & 0x7FFFFFFFu;
        if (magnitude > 0x7F800000u) {
            return 0xFFFFFFFFu;
        }
        if (magnitude == 0u) {
            return 0x80000000u;
        }
        return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
    }

    METAL_FUNC bool mererun_laguna_router_ordinal_before(
        uint a, uint a_index, uint b, uint b_index) {
        if (a < b) {
            return true;
        }
        if (b < a) {
            return false;
        }
        return a_index < b_index;
    }
    """

    private static func source(normalizing: Bool) -> String {
        let epilogue = normalizing
            ? """
            float my_score2 = lane < 8 ? original_scores[my_index2] : 0.0f;
            float total = 0.0f;
            for (uint i = 0; i < 8; ++i) {
                total = simd_shuffle(my_score2, ushort(i)) + total;
            }
            if (lane < 8) {
                router_indices[row * 8 + lane] = my_index2;
                router_scores[row * 8 + lane] = my_score2 / total;
            }
            """
            : """
            if (lane < 8) {
                router_indices[row * 8 + lane] = my_index2;
                router_scores[row * 8 + lane] = original_scores[my_index2];
            }
            """

        return """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;

        threadgroup uint xchg_ordinals[64];
        threadgroup uint xchg_indices[64];
        threadgroup uint candidate_ordinals[64];
        threadgroup uint candidate_indices[64];
        threadgroup float original_scores[256];

        float x = float(logits[row * 256 + lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        original_scores[lane] = score;
        float key = -(score + float(correction_bias[lane]));
        uint my_ordinal = mererun_laguna_router_key_ordinal(key);
        uint my_index = lane;

        for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                uint other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride));
                uint other_index = simd_shuffle_xor(my_index, ushort(stride));

                bool is_lower = (lane & stride) == 0;
                bool lower_wants_better = (lane & sequence) == 0;
                bool want_better = lower_wants_better == is_lower;
                bool other_before_my = mererun_laguna_router_ordinal_before(
                    other_ordinal, other_index, my_ordinal, my_index);
                bool take_other = want_better ? other_before_my : !other_before_my;
                if (take_other) {
                    my_ordinal = other_ordinal;
                    my_index = other_index;
                }
            }
        }

        uint block = lane >> 5;
        uint within_block = lane & 31;
        bool block_ascending = (block & 1) == 0;
        uint rank_in_block = block_ascending ? within_block : (31 - within_block);
        bool is_local_top8 = block_ascending ? (within_block < 8) : (within_block >= 24);
        if (is_local_top8) {
            candidate_ordinals[block * 8 + rank_in_block] = my_ordinal;
            candidate_indices[block * 8 + rank_in_block] = my_index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_ordinal2 = 0u;
        uint my_index2 = 0u;
        if (lane < 64) {
            my_ordinal2 = candidate_ordinals[lane];
            my_index2 = candidate_indices[lane];
        }

        if (lane < 64) {
            for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
                for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                    uint other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));
                    uint other_index = simd_shuffle_xor(my_index2, ushort(stride));

                    bool is_lower = (lane & stride) == 0;
                    bool lower_wants_better = (lane & sequence) == 0;
                    bool want_better = lower_wants_better == is_lower;
                    bool other_before_my = mererun_laguna_router_ordinal_before(
                        other_ordinal, other_index, my_ordinal2, my_index2);
                    bool take_other = want_better ? other_before_my : !other_before_my;
                    if (take_other) {
                        my_ordinal2 = other_ordinal;
                        my_index2 = other_index;
                    }
                }
            }
        }

        if (lane < 64) {
            xchg_ordinals[lane] = my_ordinal2;
            xchg_indices[lane] = my_index2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 64) {
            uint partner = lane ^ 32u;
            uint other_ordinal = xchg_ordinals[partner];
            uint other_index = xchg_indices[partner];
            bool is_lower = (lane & 32u) == 0;
            bool other_before_my = mererun_laguna_router_ordinal_before(
                other_ordinal, other_index, my_ordinal2, my_index2);
            bool take_other = is_lower ? other_before_my : !other_before_my;
            if (take_other) {
                my_ordinal2 = other_ordinal;
                my_index2 = other_index;
            }

            for (uint stride = 16; stride > 0; stride >>= 1) {
                other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));
                other_index = simd_shuffle_xor(my_index2, ushort(stride));
                is_lower = (lane & stride) == 0;
                other_before_my = mererun_laguna_router_ordinal_before(
                    other_ordinal, other_index, my_ordinal2, my_index2);
                take_other = is_lower ? other_before_my : !other_before_my;
                if (take_other) {
                    my_ordinal2 = other_ordinal;
                    my_index2 = other_index;
                }
            }
        }

        \(epilogue)
        """
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mererun_laguna_router_tournament_ordinal_active64_v1",
        inputNames: ["logits", "correction_bias"],
        outputNames: ["router_indices", "router_scores"],
        source: source(normalizing: false),
        header: ordinalHeader,
        ensureRowContiguous: true
    )

    private static let normalizingKernel = MLXFast.metalKernel(
        name: "mererun_laguna_router_tournament_ordinal_norm_active64_v1",
        inputNames: ["logits", "correction_bias"],
        outputNames: ["router_indices", "router_scores"],
        source: source(normalizing: true),
        header: ordinalHeader,
        ensureRowContiguous: true
    )

    static func routeIfEnabled(
        logits: MLXArray,
        correctionBias: MLXArray,
        expertCount: Int,
        topK: Int,
        normalizing: Bool,
        useCustomKernels: Bool
    ) -> (indices: MLXArray, weights: MLXArray)? {
        guard useCustomKernels,
              LagunaMoEAccelerationPolicy.active64RouterTournamentEnabled,
              Device.defaultDevice().deviceType == .gpu,
              expertCount == 256,
              topK == 8,
              logits.ndim >= 2,
              logits.dim(-1) == expertCount,
              logits.dtype == .float32,
              correctionBias.dtype == .float32,
              correctionBias.size == expertCount else {
            return nil
        }
        return routeForTesting(
            logits: logits,
            correctionBias: correctionBias,
            normalizing: normalizing
        )
    }

    static func routeForTesting(
        logits: MLXArray,
        correctionBias: MLXArray,
        normalizing: Bool
    ) -> (indices: MLXArray, weights: MLXArray) {
        precondition(logits.ndim >= 2)
        precondition(logits.dim(-1) == 256)
        precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
        precondition(correctionBias.dtype == .float32)
        precondition(correctionBias.size == 256)

        let rows = logits.size / 256
        precondition(rows > 0)
        let outputShape = Array(logits.shape.dropLast()) + [8]
        let selectedKernel = normalizing ? normalizingKernel : kernel
        let outputs = selectedKernel(
            [logits, correctionBias],
            grid: (256, rows, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [outputShape, outputShape],
            outputDTypes: [.uint32, .float32]
        )
        return (stopGradient(outputs[0]), outputs[1])
    }
}
