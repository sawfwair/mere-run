#if os(macOS) || os(iOS)
import MLX
import MLXFast

/// Proposal-only shortlist and exact rerank for the promoted Qwen3.8 MTP head.
/// Target verification remains the sole authority for emitted tokens.
enum Q35DraftRerank {
    static let realCount = 98_330
    static let candidateCount = 32

    private static let threadGroupSize = 256
    private static let tileCount = 64
    private static let partialCandidateCount = tileCount * candidateCount

    // Proposal-only affine-Q2 readout. Its wider per-lane reduction is safe:
    // exact Q4 reranking and target verification decide every emitted token.
    private static let coarseReadoutKernel = MLXFast.metalKernel(
        name: "mere_q38_mtp_affine2_g64_readout",
        inputNames: ["weight", "scales", "biases", "hidden"],
        outputNames: ["logits"],
        source: """
            constexpr int ROWS_PER_SIMD = 4;
            constexpr int VALUES_PER_THREAD = 32;
            constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
            constexpr int BYTES_PER_LANE = 8;

            const int input_size = hidden_shape[hidden_ndim - 1];
            const int output_size = weight_shape[0];
            const int weight_bytes_per_row = input_size / 4;
            const int groups_per_row = input_size / 64;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int output_row = int(threadgroup_position_in_grid.y) * 8
                + int(simd_group) * ROWS_PER_SIMD;

            float results[ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int k = 0; k < input_size; k += BLOCK_SIZE) {
                ulong packed[ROWS_PER_SIMD];
                float local_scales[ROWS_PER_SIMD];
                float local_biases[ROWS_PER_SIMD];
                for (int row_offset = 0; row_offset < ROWS_PER_SIMD; ++row_offset) {
                    const int row = output_row + row_offset;
                    const device uint8_t* bytes =
                        reinterpret_cast<const device uint8_t*>(weight)
                        + row * weight_bytes_per_row + k / 4 + lane * BYTES_PER_LANE;
                    packed[row_offset] =
                        *reinterpret_cast<const device ulong*>(bytes);
                    const int group_index = row * groups_per_row + k / 64
                        + (int(lane) * VALUES_PER_THREAD) / 64;
                    local_scales[row_offset] = scales[group_index];
                    local_biases[row_offset] = biases[group_index];
                }

                float activations[VALUES_PER_THREAD];
                const device bfloat16_t* input = hidden + k + lane * VALUES_PER_THREAD;
                float activation_sum = 0.0f;
                for (int index = 0; index < VALUES_PER_THREAD; index += 4) {
                    activations[index] = input[index];
                    activations[index + 1] = input[index + 1];
                    activations[index + 2] = input[index + 2];
                    activations[index + 3] = input[index + 3];
                    activation_sum += input[index] + input[index + 1]
                        + input[index + 2] + input[index + 3];
                }

                for (int row_offset = 0; row_offset < ROWS_PER_SIMD; ++row_offset) {
                    float accumulation = 0.0f;
                    #pragma unroll
                    for (int index = 0; index < VALUES_PER_THREAD; ++index) {
                        accumulation += activations[index]
                            * float((packed[row_offset] >> (2 * index)) & 0x03ul);
                    }
                    results[row_offset] += local_scales[row_offset] * accumulation
                        + activation_sum * local_biases[row_offset];
                }
            }

            for (int row_offset = 0; row_offset < ROWS_PER_SIMD; ++row_offset) {
                float reduced = simd_sum(results[row_offset]);
                if (lane == 0) {
                    logits[output_row + row_offset] = bfloat16_t(reduced);
                }
            }
        """,
        header: "",
        ensureRowContiguous: true
    )

    private static let top32Header = """
        inline uint mere_q38_top32_ordinal(float value) {
            if (isnan(value))  { return 0xFFFFFFFFu; }
            if (value == 0.0f) { return 0x80000000u; }
            uint bits = as_type<uint>(value);
            return (bits & 0x80000000u) ? (~bits) : (bits | 0x80000000u);
        }
        """

    // MIT-derived from Layr-Labs/qwen-3.8-mtp-challenge@0863b06. See
    // THIRD_PARTY_NOTICES.md. This replaces a full 98,330-element GPU sort.
    private static let top32PartialKernel = MLXFast.metalKernel(
        name: "mere_q38_mtp_top32_partial",
        inputNames: ["logits"],
        outputNames: ["candidate_ordinals", "candidate_indices"],
        source: """
            constexpr uint REAL_COUNT = 98330;
            constexpr uint TG_SIZE = 256;
            constexpr uint STRIDE = 16384;
            constexpr uint PER_THREAD = 7;
            constexpr uint TOPK = 32;
            constexpr uint SIMD_SIZE = 32;
            constexpr uint NSIMD = 8;
            constexpr uint PER_REDUCER = 8;

            uint tile = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;

            uint ordinals[PER_THREAD];
            uint indices[PER_THREAD];
            for (uint slot = 0; slot < PER_THREAD; ++slot) {
                ordinals[slot] = 0u;
                indices[slot] = 0u;
            }
            uint count = 0;
            for (uint index = tile * TG_SIZE + tid;
                 index < REAL_COUNT;
                 index += STRIDE) {
                ordinals[count] = mere_q38_top32_ordinal(float(logits[index]));
                indices[count] = index;
                count++;
            }

            threadgroup uint simd_ordinals[NSIMD * TOPK];
            threadgroup uint simd_indices[NSIMD * TOPK];

            uint taken = 0u;
            for (uint rank = 0; rank < TOPK; ++rank) {
                uint best_ordinal = 0u;
                uint best_index = 0u;
                uint best_slot = 0xFFFFFFFFu;
                for (uint slot = 0; slot < PER_THREAD; ++slot) {
                    if ((taken & (1u << slot)) != 0u) { continue; }
                    if (ordinals[slot] > best_ordinal
                        || (ordinals[slot] == best_ordinal
                            && indices[slot] > best_index)) {
                        best_ordinal = ordinals[slot];
                        best_index = indices[slot];
                        best_slot = slot;
                    }
                }
                uint simd_ordinal = simd_max(best_ordinal);
                uint simd_index = simd_max(
                    (best_ordinal == simd_ordinal) ? best_index : 0u);
                if (best_slot != 0xFFFFFFFFu
                    && best_ordinal == simd_ordinal
                    && best_index == simd_index) {
                    taken |= (1u << best_slot);
                }
                if (lane == 0) {
                    simd_ordinals[simd_group * TOPK + rank] = simd_ordinal;
                    simd_indices[simd_group * TOPK + rank] = simd_index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_group == 0) {
                uint reducer_ordinals[PER_REDUCER];
                uint reducer_indices[PER_REDUCER];
                for (uint slot = 0; slot < PER_REDUCER; ++slot) {
                    uint position = slot * SIMD_SIZE + lane;
                    reducer_ordinals[slot] = simd_ordinals[position];
                    reducer_indices[slot] = simd_indices[position];
                }
                uint reducer_taken = 0u;
                for (uint rank = 0; rank < TOPK; ++rank) {
                    uint best_ordinal = 0u;
                    uint best_index = 0u;
                    uint best_slot = 0xFFFFFFFFu;
                    for (uint slot = 0; slot < PER_REDUCER; ++slot) {
                        if ((reducer_taken & (1u << slot)) != 0u) { continue; }
                        if (reducer_ordinals[slot] > best_ordinal
                            || (reducer_ordinals[slot] == best_ordinal
                                && reducer_indices[slot] > best_index)) {
                            best_ordinal = reducer_ordinals[slot];
                            best_index = reducer_indices[slot];
                            best_slot = slot;
                        }
                    }
                    uint simd_ordinal = simd_max(best_ordinal);
                    uint simd_index = simd_max(
                        (best_ordinal == simd_ordinal) ? best_index : 0u);
                    if (best_slot != 0xFFFFFFFFu
                        && best_ordinal == simd_ordinal
                        && best_index == simd_index) {
                        reducer_taken |= (1u << best_slot);
                    }
                    if (lane == 0) {
                        candidate_ordinals[tile * TOPK + rank] = simd_ordinal;
                        candidate_indices[tile * TOPK + rank] = simd_index;
                    }
                }
            }
        """,
        header: top32Header,
        ensureRowContiguous: false
    )

    private static let top32FinalizeKernel = MLXFast.metalKernel(
        name: "mere_q38_mtp_top32_finalize",
        inputNames: ["candidate_ordinals", "candidate_indices"],
        outputNames: ["token_ids"],
        source: """
            constexpr uint TG_SIZE = 256;
            constexpr uint PER_THREAD = 8;
            constexpr uint TOPK = 32;
            constexpr uint SIMD_SIZE = 32;
            constexpr uint NSIMD = 8;
            constexpr uint PER_REDUCER = 8;

            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;

            uint ordinals[PER_THREAD];
            uint indices[PER_THREAD];
            for (uint slot = 0; slot < PER_THREAD; ++slot) {
                uint position = slot * TG_SIZE + tid;
                ordinals[slot] = candidate_ordinals[position];
                indices[slot] = candidate_indices[position];
            }

            threadgroup uint simd_ordinals[NSIMD * TOPK];
            threadgroup uint simd_indices[NSIMD * TOPK];

            uint taken = 0u;
            for (uint rank = 0; rank < TOPK; ++rank) {
                uint best_ordinal = 0u;
                uint best_index = 0u;
                uint best_slot = 0xFFFFFFFFu;
                for (uint slot = 0; slot < PER_THREAD; ++slot) {
                    if ((taken & (1u << slot)) != 0u) { continue; }
                    if (ordinals[slot] > best_ordinal
                        || (ordinals[slot] == best_ordinal
                            && indices[slot] > best_index)) {
                        best_ordinal = ordinals[slot];
                        best_index = indices[slot];
                        best_slot = slot;
                    }
                }
                uint simd_ordinal = simd_max(best_ordinal);
                uint simd_index = simd_max(
                    (best_ordinal == simd_ordinal) ? best_index : 0u);
                if (best_slot != 0xFFFFFFFFu
                    && best_ordinal == simd_ordinal
                    && best_index == simd_index) {
                    taken |= (1u << best_slot);
                }
                if (lane == 0) {
                    simd_ordinals[simd_group * TOPK + rank] = simd_ordinal;
                    simd_indices[simd_group * TOPK + rank] = simd_index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_group == 0) {
                uint reducer_ordinals[PER_REDUCER];
                uint reducer_indices[PER_REDUCER];
                for (uint slot = 0; slot < PER_REDUCER; ++slot) {
                    uint position = slot * SIMD_SIZE + lane;
                    reducer_ordinals[slot] = simd_ordinals[position];
                    reducer_indices[slot] = simd_indices[position];
                }
                uint reducer_taken = 0u;
                for (uint rank = 0; rank < TOPK; ++rank) {
                    uint best_ordinal = 0u;
                    uint best_index = 0u;
                    uint best_slot = 0xFFFFFFFFu;
                    for (uint slot = 0; slot < PER_REDUCER; ++slot) {
                        if ((reducer_taken & (1u << slot)) != 0u) { continue; }
                        if (reducer_ordinals[slot] > best_ordinal
                            || (reducer_ordinals[slot] == best_ordinal
                                && reducer_indices[slot] > best_index)) {
                            best_ordinal = reducer_ordinals[slot];
                            best_index = reducer_indices[slot];
                            best_slot = slot;
                        }
                    }
                    uint simd_ordinal = simd_max(best_ordinal);
                    uint simd_index = simd_max(
                        (best_ordinal == simd_ordinal) ? best_index : 0u);
                    if (best_slot != 0xFFFFFFFFu
                        && best_ordinal == simd_ordinal
                        && best_index == simd_index) {
                        reducer_taken |= (1u << best_slot);
                    }
                    if (lane == 0) {
                        token_ids[TOPK - 1u - rank] = simd_index;
                    }
                }
            }
        """,
        header: "",
        ensureRowContiguous: false
    )

    private static let exactRerankKernel = MLXFast.metalKernel(
        name: "mere_q38_mtp_exact_q4_rerank",
        inputNames: ["hidden", "candidate_ids", "weight", "scales", "biases"],
        outputNames: ["token_id"],
        source: """
            constexpr uint TOPK = 32;
            constexpr uint HIDDEN = 5120;
            constexpr uint PACKED_WORDS = 640;
            constexpr uint GROUPS = 80;
            constexpr uint VALUES_PER_LANE = 16;
            constexpr uint BLOCK = 512;

            uint lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint candidate_base = simd_group * 4;
            float results[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint k = 0; k < HIDDEN; k += BLOCK) {
                float activations[VALUES_PER_LANE];
                uint activation_base = k + lane * VALUES_PER_LANE;
                float activation_sum = 0.0f;
                for (uint index = 0; index < VALUES_PER_LANE; index += 4) {
                    activation_sum += hidden[activation_base + index]
                        + hidden[activation_base + index + 1]
                        + hidden[activation_base + index + 2]
                        + hidden[activation_base + index + 3];
                    activations[index] = hidden[activation_base + index];
                    activations[index + 1] = hidden[activation_base + index + 1] / 16.0f;
                    activations[index + 2] = hidden[activation_base + index + 2] / 256.0f;
                    activations[index + 3] = hidden[activation_base + index + 3] / 4096.0f;
                }
                for (uint row_index = 0; row_index < 4; ++row_index) {
                    uint row = uint(candidate_ids[candidate_base + row_index]);
                    uint word_base = row * PACKED_WORDS + k / 8 + lane * 2;
                    uint packed0 = weight[word_base];
                    uint packed1 = weight[word_base + 1];
                    ushort packed[4] = {
                        ushort(packed0 & 0xffffu), ushort(packed0 >> 16),
                        ushort(packed1 & 0xffffu), ushort(packed1 >> 16)
                    };
                    uint group_index = row * GROUPS + k / 64 + lane / 4;
                    float scale = scales[group_index];
                    float bias = biases[group_index];
                    float accumulation = 0.0f;
                    for (uint index = 0; index < 4; ++index) {
                        accumulation +=
                            activations[4 * index] * (packed[index] & 0x000f)
                            + activations[4 * index + 1] * (packed[index] & 0x00f0)
                            + activations[4 * index + 2] * (packed[index] & 0x0f00)
                            + activations[4 * index + 3] * (packed[index] & 0xf000);
                    }
                    results[row_index] += scale * accumulation + activation_sum * bias;
                }
            }

            threadgroup float exact_scores[TOPK];
            for (uint row_index = 0; row_index < 4; ++row_index) {
                float reduced = simd_sum(results[row_index]);
                if (lane == 0) {
                    exact_scores[candidate_base + row_index] = float(bfloat16_t(reduced));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_group == 0) {
                float best_value = exact_scores[lane];
                uint best_id = uint(candidate_ids[lane]);
                for (uint offset = 16; offset > 0; offset >>= 1) {
                    float other_value = simd_shuffle_down(best_value, offset);
                    uint other_id = simd_shuffle_down(best_id, offset);
                    bool other_nan = isnan(other_value);
                    bool best_nan = isnan(best_value);
                    bool take = other_nan != best_nan
                        ? !other_nan
                        : (other_value > best_value
                            || (other_value == best_value && other_id < best_id));
                    if (lane < offset && take) {
                        best_value = other_value;
                        best_id = other_id;
                    }
                }
                if (lane == 0) {
                    token_id[0] = int(best_id < PREFIX_COUNT
                        ? best_id
                        : best_id + CONTROL_OFFSET);
                }
            }
        """,
        header: "",
        ensureRowContiguous: false
    )

    static func topCandidates(_ logits: MLXArray) -> MLXArray {
        precondition(logits.size == realCount)
        let partial = top32PartialKernel(
            [logits.reshaped([realCount])],
            grid: (tileCount * threadGroupSize, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [[partialCandidateCount], [partialCandidateCount]],
            outputDTypes: [.uint32, .uint32]
        )
        return top32FinalizeKernel(
            [partial[0], partial[1]],
            grid: (threadGroupSize, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [[candidateCount]],
            outputDTypes: [.uint32]
        )[0]
    }

    static func coarseLogits(
        hidden: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray
    ) -> MLXArray? {
        guard hidden.shape == [1, 1, 5_120],
              hidden.dtype == .bfloat16,
              weight.shape == [98_336, 320],
              weight.dtype == .uint32,
              scales.shape == [98_336, 80],
              scales.dtype == .bfloat16,
              biases.shape == scales.shape,
              biases.dtype == .bfloat16 else {
            return nil
        }
        return coarseReadoutKernel(
            [weight, scales, biases, hidden],
            grid: (32, (98_336 / 8) * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[1, 1, 98_336]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    static func token(
        hidden: MLXArray,
        coarseLogits: MLXArray,
        exactWeight: MLXArray,
        exactScales: MLXArray,
        exactBiases: MLXArray,
        prefixCount: Int,
        controlOffset: Int
    ) -> MLXArray {
        let candidates = topCandidates(coarseLogits)
        return exactRerankKernel(
            [hidden.reshaped([5_120]), candidates, exactWeight, exactScales, exactBiases],
            template: [
                ("PREFIX_COUNT", prefixCount),
                ("CONTROL_OFFSET", controlOffset),
            ],
            grid: (threadGroupSize, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [[1, 1]],
            outputDTypes: [.int32]
        )[0]
    }
}
#endif
