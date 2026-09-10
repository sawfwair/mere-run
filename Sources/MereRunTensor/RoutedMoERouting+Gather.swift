import Foundation
import MLX
import MLXFast

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
extension RoutedMoERouting {
    static let fusedGatherNVFP4SwiGLUKernel = MLXFast.metalKernel(
        name: lagunaNVFP4QDotFastEnabled
            ? "mere_routed_moe_gather_nvfp4_swiglu_qf_v1"
            : "mere_routed_moe_gather_nvfp4_swiglu",
        inputNames: [
            "x",
            "gate_weight",
            "gate_scales",
            "up_weight",
            "up_scales",
            "expert_indices",
        ],
        outputNames: ["output"],
        source: """
            constexpr int packs_per_thread = 2;
            constexpr int results_per_simdgroup = RESULTS_PER_SIMDGROUP;
            constexpr int outputs_per_threadgroup =
                2 * results_per_simdgroup;
            constexpr int pack_factor = get_pack_factor<32, BITS>();
            constexpr int bytes_per_pack = get_bytes_per_pack<32>();
            constexpr int values_per_thread = pack_factor * packs_per_thread;
            constexpr int block_size = values_per_thread * SIMD_SIZE;
            constexpr int scale_step_per_thread =
                GROUP_SIZE / values_per_thread;
            constexpr int packed_input =
                INPUT_DIMENSIONS * bytes_per_pack / pack_factor;
            constexpr int scale_input = INPUT_DIMENSIONS / GROUP_SIZE;
            constexpr size_t weight_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * packed_input;
            constexpr size_t scale_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * scale_input;

            threadgroup DataT projection_values[
                2 * outputs_per_threadgroup
            ];

            const uint3 tile = threadgroup_position_in_grid;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint simd_lane = thread_index_in_simdgroup;
            const uint thread_index = thread_index_in_threadgroup;
            const bool gate_projection = simd_group < 2;
            const uint local_simd_group = simd_group % 2;
            const uint route = tile.z;
            if (route >= ROUTE_COUNT) {
                return;
            }

            const uint expert = uint(expert_indices[route]);
            const int output_row =
                int(tile.y) * outputs_per_threadgroup
                    + int(local_simd_group) * results_per_simdgroup;
            const device uint8_t* selected_weight =
                reinterpret_cast<const device uint8_t*>(
                    gate_projection ? gate_weight : up_weight);
            const device uint8_t* selected_scales =
                gate_projection ? gate_scales : up_scales;
            selected_weight +=
                size_t(expert) * weight_expert_stride
                    + size_t(output_row) * packed_input
                    + simd_lane * packs_per_thread * bytes_per_pack;
            selected_scales +=
                size_t(expert) * scale_expert_stride
                    + size_t(output_row) * scale_input
                    + simd_lane / scale_step_per_thread;
            const device DataT* input =
                x
                    + size_t(route / TOP_K) * INPUT_DIMENSIONS
                    + simd_lane * values_per_thread;

            float input_values[values_per_thread];
            float results[results_per_simdgroup] = {0};
            for (int k = 0; k < INPUT_DIMENSIONS; k += block_size) {
                load_vector<
                    DataT,
                    float,
                    values_per_thread>(input, input_values);

                STEEL_PRAGMA_UNROLL
                for (int row = 0; row < results_per_simdgroup; ++row) {
                    const device uint8_t* row_weight =
                        selected_weight + size_t(row) * packed_input;
                    const device uint8_t* row_scales =
                        selected_scales + size_t(row) * scale_input;
                    results[row] += mere_laguna_nvfp4_qdot_16(
                        row_weight,
                        input_values,
                        mere_laguna_nvfp4_scale(row_scales[0]));
                }

                selected_weight +=
                    block_size * bytes_per_pack / pack_factor;
                selected_scales += block_size / GROUP_SIZE;
                input += block_size;
            }

            STEEL_PRAGMA_UNROLL
            for (int row = 0; row < results_per_simdgroup; ++row) {
                results[row] = simd_sum(results[row]);
                if (simd_lane == 0) {
                    const uint projection_offset =
                        gate_projection ? 0 : outputs_per_threadgroup;
                    projection_values[
                        projection_offset
                            + local_simd_group * results_per_simdgroup
                            + row
                    ] = DataT(results[row]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (thread_index < outputs_per_threadgroup) {
                const DataT gate_value = projection_values[thread_index];
                const DataT up_value =
                    projection_values[
                        outputs_per_threadgroup + thread_index
                    ];
                const DataT sigmoid_base =
                    DataT(1)
                        / (
                            DataT(1)
                                + metal::exp(metal::abs(gate_value))
                        );
                const DataT sigmoid_value =
                    gate_value < DataT(0)
                        ? sigmoid_base
                        : DataT(1) - sigmoid_base;
                output[
                    size_t(route) * OUTPUT_DIMENSIONS
                        + size_t(tile.y) * outputs_per_threadgroup
                        + thread_index
                ] = DataT(DataT(gate_value * sigmoid_value) * up_value);
            }
        """,
        header: "// MLX_INCLUDE_FP_QUANTIZED_HEADERS\n" + lagunaXSDownHeader,
        ensureRowContiguous: true
    )

    static let fusedGatherAffine8SwiGLUKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_gather_affine8_swiglu",
        inputNames: [
            "x",
            "gate_weight",
            "gate_scales",
            "gate_biases",
            "up_weight",
            "up_scales",
            "up_biases",
            "expert_indices",
        ],
        outputNames: ["output"],
        source: """
            constexpr int packs_per_thread = BITS <= 2 ? 1 : 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int outputs_per_threadgroup = 8;
            constexpr int pack_factor = get_pack_factor<BITS, 32>();
            constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
            constexpr int values_per_thread = pack_factor * packs_per_thread;
            constexpr int block_size = values_per_thread * SIMD_SIZE;
            constexpr int scale_step_per_thread =
                GROUP_SIZE / values_per_thread;
            constexpr int packed_input =
                INPUT_DIMENSIONS * bytes_per_pack / pack_factor;
            constexpr int scale_input = INPUT_DIMENSIONS / GROUP_SIZE;
            constexpr size_t weight_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * packed_input;
            constexpr size_t scale_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * scale_input;

            threadgroup DataT projection_values[
                2 * outputs_per_threadgroup
            ];

            const uint3 tile = threadgroup_position_in_grid;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint simd_lane = thread_index_in_simdgroup;
            const uint thread_index = thread_index_in_threadgroup;
            const bool gate_projection = simd_group < 2;
            const uint local_simd_group = simd_group % 2;
            const uint route = tile.z;
            if (route >= ROUTE_COUNT) {
                return;
            }

            const uint expert = uint(expert_indices[route]);
            const int output_row =
                int(tile.y) * outputs_per_threadgroup
                    + int(local_simd_group) * results_per_simdgroup;
            const device uint8_t* selected_weight =
                reinterpret_cast<const device uint8_t*>(
                    gate_projection ? gate_weight : up_weight);
            const device DataT* selected_scales =
                gate_projection ? gate_scales : up_scales;
            const device DataT* selected_biases =
                gate_projection ? gate_biases : up_biases;
            selected_weight +=
                size_t(expert) * weight_expert_stride
                    + size_t(output_row) * packed_input
                    + simd_lane * packs_per_thread * bytes_per_pack;
            selected_scales +=
                size_t(expert) * scale_expert_stride
                    + size_t(output_row) * scale_input
                    + simd_lane / scale_step_per_thread;
            selected_biases +=
                size_t(expert) * scale_expert_stride
                    + size_t(output_row) * scale_input
                    + simd_lane / scale_step_per_thread;
            const device DataT* input =
                x
                    + size_t(route / TOP_K) * INPUT_DIMENSIONS
                    + simd_lane * values_per_thread;

            float input_values[values_per_thread];
            float results[results_per_simdgroup] = {0};
            for (int k = 0; k < INPUT_DIMENSIONS; k += block_size) {
                const float input_sum = load_vector<
                    DataT,
                    float,
                    values_per_thread,
                    BITS>(input, input_values);

                STEEL_PRAGMA_UNROLL
                for (int row = 0; row < results_per_simdgroup; ++row) {
                    const device uint8_t* row_weight =
                        selected_weight + size_t(row) * packed_input;
                    const device DataT* row_scales =
                        selected_scales + size_t(row) * scale_input;
                    const device DataT* row_biases =
                        selected_biases + size_t(row) * scale_input;
                    results[row] += qdot<
                        float,
                        values_per_thread,
                        BITS>(
                            row_weight,
                            input_values,
                            float(row_scales[0]),
                            float(row_biases[0]),
                            input_sum);
                }

                selected_weight +=
                    block_size * bytes_per_pack / pack_factor;
                selected_scales += block_size / GROUP_SIZE;
                selected_biases += block_size / GROUP_SIZE;
                input += block_size;
            }

            STEEL_PRAGMA_UNROLL
            for (int row = 0; row < results_per_simdgroup; ++row) {
                results[row] = simd_sum(results[row]);
                if (simd_lane == 0) {
                    const uint projection_offset =
                        gate_projection ? 0 : outputs_per_threadgroup;
                    projection_values[
                        projection_offset
                            + local_simd_group * results_per_simdgroup
                            + row
                    ] = DataT(results[row]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (thread_index < outputs_per_threadgroup) {
                const DataT gate_value = projection_values[thread_index];
                const DataT up_value =
                    projection_values[
                        outputs_per_threadgroup + thread_index
                    ];
                const DataT sigmoid_base =
                    DataT(1)
                        / (
                            DataT(1)
                                + metal::exp(metal::abs(gate_value))
                        );
                const DataT sigmoid_value =
                    gate_value < DataT(0)
                        ? sigmoid_base
                        : DataT(1) - sigmoid_base;
                output[
                    size_t(route) * OUTPUT_DIMENSIONS
                        + size_t(tile.y) * outputs_per_threadgroup
                        + thread_index
                ] = DataT(DataT(gate_value * sigmoid_value) * up_value);
            }
        """,
        header: "// MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS\n",
        ensureRowContiguous: true
    )

}

#endif
