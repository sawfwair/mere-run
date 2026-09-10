import MLX
import MLXFast

#if os(macOS) || os(iOS)
extension MiniMaxH3FusedKernels {
    static let projectFeedForwardOutputAffineInt8TiledKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc2_i8g64_tiled_v5",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["projected"],
        source: """
            constexpr uint input_width = 14336;
            constexpr uint output_width = 5376;
            constexpr uint group_size = 64;
            constexpr uint scale_groups = input_width / group_size;
            constexpr uint output_tile_columns = 64;
            constexpr uint row_tile_rows = 32;
            constexpr uint reduction_tile = 32;
            constexpr uint simdgroups_per_workgroup = 4;

            uint output_tile = threadgroup_position_in_grid.x;
            uint row_tile = threadgroup_position_in_grid.y;
            uint output_start = output_tile * output_tile_columns;
            uint row_start = row_tile * row_tile_rows;
            uint rows = uint(input_shape[1]);
            uint valid_rows = metal::min(row_tile_rows, rows - row_start);
            uint thread_linear = thread_position_in_threadgroup.y * 32
                + thread_position_in_threadgroup.x;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint quad = lane / 4;
            uint matrix_row = (quad & 4) + ((lane / 2) % 4);
            uint matrix_column = (quad & 2) * 2 + (lane % 2) * 2;

            threadgroup T weight_tile[output_tile_columns * reduction_tile];
            threadgroup T input_tile[row_tile_rows * reduction_tile];

            thread simdgroup_matrix<float, 8, 8> accumulated[8];
            #pragma clang loop unroll(full)
            for (uint index = 0; index < 8; ++index) {
                accumulated[index].thread_elements()[0] = 0.0f;
                accumulated[index].thread_elements()[1] = 0.0f;
            }

            const device uint8_t* codes = (const device uint8_t*)weight_codes;
            uint local_output = thread_linear / 2;
            uint weight_half = thread_linear % 2;
            uint local_input_row = thread_linear / 4;
            uint input_eight = 8 * (thread_linear % 4);
            for (uint reduction_start = 0;
                 reduction_start < input_width;
                 reduction_start += reduction_tile) {
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint output_column = output_start + local_output;
                uint scale_index = output_column * scale_groups
                    + reduction_start / group_size;
                float scale = float(weight_scales[scale_index]);
                float bias = float(weight_biases[scale_index]);
                uint weight_reduction_start = reduction_start + 16 * weight_half;
                const device uint8_t* weight_values = codes
                    + output_column * input_width + weight_reduction_start;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 16; ++index) {
                    uint section_x = 2 * weight_half + index / 8;
                    uint section_y = local_output / 8;
                    uint local_x = local_output % 8;
                    uint local_y = index % 8;
                    uint block = 8 * section_x + section_y;
                    weight_tile[64 * block + 8 * local_y + local_x] = T(
                        scale * float(weight_values[index]) + bias
                    );
                }

                uint safe_row = metal::min(local_input_row, valid_rows - 1);
                uint input_section_x = thread_linear % 4;
                uint input_section_y = local_input_row / 8;
                uint input_local_y = local_input_row % 8;
                uint input_block = 4 * input_section_x + input_section_y;
                const device T* input_values = input
                    + uint64_t(row_start + safe_row) * uint64_t(input_width)
                    + reduction_start + input_eight;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 8; ++index) {
                    input_tile[64 * input_block + 8 * input_local_y + index]
                        = T(input_values[index]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup const T* weight_fragments = weight_tile
                    + 4 * 64 * (simd_group % 2);
                threadgroup const T* input_fragments = input_tile
                    + 2 * 64 * (simd_group / 2);
                thread simdgroup_matrix<T, 8, 8> weights[4];
                thread simdgroup_matrix<T, 8, 8> inputs[2];
                #pragma clang loop unroll(full)
                for (uint reduction_fragment = 0;
                     reduction_fragment < 4;
                     ++reduction_fragment) {
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint output_fragment = 0;
                         output_fragment < 4;
                         ++output_fragment) {
                        simdgroup_load(
                            weights[output_fragment],
                            weight_fragments + 64 * output_fragment,
                            8,
                            0,
                            false
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint row_fragment = 0; row_fragment < 2; ++row_fragment) {
                        simdgroup_load(
                            inputs[row_fragment],
                            input_fragments + 64 * row_fragment,
                            8,
                            0,
                            false
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint result = 0; result < 8; ++result) {
                        thread simdgroup_matrix<float, 8, 8> next;
                        simdgroup_multiply_accumulate(
                            next,
                            inputs[result / 4],
                            weights[result % 4],
                            accumulated[result]
                        );
                        accumulated[result] = next;
                    }
                    weight_fragments += 8 * 64;
                    input_fragments += 4 * 64;
                }
            }

            uint simd_output_start = (simd_group & 1) * 32;
            uint simd_row_start = (simd_group >> 1) * 16;
            #pragma clang loop unroll(full)
            for (uint result = 0; result < 8; ++result) {
                uint output_row = simd_row_start + (result / 4) * 8 + matrix_row;
                uint output_column = simd_output_start
                    + (result % 4) * 8 + matrix_column;
                if (output_row < valid_rows) {
                    uint64_t output_index = uint64_t(row_start + output_row)
                        * uint64_t(output_width) + uint64_t(output_start + output_column);
                    projected[output_index] = T(
                        accumulated[result].thread_elements()[0]
                    );
                    projected[output_index + 1] = T(
                        accumulated[result].thread_elements()[1]
                    );
                }
            }
        """,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    static let projectFeedForwardOutputAffineInt8FloatKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc2_i8g64_f32_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["projected"],
        source: """
            constexpr uint input_width = 14336;
            constexpr uint output_width = 5376;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device bfloat16_t* scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            uint input_column = lane * values_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = input[row * input_width + input_column + index];
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* row_codes = codes + output * input_width;
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output] += float(scales[output * scale_groups]) * dot
                        + float(biases[output * scale_groups]) * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                input_column += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                float value = simd_sum(result[output]);
                if (lane == 0) {
                    projected[row * output_width + output_row + output] = value;
                }
            }
        """,
        ensureRowContiguous: true
    )

}

#endif
