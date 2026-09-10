import MLX
import MLXFast

#if os(macOS) || os(iOS)
extension MiniMaxH3FusedKernels {
    static let projectFeedForwardInputAffineInt8SwiGLUTiledKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc1_swiglu_i8g64_tiled_v5",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["activated"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint output_width = 14336;
            constexpr uint group_size = 64;
            constexpr uint scale_groups = input_width / group_size;
            constexpr uint output_tile_columns = 32;
            constexpr uint row_tile_rows = 32;
            constexpr uint reduction_tile = 32;

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

            threadgroup T gate_weight_tile[
                output_tile_columns * reduction_tile
            ];
            threadgroup T up_weight_tile[
                output_tile_columns * reduction_tile
            ];
            threadgroup T input_tile[row_tile_rows * reduction_tile];

            thread simdgroup_matrix<float, 8, 8> gate_accumulated[4];
            thread simdgroup_matrix<float, 8, 8> up_accumulated[4];
            #pragma clang loop unroll(full)
            for (uint index = 0; index < 4; ++index) {
                gate_accumulated[index].thread_elements()[0] = 0.0f;
                gate_accumulated[index].thread_elements()[1] = 0.0f;
                up_accumulated[index].thread_elements()[0] = 0.0f;
                up_accumulated[index].thread_elements()[1] = 0.0f;
            }

            const device uint8_t* codes = (const device uint8_t*)weight_codes;
            uint local_output = thread_linear / 4;
            uint weight_eight = 8 * (thread_linear % 4);
            uint local_input_row = thread_linear / 4;
            uint input_eight = 8 * (thread_linear % 4);
            for (uint reduction_start = 0;
                 reduction_start < input_width;
                 reduction_start += reduction_tile) {
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint gate_column = output_start + local_output;
                uint up_column = gate_column + output_width;
                uint scale_group = reduction_start / group_size;
                uint gate_scale_index = gate_column * scale_groups + scale_group;
                uint up_scale_index = up_column * scale_groups + scale_group;
                float gate_scale = float(weight_scales[gate_scale_index]);
                float gate_bias = float(weight_biases[gate_scale_index]);
                float up_scale = float(weight_scales[up_scale_index]);
                float up_bias = float(weight_biases[up_scale_index]);
                uint weight_reduction_start = reduction_start + weight_eight;
                const device uint8_t* gate_values = codes
                    + gate_column * input_width + weight_reduction_start;
                const device uint8_t* up_values = codes
                    + up_column * input_width + weight_reduction_start;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 8; ++index) {
                    uint tile_index = local_output * reduction_tile
                        + weight_eight + index;
                    gate_weight_tile[tile_index] = T(
                        gate_scale * float(gate_values[index]) + gate_bias
                    );
                    up_weight_tile[tile_index] = T(
                        up_scale * float(up_values[index]) + up_bias
                    );
                }

                uint safe_row = metal::min(local_input_row, valid_rows - 1);
                const device T* input_values = input
                    + uint64_t(row_start + safe_row) * uint64_t(input_width)
                    + reduction_start + input_eight;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 8; ++index) {
                    input_tile[local_input_row * reduction_tile + input_eight + index]
                        = T(input_values[index]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup const T* gate_fragments = gate_weight_tile
                    + (simd_group % 2) * 16 * reduction_tile;
                threadgroup const T* up_fragments = up_weight_tile
                    + (simd_group % 2) * 16 * reduction_tile;
                threadgroup const T* input_fragments = input_tile
                    + (simd_group / 2) * 16 * reduction_tile;
                thread simdgroup_matrix<T, 8, 8> gate_weights[2];
                thread simdgroup_matrix<T, 8, 8> up_weights[2];
                thread simdgroup_matrix<T, 8, 8> inputs[2];
                #pragma clang loop unroll(full)
                for (uint reduction_fragment = 0;
                     reduction_fragment < 4;
                     ++reduction_fragment) {
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint output_fragment = 0;
                         output_fragment < 2;
                         ++output_fragment) {
                        simdgroup_load(
                            gate_weights[output_fragment],
                            gate_fragments
                                + output_fragment * 8 * reduction_tile
                                + reduction_fragment * 8,
                            reduction_tile,
                            0,
                            true
                        );
                        simdgroup_load(
                            up_weights[output_fragment],
                            up_fragments
                                + output_fragment * 8 * reduction_tile
                                + reduction_fragment * 8,
                            reduction_tile,
                            0,
                            true
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint row_fragment = 0; row_fragment < 2; ++row_fragment) {
                        simdgroup_load(
                            inputs[row_fragment],
                            input_fragments
                                + row_fragment * 8 * reduction_tile
                                + reduction_fragment * 8,
                            reduction_tile,
                            0,
                            false
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint result = 0; result < 4; ++result) {
                        thread simdgroup_matrix<float, 8, 8> next_gate;
                        thread simdgroup_matrix<float, 8, 8> next_up;
                        simdgroup_multiply_accumulate(
                            next_gate,
                            inputs[result / 2],
                            gate_weights[result % 2],
                            gate_accumulated[result]
                        );
                        simdgroup_multiply_accumulate(
                            next_up,
                            inputs[result / 2],
                            up_weights[result % 2],
                            up_accumulated[result]
                        );
                        gate_accumulated[result] = next_gate;
                        up_accumulated[result] = next_up;
                    }
                }
            }

            uint simd_output_start = (simd_group & 1) * 16;
            uint simd_row_start = (simd_group >> 1) * 16;
            #pragma clang loop unroll(full)
            for (uint result = 0; result < 4; ++result) {
                uint output_row = simd_row_start + (result / 2) * 8 + matrix_row;
                uint output_column = simd_output_start
                    + (result % 2) * 8 + matrix_column;
                if (output_row < valid_rows) {
                    uint64_t output_index = uint64_t(row_start + output_row)
                        * uint64_t(output_width) + uint64_t(output_start + output_column);
                    float gate0 = float(T(
                        gate_accumulated[result].thread_elements()[0]
                    ));
                    float gate1 = float(T(
                        gate_accumulated[result].thread_elements()[1]
                    ));
                    float up0 = float(T(
                        up_accumulated[result].thread_elements()[0]
                    ));
                    float up1 = float(T(
                        up_accumulated[result].thread_elements()[1]
                    ));
                    float sigmoid0 = 1.0f / (1.0f + metal::precise::exp(-gate0));
                    float sigmoid1 = 1.0f / (1.0f + metal::precise::exp(-gate1));
                    activated[output_index] = T(gate0 * sigmoid0 * up0);
                    activated[output_index + 1] = T(
                        gate1 * sigmoid1 * up1
                    );
                }
            }
        """,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

}

#endif
