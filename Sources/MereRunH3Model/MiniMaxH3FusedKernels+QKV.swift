import MLX
import MLXFast

#if os(macOS) || os(iOS)
extension MiniMaxH3FusedKernels {
    static let prepareHeadMajorQKVKernel = MLXFast.metalKernel(
        name: "mere_h3_qkv_norm_rope_head_major_mixed_v2",
        inputNames: [
            "projected", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query", "key", "value"],
        source: """
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint inner_dimension = heads * head_dimension;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(projected_shape[1]);
            uint projected_row = row * 3 * inner_dimension;
            uint query_base = projected_row + head * head_dimension;
            uint key_base = query_base + inner_dimension;
            uint value_base = key_base + inner_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = float(projected[query_base + dimension]);
                float key_element = float(projected[key_base + dimension]);
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint output_base = (head * rows + row) * head_dimension;
            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                T query_normalized = T(
                    float(projected[query_base + dimension]) * query_inverse);
                T key_normalized = T(
                    float(projected[key_base + dimension]) * key_inverse);
                T query_weighted = T(
                    float(query_normalized) * float(query_norm_weight[dimension]));
                T key_weighted = T(
                    float(key_normalized) * float(key_norm_weight[dimension]));
                float query_output = float(query_weighted);
                float key_output = float(key_weighted);

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    T query_pair_normalized = T(
                        float(projected[query_base + pair]) * query_inverse);
                    T key_pair_normalized = T(
                        float(projected[key_base + pair]) * key_inverse);
                    T query_pair = T(
                        float(query_pair_normalized) * float(query_norm_weight[pair]));
                    T key_pair = T(
                        float(key_pair_normalized) * float(key_norm_weight[pair]));
                    float cosine = float(rope_cosine[rope_base + dimension]);
                    float sine = float(rope_sine[rope_base + dimension]);
                    if (dimension < rotary_half) {
                        query_output = float(query_weighted) * cosine
                            - float(query_pair) * sine;
                        key_output = float(key_weighted) * cosine
                            - float(key_pair) * sine;
                    } else {
                        query_output = float(query_weighted) * cosine
                            + float(query_pair) * sine;
                        key_output = float(key_weighted) * cosine
                            + float(key_pair) * sine;
                    }
                }

                query[output_base + dimension] = Q(query_output);
                key[output_base + dimension] = Q(key_output);
                value[output_base + dimension] = T(
                    projected[value_base + dimension]);
            }
        """,
        ensureRowContiguous: true
    )

    static let projectHeadMajorQKVAffineInt8Kernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qkv_projection_i8g64_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["query", "key", "value"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint inner_dimension = heads * head_dimension;
            constexpr uint projection_width = 3 * inner_dimension;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint rows = uint(input_shape[1]);
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
            const device bfloat16_t* values = input
                + row * input_width + lane * values_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value_element = float(values[index]);
                    input_values[index] = value_element;
                    input_sum += value_element;
                }

                for (uint output_index = 0;
                     output_index < results_per_simdgroup;
                     ++output_index) {
                    const device uint8_t* row_codes = codes
                        + output_index * input_width;
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output_index] += float(
                        scales[output_index * scale_groups]) * dot
                        + float(biases[output_index * scale_groups]) * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                values += block_size;
            }

            for (uint output_index = 0;
                 output_index < results_per_simdgroup;
                 ++output_index) {
                float projected = simd_sum(result[output_index]);
                if (lane == 0) {
                    uint global_dimension = output_row + output_index;
                    uint slab = global_dimension / inner_dimension;
                    uint head_dimension_index = global_dimension
                        - slab * inner_dimension;
                    uint head = head_dimension_index / head_dimension;
                    uint dimension = head_dimension_index - head * head_dimension;
                    uint destination = (head * rows + row) * head_dimension + dimension;
                    if (slab == 0) {
                        query[destination] = bfloat16_t(projected);
                    } else if (slab == 1) {
                        key[destination] = bfloat16_t(projected);
                    } else {
                        value[destination] = bfloat16_t(projected);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    static let projectHeadMajorQKVAffineInt8MixedKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qkv_projection_i8g64_mixed_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["query", "key", "value"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint inner_dimension = heads * head_dimension;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint rows = uint(input_shape[1]);
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
                    float value_element = float(input[
                        row * input_width + input_column + index]);
                    input_values[index] = value_element;
                    input_sum += value_element;
                }

                for (uint output_index = 0;
                     output_index < results_per_simdgroup;
                     ++output_index) {
                    const device uint8_t* row_codes = codes
                        + output_index * input_width;
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output_index] += float(
                        scales[output_index * scale_groups]) * dot
                        + float(biases[output_index * scale_groups]) * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                input_column += block_size;
            }

            for (uint output_index = 0;
                 output_index < results_per_simdgroup;
                 ++output_index) {
                float projected = simd_sum(result[output_index]);
                if (lane == 0) {
                    uint global_dimension = output_row + output_index;
                    uint slab = global_dimension / inner_dimension;
                    uint head_dimension_index = global_dimension
                        - slab * inner_dimension;
                    uint head = head_dimension_index / head_dimension;
                    uint dimension = head_dimension_index - head * head_dimension;
                    uint destination = (head * rows + row) * head_dimension + dimension;
                    if (slab == 0) {
                        query[destination] = projected;
                    } else if (slab == 1) {
                        key[destination] = projected;
                    } else {
                        value[destination] = projected;
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

}

#endif
