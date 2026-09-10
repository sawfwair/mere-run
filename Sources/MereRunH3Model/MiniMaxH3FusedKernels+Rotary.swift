import MLX
import MLXFast

#if os(macOS) || os(iOS)
extension MiniMaxH3FusedKernels {
    static let normalizeHeadMajorQKVRoPEKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qk_norm_rope_bf16_v1",
        inputNames: [
            "query_input", "key_input", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query_output", "key_output"],
        source: """
            constexpr uint head_dimension = 128;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(query_input_shape[2]);
            uint input_base = (head * rows + row) * head_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = float(query_input[input_base + dimension]);
                float key_element = float(key_input[input_base + dimension]);
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                bfloat16_t query_normalized = bfloat16_t(
                    float(query_input[input_base + dimension]) * query_inverse);
                bfloat16_t key_normalized = bfloat16_t(
                    float(key_input[input_base + dimension]) * key_inverse);
                bfloat16_t query_weighted = bfloat16_t(
                    float(query_normalized) * float(query_norm_weight[dimension]));
                bfloat16_t key_weighted = bfloat16_t(
                    float(key_normalized) * float(key_norm_weight[dimension]));
                float query_value = float(query_weighted);
                float key_value = float(key_weighted);

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    bfloat16_t query_pair_normalized = bfloat16_t(
                        float(query_input[input_base + pair]) * query_inverse);
                    bfloat16_t key_pair_normalized = bfloat16_t(
                        float(key_input[input_base + pair]) * key_inverse);
                    bfloat16_t query_pair = bfloat16_t(
                        float(query_pair_normalized) * float(query_norm_weight[pair]));
                    bfloat16_t key_pair = bfloat16_t(
                        float(key_pair_normalized) * float(key_norm_weight[pair]));
                    float cosine = float(rope_cosine[rope_base + dimension]);
                    float sine = float(rope_sine[rope_base + dimension]);
                    if (dimension < rotary_half) {
                        query_value = float(query_weighted) * cosine
                            - float(query_pair) * sine;
                        key_value = float(key_weighted) * cosine
                            - float(key_pair) * sine;
                    } else {
                        query_value = float(query_weighted) * cosine
                            + float(query_pair) * sine;
                        key_value = float(key_weighted) * cosine
                            + float(key_pair) * sine;
                    }
                }

                query_output[input_base + dimension] = bfloat16_t(query_value);
                key_output[input_base + dimension] = bfloat16_t(key_value);
            }
        """,
        ensureRowContiguous: true
    )

    static let normalizeHeadMajorQKVRoPEBF16ToFloatKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qk_norm_rope_bf16_f32_v1",
        inputNames: [
            "query_input", "key_input", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query_output", "key_output"],
        source: """
            constexpr uint head_dimension = 128;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(query_input_shape[2]);
            uint input_base = (head * rows + row) * head_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = float(query_input[input_base + dimension]);
                float key_element = float(key_input[input_base + dimension]);
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                bfloat16_t query_weighted = bfloat16_t(
                    float(query_input[input_base + dimension]) * query_inverse
                        * float(query_norm_weight[dimension]));
                bfloat16_t key_weighted = bfloat16_t(
                    float(key_input[input_base + dimension]) * key_inverse
                        * float(key_norm_weight[dimension]));
                float query_value = float(query_weighted);
                float key_value = float(key_weighted);

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    bfloat16_t query_pair = bfloat16_t(
                        float(query_input[input_base + pair]) * query_inverse
                            * float(query_norm_weight[pair]));
                    bfloat16_t key_pair = bfloat16_t(
                        float(key_input[input_base + pair]) * key_inverse
                            * float(key_norm_weight[pair]));
                    float cosine = float(rope_cosine[rope_base + dimension]);
                    float sine = float(rope_sine[rope_base + dimension]);
                    float sign = dimension < rotary_half ? -1.0f : 1.0f;
                    query_value = float(query_weighted) * cosine
                        + sign * float(query_pair) * sine;
                    key_value = float(key_weighted) * cosine
                        + sign * float(key_pair) * sine;
                }

                query_output[input_base + dimension] = query_value;
                key_output[input_base + dimension] = key_value;
            }
        """,
        ensureRowContiguous: true
    )

    static let normalizeHeadMajorQKVRoPEFloatKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qk_norm_rope_f32_v1",
        inputNames: [
            "query_input", "key_input", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query_output", "key_output"],
        source: """
            constexpr uint head_dimension = 128;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(query_input_shape[2]);
            uint input_base = (head * rows + row) * head_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = query_input[input_base + dimension];
                float key_element = key_input[input_base + dimension];
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_weighted = query_input[input_base + dimension]
                    * query_inverse * float(query_norm_weight[dimension]);
                float key_weighted = key_input[input_base + dimension]
                    * key_inverse * float(key_norm_weight[dimension]);
                float query_value = query_weighted;
                float key_value = key_weighted;

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    float query_pair = query_input[input_base + pair]
                        * query_inverse * float(query_norm_weight[pair]);
                    float key_pair = key_input[input_base + pair]
                        * key_inverse * float(key_norm_weight[pair]);
                    float cosine = rope_cosine[rope_base + dimension];
                    float sine = rope_sine[rope_base + dimension];
                    float sign = dimension < rotary_half ? -1.0f : 1.0f;
                    query_value = query_weighted * cosine + sign * query_pair * sine;
                    key_value = key_weighted * cosine + sign * key_pair * sine;
                }

                query_output[input_base + dimension] = query_value;
                key_output[input_base + dimension] = key_value;
            }
        """,
        ensureRowContiguous: true
    )

    static let projectHeadMajorAttentionAffineInt8Kernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_affine_oproj_i8g64_v1",
        inputNames: [
            "attention", "weight_codes", "weight_scales", "weight_biases",
        ],
        outputNames: ["projected"],
        source: """
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint input_width = heads * head_dimension;
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
            uint rows = uint(attention_shape[2]);
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

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            uint column = lane * values_per_thread;
            for (uint block = 0; block < input_width; block += block_size) {
                uint head = column / head_dimension;
                uint dimension = column - head * head_dimension;
                uint input_offset = (head * rows + row) * head_dimension + dimension;
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = float(attention[input_offset + index]);
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* row_codes = codes + output * input_width;
                    float scale = float(scales[output * scale_groups]);
                    float bias = float(biases[output * scale_groups]);
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output] += scale * dot + bias * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                column += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                result[output] = simd_sum(result[output]);
                if (lane == 0) {
                    projected[row * output_width + output_row + output]
                        = T(result[output]);
                }
            }
        """,
        ensureRowContiguous: true
    )

}

#endif
