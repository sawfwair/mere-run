import MLX
import MLXFast

#if os(macOS) || os(iOS)
extension MiniMaxH3FusedKernels {
    static let projectFeedForwardInputAffineInt8SwiGLUKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc1_swiglu_i8g64_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["activated"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint output_width = 14336;
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

            const device uint8_t* gate_codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device uint8_t* up_codes = (const device uint8_t*)weight_codes
                + (output_row + output_width) * input_width
                + lane * values_per_thread;
            const device bfloat16_t* gate_scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_scales = weight_scales
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            const device bfloat16_t* gate_biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_biases = weight_biases
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            const device bfloat16_t* values = input
                + row * input_width + lane * values_per_thread;

            thread float gate_result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            thread float up_result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = float(values[index]);
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* gate_row = gate_codes + output * input_width;
                    const device uint8_t* up_row = up_codes + output * input_width;
                    float gate_dot = 0.0f;
                    float up_dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        float value = input_values[index];
                        gate_dot += value * float(gate_row[index]);
                        up_dot += value * float(up_row[index]);
                    }
                    gate_result[output] += float(
                        gate_scales[output * scale_groups]) * gate_dot
                        + float(gate_biases[output * scale_groups]) * input_sum;
                    up_result[output] += float(
                        up_scales[output * scale_groups]) * up_dot
                        + float(up_biases[output * scale_groups]) * input_sum;
                }

                gate_codes += block_size;
                up_codes += block_size;
                gate_scales += block_size / group_size;
                up_scales += block_size / group_size;
                gate_biases += block_size / group_size;
                up_biases += block_size / group_size;
                values += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                float gate_sum = simd_sum(gate_result[output]);
                float up_sum = simd_sum(up_result[output]);
                if (lane == 0) {
                    float gate = float(bfloat16_t(gate_sum));
                    float up = float(bfloat16_t(up_sum));
                    float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate));
                    activated[row * output_width + output_row + output]
                        = bfloat16_t(gate * sigmoid * up);
                }
            }
        """,
        ensureRowContiguous: true
    )

    static let projectFeedForwardInputAffineInt8SwiGLUFloatKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc1_swiglu_i8g64_f32_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["activated"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint output_width = 14336;
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

            const device uint8_t* gate_codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device uint8_t* up_codes = (const device uint8_t*)weight_codes
                + (output_row + output_width) * input_width
                + lane * values_per_thread;
            const device bfloat16_t* gate_scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_scales = weight_scales
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            const device bfloat16_t* gate_biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_biases = weight_biases
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            uint input_column = lane * values_per_thread;

            thread float gate_result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            thread float up_result[results_per_simdgroup] = {
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
                    const device uint8_t* gate_row = gate_codes + output * input_width;
                    const device uint8_t* up_row = up_codes + output * input_width;
                    float gate_dot = 0.0f;
                    float up_dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        gate_dot += input_values[index] * float(gate_row[index]);
                        up_dot += input_values[index] * float(up_row[index]);
                    }
                    gate_result[output] += float(
                        gate_scales[output * scale_groups]) * gate_dot
                        + float(gate_biases[output * scale_groups]) * input_sum;
                    up_result[output] += float(
                        up_scales[output * scale_groups]) * up_dot
                        + float(up_biases[output * scale_groups]) * input_sum;
                }

                gate_codes += block_size;
                up_codes += block_size;
                gate_scales += block_size / group_size;
                up_scales += block_size / group_size;
                gate_biases += block_size / group_size;
                up_biases += block_size / group_size;
                input_column += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                float gate = simd_sum(gate_result[output]);
                float up = simd_sum(up_result[output]);
                if (lane == 0) {
                    float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate));
                    activated[row * output_width + output_row + output]
                        = gate * sigmoid * up;
                }
            }
        """,
        ensureRowContiguous: true
    )

    static let projectFeedForwardOutputAffineInt8Kernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc2_i8g64_v1",
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
            const device bfloat16_t* values = input
                + row * input_width + lane * values_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = float(values[index]);
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
                values += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                result[output] = simd_sum(result[output]);
                if (lane == 0) {
                    projected[row * output_width + output_row + output]
                        = bfloat16_t(result[output]);
                }
            }
        """,
        ensureRowContiguous: true
    )

}

#endif
