import MLX
import MLXFast

#if os(macOS) || os(iOS)
extension MiniMaxH3FusedKernels {
    static let attentionAdaLNKernel = MLXFast.metalKernel(
        name: "mere_h3_attention_adaln_bf16_h5376_v1",
        inputNames: [
            "input", "norm_weight", "modulation", "row_indices", "epsilon",
        ],
        outputNames: ["output"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float value = float(input[row_offset + dimension]);
                sum += value * value;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                bfloat16_t normalized = bfloat16_t(
                    float(input[row_offset + dimension]) * inverse_rms[0]);
                bfloat16_t weighted = bfloat16_t(
                    float(norm_weight[dimension]) * float(normalized));
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + hidden_size + dimension]));
                bfloat16_t scaled = bfloat16_t(
                    float(weighted) * float(one_plus_scale));
                output[row_offset + dimension] = bfloat16_t(
                    float(scaled) + float(modulation[
                        modulation_offset + dimension]));
            }
        """,
        ensureRowContiguous: true
    )

    static let attentionAdaLNMixedKernel = MLXFast.metalKernel(
        name: "mere_h3_attention_adaln_mixed_h5376_v1",
        inputNames: [
            "input", "norm_weight", "modulation", "row_indices", "epsilon",
        ],
        outputNames: ["output"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float value = float(input[row_offset + dimension]);
                sum += value * value;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float weighted = float(input[row_offset + dimension])
                    * inverse_rms[0] * float(norm_weight[dimension]);
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + hidden_size + dimension]));
                output[row_offset + dimension] = weighted
                    * float(one_plus_scale) + float(modulation[
                        modulation_offset + dimension]);
            }
        """,
        ensureRowContiguous: true
    )

    static let gateAdaLNKernel = MLXFast.metalKernel(
        name: "mere_h3_gate_adaln_bf16_h5376_v1",
        inputNames: [
            "residual", "attention_output", "norm_weight", "modulation",
            "row_indices", "epsilon",
        ],
        outputNames: ["residual_out", "feed_forward_input", "feed_forward_gate"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup bfloat16_t rounded_residual[hidden_size];
            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float gate = float(modulation[
                    modulation_offset + 2 * hidden_size + dimension]);
                bfloat16_t gated_attention = bfloat16_t(
                    gate * float(attention_output[row_offset + dimension]));
                bfloat16_t value = bfloat16_t(
                    float(residual[row_offset + dimension])
                    + float(gated_attention));
                rounded_residual[dimension] = value;
                residual_out[row_offset + dimension] = value;
                float widened = float(value);
                sum += widened * widened;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                bfloat16_t normalized = bfloat16_t(
                    float(rounded_residual[dimension]) * inverse_rms[0]);
                bfloat16_t weighted = bfloat16_t(
                    float(norm_weight[dimension]) * float(normalized));
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + 4 * hidden_size + dimension]));
                bfloat16_t scaled = bfloat16_t(
                    float(weighted) * float(one_plus_scale));
                feed_forward_input[row_offset + dimension] = bfloat16_t(
                    float(scaled) + float(modulation[
                        modulation_offset + 3 * hidden_size + dimension]));
                feed_forward_gate[row_offset + dimension] = modulation[
                    modulation_offset + 5 * hidden_size + dimension];
            }
        """,
        ensureRowContiguous: true
    )

    static let gateAdaLNMixedKernel = MLXFast.metalKernel(
        name: "mere_h3_gate_adaln_mixed_h5376_v1",
        inputNames: [
            "residual", "attention_output", "norm_weight", "modulation",
            "row_indices", "epsilon",
        ],
        outputNames: ["residual_out", "feed_forward_input", "feed_forward_gate"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup float attended[hidden_size];
            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float gate = float(modulation[
                    modulation_offset + 2 * hidden_size + dimension]);
                float value = float(residual[row_offset + dimension])
                    + gate * float(attention_output[row_offset + dimension]);
                attended[dimension] = value;
                residual_out[row_offset + dimension] = value;
                sum += value * value;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float weighted = attended[dimension] * inverse_rms[0]
                    * float(norm_weight[dimension]);
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + 4 * hidden_size + dimension]));
                feed_forward_input[row_offset + dimension] = weighted
                    * float(one_plus_scale) + float(modulation[
                        modulation_offset + 3 * hidden_size + dimension]);
                feed_forward_gate[row_offset + dimension] = modulation[
                    modulation_offset + 5 * hidden_size + dimension];
            }
        """,
        ensureRowContiguous: true
    )

    static let gateAdaLNQuantizeKernel = MLXFast.metalKernel(
        name: "mere_h3_gate_adaln_quantize_i8_h5376_v1",
        inputNames: [
            "residual", "attention_output", "norm_weight", "modulation",
            "row_indices", "epsilon",
        ],
        outputNames: ["residual_out", "quantized_input", "quantized_scales"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup bfloat16_t rounded_values[hidden_size];
            threadgroup float partial_values[simd_width];
            threadgroup float shared_value[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float gate = float(modulation[
                    modulation_offset + 2 * hidden_size + dimension]);
                bfloat16_t gated_attention = bfloat16_t(
                    gate * float(attention_output[row_offset + dimension]));
                bfloat16_t value = bfloat16_t(
                    float(residual[row_offset + dimension])
                    + float(gated_attention));
                rounded_values[dimension] = value;
                residual_out[row_offset + dimension] = value;
                float widened = float(value);
                sum += widened * widened;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_values[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_values[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_values[simd_lane]);
                if (simd_lane == 0) {
                    shared_value[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            float local_max = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                bfloat16_t normalized = bfloat16_t(
                    float(rounded_values[dimension]) * shared_value[0]);
                bfloat16_t weighted = bfloat16_t(
                    float(norm_weight[dimension]) * float(normalized));
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + 4 * hidden_size + dimension]));
                bfloat16_t scaled = bfloat16_t(
                    float(weighted) * float(one_plus_scale));
                bfloat16_t value = bfloat16_t(
                    float(scaled) + float(modulation[
                        modulation_offset + 3 * hidden_size + dimension]));
                rounded_values[dimension] = value;
                local_max = metal::max(local_max, metal::fabs(float(value)));
            }

            local_max = simd_max(local_max);
            if (simd_group == 0) {
                partial_values[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_values[simd_group] = local_max;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                local_max = simd_max(partial_values[simd_lane]);
                if (simd_lane == 0) {
                    shared_value[0] = local_max;
                    quantized_scales[row] = local_max > 0.0f
                        ? local_max / 127.0f
                        : 1.0f / 127.0f;
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            float quantize_scale = shared_value[0] > 0.0f
                ? 127.0f / shared_value[0]
                : 127.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                int value = int(rint(float(rounded_values[dimension]) * quantize_scale));
                quantized_input[row_offset + dimension] = int8_t(
                    metal::clamp(value, -127, 127));
            }
        """,
        ensureRowContiguous: true
    )

    static let quantizeRowsKernel = MLXFast.metalKernel(
        name: "mere_h3_quantize_rows_i8_h5376_v1",
        inputNames: ["input"],
        outputNames: ["quantized", "scales"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            threadgroup float partial_maxima[simd_width];
            threadgroup float maximum[1];

            float local_max = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                local_max = metal::max(
                    local_max,
                    metal::fabs(float(input[row_offset + dimension])));
            }
            local_max = simd_max(local_max);
            if (simd_group == 0) {
                partial_maxima[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_maxima[simd_group] = local_max;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                local_max = simd_max(partial_maxima[simd_lane]);
                if (simd_lane == 0) {
                    maximum[0] = local_max;
                    scales[row] = local_max > 0.0f
                        ? local_max / 127.0f
                        : 1.0f / 127.0f;
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            float quantize_scale = maximum[0] > 0.0f
                ? 127.0f / maximum[0]
                : 127.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                int value = int(rint(float(input[row_offset + dimension]) * quantize_scale));
                quantized[row_offset + dimension] = int8_t(
                    metal::clamp(value, -127, 127));
            }
        """,
        ensureRowContiguous: true
    )

}

#endif
