import Foundation
import MLX
import MLXFast

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
extension RoutedMoERouting {
    static let lagunaXSDownHeader = lagunaNVFP4QDotFastEnabled
        ? lagunaXSFastQDotHeader
        : lagunaXSReferenceQDotHeader

    static let lagunaXSReferenceQDotHeader = """
        static inline float mere_laguna_nvfp4_scale(uint8_t bits) {
            // E4M3 and IEEE half are both sign-magnitude here. Adding the
            // source sign bit before the shift carries it directly into the
            // half sign position, including negative zero, and removes the
            // post-conversion conditional negate without changing any bits.
            ushort raw = ushort((uint(bits) + (bits & 128u)) << 7);
            return float(as_type<half>(raw)) * 4194304.0f;
        }

        static inline float mere_laguna_nvfp4_qdot_codes_16(
            uint2 codes,
            const thread float* input,
            float scale
        ) {
            float accum = 0.0f;
            for (uint j = 0; j < 2; ++j) {
                const uint c = (j == 0) ? codes.x : codes.y;
                const uint p0 =
                    ((c & 0x00070007u) << 9) | ((c & 0x00080008u) << 12);
                const uint p1 =
                    ((c & 0x00700070u) << 5) | ((c & 0x00800080u) << 8);
                const uint p2 =
                    ((c & 0x07000700u) << 1) | ((c & 0x08000800u) << 4);
                const uint p3 =
                    ((c & 0x70007000u) >> 3) | (c & 0x80008000u);
                const float2 v04 = float2(as_type<half2>(p0));
                const float2 v15 = float2(as_type<half2>(p1));
                const float2 v26 = float2(as_type<half2>(p2));
                const float2 v37 = float2(as_type<half2>(p3));
                accum +=
                    (input[8 * j] * v04.x
                     + input[8 * j + 1] * v15.x
                     + input[8 * j + 2] * v26.x
                     + input[8 * j + 3] * v37.x);
                accum +=
                    (input[8 * j + 4] * v04.y
                     + input[8 * j + 5] * v15.y
                     + input[8 * j + 6] * v26.y
                     + input[8 * j + 7] * v37.y);
            }
            return scale * accum;
        }

        static inline float mere_laguna_nvfp4_qdot_16(
            const device uint8_t* weight,
            const thread float* input,
            float scale
        ) {
            const device uint2* packed = (const device uint2*)weight;
            return mere_laguna_nvfp4_qdot_codes_16(packed[0], input, scale);
        }
        """

    /// Laguna-only NVFP4 group-16 qdot specialization. It constructs the same
    /// half bit patterns with a split-nibble integer sequence, moves the exact
    /// 2^14 renormalization into the group scale, and seeds the accumulator
    /// from the first four-term product group. The only exceptional case is
    /// the sign of an all-negative-zero partial, which every caller absorbs in
    /// its existing +0.0 row accumulator before the BF16 output boundary.
    static let lagunaXSFastQDotHeader = """
        static inline float mere_laguna_nvfp4_scale(uint8_t bits) {
            if (bits < 16u) {
                return float(uint(bits) << 5);
            }
            ushort raw = ushort(bits & 127) << 7;
            half converted = as_type<half>(raw);
            half signed_value = (bits & 128) ? -converted : converted;
            return float(signed_value) * 4194304.0f;
        }

        static inline float mere_laguna_nvfp4_qdot_codes_16(
            uint2 codes,
            const thread float* input,
            float scale
        ) {
            float accum;
            {
                const uint c = codes.x;
                const uint xe = c & 0x0F0F0F0Fu;
                const uint ge = xe | (xe << 3);
                const uint yo = c & 0xF0F0F0F0u;
                const uint go = yo | (yo >> 3);
                const uint p0 = (ge << 9) & 0x8E008E00u;
                const uint p1 = (go << 8) & 0x8E008E00u;
                const uint p2 = (ge << 1) & 0x8E008E00u;
                const uint p3 = go & 0x8E008E00u;
                const float2 v04 = float2(as_type<half2>(p0));
                const float2 v15 = float2(as_type<half2>(p1));
                const float2 v26 = float2(as_type<half2>(p2));
                const float2 v37 = float2(as_type<half2>(p3));
                accum =
                    (input[0] * v04.x
                     + input[1] * v15.x
                     + input[2] * v26.x
                     + input[3] * v37.x);
                accum +=
                    (input[4] * v04.y
                     + input[5] * v15.y
                     + input[6] * v26.y
                     + input[7] * v37.y);
            }
            {
                const uint c = codes.y;
                const uint xe = c & 0x0F0F0F0Fu;
                const uint ge = xe | (xe << 3);
                const uint yo = c & 0xF0F0F0F0u;
                const uint go = yo | (yo >> 3);
                const uint p0 = (ge << 9) & 0x8E008E00u;
                const uint p1 = (go << 8) & 0x8E008E00u;
                const uint p2 = (ge << 1) & 0x8E008E00u;
                const uint p3 = go & 0x8E008E00u;
                const float2 v04 = float2(as_type<half2>(p0));
                const float2 v15 = float2(as_type<half2>(p1));
                const float2 v26 = float2(as_type<half2>(p2));
                const float2 v37 = float2(as_type<half2>(p3));
                accum +=
                    (input[8] * v04.x
                     + input[9] * v15.x
                     + input[10] * v26.x
                     + input[11] * v37.x);
                accum +=
                    (input[12] * v04.y
                     + input[13] * v15.y
                     + input[14] * v26.y
                     + input[15] * v37.y);
            }
            return scale * accum;
        }

        static inline float mere_laguna_nvfp4_qdot_16(
            const device uint8_t* weight,
            const thread float* input,
            float scale
        ) {
            const device uint2* packed = (const device uint2*)weight;
            return mere_laguna_nvfp4_qdot_codes_16(packed[0], input, scale);
        }
        """

    static let lagunaXSRoutedSharedDownResidualKernel = MLXFast.metalKernel(
        name: lagunaNVFP4QDotFastEnabled
            ? "mere_laguna_xs_routed_shared_nvfp4_down_residual_bf16_r1_qf_v1"
            : "mere_laguna_xs_routed_shared_nvfp4_down_residual_bf16_r1_v1",
        inputNames: [
            "routed_activated",
            "routed_down_weight",
            "routed_down_scales",
            "indices",
            "router_weights",
            "shared_activated",
            "shared_down_weight",
            "shared_down_scales",
            "residual",
        ],
        outputNames: ["output"],
        source: """
            constexpr uint input_width = 512;
            constexpr uint output_width = 2048;
            constexpr uint routed_experts = 8;
            constexpr uint shared_slot = 8;
            constexpr uint outputs_per_simd = 1;
            constexpr uint values_per_lane = 16;
            constexpr uint packed_row_bytes = 256;
            constexpr uint scale_row_bytes = 32;
            constexpr uint packed_expert_bytes =
                output_width * packed_row_bytes;
            constexpr uint scale_expert_bytes =
                output_width * scale_row_bytes;

            uint tile = threadgroup_position_in_grid.x;
            uint slot = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint first_row = tile * outputs_per_simd;
            bool is_shared = slot == shared_slot;
            uint expert = is_shared ? 0 : uint(indices[slot]);

            const device bfloat* expert_input = is_shared
                ? shared_activated
                : routed_activated + slot * input_width;
            const device uint8_t* expert_weight = is_shared
                ? (const device uint8_t*)shared_down_weight
                : (const device uint8_t*)routed_down_weight
                    + expert * packed_expert_bytes;
            const device uint8_t* expert_scales = is_shared
                ? shared_down_scales
                : routed_down_scales + expert * scale_expert_bytes;

            thread float input_values[values_per_lane];
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*) (
                    expert_input + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = values[0];
                input_values[4 * i + 1] = values[1];
                input_values[4 * i + 2] = values[2];
                input_values[4 * i + 3] = values[3];
            }

            thread float result[outputs_per_simd] = {0.0f};
            for (uint row = 0; row < outputs_per_simd; ++row) {
                uint output_row = first_row + row;
                const device uint8_t* weight =
                    expert_weight + output_row * packed_row_bytes + lane * 8;
                const device uint8_t* scale =
                    expert_scales + output_row * scale_row_bytes + lane;
                result[row] = mere_laguna_nvfp4_qdot_16(
                    weight,
                    input_values,
                    mere_laguna_nvfp4_scale(scale[0]));
                result[row] = simd_sum(result[row]);
            }

            threadgroup bfloat down_outputs[
                (routed_experts + 1) * outputs_per_simd
            ];
            if (lane == 0) {
                for (uint row = 0; row < outputs_per_simd; ++row) {
                    down_outputs[slot * outputs_per_simd + row] =
                        bfloat(result[row]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (slot == 0 && lane < outputs_per_simd) {
                bfloat routed_total = bfloat(0);
                for (uint routed_slot = 0;
                     routed_slot < routed_experts;
                     ++routed_slot) {
                    bfloat route_weight = router_weights[routed_slot];
                    bfloat product = bfloat(
                        down_outputs[
                            routed_slot * outputs_per_simd + lane
                        ] * route_weight);
                    routed_total = bfloat(product + routed_total);
                }
                bfloat routed = bfloat(routed_total * bfloat(2.5f));
                bfloat shared =
                    down_outputs[shared_slot * outputs_per_simd + lane];
                bfloat branch = bfloat(routed + shared);
                output[first_row + lane] =
                    bfloat(residual[first_row + lane] + branch);
            }
        """,
        header: lagunaXSDownHeader,
        ensureRowContiguous: true
    )

}

#endif
