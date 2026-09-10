import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package enum LagunaFusedPrefill {
    package static let ropeAngleAtlasLength = 4_096

    package enum QKNormRoPEKind {
        case fullYaRN
        case sliding
    }

    package static func residualRMSNorm(
        residual: MLXArray,
        branch: MLXArray,
        weight: MLXArray
    ) -> (summed: MLXArray, normalized: MLXArray)? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let hiddenSize = 2_048
        guard Device.defaultDevice().deviceType == .gpu,
              residual.dtype == .bfloat16,
              branch.dtype == .bfloat16,
              weight.dtype == .bfloat16,
              residual.shape == branch.shape,
              residual.ndim == 3,
              residual.dim(0) == 1,
              residual.dim(1) > 1,
              residual.dim(2) == hiddenSize,
              weight.shape == [hiddenSize] else {
            return nil
        }

        let rows = residual.size / hiddenSize
        let outputs = residualRMSNormKernel(
            [residual, branch, weight],
            grid: (rows * 512, 1, 1),
            threadGroup: (512, 1, 1),
            outputShapes: [residual.shape, residual.shape],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
        #else
        return nil
        #endif
    }

    package static func qkNormRoPE(
        kind: QKNormRoPEKind,
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        queryWeight: MLXArray,
        keyWeight: MLXArray,
        angleAtlas: MLXArray,
        offset: Int,
        length: Int
    ) -> (queries: MLXArray, keys: MLXArray)? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let headDimension = 128
        let keyValueHeads = 8
        let queryHeads = kind == .fullYaRN ? 48 : 64
        let angleWidth = kind == .fullYaRN ? 64 : 128
        guard Device.defaultDevice().deviceType == .gpu,
              length > 1,
              offset >= 0,
              offset + length <= ropeAngleAtlasLength,
              rawQueries.dtype == .bfloat16,
              rawKeys.dtype == .bfloat16,
              queryWeight.dtype == .bfloat16,
              keyWeight.dtype == .bfloat16,
              angleAtlas.dtype == .float32,
              rawQueries.shape == [1, length, queryHeads * headDimension],
              rawKeys.shape == [1, length, keyValueHeads * headDimension],
              queryWeight.shape == [headDimension],
              keyWeight.shape == [headDimension],
              angleAtlas.shape == [1, 1, ropeAngleAtlasLength, angleWidth] else {
            return nil
        }

        let offsets = MLXArray([Int32(offset)])
        let kernel = kind == .fullYaRN
            ? prefillFullQKNormYaRNKernel
            : prefillSlidingQKNormRoPEKernel
        let outputs = kernel(
            [rawQueries, rawKeys, queryWeight, keyWeight, angleAtlas, offsets],
            grid: ((queryHeads + keyValueHeads) / 4 * 128, length, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [
                [1, queryHeads, length, headDimension],
                [1, keyValueHeads, length, headDimension],
            ],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
        #else
        return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    static let residualRMSNormKernel = MLXFast.metalKernel(
        name: "mere_laguna_prefill_residual_rms_bf16_2048_v1",
        inputNames: ["residual", "branch", "weight"],
        outputNames: ["summed", "normalized"],
        source: """
            constexpr uint axis_size = 2048;
            constexpr uint n_reads = 4;
            constexpr uint simd_size = 32;

            uint row = threadgroup_position_in_grid.x;
            uint lid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint base = row * axis_size + lid * n_reads;

            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[simd_size];

            thread bfloat values[n_reads];
            float acc = 0.0f;
            for (uint i = 0; i < n_reads; ++i) {
                bfloat value = bfloat(residual[base + i] + branch[base + i]);
                values[i] = value;
                summed[base + i] = value;
                float fv = float(value);
                acc += fv * fv;
            }

            acc = simd_sum(acc);
            if (simd_group == 0) {
                local_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                local_sums[simd_group] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc = simd_sum(local_sums[simd_lane]);
                if (simd_lane == 0) {
                    local_inv_mean[0] =
                        metal::precise::rsqrt(acc / 2048.0f + 1.0e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inverse_mean = local_inv_mean[0];

            for (uint i = 0; i < n_reads; ++i) {
                normalized[base + i] =
                    weight[lid * n_reads + i]
                    * bfloat(float(values[i]) * inverse_mean);
            }
        """,
        ensureRowContiguous: true
    )

    static let prefillSlidingQKNormRoPEKernel = MLXFast.metalKernel(
        name: "mere_laguna_prefill_sliding_qk_norm_rope_bf16_128_v1",
        inputNames: [
            "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
            "offsets",
        ],
        outputNames: ["queries", "keys"],
        source: """
            constexpr uint head_dim = 128;
            constexpr uint rotary_pairs = 64;
            constexpr uint query_heads = 64;
            constexpr uint kv_heads = 8;

            uint token = threadgroup_position_in_grid.y;
            uint length = threadgroups_per_grid.y;
            uint head = threadgroup_position_in_grid.x * 4
                + simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;

            const device bfloat* input;
            const device bfloat* weight;
            device bfloat* output;
            if (head < query_heads) {
                input = raw_queries + (token * query_heads + head) * head_dim;
                weight = query_weight;
                output = queries + (head * length + token) * head_dim;
            } else {
                uint key_head = head - query_heads;
                input = raw_keys + (token * kv_heads + key_head) * head_dim;
                weight = key_weight;
                output = keys + (key_head * length + token) * head_dim;
            }

            uint base = lane * 4;
            thread bfloat normalized[4];
            float sum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                float value = float(input[base + i]);
                sum += value * value;
            }
            sum = simd_sum(sum);
            float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

            for (uint i = 0; i < 4; ++i) {
                normalized[i] =
                    weight[base + i]
                    * bfloat(float(input[base + i]) * inverse_rms);
            }

            thread float paired[4];
            for (uint i = 0; i < 4; ++i) {
                paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
            }

            const device float* angle_row =
                angles + (uint(offsets[0]) + token) * (2 * rotary_pairs);
            if (lane < 16) {
                for (uint i = 0; i < 4; ++i) {
                    uint pair = base + i;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    float cosine = angle_row[pair];
                    float sine = angle_row[pair + rotary_pairs];
                    output[pair] = bfloat(first * cosine - second * sine);
                    output[pair + rotary_pairs] =
                        bfloat(first * sine + second * cosine);
                }
            }
        """,
        ensureRowContiguous: true
    )

    static let prefillFullQKNormYaRNKernel = MLXFast.metalKernel(
        name: "mere_laguna_prefill_full_qk_norm_yarn_bf16_128_v1",
        inputNames: [
            "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
            "offsets",
        ],
        outputNames: ["queries", "keys"],
        source: """
            constexpr uint head_dim = 128;
            constexpr uint rotary_pairs = 32;
            constexpr uint query_heads = 48;
            constexpr uint kv_heads = 8;
            constexpr float yarn_mscale = 1.3465735912322998f;

            uint token = threadgroup_position_in_grid.y;
            uint length = threadgroups_per_grid.y;
            uint head = threadgroup_position_in_grid.x * 4
                + simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;

            const device bfloat* input;
            const device bfloat* weight;
            device bfloat* output;
            if (head < query_heads) {
                input = raw_queries + (token * query_heads + head) * head_dim;
                weight = query_weight;
                output = queries + (head * length + token) * head_dim;
            } else {
                uint key_head = head - query_heads;
                input = raw_keys + (token * kv_heads + key_head) * head_dim;
                weight = key_weight;
                output = keys + (key_head * length + token) * head_dim;
            }

            uint base = lane * 4;
            thread bfloat normalized[4];
            float sum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                float value = float(input[base + i]);
                sum += value * value;
            }
            sum = simd_sum(sum);
            float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

            for (uint i = 0; i < 4; ++i) {
                normalized[i] =
                    weight[base + i]
                    * bfloat(float(input[base + i]) * inverse_rms);
            }

            thread float paired[4];
            for (uint i = 0; i < 4; ++i) {
                paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
            }

            const device float* angle_row =
                angles + (uint(offsets[0]) + token) * (2 * rotary_pairs);
            if (lane < 8) {
                bfloat rounded_mscale = bfloat(yarn_mscale);
                for (uint i = 0; i < 4; ++i) {
                    uint pair = base + i;
                    float first = float(bfloat(normalized[i] * rounded_mscale));
                    float second =
                        float(bfloat(bfloat(paired[i]) * rounded_mscale));
                    float cosine = angle_row[pair];
                    float sine = angle_row[pair + rotary_pairs];
                    output[pair] = bfloat(first * cosine - second * sine);
                    output[pair + rotary_pairs] =
                        bfloat(first * sine + second * cosine);
                }
            } else if (lane >= 16) {
                for (uint i = 0; i < 4; ++i) {
                    output[base + i] = normalized[i];
                }
            }
        """,
        ensureRowContiguous: true
    )
    #endif
}
