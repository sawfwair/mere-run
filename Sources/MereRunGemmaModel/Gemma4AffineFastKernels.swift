import Foundation
import MLX
import MLXFast

struct Gemma4AffineFastKernelKey: Hashable {
    enum Kind: Hashable {
        case score
        case weightedValue
        case weightedValueFromScores
        case weightedValueAndNormalizerFromScores
        case fusedChunkDecode
    }

    let kind: Kind
    let bits: Int
    let groupSize: Int
    let dim: Int
    let packedWidth: Int
    let groupCount: Int
    let repeats: Int
}

enum Gemma4AffineFastKernels {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [Gemma4AffineFastKernelKey: MLXFast.MLXFastKernel] = [:]

    static func scoreKernel(
        bits: Int,
        groupSize: Int,
        dim: Int,
        packedWidth: Int,
        groupCount: Int,
        repeats: Int
        ) -> MLXFast.MLXFastKernel {
        let key = Gemma4AffineFastKernelKey(
                kind: .score,
                bits: bits,
                groupSize: groupSize,
                dim: dim,
                packedWidth: packedWidth,
                groupCount: groupCount,
                repeats: repeats
            )
        return kernel(
            key: key,
            inputNames: ["queries", "packed", "scales", "biases", "scale", "token_counts"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                int token_count = int(token_counts[0]);
                int token_capacity = packed_shape[2];
                auto head_count = queries_shape[1];
                auto batch = n / token_count;
                auto token = n % token_count;
                if (batch >= queries_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto query_ptr = queries + ((batch * head_count + head_idx) * Dim);
                auto packed_ptr = packed + (((batch * packed_shape[1] + kv_head) * token_capacity + token) * PackedWidth);
                auto scales_ptr = scales + (((batch * scales_shape[1] + kv_head) * token_capacity + token) * GroupCount);
                auto biases_ptr = biases + (((batch * biases_shape[1] + kv_head) * token_capacity + token) * GroupCount);

                constexpr uint value_mask = (1u << Bits) - 1u;
                float acc = 0.0f;
                for (int d = lane; d < Dim; d += 32) {
                    int bit_offset = d * Bits;
                    int word_idx = bit_offset / 32;
                    int offset = bit_offset % 32;
                    uint packed_value = packed_ptr[word_idx] >> offset;
                    int spill = offset + Bits - 32;
                    if (spill > 0 && (word_idx + 1) < PackedWidth) {
                        packed_value |= packed_ptr[word_idx + 1] << (Bits - spill);
                    }
                    packed_value &= value_mask;

                    int group_idx = d / GroupSize;
                    float decoded = static_cast<float>(packed_value) * static_cast<float>(scales_ptr[group_idx])
                        + static_cast<float>(biases_ptr[group_idx]);
                    acc += static_cast<float>(query_ptr[d]) * decoded;
                }

                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[((batch * head_count + head_idx) * token_count) + token] =
                        acc * static_cast<float>(scale);
                }
                """
        )
    }

    static func weightedValueKernel(
        bits: Int,
        groupSize: Int,
        dim: Int,
        packedWidth: Int,
        groupCount: Int,
        repeats: Int
        ) -> MLXFast.MLXFastKernel {
        let key = Gemma4AffineFastKernelKey(
                kind: .weightedValue,
                bits: bits,
                groupSize: groupSize,
                dim: dim,
                packedWidth: packedWidth,
                groupCount: groupCount,
                repeats: repeats
            )
        return kernel(
            key: key,
            inputNames: ["weights", "packed", "scales", "biases", "token_counts"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                int token_count = int(token_counts[0]);
                int token_capacity = packed_shape[2];
                auto head_count = weights_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= weights_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto weights_ptr = weights + ((batch * head_count + head_idx) * token_count);
                auto packed_ptr = packed + ((batch * packed_shape[1] + kv_head) * token_capacity * PackedWidth);
                auto scales_ptr = scales + ((batch * scales_shape[1] + kv_head) * token_capacity * GroupCount);
                auto biases_ptr = biases + ((batch * biases_shape[1] + kv_head) * token_capacity * GroupCount);

                int group_idx = dim_idx / GroupSize;
                int bit_offset = dim_idx * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                constexpr uint value_mask = (1u << Bits) - 1u;

                float acc = 0.0f;
                for (int token = lane; token < token_count; token += 32) {
                    auto token_packed = packed_ptr + token * PackedWidth;
                    auto token_scales = scales_ptr + token * GroupCount;
                    auto token_biases = biases_ptr + token * GroupCount;

                    uint packed_value = token_packed[word_idx] >> offset;
                    int spill = offset + Bits - 32;
                    if (spill > 0 && (word_idx + 1) < PackedWidth) {
                        packed_value |= token_packed[word_idx + 1] << (Bits - spill);
                    }
                    packed_value &= value_mask;

                    float decoded = static_cast<float>(packed_value) * static_cast<float>(token_scales[group_idx])
                        + static_cast<float>(token_biases[group_idx]);
                    acc += static_cast<float>(weights_ptr[token]) * decoded;
                }

                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[((batch * head_count + head_idx) * Dim) + dim_idx] = acc;
                }
                """
        )
    }

    static func weightedValueFromScoresKernel(
        bits: Int,
        groupSize: Int,
        dim: Int,
        packedWidth: Int,
        groupCount: Int,
        repeats: Int
    ) -> MLXFast.MLXFastKernel {
        let key = Gemma4AffineFastKernelKey(
            kind: .weightedValueFromScores,
            bits: bits,
            groupSize: groupSize,
            dim: dim,
            packedWidth: packedWidth,
            groupCount: groupCount,
            repeats: repeats
        )
        return kernel(
            key: key,
            inputNames: ["scores", "packed", "scales", "biases", "score_max", "token_counts"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                int token_count = int(token_counts[0]);
                int token_capacity = packed_shape[2];
                auto head_count = scores_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= scores_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto scores_ptr = scores + ((batch * head_count + head_idx) * token_count);
                auto packed_ptr = packed + ((batch * packed_shape[1] + kv_head) * token_capacity * PackedWidth);
                auto scales_ptr = scales + ((batch * scales_shape[1] + kv_head) * token_capacity * GroupCount);
                auto biases_ptr = biases + ((batch * biases_shape[1] + kv_head) * token_capacity * GroupCount);
                float max_score = static_cast<float>(score_max[batch * head_count + head_idx]);

                int group_idx = dim_idx / GroupSize;
                int bit_offset = dim_idx * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                constexpr uint value_mask = (1u << Bits) - 1u;

                float acc = 0.0f;
                for (int token = lane; token < token_count; token += 32) {
                    auto token_packed = packed_ptr + token * PackedWidth;
                    auto token_scales = scales_ptr + token * GroupCount;
                    auto token_biases = biases_ptr + token * GroupCount;

                    uint packed_value = token_packed[word_idx] >> offset;
                    int spill = offset + Bits - 32;
                    if (spill > 0 && (word_idx + 1) < PackedWidth) {
                        packed_value |= token_packed[word_idx + 1] << (Bits - spill);
                    }
                    packed_value &= value_mask;

                    float decoded = static_cast<float>(packed_value) * static_cast<float>(token_scales[group_idx])
                        + static_cast<float>(token_biases[group_idx]);
                    float weight = exp(static_cast<float>(scores_ptr[token]) - max_score);
                    acc += weight * decoded;
                }

                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[((batch * head_count + head_idx) * Dim) + dim_idx] = acc;
                }
                """
        )
    }

    static func weightedValueAndNormalizerFromScoresKernel(
        bits: Int,
        groupSize: Int,
        dim: Int,
        packedWidth: Int,
        groupCount: Int,
        repeats: Int
    ) -> MLXFast.MLXFastKernel {
        let key = Gemma4AffineFastKernelKey(
            kind: .weightedValueAndNormalizerFromScores,
            bits: bits,
            groupSize: groupSize,
            dim: dim,
            packedWidth: packedWidth,
            groupCount: groupCount,
            repeats: repeats
        )
        return kernel(
            key: key,
            inputNames: ["scores", "packed", "scales", "biases", "score_max", "token_counts"],
            outputNames: ["weighted", "normalizer"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                int token_count = int(token_counts[0]);
                int token_capacity = packed_shape[2];
                auto head_count = scores_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= scores_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto scores_ptr = scores + ((batch * head_count + head_idx) * token_count);
                auto packed_ptr = packed + ((batch * packed_shape[1] + kv_head) * token_capacity * PackedWidth);
                auto scales_ptr = scales + ((batch * scales_shape[1] + kv_head) * token_capacity * GroupCount);
                auto biases_ptr = biases + ((batch * biases_shape[1] + kv_head) * token_capacity * GroupCount);
                float max_score = static_cast<float>(score_max[batch * head_count + head_idx]);

                int group_idx = dim_idx / GroupSize;
                int bit_offset = dim_idx * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                constexpr uint value_mask = (1u << Bits) - 1u;

                float weighted_acc = 0.0f;
                float normalizer_acc = 0.0f;
                for (int token = lane; token < token_count; token += 32) {
                    auto token_packed = packed_ptr + token * PackedWidth;
                    auto token_scales = scales_ptr + token * GroupCount;
                    auto token_biases = biases_ptr + token * GroupCount;

                    uint packed_value = token_packed[word_idx] >> offset;
                    int spill = offset + Bits - 32;
                    if (spill > 0 && (word_idx + 1) < PackedWidth) {
                        packed_value |= token_packed[word_idx + 1] << (Bits - spill);
                    }
                    packed_value &= value_mask;

                    float decoded = static_cast<float>(packed_value) * static_cast<float>(token_scales[group_idx])
                        + static_cast<float>(token_biases[group_idx]);
                    float weight = exp(static_cast<float>(scores_ptr[token]) - max_score);
                    weighted_acc += weight * decoded;
                    if (dim_idx == 0) {
                        normalizer_acc += weight;
                    }
                }

                weighted_acc = simd_sum(weighted_acc);
                if (thread_index_in_simdgroup == 0) {
                    weighted[((batch * head_count + head_idx) * Dim) + dim_idx] = weighted_acc;
                }

                if (dim_idx == 0) {
                    normalizer_acc = simd_sum(normalizer_acc);
                    if (thread_index_in_simdgroup == 0) {
                        normalizer[batch * head_count + head_idx] = normalizer_acc;
                    }
                }
                """
        )
    }

    static func fusedChunkDecodeKernel(
        keyBits: Int,
        valueBits: Int,
        groupSize: Int,
        dim: Int,
        keyPackedWidth: Int,
        valuePackedWidth: Int,
        keyGroupCount: Int,
        valueGroupCount: Int,
        repeats: Int
    ) -> MLXFast.MLXFastKernel {
        let key = Gemma4AffineFastKernelKey(
            kind: .fusedChunkDecode,
            bits: keyBits * 100 + valueBits,
            groupSize: groupSize,
            dim: dim,
            packedWidth: max(keyPackedWidth, valuePackedWidth),
            groupCount: max(keyGroupCount, valueGroupCount),
            repeats: repeats
        )
        return kernel(
            key: key,
            inputNames: [
                "queries",
                "key_packed",
                "key_scales",
                "key_biases",
                "value_packed",
                "value_scales",
                "value_biases",
                "scale",
                "token_counts",
            ],
            outputNames: ["weighted", "normalizer", "score_max"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                int token_count = int(token_counts[0]);
                int token_start = int(token_counts[1]);
                int key_capacity = key_packed_shape[2];
                int value_capacity = value_packed_shape[2];
                auto head_count = queries_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= queries_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto query_ptr = queries + ((batch * head_count + head_idx) * Dim);

                auto key_packed_ptr = key_packed + (((batch * key_packed_shape[1] + kv_head) * key_capacity + token_start) * KeyPackedWidth);
                auto key_scales_ptr = key_scales + (((batch * key_scales_shape[1] + kv_head) * key_capacity + token_start) * KeyGroupCount);
                auto key_biases_ptr = key_biases + (((batch * key_biases_shape[1] + kv_head) * key_capacity + token_start) * KeyGroupCount);

                auto value_packed_ptr = value_packed + (((batch * value_packed_shape[1] + kv_head) * value_capacity + token_start) * ValuePackedWidth);
                auto value_scales_ptr = value_scales + (((batch * value_scales_shape[1] + kv_head) * value_capacity + token_start) * ValueGroupCount);
                auto value_biases_ptr = value_biases + (((batch * value_biases_shape[1] + kv_head) * value_capacity + token_start) * ValueGroupCount);

                int value_group_idx = dim_idx / GroupSize;
                int value_bit_offset = dim_idx * ValueBits;
                int value_word_idx = value_bit_offset / 32;
                int value_offset = value_bit_offset % 32;
                constexpr uint key_mask = (1u << KeyBits) - 1u;
                constexpr uint value_mask = (1u << ValueBits) - 1u;

                float running_max = -INFINITY;
                float running_sum = 0.0f;
                float running_acc = 0.0f;

                for (int token = 0; token < token_count; token++) {
                    auto key_token_packed = key_packed_ptr + token * KeyPackedWidth;
                    auto key_token_scales = key_scales_ptr + token * KeyGroupCount;
                    auto key_token_biases = key_biases_ptr + token * KeyGroupCount;

                    float score = 0.0f;
                    for (int d = lane; d < Dim; d += 32) {
                        int group_idx = d / GroupSize;
                        int bit_offset = d * KeyBits;
                        int word_idx = bit_offset / 32;
                        int offset = bit_offset % 32;

                        uint packed_value = key_token_packed[word_idx] >> offset;
                        int spill = offset + KeyBits - 32;
                        if (spill > 0 && (word_idx + 1) < KeyPackedWidth) {
                            packed_value |= key_token_packed[word_idx + 1] << (KeyBits - spill);
                        }
                        packed_value &= key_mask;

                        float decoded = static_cast<float>(packed_value) * static_cast<float>(key_token_scales[group_idx])
                            + static_cast<float>(key_token_biases[group_idx]);
                        score += static_cast<float>(query_ptr[d]) * decoded;
                    }

                    score = simd_sum(score) * static_cast<float>(scale);

                    auto value_token_packed = value_packed_ptr + token * ValuePackedWidth;
                    auto value_token_scales = value_scales_ptr + token * ValueGroupCount;
                    auto value_token_biases = value_biases_ptr + token * ValueGroupCount;

                    uint value_packed_word = value_token_packed[value_word_idx] >> value_offset;
                    int value_spill = value_offset + ValueBits - 32;
                    if (value_spill > 0 && (value_word_idx + 1) < ValuePackedWidth) {
                        value_packed_word |= value_token_packed[value_word_idx + 1] << (ValueBits - value_spill);
                    }
                    value_packed_word &= value_mask;

                    float decoded_value = static_cast<float>(value_packed_word) * static_cast<float>(value_token_scales[value_group_idx])
                        + static_cast<float>(value_token_biases[value_group_idx]);

                    float previous_max = running_max;
                    running_max = max(running_max, score);
                    float rescale = isfinite(previous_max) ? exp(previous_max - running_max) : 0.0f;
                    float weight = exp(score - running_max);
                    running_sum = running_sum * rescale + weight;
                    running_acc = running_acc * rescale + weight * decoded_value;
                }

                if (thread_index_in_simdgroup == 0) {
                    weighted[((batch * head_count + head_idx) * Dim) + dim_idx] = running_acc;
                    if (dim_idx == 0) {
                        normalizer[batch * head_count + head_idx] = running_sum;
                        score_max[batch * head_count + head_idx] = running_max;
                    }
                }
                """
        )
    }

    private static func kernel(
        key: Gemma4AffineFastKernelKey,
        inputNames: [String],
        outputNames: [String] = ["out"],
        source: @autoclosure () -> String
    ) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }

        if let existing = kernels[key] {
            return existing
        }

        let kernel = MLXFast.metalKernel(
            name: "gemma4_affine_\(key.kind)_b\(key.bits)_g\(key.groupSize)_d\(key.dim)_pw\(key.packedWidth)_gc\(key.groupCount)_r\(key.repeats)",
            inputNames: inputNames,
            outputNames: outputNames,
            source: source()
        )
        kernels[key] = kernel
        return kernel
    }
}
