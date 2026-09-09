import Foundation
import MLX
import MLXFast

struct Gemma4PolarFastKernelKey: Hashable {
    enum Kind: Hashable {
        case pack
        case unpack
        case score
        case weightedValue
        case fusedChunkDecode
    }

    let kind: Kind
    let bits: Int
    let dim: Int
    let packedWidth: Int
    let repeats: Int
}

enum Gemma4PolarFastKernels {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [Gemma4PolarFastKernelKey: MLXFast.MLXFastKernel] = [:]

    static func packKernel(bits: Int, dim: Int, packedWidth: Int) -> MLXFast.MLXFastKernel {
        let key = Gemma4PolarFastKernelKey(kind: .pack, bits: bits, dim: dim, packedWidth: packedWidth, repeats: 1)
        return kernel(
            key: key,
            inputNames: ["rotated", "inner_boundaries"],
            outputNames: ["packed"],
            source: """
                auto packed_word = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                auto token_count = rotated_shape[2];
                auto batch = n / token_count;
                auto token = n % token_count;
                if (packed_word >= PackedWidth || head_idx >= rotated_shape[1] || batch >= rotated_shape[0]) {
                    return;
                }

                constexpr uint value_mask = (1u << Bits) - 1u;
                constexpr int level_bound_count = (1 << Bits) - 1;
                int word_start = int(packed_word) * 32;
                int word_end = word_start + 32;
                int start_dim = max(0, ((word_start - Bits) / Bits) + 1);
                int end_dim = min(Dim, (word_end + Bits - 1) / Bits);

                auto rotated_ptr = rotated + (((batch * rotated_shape[1] + head_idx) * token_count + token) * Dim);

                uint packed_value = 0u;
                for (int d = start_dim; d < end_dim; d++) {
                    float v = static_cast<float>(rotated_ptr[d]);
                    uint index = 0u;
                    for (int boundary = 0; boundary < level_bound_count; boundary++) {
                        index += static_cast<uint>(v > static_cast<float>(inner_boundaries[boundary]));
                    }
                    index &= value_mask;

                    int bit_offset = d * Bits - word_start;
                    if (bit_offset >= 0) {
                        packed_value |= index << bit_offset;
                    } else {
                        packed_value |= index >> (-bit_offset);
                    }
                }

                packed[((batch * rotated_shape[1] + head_idx) * token_count + token) * PackedWidth + packed_word] = packed_value;
                """
        )
    }

    static func unpackKernel(bits: Int, dim: Int, packedWidth: Int) -> MLXFast.MLXFastKernel {
        let key = Gemma4PolarFastKernelKey(kind: .unpack, bits: bits, dim: dim, packedWidth: packedWidth, repeats: 1)
        return kernel(
            key: key,
            inputNames: ["packed", "norms", "centroids", "token_counts"],
            source: """
                auto dim_idx = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                int token_count = int(token_counts[0]);
                int token_start = int(token_counts[1]);
                int token_capacity = packed_shape[2];
                auto batch = n / token_count;
                auto token = n % token_count;
                if (dim_idx >= Dim || head_idx >= packed_shape[1] || batch >= packed_shape[0]) {
                    return;
                }

                auto packed_ptr = packed + (((batch * packed_shape[1] + head_idx) * token_capacity + token_start + token) * PackedWidth);
                int bit_offset = int(dim_idx) * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                constexpr uint value_mask = (1u << Bits) - 1u;

                uint packed_value = packed_ptr[word_idx] >> offset;
                int spill = offset + Bits - 32;
                if (spill > 0 && (word_idx + 1) < PackedWidth) {
                    packed_value |= packed_ptr[word_idx + 1] << (Bits - spill);
                }
                packed_value &= value_mask;

                float norm = static_cast<float>(norms[((batch * norms_shape[1] + head_idx) * token_capacity + token_start + token)]);
                out[((batch * packed_shape[1] + head_idx) * token_count + token) * Dim + dim_idx] =
                    static_cast<float>(centroids[packed_value]) * norm;
                """
        )
    }

    static func scoreKernel(bits: Int, dim: Int, packedWidth: Int, repeats: Int) -> MLXFast.MLXFastKernel {
        let key = Gemma4PolarFastKernelKey(
            kind: .score,
            bits: bits,
            dim: dim,
            packedWidth: packedWidth,
            repeats: repeats
        )
        return kernel(
            key: key,
            inputNames: ["queries", "packed", "norms", "centroids", "scale", "token_counts"],
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
                auto norms_ptr = norms + ((batch * norms_shape[1] + kv_head) * token_capacity);
                float norm = static_cast<float>(norms_ptr[token]);

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

                    float decoded = static_cast<float>(centroids[packed_value]) * norm;
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

    static func weightedValueKernel(bits: Int, dim: Int, packedWidth: Int, repeats: Int) -> MLXFast.MLXFastKernel {
        let key = Gemma4PolarFastKernelKey(
            kind: .weightedValue,
            bits: bits,
            dim: dim,
            packedWidth: packedWidth,
            repeats: repeats
        )
        return kernel(
            key: key,
            inputNames: ["weights", "packed", "norms", "centroids", "token_counts"],
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
                auto norms_ptr = norms + ((batch * norms_shape[1] + kv_head) * token_capacity);

                int bit_offset = dim_idx * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                constexpr uint value_mask = (1u << Bits) - 1u;

                float acc = 0.0f;
                for (int token = lane; token < token_count; token += 32) {
                    auto token_packed = packed_ptr + token * PackedWidth;
                    uint packed_value = token_packed[word_idx] >> offset;
                    int spill = offset + Bits - 32;
                    if (spill > 0 && (word_idx + 1) < PackedWidth) {
                        packed_value |= token_packed[word_idx + 1] << (Bits - spill);
                    }
                    packed_value &= value_mask;

                    float decoded = static_cast<float>(centroids[packed_value]) * static_cast<float>(norms_ptr[token]);
                    acc += static_cast<float>(weights_ptr[token]) * decoded;
                }

                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[((batch * head_count + head_idx) * Dim) + dim_idx] = acc;
                }
                """
        )
    }

    static func fusedChunkDecodeKernel(bits: Int, dim: Int, packedWidth: Int, repeats: Int) -> MLXFast.MLXFastKernel {
        let key = Gemma4PolarFastKernelKey(
            kind: .fusedChunkDecode,
            bits: bits,
            dim: dim,
            packedWidth: packedWidth,
            repeats: repeats
        )
        return kernel(
            key: key,
            inputNames: [
                "queries",
                "key_packed",
                "key_norms",
                "value_packed",
                "value_norms",
                "centroids",
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
                auto key_packed_ptr = key_packed + (((batch * key_packed_shape[1] + kv_head) * key_capacity + token_start) * PackedWidth);
                auto value_packed_ptr = value_packed + (((batch * value_packed_shape[1] + kv_head) * value_capacity + token_start) * PackedWidth);
                auto key_norms_ptr = key_norms + ((batch * key_norms_shape[1] + kv_head) * key_capacity + token_start);
                auto value_norms_ptr = value_norms + ((batch * value_norms_shape[1] + kv_head) * value_capacity + token_start);

                constexpr uint value_mask = (1u << Bits) - 1u;
                int value_bit_offset = dim_idx * Bits;
                int value_word_idx = value_bit_offset / 32;
                int value_offset = value_bit_offset % 32;

                float running_max = -INFINITY;
                float running_sum = 0.0f;
                float running_acc = 0.0f;

                for (int token = 0; token < token_count; token++) {
                    auto key_token_packed = key_packed_ptr + token * PackedWidth;
                    float key_norm = static_cast<float>(key_norms_ptr[token]);

                    float score = 0.0f;
                    for (int d = lane; d < Dim; d += 32) {
                        int bit_offset = d * Bits;
                        int word_idx = bit_offset / 32;
                        int offset = bit_offset % 32;
                        uint packed_value = key_token_packed[word_idx] >> offset;
                        int spill = offset + Bits - 32;
                        if (spill > 0 && (word_idx + 1) < PackedWidth) {
                            packed_value |= key_token_packed[word_idx + 1] << (Bits - spill);
                        }
                        packed_value &= value_mask;

                        float decoded = static_cast<float>(centroids[packed_value]) * key_norm;
                        score += static_cast<float>(query_ptr[d]) * decoded;
                    }

                    score = simd_sum(score) * static_cast<float>(scale);

                    auto value_token_packed = value_packed_ptr + token * PackedWidth;
                    uint value_packed_word = value_token_packed[value_word_idx] >> value_offset;
                    int value_spill = value_offset + Bits - 32;
                    if (value_spill > 0 && (value_word_idx + 1) < PackedWidth) {
                        value_packed_word |= value_token_packed[value_word_idx + 1] << (Bits - value_spill);
                    }
                    value_packed_word &= value_mask;

                    float decoded_value = static_cast<float>(centroids[value_packed_word])
                        * static_cast<float>(value_norms_ptr[token]);

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
        key: Gemma4PolarFastKernelKey,
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
            name: "gemma4_polar_\(key.kind)_b\(key.bits)_d\(key.dim)_pw\(key.packedWidth)_r\(key.repeats)",
            inputNames: inputNames,
            outputNames: outputNames,
            source: source()
        )
        kernels[key] = kernel
        return kernel
    }
}
