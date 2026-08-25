import Foundation
import MLX
import MLXFast

public enum Gemma4KVQuantizationScheme: String, Sendable, Hashable {
    case uniform
    case polar
    case turboquant
}

public struct Gemma4KVCacheQuantization: Sendable, Hashable {
    public var bits: Double?
    public var scheme: Gemma4KVQuantizationScheme
    public var groupSize: Int
    public var quantizedStart: Int

    public init(
        bits: Double? = nil,
        scheme: Gemma4KVQuantizationScheme = Gemma4Resources.defaultKVQuantizationScheme,
        groupSize: Int = Gemma4Resources.defaultKVGroupSize,
        quantizedStart: Int = Gemma4Resources.defaultQuantizedKVStart
    ) {
        self.bits = bits
        self.scheme = scheme
        self.groupSize = groupSize
        self.quantizedStart = quantizedStart
    }

    public var isEnabled: Bool {
        bits != nil
    }

    var statusDescription: String {
        guard let bits else {
            return "full-precision"
        }
        return "\(scheme.rawValue):\(bits)-bit:start-\(quantizedStart)"
    }

    func validated() throws -> Gemma4KVCacheQuantization {
        guard let bits else {
            return self
        }

        guard bits >= 2, bits <= 8 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 KV quantization bits must be between 2 and 8 (received \(bits)).")
        }
        guard groupSize > 0 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 KV quantization group size must be greater than zero.")
        }
        guard quantizedStart >= 0 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 quantized KV start must be zero or greater.")
        }

        let roundedHalf = (bits * 2).rounded() / 2
        switch scheme {
        case .uniform:
            guard Swift.abs(bits.rounded() - bits) < 0.000_001 else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 uniform KV quantization requires an integer bit width. Use turboquant for fractional .5 widths.")
            }
        case .polar:
            guard Swift.abs(bits.rounded() - bits) < 0.000_001,
                  [2, 3, 4].contains(Int(bits.rounded())) else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 PolarKV quantization currently supports 2, 3, or 4 bits (received \(bits)).")
            }
        case .turboquant:
            guard Swift.abs(roundedHalf - bits) < 0.000_001 else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 turboquant currently supports integer and .5 bit widths (received \(bits)).")
            }
        }

        return self
    }

    var keyBits: Int? {
        guard let bits else { return nil }
        switch scheme {
        case .uniform, .polar:
            return Int(bits.rounded())
        case .turboquant:
            return Int(floor(bits))
        }
    }

    var valueBits: Int? {
        guard let bits else { return nil }
        switch scheme {
        case .uniform, .polar:
            return Int(bits.rounded())
        case .turboquant:
            return Int(ceil(bits))
        }
    }
}

private struct Gemma4AffineFastKernelKey: Hashable {
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

private enum Gemma4AffineFastKernels {
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

private struct Gemma4PolarFastKernelKey: Hashable {
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

private enum Gemma4PolarCodebook {
    static func centroids(bits: Int, dim: Int) -> MLXArray {
        MLXArray(values(for: bits).map { $0 / sqrt(Float32(dim)) }, [1 << bits]).asType(.float32)
    }

    static func innerBoundaries(bits: Int, dim: Int) -> MLXArray {
        let values = boundaries(for: bits)
        let scale = Float32(1) / sqrt(Float32(dim))
        return MLXArray(values.dropFirst().dropLast().map { $0 * scale }, [(1 << bits) - 1]).asType(.float32)
    }

    private static func values(for bits: Int) -> [Float32] {
        switch bits {
        case 2:
            return [-1.5104, -0.4528, 0.4528, 1.5104]
        case 3:
            return [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519]
        case 4:
            return [
                -2.7331, -2.0698, -1.6189, -1.2570, -0.9431, -0.6573, -0.3884, -0.1285,
                0.1285, 0.3884, 0.6573, 0.9431, 1.2570, 1.6189, 2.0698, 2.7331,
            ]
        default:
            preconditionFailure("Unsupported PolarKV bit width \(bits).")
        }
    }

    private static func boundaries(for bits: Int) -> [Float32] {
        switch bits {
        case 2:
            return [-5.0, -0.9816, 0.0, 0.9816, 5.0]
        case 3:
            return [-5.0, -1.7479, -1.0499, -0.5005, 0.0, 0.5005, 1.0499, 1.7479, 5.0]
        case 4:
            return [
                -5.0, -2.4015, -1.8443, -1.4380, -1.1001, -0.8002, -0.5229, -0.2585,
                0.0, 0.2585, 0.5229, 0.8002, 1.1001, 1.4380, 1.8443, 2.4015, 5.0,
            ]
        default:
            preconditionFailure("Unsupported PolarKV bit width \(bits).")
        }
    }
}

private enum Gemma4PolarRotation {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var matrices: [Int: MLXArray] = [:]

    static func matrix(dim: Int) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }

        if let existing = matrices[dim] {
            return existing
        }

        let scale = Float32(1) / sqrt(Float32(dim))
        let values: [Float32]
        // The Python prototype uses seeded QR; a deterministic Hadamard rotation
        // keeps this native prototype cheap to build for Gemma's power-of-two head dims.
        if dim > 0 && (dim & (dim - 1)) == 0 {
            values = (0..<dim).flatMap { row in
                (0..<dim).map { column in
                    ((row & column).nonzeroBitCount.isMultiple(of: 2) ? scale : -scale)
                }
            }
        } else {
            values = (0..<dim).flatMap { row in
                (0..<dim).map { column in row == column ? Float32(1) : Float32(0) }
            }
        }

        let matrix = MLXArray(values, [dim, dim]).asType(.float32)
        matrices[dim] = matrix
        return matrix
    }
}

private enum Gemma4PolarFastKernels {
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

private enum Gemma4KVTokenStorage {
    static let allocationStep = 256

    // Tensor states returned by `appending` share these capacity-padded buffers with their
    // ancestors: new rows are written in place at indices >= the writer's previous tokenCount,
    // so a snapshot must only ever read rows below its own tokenCount. Growth reallocates for
    // the writer, leaving older snapshots on the previous buffer.
    static func appended(_ array: MLXArray, rows: MLXArray, validCount: Int) -> MLXArray {
        let newCount = validCount + rows.dim(2)
        var target = array
        if newCount > array.dim(2) {
            let steps = max(1, (newCount - validCount + allocationStep - 1) / allocationStep)
            let valid = validCount < array.dim(2) ? array[0..., 0..., 0..<validCount, 0...] : array
            let padding = MLXArray.zeros(
                [array.dim(0), array.dim(1), steps * allocationStep, array.dim(3)],
                dtype: array.dtype
            )
            target = concatenated([valid, padding], axis: 2)
        }
        target[0..., 0..., validCount..<newCount, 0...] = rows
        return target
    }
}

final class Gemma4QuantizedTensorState {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let dtype: DType
    let tokenCount: Int

    var tokenCapacity: Int {
        weight.dim(2)
    }

    var packedWidth: Int {
        weight.dim(3)
    }

    var groupCount: Int {
        scales.dim(3)
    }

    init(source: MLXArray, groupSize: Int, bits: Int) {
        let quantized = MLX.quantized(
            source,
            groupSize: groupSize,
            bits: bits,
            mode: .affine
        )
        self.weight = quantized.wq
        self.scales = quantized.scales
        self.biases = quantized.biases
        self.groupSize = groupSize
        self.bits = bits
        self.dtype = source.dtype
        self.tokenCount = source.dim(2)
    }

    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        dtype: DType,
        tokenCount: Int
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.dtype = dtype
        self.tokenCount = tokenCount
    }

    func appending(_ source: MLXArray) -> Gemma4QuantizedTensorState {
        let next = Gemma4QuantizedTensorState(source: source, groupSize: groupSize, bits: bits)
        var appendedBiases: MLXArray?
        if let biases, let nextBiases = next.biases {
            appendedBiases = Gemma4KVTokenStorage.appended(biases, rows: nextBiases, validCount: tokenCount)
        }
        return Gemma4QuantizedTensorState(
            weight: Gemma4KVTokenStorage.appended(weight, rows: next.weight, validCount: tokenCount),
            scales: Gemma4KVTokenStorage.appended(scales, rows: next.scales, validCount: tokenCount),
            biases: appendedBiases,
            groupSize: groupSize,
            bits: bits,
            dtype: dtype,
            tokenCount: tokenCount + next.tokenCount
        )
    }

    func dequantized() -> MLXArray {
        dequantized(tokenRange: 0..<tokenCount)
    }

    func dequantized(tokenRange: Range<Int>) -> MLXArray {
        let slicedWeight = weight[0..., 0..., tokenRange, 0...]
        let slicedScales = scales[0..., 0..., tokenRange, 0...]
        let slicedBiases = biases.map { $0[0..., 0..., tokenRange, 0...] }
        return MLX.dequantized(
            slicedWeight,
            scales: slicedScales,
            biases: slicedBiases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: dtype
        )
    }
}

final class Gemma4PolarTensorState {
    let packed: MLXArray
    let norms: MLXArray
    let bits: Int
    let dtype: DType
    let headDim: Int
    let rotation: MLXArray
    let rotationTransposed: MLXArray
    let centroids: MLXArray
    let innerBoundaries: MLXArray
    let tokenCount: Int

    var tokenCapacity: Int {
        packed.dim(2)
    }

    var packedWidth: Int {
        packed.dim(3)
    }

    init(source: MLXArray, bits: Int) {
        let headDim = source.dim(3)
        let rotation = Gemma4PolarRotation.matrix(dim: headDim)
        let rotationTransposed = rotation.transposed()
        let centroids = Gemma4PolarCodebook.centroids(bits: bits, dim: headDim)
        let innerBoundaries = Gemma4PolarCodebook.innerBoundaries(bits: bits, dim: headDim)
        let source32 = source.asType(.float32)
        let norms = MLX.sqrt(MLX.sum(source32 * source32, axis: -1, keepDims: true))
        let normalized = source32 / MLX.maximum(norms, MLXArray(Float32(1e-8)))
        let rotated = MLX.matmul(normalized, rotationTransposed)
        let packedWidth = (headDim * bits + 31) / 32
        let packKernel = Gemma4PolarFastKernels.packKernel(bits: bits, dim: headDim, packedWidth: packedWidth)
        let packed = packKernel(
            [rotated, innerBoundaries],
            template: [
                ("Bits", bits),
                ("Dim", headDim),
                ("PackedWidth", packedWidth),
            ],
            grid: (packedWidth, source.dim(1), source.dim(0) * source.dim(2)),
            threadGroup: (max(1, min(32, packedWidth)), 1, 1),
            outputShapes: [[source.dim(0), source.dim(1), source.dim(2), packedWidth]],
            outputDTypes: [.uint32]
        )[0]

        self.packed = packed
        self.norms = norms
        self.bits = bits
        self.dtype = source.dtype
        self.headDim = headDim
        self.rotation = rotation
        self.rotationTransposed = rotationTransposed
        self.centroids = centroids
        self.innerBoundaries = innerBoundaries
        self.tokenCount = source.dim(2)
    }

    init(
        packed: MLXArray,
        norms: MLXArray,
        bits: Int,
        dtype: DType,
        headDim: Int,
        rotation: MLXArray,
        rotationTransposed: MLXArray,
        centroids: MLXArray,
        innerBoundaries: MLXArray,
        tokenCount: Int
    ) {
        self.packed = packed
        self.norms = norms
        self.bits = bits
        self.dtype = dtype
        self.headDim = headDim
        self.rotation = rotation
        self.rotationTransposed = rotationTransposed
        self.centroids = centroids
        self.innerBoundaries = innerBoundaries
        self.tokenCount = tokenCount
    }

    func appending(_ source: MLXArray) -> Gemma4PolarTensorState {
        let next = Gemma4PolarTensorState(source: source, bits: bits)
        return Gemma4PolarTensorState(
            packed: Gemma4KVTokenStorage.appended(packed, rows: next.packed, validCount: tokenCount),
            norms: Gemma4KVTokenStorage.appended(norms, rows: next.norms, validCount: tokenCount),
            bits: bits,
            dtype: dtype,
            headDim: headDim,
            rotation: rotation,
            rotationTransposed: rotationTransposed,
            centroids: centroids,
            innerBoundaries: innerBoundaries,
            tokenCount: tokenCount + next.tokenCount
        )
    }

    func dequantized() -> MLXArray {
        dequantized(tokenRange: 0..<tokenCount)
    }

    func dequantized(tokenRange: Range<Int>) -> MLXArray {
        let unpackKernel = Gemma4PolarFastKernels.unpackKernel(bits: bits, dim: headDim, packedWidth: packedWidth)
        let roundedDim = ((headDim + 31) / 32) * 32
        let tokenCounts = MLXArray([UInt32(tokenRange.count), UInt32(tokenRange.lowerBound)])
        let rotated = unpackKernel(
            [packed, norms, centroids, tokenCounts],
            template: [
                ("Bits", bits),
                ("Dim", headDim),
                ("PackedWidth", packedWidth),
            ],
            grid: (roundedDim, packed.dim(1), packed.dim(0) * tokenRange.count),
            threadGroup: (32, 1, 1),
            outputShapes: [[packed.dim(0), packed.dim(1), tokenRange.count, headDim]],
            outputDTypes: [.float32]
        )[0]
        return MLX.matmul(rotated, rotation).asType(dtype)
    }
}

final class Gemma4PolarKVCache: Gemma4AttentionCache {
    private static let decodeChunkSize = 2_048

    private let configuration: Gemma4KVCacheQuantization
    private let maxSize: Int?

    private var leadingKeys: MLXArray?
    private var leadingValues: MLXArray?
    private var polarKeys: Gemma4PolarTensorState?
    private var polarValues: Gemma4PolarTensorState?

    private(set) var offset: Int = 0

    init(configuration: Gemma4KVCacheQuantization, maxSize: Int?) {
        self.configuration = configuration
        self.maxSize = maxSize
    }

    func currentState() -> (MLXArray, MLXArray)? {
        reconstructState()
    }

    func append(keys: MLXArray, values: MLXArray) {
        guard maxSize != nil else {
            appendUnbounded(keys: keys, values: values)
            return
        }

        let existing = reconstructState()
        let combinedKeys: MLXArray
        let combinedValues: MLXArray
        if let existing {
            combinedKeys = concatenated([existing.0, keys], axis: 2)
            combinedValues = concatenated([existing.1, values], axis: 2)
        } else {
            combinedKeys = keys
            combinedValues = values
        }

        let newOffset = offset + keys.dim(2)
        var keptKeys = combinedKeys
        var keptValues = combinedValues
        if let maxSize {
            let totalLength = combinedKeys.dim(2)
            if totalLength > maxSize {
                let start = totalLength - maxSize
                keptKeys = combinedKeys[0..., 0..., start..., 0...]
                keptValues = combinedValues[0..., 0..., start..., 0...]
            }
        }

        repartition(keys: keptKeys, values: keptValues, newOffset: newOffset)
    }

    private func appendUnbounded(keys: MLXArray, values: MLXArray) {
        let tokenCount = keys.dim(2)
        let newOffset = offset + tokenCount
        defer { offset = newOffset }

        guard let bits = configuration.keyBits else {
            appendLeading(keys: keys, values: values)
            return
        }

        let plainCount = max(0, min(tokenCount, configuration.quantizedStart - offset))
        if plainCount > 0 {
            appendLeading(
                keys: keys[0..., 0..., ..<plainCount, 0...],
                values: values[0..., 0..., ..<plainCount, 0...]
            )
        }

        guard plainCount < tokenCount else {
            return
        }

        appendPolar(
            keys: keys[0..., 0..., plainCount..., 0...],
            values: values[0..., 0..., plainCount..., 0...],
            bits: bits
        )
    }

    private func appendLeading(keys: MLXArray, values: MLXArray) {
        if let existingKeys = leadingKeys, let existingValues = leadingValues {
            leadingKeys = concatenated([existingKeys, keys], axis: 2)
            leadingValues = concatenated([existingValues, values], axis: 2)
        } else {
            leadingKeys = keys
            leadingValues = values
        }
    }

    private func appendPolar(keys: MLXArray, values: MLXArray, bits: Int) {
        if let polarKeys {
            self.polarKeys = polarKeys.appending(keys)
        } else {
            polarKeys = Gemma4PolarTensorState(source: keys, bits: bits)
        }

        if let polarValues {
            self.polarValues = polarValues.appending(values)
        } else {
            polarValues = Gemma4PolarTensorState(source: values, bits: bits)
        }
    }

    func fork() -> Gemma4AttentionCache {
        let copy = Gemma4PolarKVCache(configuration: configuration, maxSize: maxSize)
        copy.leadingKeys = leadingKeys
        copy.leadingValues = leadingValues
        copy.polarKeys = polarKeys
        copy.polarValues = polarValues
        copy.offset = offset
        return copy
    }

    static func reencoded(
        keys: MLXArray,
        values: MLXArray,
        configuration: Gemma4KVCacheQuantization,
        maxSize: Int?,
        offset: Int
    ) -> Gemma4PolarKVCache {
        let cache = Gemma4PolarKVCache(configuration: configuration, maxSize: maxSize)
        cache.repartition(keys: keys, values: values, newOffset: offset)
        return cache
    }

    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        if quantization == configuration {
            return fork()
        }
        guard let state = currentState() else { return nil }
        return makeGemma4AttentionCache(
            keys: state.0,
            values: state.1,
            offset: offset,
            maxSize: maxSize,
            quantization: quantization
        )
    }

    func evaluateStorage() {
        if let leadingKeys, let leadingValues {
            MLX.eval(leadingKeys, leadingValues)
        }
        if let polarKeys {
            MLX.eval(
                polarKeys.packed,
                polarKeys.norms,
                polarKeys.rotation,
                polarKeys.rotationTransposed,
                polarKeys.centroids
            )
        }
        if let polarValues {
            MLX.eval(
                polarValues.packed,
                polarValues.norms,
                polarValues.rotation,
                polarValues.rotationTransposed,
                polarValues.centroids
            )
        }
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4PolarKVCache],
              !typed.isEmpty,
              typed.allSatisfy({
                  $0.offset == offset
                      && $0.configuration == configuration
                      && $0.maxSize == maxSize
              }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4PolarKVCache(configuration: configuration, maxSize: maxSize)
        copy.repartition(
            keys: concatenated(states.map(\.0), axis: 0),
            values: concatenated(states.map(\.1), axis: 0),
            newOffset: offset
        )
        return copy
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0,
              let state = currentState(),
              state.0.dim(0) == count,
              state.1.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4PolarKVCache(configuration: configuration, maxSize: maxSize)
            copy.repartition(
                keys: state.0[index..<(index + 1), 0..., 0..., 0...],
                values: state.1[index..<(index + 1), 0..., 0..., 0...],
                newOffset: offset
            )
            return copy
        }
    }

    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        guard queries.dim(2) == 1 else {
            return nil
        }

        if let fused = fusedSpecializedAttention(queries: queries, repeats: repeats, scale: scale) {
            return fused
        }
        return chunkedSpecializedAttention(queries: queries, repeats: repeats, scale: scale)
    }

    func fusedSpecializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        guard Self.supportsFastKernels,
              queries.dim(2) == 1,
              leadingKeys == nil,
              leadingValues == nil,
              let polarKeys,
              let polarValues else {
            return nil
        }

        if let scoreValueOutput = scoreValueSpecializedAttention(
            queries: queries,
            keyState: polarKeys,
            valueState: polarValues,
            repeats: repeats,
            scale: scale
        ) {
            return scoreValueOutput
        }

        let queries32 = queries.asType(.float32)
        let rotatedQueries = MLX.matmul(queries32, polarKeys.rotationTransposed)
        var runningWeighted: MLXArray?
        var runningNormalizer: MLXArray?
        var runningMax: MLXArray?

        let totalTokens = polarKeys.tokenCount
        var start = 0
        while start < totalTokens {
            let end = min(start + Self.decodeChunkSize, totalTokens)
            applyPolarChunkFused(
                queries: rotatedQueries,
                keyState: polarKeys,
                valueState: polarValues,
                tokenRange: start..<end,
                repeats: repeats,
                scale: scale,
                runningWeighted: &runningWeighted,
                runningNormalizer: &runningNormalizer,
                runningMax: &runningMax
            )
            start = end
        }

        guard let runningWeighted, let runningNormalizer else {
            return nil
        }
        return MLX.matmul(runningWeighted / runningNormalizer, polarValues.rotation).asType(queries.dtype)
    }

    private func scoreValueSpecializedAttention(
        queries: MLXArray,
        keyState: Gemma4PolarTensorState,
        valueState: Gemma4PolarTensorState,
        repeats: Int,
        scale: Float
    ) -> MLXArray? {
        let totalTokens = keyState.tokenCount
        guard totalTokens > 0 else {
            return nil
        }

        let queries32 = queries.asType(.float32)
        let rotatedQueries = MLX.matmul(queries32, keyState.rotationTransposed)
        let tokenCounts = MLXArray([UInt32(totalTokens)])
        let scoreKernel = Gemma4PolarFastKernels.scoreKernel(
            bits: keyState.bits,
            dim: queries.dim(3),
            packedWidth: keyState.packedWidth,
            repeats: repeats
        )
        let scores = scoreKernel(
            [rotatedQueries, keyState.packed, keyState.norms, keyState.centroids, scale, tokenCounts],
            template: [
                ("Bits", keyState.bits),
                ("Dim", queries.dim(3)),
                ("PackedWidth", keyState.packedWidth),
                ("RepeatCount", repeats),
            ],
            grid: (32, queries.dim(1), queries.dim(0) * totalTokens),
            threadGroup: (32, 1, 1),
            outputShapes: [[queries.dim(0), queries.dim(1), 1, totalTokens]],
            outputDTypes: [.float32]
        )[0]

        let scoreMax = scores.max(axis: -1, keepDims: true)
        let weights = exp(scores - scoreMax)
        let normalizer = weights.sum(axis: -1, keepDims: true)
        let weightedValueKernel = Gemma4PolarFastKernels.weightedValueKernel(
            bits: valueState.bits,
            dim: queries.dim(3),
            packedWidth: valueState.packedWidth,
            repeats: repeats
        )
        let weighted = weightedValueKernel(
            [weights, valueState.packed, valueState.norms, valueState.centroids, tokenCounts],
            template: [
                ("Bits", valueState.bits),
                ("Dim", queries.dim(3)),
                ("PackedWidth", valueState.packedWidth),
                ("RepeatCount", repeats),
            ],
            grid: (32, queries.dim(1), queries.dim(0) * queries.dim(3)),
            threadGroup: (32, 1, 1),
            outputShapes: [[queries.dim(0), queries.dim(1), 1, queries.dim(3)]],
            outputDTypes: [.float32]
        )[0]

        return MLX.matmul(weighted / normalizer, valueState.rotation).asType(queries.dtype)
    }

    private func chunkedSpecializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        let queries32 = queries.asType(.float32)

        var runningWeighted: MLXArray?
        var runningNormalizer: MLXArray?
        var runningMax: MLXArray?

        if let leadingKeys, let leadingValues {
            applyChunk(
                queries: queries32,
                keys: leadingKeys,
                values: leadingValues,
                repeats: repeats,
                scale: scale,
                runningWeighted: &runningWeighted,
                runningNormalizer: &runningNormalizer,
                runningMax: &runningMax
            )
        }

        if let polarKeys, let polarValues {
            let totalTokens = polarKeys.tokenCount
            if totalTokens > 0 {
                var start = 0
                while start < totalTokens {
                    let end = min(start + Self.decodeChunkSize, totalTokens)
                    applyChunk(
                        queries: queries32,
                        keys: polarKeys.dequantized(tokenRange: start..<end),
                        values: polarValues.dequantized(tokenRange: start..<end),
                        repeats: repeats,
                        scale: scale,
                        runningWeighted: &runningWeighted,
                        runningNormalizer: &runningNormalizer,
                        runningMax: &runningMax
                    )
                    start = end
                }
            }
        }

        guard let runningWeighted, let runningNormalizer else {
            return nil
        }
        return (runningWeighted / runningNormalizer).asType(queries.dtype)
    }

    private func applyPolarChunkFused(
        queries: MLXArray,
        keyState: Gemma4PolarTensorState,
        valueState: Gemma4PolarTensorState,
        tokenRange: Range<Int>,
        repeats: Int,
        scale: Float,
        runningWeighted: inout MLXArray?,
        runningNormalizer: inout MLXArray?,
        runningMax: inout MLXArray?
    ) {
        let tokenCounts = MLXArray([UInt32(tokenRange.count), UInt32(tokenRange.lowerBound)])
        let fusedChunkDecodeKernel = Gemma4PolarFastKernels.fusedChunkDecodeKernel(
            bits: keyState.bits,
            dim: queries.dim(3),
            packedWidth: keyState.packedWidth,
            repeats: repeats
        )

        let weightedAndStats = fusedChunkDecodeKernel(
            [queries, keyState.packed, keyState.norms, valueState.packed, valueState.norms, keyState.centroids, scale, tokenCounts],
            template: [
                ("Bits", keyState.bits),
                ("Dim", queries.dim(3)),
                ("PackedWidth", keyState.packedWidth),
                ("RepeatCount", repeats),
            ],
            grid: (32, queries.dim(1), queries.dim(0) * queries.dim(3)),
            threadGroup: (32, 1, 1),
            outputShapes: [
                [queries.dim(0), queries.dim(1), 1, queries.dim(3)],
                [queries.dim(0), queries.dim(1), 1, 1],
                [queries.dim(0), queries.dim(1), 1, 1],
            ],
            outputDTypes: [.float32, .float32, .float32]
        )
        let chunkWeighted = weightedAndStats[0]
        let chunkNormalizer = weightedAndStats[1]
        let chunkMax = weightedAndStats[2]

        guard let currentWeighted = runningWeighted,
              let currentNormalizer = runningNormalizer,
              let currentMax = runningMax else {
            runningWeighted = chunkWeighted
            runningNormalizer = chunkNormalizer
            runningMax = chunkMax
            return
        }

        let mergedMax = MLX.maximum(currentMax, chunkMax)
        let currentScale = exp(currentMax - mergedMax)
        let chunkScale = exp(chunkMax - mergedMax)
        runningWeighted = currentWeighted * currentScale + chunkWeighted * chunkScale
        runningNormalizer = currentNormalizer * currentScale + chunkNormalizer * chunkScale
        runningMax = mergedMax
    }

    private func repartition(keys: MLXArray, values: MLXArray, newOffset: Int) {
        let keptLength = keys.dim(2)
        let keptStart = newOffset - keptLength
        let plainCount = max(0, min(keptLength, configuration.quantizedStart - keptStart))

        if plainCount > 0 {
            leadingKeys = keys[0..., 0..., ..<plainCount, 0...]
            leadingValues = values[0..., 0..., ..<plainCount, 0...]
        } else {
            leadingKeys = nil
            leadingValues = nil
        }

        let polarLength = keptLength - plainCount
        if polarLength > 0, let bits = configuration.keyBits {
            polarKeys = Gemma4PolarTensorState(source: keys[0..., 0..., plainCount..., 0...], bits: bits)
            polarValues = Gemma4PolarTensorState(source: values[0..., 0..., plainCount..., 0...], bits: bits)
        } else {
            polarKeys = nil
            polarValues = nil
        }

        offset = newOffset
    }

    private func reconstructState() -> (MLXArray, MLXArray)? {
        var keyParts: [MLXArray] = []
        var valueParts: [MLXArray] = []

        if let leadingKeys, let leadingValues {
            keyParts.append(leadingKeys)
            valueParts.append(leadingValues)
        }

        if let polarKeys, let polarValues {
            keyParts.append(polarKeys.dequantized())
            valueParts.append(polarValues.dequantized())
        }

        guard !keyParts.isEmpty else {
            return nil
        }

        if keyParts.count == 1 {
            return (keyParts[0], valueParts[0])
        }

        return (concatenated(keyParts, axis: 2), concatenated(valueParts, axis: 2))
    }

    private func applyChunk(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        repeats: Int,
        scale: Float,
        runningWeighted: inout MLXArray?,
        runningNormalizer: inout MLXArray?,
        runningMax: inout MLXArray?
    ) {
        guard keys.dim(2) > 0 else {
            return
        }

        var broadcastKeys = keys.asType(.float32)
        var broadcastValues = values.asType(.float32)
        if repeats > 1 {
            broadcastKeys = MLX.repeated(broadcastKeys, count: repeats, axis: 1)
            broadcastValues = MLX.repeated(broadcastValues, count: repeats, axis: 1)
        }

        let scores = MLX.matmul(queries, broadcastKeys.transposed(0, 1, 3, 2)) * MLXArray(scale)
        let chunkMax = scores.max(axis: -1, keepDims: true)
        let shifted = exp(scores - chunkMax)
        let chunkWeighted = MLX.matmul(shifted, broadcastValues)
        let chunkNormalizer = shifted.sum(axis: -1, keepDims: true)

        guard let currentWeighted = runningWeighted,
              let currentNormalizer = runningNormalizer,
              let currentMax = runningMax else {
            runningWeighted = chunkWeighted
            runningNormalizer = chunkNormalizer
            runningMax = chunkMax
            return
        }

        let mergedMax = MLX.maximum(currentMax, chunkMax)
        let currentScale = exp(currentMax - mergedMax)
        let chunkScale = exp(chunkMax - mergedMax)
        runningWeighted = currentWeighted * currentScale + chunkWeighted * chunkScale
        runningNormalizer = currentNormalizer * currentScale + chunkNormalizer * chunkScale
        runningMax = mergedMax
    }

    private static var supportsFastKernels: Bool {
        Device.defaultDevice().deviceType == .gpu
    }
}

final class Gemma4QuantizedKVCache: Gemma4AttentionCache {
    private static let decodeChunkSize = 2_048

    private let configuration: Gemma4KVCacheQuantization
    private let maxSize: Int?

    private var leadingKeys: MLXArray?
    private var leadingValues: MLXArray?
    private var quantizedKeys: Gemma4QuantizedTensorState?
    private var quantizedValues: Gemma4QuantizedTensorState?

    private(set) var offset: Int = 0

    init(configuration: Gemma4KVCacheQuantization, maxSize: Int?) {
        self.configuration = configuration
        self.maxSize = maxSize
    }

    func currentState() -> (MLXArray, MLXArray)? {
        reconstructState()
    }

    func append(keys: MLXArray, values: MLXArray) {
        guard maxSize != nil else {
            appendUnbounded(keys: keys, values: values)
            return
        }

        let existing = reconstructState()

        let combinedKeys: MLXArray
        let combinedValues: MLXArray
        if let existing {
            combinedKeys = concatenated([existing.0, keys], axis: 2)
            combinedValues = concatenated([existing.1, values], axis: 2)
        } else {
            combinedKeys = keys
            combinedValues = values
        }

        let newOffset = offset + keys.dim(2)
        var keptKeys = combinedKeys
        var keptValues = combinedValues
        if let maxSize {
            let totalLength = combinedKeys.dim(2)
            if totalLength > maxSize {
                let start = totalLength - maxSize
                keptKeys = combinedKeys[0..., 0..., start..., 0...]
                keptValues = combinedValues[0..., 0..., start..., 0...]
            }
        }

        repartition(keys: keptKeys, values: keptValues, newOffset: newOffset)
    }

    private func appendUnbounded(keys: MLXArray, values: MLXArray) {
        let tokenCount = keys.dim(2)
        let newOffset = offset + tokenCount
        defer { offset = newOffset }

        guard let keyBits = configuration.keyBits,
              let valueBits = configuration.valueBits else {
            appendLeading(keys: keys, values: values)
            return
        }

        let plainCount = max(0, min(tokenCount, configuration.quantizedStart - offset))
        if plainCount > 0 {
            appendLeading(
                keys: keys[0..., 0..., ..<plainCount, 0...],
                values: values[0..., 0..., ..<plainCount, 0...]
            )
        }

        guard plainCount < tokenCount else {
            return
        }

        appendQuantized(
            keys: keys[0..., 0..., plainCount..., 0...],
            values: values[0..., 0..., plainCount..., 0...],
            keyBits: keyBits,
            valueBits: valueBits
        )
    }

    private func appendLeading(keys: MLXArray, values: MLXArray) {
        if let existingKeys = leadingKeys, let existingValues = leadingValues {
            leadingKeys = concatenated([existingKeys, keys], axis: 2)
            leadingValues = concatenated([existingValues, values], axis: 2)
        } else {
            leadingKeys = keys
            leadingValues = values
        }
    }

    private func appendQuantized(keys: MLXArray, values: MLXArray, keyBits: Int, valueBits: Int) {
        if let quantizedKeys {
            self.quantizedKeys = quantizedKeys.appending(keys)
        } else {
            quantizedKeys = Gemma4QuantizedTensorState(
                source: keys,
                groupSize: configuration.groupSize,
                bits: keyBits
            )
        }

        if let quantizedValues {
            self.quantizedValues = quantizedValues.appending(values)
        } else {
            quantizedValues = Gemma4QuantizedTensorState(
                source: values,
                groupSize: configuration.groupSize,
                bits: valueBits
            )
        }
    }

    func fork() -> Gemma4AttentionCache {
        let copy = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
        copy.leadingKeys = leadingKeys
        copy.leadingValues = leadingValues
        copy.quantizedKeys = quantizedKeys
        copy.quantizedValues = quantizedValues
        copy.offset = offset
        return copy
    }

    static func reencoded(
        keys: MLXArray,
        values: MLXArray,
        configuration: Gemma4KVCacheQuantization,
        maxSize: Int?,
        offset: Int
    ) -> Gemma4QuantizedKVCache {
        let cache = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
        cache.repartition(keys: keys, values: values, newOffset: offset)
        return cache
    }

    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        if quantization == configuration {
            return fork()
        }
        guard let state = currentState() else { return nil }
        return makeGemma4AttentionCache(
            keys: state.0,
            values: state.1,
            offset: offset,
            maxSize: maxSize,
            quantization: quantization
        )
    }

    func evaluateStorage() {
        if let leadingKeys, let leadingValues {
            MLX.eval(leadingKeys, leadingValues)
        }
        if let quantizedKeys {
            if let biases = quantizedKeys.biases {
                MLX.eval(quantizedKeys.weight, quantizedKeys.scales, biases)
            } else {
                MLX.eval(quantizedKeys.weight, quantizedKeys.scales)
            }
        }
        if let quantizedValues {
            if let biases = quantizedValues.biases {
                MLX.eval(quantizedValues.weight, quantizedValues.scales, biases)
            } else {
                MLX.eval(quantizedValues.weight, quantizedValues.scales)
            }
        }
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4QuantizedKVCache],
              !typed.isEmpty,
              typed.allSatisfy({
                  $0.offset == offset
                      && $0.configuration == configuration
                      && $0.maxSize == maxSize
              }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
        copy.repartition(
            keys: concatenated(states.map(\.0), axis: 0),
            values: concatenated(states.map(\.1), axis: 0),
            newOffset: offset
        )
        return copy
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0,
              let state = currentState(),
              state.0.dim(0) == count,
              state.1.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
            copy.repartition(
                keys: state.0[index..<(index + 1), 0..., 0..., 0...],
                values: state.1[index..<(index + 1), 0..., 0..., 0...],
                newOffset: offset
            )
            return copy
        }
    }

    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        guard queries.dim(2) == 1 else {
            return nil
        }

        if let fused = fusedSpecializedAttention(queries: queries, repeats: repeats, scale: scale) {
            return fused
        }
        return chunkedSpecializedAttention(queries: queries, repeats: repeats, scale: scale)
    }

    func fusedSpecializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        guard Self.supportsFastKernels else {
            return nil
        }
        guard queries.dim(2) == 1 else {
            return nil
        }

        let queries32 = queries.asType(.float32)
        var runningWeighted: MLXArray?
        var runningNormalizer: MLXArray?
        var runningMax: MLXArray?

        if let leadingKeys, let leadingValues {
            applyChunk(
                queries: queries32,
                keys: leadingKeys,
                values: leadingValues,
                repeats: repeats,
                scale: scale,
                runningWeighted: &runningWeighted,
                runningNormalizer: &runningNormalizer,
                runningMax: &runningMax
            )
        }

        if let quantizedKeys, let quantizedValues {
            let totalTokens = quantizedKeys.tokenCount
            if totalTokens > 0 {
                var start = 0
                while start < totalTokens {
                    let end = min(start + Self.decodeChunkSize, totalTokens)
                    guard applyQuantizedChunkFused(
                        queries: queries32,
                        keyState: quantizedKeys,
                        valueState: quantizedValues,
                        tokenRange: start..<end,
                        repeats: repeats,
                        scale: scale,
                        runningWeighted: &runningWeighted,
                        runningNormalizer: &runningNormalizer,
                        runningMax: &runningMax
                    ) else {
                        return nil
                    }
                    start = end
                }
            }
        }

        guard let runningWeighted, let runningNormalizer else {
            return nil
        }
        return (runningWeighted / runningNormalizer).asType(queries.dtype)
    }

    private func chunkedSpecializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        let queries32 = queries.asType(.float32)

        var runningWeighted: MLXArray?
        var runningNormalizer: MLXArray?
        var runningMax: MLXArray?

        if let leadingKeys, let leadingValues {
            applyChunk(
                queries: queries32,
                keys: leadingKeys,
                values: leadingValues,
                repeats: repeats,
                scale: scale,
                runningWeighted: &runningWeighted,
                runningNormalizer: &runningNormalizer,
                runningMax: &runningMax
            )
        }

        if let quantizedKeys, let quantizedValues {
            let totalTokens = quantizedKeys.tokenCount
            if totalTokens > 0 {
                var start = 0
                while start < totalTokens {
                    let end = min(start + Self.decodeChunkSize, totalTokens)
                    let chunkKeys = quantizedKeys.dequantized(tokenRange: start..<end)
                    let chunkValues = quantizedValues.dequantized(tokenRange: start..<end)
                    applyChunk(
                        queries: queries32,
                        keys: chunkKeys,
                        values: chunkValues,
                        repeats: repeats,
                        scale: scale,
                        runningWeighted: &runningWeighted,
                        runningNormalizer: &runningNormalizer,
                        runningMax: &runningMax
                    )
                    start = end
                }
            }
        }

        guard let runningWeighted, let runningNormalizer else {
            return nil
        }
        return (runningWeighted / runningNormalizer).asType(queries.dtype)
    }

    private func applyQuantizedChunkFused(
        queries: MLXArray,
        keyState: Gemma4QuantizedTensorState,
        valueState: Gemma4QuantizedTensorState,
        tokenRange: Range<Int>,
        repeats: Int,
        scale: Float,
        runningWeighted: inout MLXArray?,
        runningNormalizer: inout MLXArray?,
        runningMax: inout MLXArray?
    ) -> Bool {
        guard let keyBiases = keyState.biases,
              let valueBiases = valueState.biases else {
            return false
        }
        guard let fusedChunkDecodeKernel = makeFusedChunkDecodeKernel(
            keyState: keyState,
            valueState: valueState,
            repeats: repeats,
            dim: queries.dim(3)
        ) else {
            return false
        }

        let tokenCounts = MLXArray([UInt32(tokenRange.count), UInt32(tokenRange.lowerBound)])
        let weightedAndStats = fusedChunkDecodeKernel(
            [queries, keyState.weight, keyState.scales, keyBiases, valueState.weight, valueState.scales, valueBiases, scale, tokenCounts],
            template: fusedChunkDecodeTemplateArguments(
                keyState: keyState,
                valueState: valueState,
                repeats: repeats,
                dim: queries.dim(3)
            ),
            grid: (32, queries.dim(1), queries.dim(0) * queries.dim(3)),
            threadGroup: (32, 1, 1),
            outputShapes: [
                [queries.dim(0), queries.dim(1), 1, queries.dim(3)],
                [queries.dim(0), queries.dim(1), 1, 1],
                [queries.dim(0), queries.dim(1), 1, 1],
            ],
            outputDTypes: [.float32, .float32, .float32]
        )
        let chunkWeighted = weightedAndStats[0]
        let chunkNormalizer = weightedAndStats[1]
        let chunkMax = weightedAndStats[2]

        guard let currentWeighted = runningWeighted,
              let currentNormalizer = runningNormalizer,
              let currentMax = runningMax else {
            runningWeighted = chunkWeighted
            runningNormalizer = chunkNormalizer
            runningMax = chunkMax
            return true
        }

        let mergedMax = MLX.maximum(currentMax, chunkMax)
        let currentScale = exp(currentMax - mergedMax)
        let chunkScale = exp(chunkMax - mergedMax)
        runningWeighted = currentWeighted * currentScale + chunkWeighted * chunkScale
        runningNormalizer = currentNormalizer * currentScale + chunkNormalizer * chunkScale
        runningMax = mergedMax
        return true
    }

    private func makeScoreKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        return Gemma4AffineFastKernels.scoreKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    private func makeWeightedValueKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.weightedValueKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    private func makeWeightedValueFromScoresKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.weightedValueFromScoresKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    private func makeWeightedValueAndNormalizerFromScoresKernel(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.weightedValueAndNormalizerFromScoresKernel(
            bits: state.bits,
            groupSize: state.groupSize,
            dim: dim,
            packedWidth: state.packedWidth,
            groupCount: state.groupCount,
            repeats: repeats
        )
    }

    private func makeFusedChunkDecodeKernel(
        keyState: Gemma4QuantizedTensorState,
        valueState: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> MLXFast.MLXFastKernel? {
        Gemma4AffineFastKernels.fusedChunkDecodeKernel(
            keyBits: keyState.bits,
            valueBits: valueState.bits,
            groupSize: keyState.groupSize,
            dim: dim,
            keyPackedWidth: keyState.packedWidth,
            valuePackedWidth: valueState.packedWidth,
            keyGroupCount: keyState.groupCount,
            valueGroupCount: valueState.groupCount,
            repeats: repeats
        )
    }

    private func scoreTemplateArguments(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("Bits", state.bits),
            ("GroupSize", state.groupSize),
            ("Dim", dim),
            ("PackedWidth", state.packedWidth),
            ("GroupCount", state.groupCount),
            ("RepeatCount", repeats),
        ]
    }

    private func valueTemplateArguments(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("Bits", state.bits),
            ("GroupSize", state.groupSize),
            ("Dim", dim),
            ("PackedWidth", state.packedWidth),
            ("GroupCount", state.groupCount),
            ("RepeatCount", repeats),
        ]
    }

    private func weightedValueFromScoresTemplateArguments(
        state: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("Bits", state.bits),
            ("GroupSize", state.groupSize),
            ("Dim", dim),
            ("PackedWidth", state.packedWidth),
            ("GroupCount", state.groupCount),
            ("RepeatCount", repeats),
        ]
    }

    private func fusedChunkDecodeTemplateArguments(
        keyState: Gemma4QuantizedTensorState,
        valueState: Gemma4QuantizedTensorState,
        repeats: Int,
        dim: Int
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("KeyBits", keyState.bits),
            ("ValueBits", valueState.bits),
            ("GroupSize", keyState.groupSize),
            ("Dim", dim),
            ("KeyPackedWidth", keyState.packedWidth),
            ("ValuePackedWidth", valueState.packedWidth),
            ("KeyGroupCount", keyState.groupCount),
            ("ValueGroupCount", valueState.groupCount),
            ("RepeatCount", repeats),
        ]
    }

    private func repartition(keys: MLXArray, values: MLXArray, newOffset: Int) {
        let keptLength = keys.dim(2)
        let keptStart = newOffset - keptLength
        let plainCount = max(0, min(keptLength, configuration.quantizedStart - keptStart))

        if plainCount > 0 {
            leadingKeys = keys[0..., 0..., ..<plainCount, 0...]
            leadingValues = values[0..., 0..., ..<plainCount, 0...]
        } else {
            leadingKeys = nil
            leadingValues = nil
        }

        let quantizedLength = keptLength - plainCount
        if quantizedLength > 0,
           let keyBits = configuration.keyBits,
           let valueBits = configuration.valueBits {
            quantizedKeys = Gemma4QuantizedTensorState(
                source: keys[0..., 0..., plainCount..., 0...],
                groupSize: configuration.groupSize,
                bits: keyBits
            )
            quantizedValues = Gemma4QuantizedTensorState(
                source: values[0..., 0..., plainCount..., 0...],
                groupSize: configuration.groupSize,
                bits: valueBits
            )
        } else {
            quantizedKeys = nil
            quantizedValues = nil
        }

        offset = newOffset
    }

    private func reconstructState() -> (MLXArray, MLXArray)? {
        var keyParts: [MLXArray] = []
        var valueParts: [MLXArray] = []

        if let leadingKeys, let leadingValues {
            keyParts.append(leadingKeys)
            valueParts.append(leadingValues)
        }

        if let quantizedKeys, let quantizedValues {
            keyParts.append(quantizedKeys.dequantized())
            valueParts.append(quantizedValues.dequantized())
        }

        guard !keyParts.isEmpty else {
            return nil
        }

        if keyParts.count == 1 {
            return (keyParts[0], valueParts[0])
        }

        return (concatenated(keyParts, axis: 2), concatenated(valueParts, axis: 2))
    }

    private func applyChunk(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        repeats: Int,
        scale: Float,
        runningWeighted: inout MLXArray?,
        runningNormalizer: inout MLXArray?,
        runningMax: inout MLXArray?
    ) {
        guard keys.dim(2) > 0 else {
            return
        }

        var broadcastKeys = keys.asType(.float32)
        var broadcastValues = values.asType(.float32)
        if repeats > 1 {
            broadcastKeys = MLX.repeated(broadcastKeys, count: repeats, axis: 1)
            broadcastValues = MLX.repeated(broadcastValues, count: repeats, axis: 1)
        }

        let scores = MLX.matmul(queries, broadcastKeys.transposed(0, 1, 3, 2)) * MLXArray(scale)
        let chunkMax = scores.max(axis: -1, keepDims: true)
        let shifted = exp(scores - chunkMax)
        let chunkWeighted = MLX.matmul(shifted, broadcastValues)
        let chunkNormalizer = shifted.sum(axis: -1, keepDims: true)

        guard let currentWeighted = runningWeighted,
              let currentNormalizer = runningNormalizer,
              let currentMax = runningMax else {
            runningWeighted = chunkWeighted
            runningNormalizer = chunkNormalizer
            runningMax = chunkMax
            return
        }

        let mergedMax = MLX.maximum(currentMax, chunkMax)
        let currentScale = exp(currentMax - mergedMax)
        let chunkScale = exp(chunkMax - mergedMax)
        runningWeighted = currentWeighted * currentScale + chunkWeighted * chunkScale
        runningNormalizer = currentNormalizer * currentScale + chunkNormalizer * chunkScale
        runningMax = mergedMax
    }

    private static var supportsFastKernels: Bool {
        Device.defaultDevice().deviceType == .gpu
    }
}
