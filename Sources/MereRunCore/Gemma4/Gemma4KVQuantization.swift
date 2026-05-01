import Foundation
import MLX
import MLXFast

public enum Gemma4KVQuantizationScheme: String, Sendable, Hashable {
    case uniform
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
            guard abs(bits.rounded() - bits) < 0.000_001 else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 uniform KV quantization requires an integer bit width. Use turboquant for fractional .5 widths.")
            }
        case .turboquant:
            guard abs(roundedHalf - bits) < 0.000_001 else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 turboquant currently supports integer and .5 bit widths (received \(bits)).")
            }
        }

        return self
    }

    var keyBits: Int? {
        guard let bits else { return nil }
        switch scheme {
        case .uniform:
            return Int(bits.rounded())
        case .turboquant:
            return Int(floor(bits))
        }
    }

    var valueBits: Int? {
        guard let bits else { return nil }
        switch scheme {
        case .uniform:
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
            inputNames: ["queries", "packed", "scales", "biases", "scale"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                auto token_count = packed_shape[2];
                auto head_count = queries_shape[1];
                auto batch = n / token_count;
                auto token = n % token_count;
                if (batch >= queries_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto query_ptr = queries + ((batch * head_count + head_idx) * Dim);
                auto packed_ptr = packed + (((batch * packed_shape[1] + kv_head) * token_count + token) * PackedWidth);
                auto scales_ptr = scales + (((batch * scales_shape[1] + kv_head) * token_count + token) * GroupCount);
                auto biases_ptr = biases + (((batch * biases_shape[1] + kv_head) * token_count + token) * GroupCount);

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
            inputNames: ["weights", "packed", "scales", "biases"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                auto token_count = packed_shape[2];
                auto head_count = weights_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= weights_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto weights_ptr = weights + ((batch * head_count + head_idx) * token_count);
                auto packed_ptr = packed + ((batch * packed_shape[1] + kv_head) * token_count * PackedWidth);
                auto scales_ptr = scales + ((batch * scales_shape[1] + kv_head) * token_count * GroupCount);
                auto biases_ptr = biases + ((batch * biases_shape[1] + kv_head) * token_count * GroupCount);

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
            inputNames: ["scores", "packed", "scales", "biases", "score_max"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                auto token_count = packed_shape[2];
                auto head_count = scores_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= scores_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto scores_ptr = scores + ((batch * head_count + head_idx) * token_count);
                auto packed_ptr = packed + ((batch * packed_shape[1] + kv_head) * token_count * PackedWidth);
                auto scales_ptr = scales + ((batch * scales_shape[1] + kv_head) * token_count * GroupCount);
                auto biases_ptr = biases + ((batch * biases_shape[1] + kv_head) * token_count * GroupCount);
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
            inputNames: ["scores", "packed", "scales", "biases", "score_max"],
            outputNames: ["weighted", "normalizer"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                auto token_count = packed_shape[2];
                auto head_count = scores_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= scores_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto scores_ptr = scores + ((batch * head_count + head_idx) * token_count);
                auto packed_ptr = packed + ((batch * packed_shape[1] + kv_head) * token_count * PackedWidth);
                auto scales_ptr = scales + ((batch * scales_shape[1] + kv_head) * token_count * GroupCount);
                auto biases_ptr = biases + ((batch * biases_shape[1] + kv_head) * token_count * GroupCount);
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
            ],
            outputNames: ["weighted", "normalizer", "score_max"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_idx = thread_position_in_grid.y;
                auto n = thread_position_in_grid.z;

                auto token_count = key_packed_shape[2];
                auto head_count = queries_shape[1];
                auto batch = n / Dim;
                auto dim_idx = n % Dim;
                if (batch >= queries_shape[0] || head_idx >= head_count) {
                    return;
                }

                auto kv_head = head_idx / RepeatCount;
                auto query_ptr = queries + ((batch * head_count + head_idx) * Dim);

                auto key_packed_ptr = key_packed + ((batch * key_packed_shape[1] + kv_head) * token_count * KeyPackedWidth);
                auto key_scales_ptr = key_scales + ((batch * key_scales_shape[1] + kv_head) * token_count * KeyGroupCount);
                auto key_biases_ptr = key_biases + ((batch * key_biases_shape[1] + kv_head) * token_count * KeyGroupCount);

                auto value_packed_ptr = value_packed + ((batch * value_packed_shape[1] + kv_head) * token_count * ValuePackedWidth);
                auto value_scales_ptr = value_scales + ((batch * value_scales_shape[1] + kv_head) * token_count * ValueGroupCount);
                auto value_biases_ptr = value_biases + ((batch * value_biases_shape[1] + kv_head) * token_count * ValueGroupCount);

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

final class Gemma4QuantizedTensorState {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let dtype: DType

    var tokenCount: Int {
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
    }

    func dequantized() -> MLXArray {
        MLX.dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: dtype
        )
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

        let packedKeys = keyState.weight[0..., 0..., tokenRange, 0...]
        let keyScales = keyState.scales[0..., 0..., tokenRange, 0...]
        let keyBiasChunk = keyBiases[0..., 0..., tokenRange, 0...]
        let packedValues = valueState.weight[0..., 0..., tokenRange, 0...]
        let valueScales = valueState.scales[0..., 0..., tokenRange, 0...]
        let valueBiasChunk = valueBiases[0..., 0..., tokenRange, 0...]

        let weightedAndStats = fusedChunkDecodeKernel(
            [queries, packedKeys, keyScales, keyBiasChunk, packedValues, valueScales, valueBiasChunk, scale],
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
