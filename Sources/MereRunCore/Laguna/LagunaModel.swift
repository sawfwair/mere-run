import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

enum LagunaMoEAccelerationPolicy {
    private static let m5MaxDefaultsEnabled: Bool = {
        #if os(macOS)
        Device.defaultDevice().deviceType == .gpu
            && GPU.deviceInfo().architecture == "applegpu_g17s"
        #else
        false
        #endif
    }()

    private static let defaultDecodeNVFP4RowsPerSIMDGroup: Int = {
        #if os(macOS)
        defaultDecodeRowsPerSIMDGroup(
            architecture: Device.defaultDevice().deviceType == .gpu
                ? GPU.deviceInfo().architecture
                : nil
        )
        #else
        4
        #endif
    }()

    static let sortedRoutingEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_SORTED_MOE",
        default: true
    )
    static let fusedNVFP4MoEEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_NVFP4_MOE",
        default: true
    )
    static let fastSortedInverseEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FAST_SORTED_INVERSE",
        default: true
    )
    static let rankedPrefillRouteStagingEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_RANKED_PREFILL_ROUTE_STAGING",
        default: true
    )
    static let fusedSortedNVFP4MoEEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_SORTED_NVFP4_MOE",
        default: true
    )
    static let fusedSortedNVFP4DownEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_SORTED_NVFP4_DOWN",
        default: true
    )
    static let fusedRoutedSharedDownResidualEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_FUSED_ROUTED_SHARED_DOWN_RESIDUAL",
        default: m5MaxDefaultsEnabled
    )
    static let prefillExpertPairwiseScaleReuseEnabled = booleanEnvironment(
        "MERERUN_LAGUNA_PREFILL_EXPERT_PAIRWISE_SCALES",
        default: m5MaxDefaultsEnabled
    )
    static let decodeNVFP4RowsPerSIMDGroup = decodeRowsPerSIMDGroup(
        ProcessInfo.processInfo.environment[
            "MERERUN_LAGUNA_DECODE_NVFP4_ROWS_PER_SIMDGROUP"
        ],
        default: defaultDecodeNVFP4RowsPerSIMDGroup
    )
    static let fusedSortedMinimumSequenceLength = 64

    static func parseBoolean(_ raw: String?, default defaultValue: Bool) -> Bool {
        guard let raw else { return defaultValue }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return defaultValue
        }
    }

    private static func booleanEnvironment(
        _ name: String,
        default defaultValue: Bool
    ) -> Bool {
        parseBoolean(ProcessInfo.processInfo.environment[name], default: defaultValue)
    }

    static func decodeRowsPerSIMDGroup(
        _ raw: String?,
        default defaultValue: Int
    ) -> Int {
        guard let raw,
              let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              value == 1 || value == 2 || value == 4 else {
            return defaultValue
        }
        return value
    }

    static func defaultDecodeRowsPerSIMDGroup(architecture: String?) -> Int {
        architecture == "applegpu_g17s" ? 2 : 4
    }

    static func decodeRowsPerSIMDGroup(
        hiddenSize: Int,
        intermediateSize: Int,
        topK: Int,
        xsCandidate: Int
    ) -> Int {
        guard hiddenSize == 2_048,
              intermediateSize == 512,
              topK == 8 else {
            return 4
        }
        return xsCandidate
    }
}

/// Proves that every adjacent pair in an NVFP4 group-16 scale plane is
/// byte-identical except for explicitly preserved pairs. The Laguna XS expert
/// tensors shipped by Poolside were produced by a simdgroup-wide quantizer:
/// the two group-16 halves of each 32-weight span therefore share one scale,
/// while the first pair may carry the quantizer's duplicated-write exception.
///
/// This is a fail-closed, input-independent certificate over the weights that
/// were actually loaded. A false result keeps the stock prefill loader. A true
/// result permits the M5 kernel to load the even scale byte once and broadcast
/// it to the adjacent SIMD lane without changing a single dequantized value.
func lagunaNVFP4AdjacentScalePairsCertified(
    _ scales: MLXArray,
    allowedFlatPairs: Set<Int> = [0]
) -> Bool {
    let pairCount = scales.size / 2
    guard scales.dtype == .uint8,
          scales.ndim >= 1,
          scales.size.isMultiple(of: 2),
          scales.dim(-1).isMultiple(of: 2),
          allowedFlatPairs.allSatisfy({ $0 >= 0 && $0 < pairCount }) else {
        return false
    }

    let pairs = contiguous(scales).reshaped([pairCount, 2])
    let mismatch = (pairs[0..., 0] .!= pairs[0..., 1]).asType(.int32)
    var violations = mismatch.sum()
    for index in allowedFlatPairs {
        violations = violations - mismatch[index]
    }
    return violations.item(Int32.self) == 0
}

enum LagunaGraphAccelerationPolicy {
    private static let m5MaxDefaultsEnabled: Bool = {
        #if os(macOS)
        Device.defaultDevice().deviceType == .gpu
            && GPU.deviceInfo().architecture == "applegpu_g17s"
        #else
        false
        #endif
    }()

    static let sharedAttentionMasksEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_SHARED_ATTENTION_MASKS"],
        default: true
    )
    static let prefillAsyncLadderStride = parseLadderStride(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_PREFILL_ASYNC_LADDER"],
        default: 8
    )
    static let prefillFusedResidualRMSNormEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_PREFILL_FUSED_RESIDUAL_RMSNORM"],
        default: m5MaxDefaultsEnabled
    )
    static let prefillQKNormRoPEEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_PREFILL_QK_NORM_ROPE"],
        default: m5MaxDefaultsEnabled
    )
    static let terminalPrefillRowEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_TERMINAL_PREFILL_ROW"],
        default: m5MaxDefaultsEnabled
    )
    static let terminalPrefillProjectionBanksEnabled = parseBoolean(
        ProcessInfo.processInfo.environment[
            "MERERUN_LAGUNA_TERMINAL_PREFILL_PROJECTION_BANKS"
        ],
        default: m5MaxDefaultsEnabled
    )
    static let nativeAffineQKVEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_QKV"],
        default: m5MaxDefaultsEnabled
    )
    static let nativeAffineQKVLayerCount = nativeAffineQKVEnabled
        ? parseLayerCount(
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NATIVE_AFFINE_QKV_LAYERS"],
            default: 28
        )
        : 0

    static func parseBoolean(_ raw: String?, default defaultValue: Bool) -> Bool {
        LagunaMoEAccelerationPolicy.parseBoolean(raw, default: defaultValue)
    }

    static func parseLadderStride(_ raw: String?, default defaultValue: Int) -> Int {
        guard let raw else { return defaultValue }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.isEmpty || normalized == "off" || normalized == "0" {
            return 0
        }
        guard let value = Int(normalized), (1...40).contains(value) else {
            return defaultValue
        }
        return value
    }

    static func parseLayerCount(_ raw: String?, default defaultValue: Int) -> Int {
        guard let raw,
              let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return defaultValue
        }
        return min(max(value, 0), 40)
    }

    static func usesNativeAffineQKV(layerIndex: Int) -> Bool {
        layerIndex >= 0 && layerIndex < nativeAffineQKVLayerCount
    }
}

struct LagunaNativeAffineWeight {
    let packedCodes: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let originalShape: [Int]

    var arrays: [MLXArray] { [packedCodes, scales, biases] }
}

func lagunaNativeAffineWeight(_ weight: MLXArray) -> LagunaNativeAffineWeight? {
    guard weight.dtype == .bfloat16,
          weight.ndim == 2,
          weight.dim(1).isMultiple(of: 32) else {
        return nil
    }
    let quantizedWeight = MLX.quantized(
        weight,
        groupSize: 32,
        bits: 8,
        mode: .affine
    )
    guard let biases = quantizedWeight.biases else { return nil }
    return LagunaNativeAffineWeight(
        packedCodes: quantizedWeight.wq,
        scales: quantizedWeight.scales,
        biases: biases,
        originalShape: weight.shape
    )
}

enum LagunaFusedPrefill {
    static let ropeAngleAtlasLength = 4_096

    enum QKNormRoPEKind {
        case fullYaRN
        case sliding
    }

    static func residualRMSNorm(
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

    static func qkNormRoPE(
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
    private static let residualRMSNormKernel = MLXFast.metalKernel(
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

    private static let prefillSlidingQKNormRoPEKernel = MLXFast.metalKernel(
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

    private static let prefillFullQKNormYaRNKernel = MLXFast.metalKernel(
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

final class LagunaRoPE: Module, OffsetLayer {
    private let dimensions: Int
    private let traditional: Bool
    private let magnitudeScale: Float
    private let base: Float?
    private let frequencies: MLXArray?
    let prefillFusionKind: LagunaFusedPrefill.QKNormRoPEKind?

    init(headDim: Int, parameters: LagunaRopeParameters) {
        let resolvedDimensions = max(2, Int(Float(headDim) * parameters.partialRotaryFactor))
        self.dimensions = resolvedDimensions
        self.traditional = false

        if parameters.ropeType == "yarn" {
            let factor = parameters.factor ?? 1
            let originalContext = parameters.originalMaxPositionEmbeddings ?? 4_096
            let betaFast = parameters.betaFast ?? 32
            let betaSlow = parameters.betaSlow ?? 1

            func correctionDimension(rotations: Float) -> Float {
                let numerator = Float(resolvedDimensions)
                    * log(Float(originalContext) / (rotations * 2 * Float.pi))
                return numerator / (2 * log(parameters.ropeTheta))
            }

            let low = max(Int(floor(correctionDimension(rotations: betaFast))), 0)
            let high = min(Int(ceil(correctionDimension(rotations: betaSlow))), resolvedDimensions - 1)
            let denominator: Float = low == high ? 0.001 : Float(high - low)
            let half = max(1, resolvedDimensions / 2)
            let halfIndices = MLXArray((0..<half).map(Float.init))
            let evenIndices = MLXArray(
                Array(stride(from: 0, to: resolvedDimensions, by: 2)).map(Float.init)
            )
            let frequencyExtra = MLX.pow(
                MLXArray(parameters.ropeTheta),
                evenIndices / Float(resolvedDimensions)
            )
            let frequencyInterpolated = MLXArray(factor) * frequencyExtra
            let ramp = MLX.clip(
                (halfIndices - Float(low)) / denominator,
                min: 0,
                max: 1
            )
            let frequencyMask = MLXArray(1) - ramp
            self.frequencies = (frequencyInterpolated * frequencyExtra)
                / (
                    frequencyInterpolated * frequencyMask
                        + frequencyExtra * (MLXArray(1) - frequencyMask)
                )
            // Match mlx-swift-lm's YarnRoPE runtime contract. The public
            // checkpoint carries attention_factor=1.0 as metadata, while the
            // runtime derives mscale from factor (and defaults mscale=1,
            // mscale_all_dim=0). For factor 32 the applied multiplier is
            // 1 + 0.1 * log(32), not the literal metadata field.
            self.magnitudeScale = factor <= 1 ? 1 : 0.1 * log(factor) + 1
            self.base = nil
        } else {
            self.frequencies = nil
            self.magnitudeScale = 1
            self.base = parameters.ropeTheta
        }
        if headDim == 128,
           resolvedDimensions == 64,
           parameters.ropeType == "yarn",
           parameters.factor == 32,
           parameters.originalMaxPositionEmbeddings == 8_192,
           parameters.betaFast == 64,
           parameters.betaSlow == 1,
           parameters.ropeTheta == 500_000 {
            self.prefillFusionKind = .fullYaRN
        } else if headDim == 128,
                  resolvedDimensions == 128,
                  parameters.ropeType == "default",
                  parameters.ropeTheta == 10_000 {
            self.prefillFusionKind = .sliding
        } else {
            self.prefillFusionKind = nil
        }
        super.init()
    }

    func angleAtlas(length: Int) -> MLXArray? {
        guard prefillFusionKind != nil, length > 0 else {
            return nil
        }
        let seedMagnitude = magnitudeScale == 1 ? 1 : 1 / magnitudeScale
        let seed = MLXArray(
            Array(repeating: seedMagnitude, count: dimensions / 2)
                + Array(repeating: Float(0), count: dimensions / 2),
            [1, 1, 1, dimensions]
        )
        return callAsFunction(
            broadcast(seed, to: [1, 1, length, dimensions]),
            offset: 0
        )
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let input: MLXArray
        if magnitudeScale == 1 {
            input = x
        } else if dimensions < x.dim(-1) {
            input = concatenated(
                [
                    x[.ellipsis, ..<dimensions] * MLXArray(magnitudeScale).asType(x.dtype),
                    x[.ellipsis, dimensions...],
                ],
                axis: -1
            )
        } else {
            input = x * MLXArray(magnitudeScale).asType(x.dtype)
        }

        return MLXFast.RoPE(
            input,
            dimensions: dimensions,
            traditional: traditional,
            base: base,
            scale: 1,
            offset: offset,
            freqs: frequencies
        )
    }

    func callAsFunction(_ x: MLXArray, offsets: [Int]) -> MLXArray {
        precondition(
            x.dim(0) == offsets.count,
            "Laguna RoPE requires one position offset per batch row."
        )
        guard let first = offsets.first else {
            return x
        }
        if offsets.allSatisfy({ $0 == first }) {
            return callAsFunction(x, offset: first)
        }
        return concatenated(
            offsets.enumerated().map { index, offset in
                callAsFunction(x[index..<(index + 1), 0..., 0..., 0...], offset: offset)
            },
            axis: 0
        )
    }
}

/// A transient decode-time cache view that packs independently positioned
/// request rows into one MLX batch. The underlying row caches remain the
/// source of truth, so splitting after the forward is zero-copy at the cache
/// object level and each row keeps its own absolute position.
final class LagunaRaggedKVCache: Gemma4AttentionCache {
    private let rows: [Gemma4AttentionCache]
    private(set) var lastAttentionKeyLengths: [Int] = []

    init?(rows: [Gemma4AttentionCache]) {
        guard !rows.isEmpty else { return nil }
        let cacheType = String(describing: type(of: rows[0]))
        guard rows.allSatisfy({ String(describing: type(of: $0)) == cacheType }) else {
            return nil
        }
        self.rows = rows
    }

    var offset: Int {
        positionOffsets.min() ?? 0
    }

    var positionOffsets: [Int] {
        rows.map(\.offset)
    }

    func currentState() -> (MLXArray, MLXArray)? {
        paddedState(rows.compactMap { $0.currentState() })
    }

    func decodeState() -> (MLXArray, MLXArray)? {
        paddedState(rows.compactMap { $0.decodeState() })
    }

    func append(keys: MLXArray, values: MLXArray) {
        precondition(keys.dim(0) == rows.count && values.dim(0) == rows.count)
        for (index, row) in rows.enumerated() {
            row.append(
                keys: keys[index..<(index + 1), 0..., 0..., 0...],
                values: values[index..<(index + 1), 0..., 0..., 0...]
            )
        }
    }

    func attentionState(
        appending keys: MLXArray,
        values: MLXArray
    ) -> (MLXArray, MLXArray)? {
        precondition(keys.dim(0) == rows.count && values.dim(0) == rows.count)
        var states: [(MLXArray, MLXArray)] = []
        states.reserveCapacity(rows.count)
        for (index, row) in rows.enumerated() {
            guard let state = row.attentionState(
                appending: keys[index..<(index + 1), 0..., 0..., 0...],
                values: values[index..<(index + 1), 0..., 0..., 0...]
            ) else {
                return nil
            }
            states.append(state)
        }
        lastAttentionKeyLengths = states.map { $0.0.dim(2) }
        return paddedState(states)
    }

    func fork() -> Gemma4AttentionCache {
        LagunaRaggedKVCache(rows: rows.map { $0.fork() })!
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        let nestedRows = caches.compactMap { ($0 as? LagunaRaggedKVCache)?.rows }
        guard nestedRows.count == caches.count else { return nil }
        return LagunaRaggedKVCache(rows: nestedRows.flatMap { $0 })
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count == rows.count else { return nil }
        return rows
    }

    func specializedAttention(
        queries: MLXArray,
        repeats: Int,
        scale: Float
    ) -> MLXArray? {
        nil
    }

    func reencoded(
        quantization: Gemma4KVCacheQuantization
    ) -> Gemma4AttentionCache? {
        let converted = rows.compactMap { $0.reencoded(quantization: quantization) }
        guard converted.count == rows.count else { return nil }
        return LagunaRaggedKVCache(rows: converted)
    }

    func evaluateStorage() {
        rows.forEach { $0.evaluateStorage() }
    }

    func storageArraysForEvaluation() -> [MLXArray] {
        rows.flatMap { $0.storageArraysForEvaluation() }
    }

    private func paddedState(
        _ states: [(MLXArray, MLXArray)]
    ) -> (MLXArray, MLXArray)? {
        guard states.count == rows.count, let first = states.first else {
            return nil
        }
        let maximumLength = states.map { $0.0.dim(2) }.max() ?? 0
        guard maximumLength > 0 else { return nil }

        func pad(_ array: MLXArray, to length: Int) -> MLXArray {
            let missing = length - array.dim(2)
            guard missing > 0 else { return array }
            return concatenated(
                [
                    array,
                    MLXArray.zeros(
                        [1, array.dim(1), missing, array.dim(3)],
                        dtype: array.dtype
                    ),
                ],
                axis: 2
            )
        }

        let keys = concatenated(states.map { pad($0.0, to: maximumLength) }, axis: 0)
        let values = concatenated(states.map { pad($0.1, to: maximumLength) }, axis: 0)
        precondition(keys.dim(1) == first.0.dim(1))
        return (keys, values)
    }
}

class LagunaFeedForward: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("LagunaFeedForward subclasses must implement callAsFunction(_:).")
    }
}

final class LagunaDenseMLP: LagunaFeedForward {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(inputDimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, inputDimensions, bias: false)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }

    func lagunaXSDecodeDownInputs(
        _ x: MLXArray
    ) -> (activated: MLXArray, weight: MLXArray, scales: MLXArray)? {
        guard x.dtype == .bfloat16,
              x.shape == [1, 1, 2_048],
              (type(of: gateProj) == QuantizedLinear.self
                || type(of: gateProj) == PortableQuantizedLinear.self),
              (type(of: upProj) == QuantizedLinear.self
                || type(of: upProj) == PortableQuantizedLinear.self),
              (type(of: downProj) == QuantizedLinear.self
                || type(of: downProj) == PortableQuantizedLinear.self),
              let gate = gateProj as? QuantizedLinear,
              let up = upProj as? QuantizedLinear,
              let down = downProj as? QuantizedLinear,
              gate.mode == .nvfp4,
              up.mode == .nvfp4,
              down.mode == .nvfp4,
              gate.groupSize == 16,
              up.groupSize == 16,
              down.groupSize == 16,
              gate.bits == 4,
              up.bits == 4,
              down.bits == 4,
              gate.bias == nil,
              up.bias == nil,
              down.bias == nil,
              gate.biases == nil,
              up.biases == nil,
              down.biases == nil,
              gate.weight.shape == [512, 256],
              up.weight.shape == [512, 256],
              down.weight.shape == [2_048, 64],
              gate.scales.shape == [512, 128],
              up.scales.shape == [512, 128],
              down.scales.shape == [2_048, 32],
              down.weight.dtype == .uint32,
              down.scales.dtype == .uint8 else {
            return nil
        }
        return (
            MLXNN.silu(gate(x)) * up(x),
            down.weight,
            down.scales
        )
    }
}

final class LagunaSwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?

    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    init(
        inputDimensions: Int,
        outputDimensions: Int,
        expertCount: Int,
        quantization: LagunaQuantizationConfig?
    ) {
        self.groupSize = quantization?.groupSize ?? 16
        self.bits = quantization?.bits ?? 4
        self.mode = quantization.flatMap { QuantizationMode(rawValue: $0.mode) } ?? .affine

        if quantization != nil {
            let packedInputDimensions = (inputDimensions * bits + 31) / 32
            self._weight.wrappedValue = MLXArray.zeros(
                [expertCount, outputDimensions, packedInputDimensions],
                dtype: .uint32
            )
            self._scales.wrappedValue = MLXArray.zeros(
                [expertCount, outputDimensions, max(1, inputDimensions / groupSize)]
            )
        } else {
            let scale = sqrt(1 / Float(max(1, inputDimensions)))
            self._weight.wrappedValue = MLXRandom.uniform(
                low: -scale,
                high: scale,
                [expertCount, outputDimensions, inputDimensions]
            )
            self._scales.wrappedValue = nil
        }
        self._biases.wrappedValue = nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        let inputDimensions = x.dim(-1)
        let tokenCount = batch * sequenceLength

        let flatInput: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatInput = x.reshaped([tokenCount * topK, 1, inputDimensions])
        } else {
            let expanded = MLX.repeated(
                MLX.expandedDimensions(x.reshaped([tokenCount, 1, inputDimensions]), axis: 1),
                count: topK,
                axis: 1
            )
            flatInput = expanded.reshaped([tokenCount * topK, 1, inputDimensions])
        }

        let output = applyFlat(
            flatInput,
            indices: indices.reshaped([tokenCount * topK]),
            sortedIndices: false
        )
        return output.reshaped([batch, sequenceLength, topK, output.dim(-1)])
    }

    func applyFlat(
        _ x: MLXArray,
        indices: MLXArray,
        sortedIndices: Bool
    ) -> MLXArray {
        let output: MLXArray
        if let scales {
            output = portableGatherQuantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                sortedIndices: sortedIndices
            )
        } else {
            output = gatherMM(
                x,
                weight.swappedAxes(-1, -2),
                rhsIndices: indices,
                sortedIndices: sortedIndices
            )
        }
        return output
    }
}

final class LagunaSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: LagunaSwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: LagunaSwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: LagunaSwitchLinear
    private let decodeNVFP4RowsPerSIMDGroup: Int
    private(set) var prefillPairwiseScaleReuseCertified = false
    private(set) var prefillDownPairwiseScaleReuseCertified = false

    init(config: LagunaConfig) {
        self.decodeNVFP4RowsPerSIMDGroup =
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize,
                topK: config.numExpertsPerToken,
                xsCandidate:
                    LagunaMoEAccelerationPolicy.decodeNVFP4RowsPerSIMDGroup
            )
        self._gateProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.moeIntermediateSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        self._upProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.moeIntermediateSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        self._downProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.moeIntermediateSize,
            outputDimensions: config.hiddenSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        indices: MLXArray,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let routeCount = x.dim(0) * x.dim(1) * indices.dim(2)
        if LagunaMoEAccelerationPolicy.sortedRoutingEnabled, routeCount >= 64 {
            return sorted(
                x,
                indices: indices,
                useCustomKernels: useCustomKernels
            )
        }
        return unsorted(
            x,
            indices: indices,
            useCustomKernels: useCustomKernels
        )
    }

    func sorted(
        _ x: MLXArray,
        indices: MLXArray,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let tokenCount = batch * sequenceLength
        let topK = indices.dim(2)
        let routeCount = tokenCount * topK
        let inputDimensions = x.dim(-1)
        let flatIndices = indices.reshaped([routeCount])
        let order = argSort(flatIndices, axis: 0)
        let stagedRoute =
            useCustomKernels
                && LagunaMoEAccelerationPolicy.rankedPrefillRouteStagingEnabled
                ? RoutedMoERouting.stageRankedLagunaPrefillRoute(
                    x.reshaped([tokenCount, inputDimensions]),
                    flatIndices: flatIndices,
                    order: order,
                    topK: topK
                )
                : nil
        let sortedIndices: MLXArray
        let flatInput: MLXArray
        let stagedInverseOrder: MLXArray?
        if let stagedRoute {
            sortedIndices = stagedRoute.sortedIndices
            flatInput = stagedRoute.sortedInput
            stagedInverseOrder = stagedRoute.inverseOrder
        } else {
            sortedIndices = flatIndices.take(order, axis: 0)
            let tokenOrder = order.floorDivide(topK)
            flatInput = x.reshaped([tokenCount, inputDimensions])
                .take(tokenOrder, axis: 0)
                .reshaped([routeCount, 1, inputDimensions])
            stagedInverseOrder = nil
        }
        let activated: MLXArray
        if useCustomKernels,
           LagunaMoEAccelerationPolicy.fusedSortedNVFP4MoEEnabled,
           sequenceLength >= LagunaMoEAccelerationPolicy.fusedSortedMinimumSequenceLength,
           gateProj.mode == .nvfp4,
           upProj.mode == .nvfp4,
           gateProj.groupSize == upProj.groupSize,
           gateProj.bits == upProj.bits,
           gateProj.biases == nil,
           upProj.biases == nil,
           let gateScales = gateProj.scales,
           let upScales = upProj.scales,
           let fused = RoutedMoERouting.fusedSortedNVFP4SwiGLU(
               flatInput,
               gateWeight: gateProj.weight,
               gateScales: gateScales,
               upWeight: upProj.weight,
               upScales: upScales,
               sortedExpertIndices: sortedIndices,
               groupSize: gateProj.groupSize,
               bits: gateProj.bits,
               pairwiseScaleReuse: prefillPairwiseScaleReuseCertified
           ) {
            activated = fused
        } else {
            let gate = gateProj.applyFlat(
                flatInput,
                indices: sortedIndices,
                sortedIndices: true
            )
            let up = upProj.applyFlat(
                flatInput,
                indices: sortedIndices,
                sortedIndices: true
            )
            activated = MLXNN.silu(gate) * up
        }
        let sortedOutput: MLXArray
        if useCustomKernels,
           LagunaMoEAccelerationPolicy.fusedSortedNVFP4DownEnabled,
           sequenceLength >= LagunaMoEAccelerationPolicy.fusedSortedMinimumSequenceLength,
           downProj.mode == .nvfp4,
           downProj.biases == nil,
           let downScales = downProj.scales,
           let fusedDown = RoutedMoERouting.sortedNVFP4Projection(
               activated,
               weight: downProj.weight,
               scales: downScales,
               sortedExpertIndices: sortedIndices,
               groupSize: downProj.groupSize,
               bits: downProj.bits,
               pairwiseScaleReuse: prefillDownPairwiseScaleReuseCertified
           ) {
            sortedOutput = fusedDown
        } else {
            sortedOutput = downProj.applyFlat(
                activated,
                indices: sortedIndices,
                sortedIndices: true
            )
        }
        let inverseOrder = stagedInverseOrder
            ?? (
                useCustomKernels
                    && LagunaMoEAccelerationPolicy.fastSortedInverseEnabled
                    ? RoutedMoERouting.invertPermutation(order) ?? argSort(order, axis: 0)
                    : argSort(order, axis: 0)
            )
        return sortedOutput.take(inverseOrder, axis: 0)
            .reshaped([batch, sequenceLength, topK, sortedOutput.dim(-1)])
    }

    func unsorted(
        _ x: MLXArray,
        indices: MLXArray,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        if useCustomKernels,
           let fused = lagunaXSDecodeActivation(x, indices: indices) {
            return downProj(
                fused.reshaped([
                    batch,
                    sequenceLength,
                    topK,
                    fused.dim(-1),
                ]),
                indices: indices
            )
        }
        let gate = gateProj(x, indices: indices)
        let up = upProj(x, indices: indices)
        return downProj(MLXNN.silu(gate) * up, indices: indices)
    }

    func lagunaXSDecodeActivation(
        _ x: MLXArray,
        indices: MLXArray
    ) -> MLXArray? {
        let topK = indices.dim(2)
        guard LagunaMoEAccelerationPolicy.fusedNVFP4MoEEnabled,
           gateProj.mode == .nvfp4,
           upProj.mode == .nvfp4,
           gateProj.groupSize == upProj.groupSize,
           gateProj.bits == upProj.bits,
           gateProj.biases == nil,
           upProj.biases == nil,
           let gateScales = gateProj.scales,
           let upScales = upProj.scales,
           let fused = RoutedMoERouting.fusedGatherNVFP4SwiGLU(
               x,
               gateWeight: gateProj.weight,
               gateScales: gateScales,
               upWeight: upProj.weight,
               upScales: upScales,
               expertIndices: indices,
               topK: topK,
               groupSize: gateProj.groupSize,
               bits: gateProj.bits,
               rowsPerSIMDGroup: decodeNVFP4RowsPerSIMDGroup
           ) else {
            return nil
        }
        return fused
    }

    func lagunaXSDecodeDownInputs() -> (weight: MLXArray, scales: MLXArray)? {
        guard downProj.mode == .nvfp4,
              downProj.groupSize == 16,
              downProj.bits == 4,
              downProj.biases == nil,
              let scales = downProj.scales,
              downProj.weight.dtype == .uint32,
              downProj.weight.shape == [256, 2_048, 64],
              scales.dtype == .uint8,
              scales.shape == [256, 2_048, 32] else {
            return nil
        }
        return (downProj.weight, scales)
    }

    func prepareSortedDownWarmUp() -> MLXArray? {
        guard LagunaMoEAccelerationPolicy.fusedSortedNVFP4DownEnabled,
              downProj.mode == .nvfp4,
              downProj.biases == nil,
              let scales = downProj.scales else {
            return nil
        }
        let routeCount = LagunaMoEAccelerationPolicy.fusedSortedMinimumSequenceLength
        let inputDimensions = downProj.weight.dim(2) * 8
        let input = MLXArray.zeros(
            [routeCount, 1, inputDimensions],
            dtype: .bfloat16
        )
        let indices = MLXArray.zeros([routeCount], dtype: .int32)
        return RoutedMoERouting.sortedNVFP4Projection(
            input,
            weight: downProj.weight,
            scales: scales,
            sortedExpertIndices: indices,
            groupSize: downProj.groupSize,
            bits: downProj.bits,
            pairwiseScaleReuse: prefillDownPairwiseScaleReuseCertified
        )
    }

    /// Certifies the loaded routed scale planes once, before warmup. The
    /// results are retained as Booleans only; unlike the challenge runtime's
    /// packed banks, production needs no additional scale storage.
    func preparePrefillPairwiseScaleReuse() {
        prefillPairwiseScaleReuseCertified = false
        prefillDownPairwiseScaleReuseCertified = false
        guard LagunaMoEAccelerationPolicy.prefillExpertPairwiseScaleReuseEnabled else {
            return
        }
        if gateProj.mode == .nvfp4,
           upProj.mode == .nvfp4,
           gateProj.groupSize == 16,
           upProj.groupSize == 16,
           gateProj.bits == 4,
           upProj.bits == 4,
           gateProj.biases == nil,
           upProj.biases == nil,
           let gateScales = gateProj.scales,
           let upScales = upProj.scales,
           gateScales.dtype == .uint8,
           upScales.dtype == .uint8,
           gateScales.shape == upScales.shape {
            prefillPairwiseScaleReuseCertified =
                lagunaNVFP4AdjacentScalePairsCertified(gateScales)
                && lagunaNVFP4AdjacentScalePairsCertified(upScales)
        }
        if downProj.mode == .nvfp4,
           downProj.groupSize == 16,
           downProj.bits == 4,
           downProj.biases == nil,
           let downScales = downProj.scales,
           downScales.dtype == .uint8 {
            prefillDownPairwiseScaleReuseCertified =
                lagunaNVFP4AdjacentScalePairsCertified(downScales)
        }
    }
}

final class LagunaRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    private let topK: Int
    private let normalize: Bool
    private let softcap: Float

    init(config: LagunaConfig) {
        self.topK = config.numExpertsPerToken
        self.normalize = config.normTopKProbability
        self.softcap = config.moeRouterLogitSoftcapping
        self._weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize])
        self._correctionBias.wrappedValue = MLXArray.zeros([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        var logits = x.matmul(weight.T).asType(.float32)
        if softcap > 0 {
            logits = tanh(logits / softcap) * softcap
        }
        let scores = sigmoid(logits)
        let selectionScores = scores + correctionBias.asType(scores.dtype)
        let count = min(topK, selectionScores.dim(-1))
        // Hard expert selection is discrete. Keep gradients through the
        // selected score values, but do not ask gather kernels for an
        // undefined VJP with respect to their integer indices.
        let indices = stopGradient(
            argPartition(-selectionScores, kth: count - 1, axis: -1)[
                .ellipsis,
                ..<count
            ]
        )
        var weights = takeAlong(scores, indices, axis: -1)
        if normalize {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (indices, weights.asType(x.dtype))
    }
}

final class LagunaSparseMoE: LagunaFeedForward {
    @ModuleInfo(key: "gate") var gate: LagunaRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: LagunaSwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaDenseMLP

    private let scalingFactor: Float

    init(config: LagunaConfig) {
        self._gate.wrappedValue = LagunaRouter(config: config)
        self._switchMLP.wrappedValue = LagunaSwitchGLU(config: config)
        self._sharedExpert.wrappedValue = LagunaDenseMLP(
            inputDimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
        self.scalingFactor = config.moeRoutedScalingFactor
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, residual: nil)
    }

    func callAsFunction(
        _ x: MLXArray,
        residual: MLXArray?,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let routed = gate(x)
        if useCustomKernels,
           LagunaMoEAccelerationPolicy.fusedRoutedSharedDownResidualEnabled,
           let residual,
           scalingFactor == 2.5,
           let routedActivated = switchMLP.lagunaXSDecodeActivation(
               x,
               indices: routed.indices
           ),
           let routedDown = switchMLP.lagunaXSDecodeDownInputs(),
           let sharedDown = sharedExpert.lagunaXSDecodeDownInputs(x),
           let fused = RoutedMoERouting.fusedLagunaXSRoutedSharedDownResidual(
               routedActivated: routedActivated,
               routedDownWeight: routedDown.weight,
               routedDownScales: routedDown.scales,
               indices: routed.indices,
               routerWeights: routed.weights,
               sharedActivated: sharedDown.activated,
               sharedDownWeight: sharedDown.weight,
               sharedDownScales: sharedDown.scales,
               residual: residual
           ) {
            return fused
        }
        var expertOutput = switchMLP(
            x,
            indices: routed.indices,
            useCustomKernels: useCustomKernels
        )
        expertOutput = (
            expertOutput * MLX.expandedDimensions(routed.weights, axis: routed.weights.ndim)
        ).sum(axis: -2)
        if scalingFactor != 1 {
            expertOutput = expertOutput * scalingFactor
        }
        let branch = expertOutput + sharedExpert(x)
        return residual.map { $0 + branch } ?? branch
    }

    func preparePrefillAcceleration() -> MLXArray? {
        switchMLP.prepareSortedDownWarmUp()
    }

    func preparePrefillPairwiseScaleReuse() {
        switchMLP.preparePrefillPairwiseScaleReuse()
    }
}

final class LagunaAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDim: Int
    private let scale: Float
    private let slidingWindow: Int?
    private let gatePerHead: Bool
    private let rope: LagunaRoPE
    private let layerIndex: Int
    private let isTerminalLayer: Bool
    private var _nativeAffineQKV: LagunaNativeAffineWeight?
    private var _terminalPrefillQGateWeight: MLXArray?
    private var _terminalPrefillKVWeight: MLXArray?

    init(config: LagunaConfig, layerIndex: Int) {
        self.layerIndex = layerIndex
        self.headCount = config.attentionHeads(layerIndex: layerIndex)
        self.keyValueHeadCount = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.slidingWindow = config.layerTypes[layerIndex] == "sliding_attention"
            ? config.slidingWindow
            : nil
        self.gatePerHead = config.gating == "per-head"
        self.isTerminalLayer = layerIndex == config.numHiddenLayers - 1
        self.rope = LagunaRoPE(
            headDim: config.headDim,
            parameters: config.ropeParameters(layerIndex: layerIndex)
        )

        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            headCount * config.headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            keyValueHeadCount * config.headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            keyValueHeadCount * config.headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(headCount * config.headDim, config.hiddenSize, bias: false)
        if config.gating == "none" || config.gating == "false" {
            self._gProj.wrappedValue = nil
        } else {
            let gateDimensions = gatePerHead ? headCount : headCount * config.headDim
            self._gProj.wrappedValue = Linear(config.hiddenSize, gateDimensions, bias: false)
        }
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    func prepareNativeAffineQKV() -> [MLXArray] {
        guard _nativeAffineQKV == nil,
              LagunaGraphAccelerationPolicy.usesNativeAffineQKV(layerIndex: layerIndex),
              headDim == 128,
              keyValueHeadCount == 8,
              headCount == 48 || headCount == 64,
              type(of: qProj) == Linear.self,
              type(of: kProj) == Linear.self,
              type(of: vProj) == Linear.self,
              qProj.bias == nil,
              kProj.bias == nil,
              vProj.bias == nil,
              qProj.weight.shape == [headCount * headDim, 2_048],
              kProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              vProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              let query = lagunaNativeAffineWeight(qProj.weight),
              let key = lagunaNativeAffineWeight(kProj.weight),
              let value = lagunaNativeAffineWeight(vProj.weight) else {
            return []
        }
        let fused = LagunaNativeAffineWeight(
            packedCodes: concatenated(
                [query.packedCodes, key.packedCodes, value.packedCodes],
                axis: 0
            ),
            scales: concatenated([query.scales, key.scales, value.scales], axis: 0),
            biases: concatenated([query.biases, key.biases, value.biases], axis: 0),
            originalShape: [
                qProj.weight.dim(0) + kProj.weight.dim(0) + vProj.weight.dim(0),
                qProj.weight.dim(1),
            ]
        )
        _nativeAffineQKV = fused
        return fused.arrays
    }

    /// Retain two bias-free BF16 side banks for the terminal XS prefill layer.
    /// The final query row and per-head gate share one input, while K/V still
    /// consume every supplied row so the cache advances normally. Concatenating
    /// output rows does not change any contraction or reduction order.
    func prepareTerminalPrefillProjectionWeights(enabled: Bool? = nil) -> [MLXArray] {
        let enabled = enabled
            ?? (
                LagunaGraphAccelerationPolicy.terminalPrefillRowEnabled
                    && LagunaGraphAccelerationPolicy.terminalPrefillProjectionBanksEnabled
            )
        guard enabled,
              _terminalPrefillQGateWeight == nil,
              _terminalPrefillKVWeight == nil,
              isTerminalLayer,
              slidingWindow != nil,
              gatePerHead,
              headDim == 128,
              keyValueHeadCount == 8,
              headCount == 48 || headCount == 64,
              let gProj,
              type(of: qProj) == Linear.self,
              type(of: kProj) == Linear.self,
              type(of: vProj) == Linear.self,
              type(of: gProj) == Linear.self,
              qProj.bias == nil,
              kProj.bias == nil,
              vProj.bias == nil,
              gProj.bias == nil,
              qProj.weight.dtype == .bfloat16,
              kProj.weight.dtype == .bfloat16,
              vProj.weight.dtype == .bfloat16,
              gProj.weight.dtype == .bfloat16,
              qProj.weight.shape == [headCount * headDim, 2_048],
              kProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              vProj.weight.shape == [keyValueHeadCount * headDim, 2_048],
              gProj.weight.shape == [headCount, 2_048] else {
            return []
        }

        let queryGate = concatenated([qProj.weight, gProj.weight], axis: 0)
        let keyValue = concatenated([kProj.weight, vProj.weight], axis: 0)
        _terminalPrefillQGateWeight = queryGate
        _terminalPrefillKVWeight = keyValue
        return [queryGate, keyValue]
    }

    /// LoRA-wrapped projections must remain the only source of Q/K/V/O values.
    /// Discard retained base-weight layouts that would otherwise bypass them.
    func invalidateTextLoRAUnsafeAcceleration() {
        _nativeAffineQKV = nil
        _terminalPrefillQGateWeight = nil
        _terminalPrefillKVWeight = nil
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        precomputedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        precomputedRoPEAtlas: MLXArray? = nil,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let offset = cache?.offset ?? 0
        let positionOffsets = (cache as? LagunaRaggedKVCache)?.positionOffsets
            ?? Array(repeating: offset, count: batch)

        let rawQueries: MLXArray
        let rawKeys: MLXArray
        var values: MLXArray
        let queryDimensions = headCount * headDim
        let keyValueDimensions = keyValueHeadCount * headDim
        if useCustomKernels,
           batch == 1,
           sequenceLength == 1,
           x.dtype == .bfloat16,
           x.shape == [1, 1, 2_048],
           let affine = _nativeAffineQKV,
           affine.originalShape == [queryDimensions + 2 * keyValueDimensions, 2_048] {
            let qkv = MLX.quantizedMM(
                x,
                affine.packedCodes,
                scales: affine.scales,
                biases: affine.biases,
                transpose: true,
                groupSize: 32,
                bits: 8,
                mode: .affine
            )
            rawQueries = qkv[.ellipsis, 0..<queryDimensions]
            rawKeys = qkv[
                .ellipsis,
                queryDimensions..<(queryDimensions + keyValueDimensions)
            ]
            values = qkv[
                .ellipsis,
                (queryDimensions + keyValueDimensions)...
            ].reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        } else {
            rawQueries = qProj(x)
            rawKeys = kProj(x)
            values = vProj(x).reshaped(
                batch,
                sequenceLength,
                keyValueHeadCount,
                headDim
            )
        }
        var queries: MLXArray
        var keys: MLXArray
        let fusedQK: (queries: MLXArray, keys: MLXArray)? =
            useCustomKernels
                && LagunaGraphAccelerationPolicy.prefillQKNormRoPEEnabled
            ? rope.prefillFusionKind.flatMap { kind in
                guard let precomputedRoPEAtlas,
                      batch == 1,
                      positionOffsets.allSatisfy({ $0 == offset }) else {
                    return nil
                }
                return LagunaFusedPrefill.qkNormRoPE(
                    kind: kind,
                    rawQueries: rawQueries,
                    rawKeys: rawKeys,
                    queryWeight: qNorm.weight,
                    keyWeight: kNorm.weight,
                    angleAtlas: precomputedRoPEAtlas,
                    offset: offset,
                    length: sequenceLength
                )
            } : nil
        if let fusedQK {
            queries = fusedQK.queries
            keys = fusedQK.keys
        } else {
            queries = rawQueries.reshaped(batch, sequenceLength, headCount, headDim)
            keys = rawKeys.reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            queries = rope(queries, offsets: positionOffsets)
            keys = rope(keys, offsets: positionOffsets)
        }
        values = values.transposed(0, 2, 1, 3)

        var keyLengths: [Int]?
        if let cache {
            let state = cache.attentionState(appending: keys, values: values)
            keys = state!.0
            values = state!.1
            keyLengths = (cache as? LagunaRaggedKVCache)?.lastAttentionKeyLengths
        }

        let mask = precomputedMask ?? attentionMask(
            queryLength: sequenceLength,
            queryOffsets: positionOffsets,
            keyLengths: keyLengths ?? Array(repeating: keys.dim(2), count: batch),
            keyLength: keys.dim(2),
            dtype: x.dtype
        )
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3)

        if let gProj {
            let gate = MLXNN.softplus(gProj(x).asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output = output * MLX.expandedDimensions(gate, axis: gate.ndim)
            } else {
                output = output.reshaped(batch, sequenceLength, -1) * gate
                return oProj(output)
            }
        }
        return oProj(output.reshaped(batch, sequenceLength, -1))
    }

    /// Final-layer multi-token specialization for callers that consume only
    /// the last hidden row. All K/V rows are produced and committed; Q,
    /// attention output, gating, and O projection run only for the final row.
    func callLastPrefillRow(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        useProjectionBanks: Bool? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        precondition(batch == 1 && sequenceLength > 1)

        let offset = cache?.offset ?? 0
        let lastInput = x[0..., (sequenceLength - 1)..., 0...]
        let queryDimensions = headCount * headDim
        let keyValueDimensions = keyValueHeadCount * headDim

        let rawQueries: MLXArray
        let rawKeys: MLXArray
        var values: MLXArray
        let bankedGate: MLXArray?
        let useProjectionBanks = useProjectionBanks
            ?? LagunaGraphAccelerationPolicy.terminalPrefillProjectionBanksEnabled
        if useProjectionBanks,
           let queryGateWeight = _terminalPrefillQGateWeight,
           let keyValueWeight = _terminalPrefillKVWeight,
           x.dtype == .bfloat16,
           lastInput.dtype == .bfloat16,
           queryGateWeight.dtype == .bfloat16,
           keyValueWeight.dtype == .bfloat16,
           queryGateWeight.shape == [queryDimensions + headCount, 2_048],
           keyValueWeight.shape == [2 * keyValueDimensions, 2_048] {
            let queryGate = matmul(lastInput, queryGateWeight.T)
            rawQueries = queryGate[.ellipsis, 0..<queryDimensions]
            bankedGate = queryGate[
                .ellipsis,
                queryDimensions..<(queryDimensions + headCount)
            ]

            let keyValue = matmul(x, keyValueWeight.T)
            rawKeys = keyValue[.ellipsis, 0..<keyValueDimensions]
            values = keyValue[
                .ellipsis,
                keyValueDimensions..<(2 * keyValueDimensions)
            ].reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        } else {
            rawQueries = qProj(lastInput)
            rawKeys = kProj(x)
            values = vProj(x).reshaped(
                batch,
                sequenceLength,
                keyValueHeadCount,
                headDim
            )
            bankedGate = nil
        }

        var queries = qNorm(
            rawQueries.reshaped(batch, 1, headCount, headDim)
        ).transposed(0, 2, 1, 3)
        var keys = kNorm(
            rawKeys.reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        ).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset + sequenceLength - 1)
        keys = rope(keys, offset: offset)

        var keyLengths: [Int]?
        if let cache {
            let state = cache.attentionState(appending: keys, values: values)
            keys = state!.0
            values = state!.1
            keyLengths = (cache as? LagunaRaggedKVCache)?.lastAttentionKeyLengths
        }
        let queryOffset = offset + sequenceLength - 1
        let mask = attentionMask(
            queryLength: 1,
            queryOffsets: [queryOffset],
            keyLengths: keyLengths ?? [keys.dim(2)],
            keyLength: keys.dim(2),
            dtype: x.dtype
        )
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3)

        if let gProj {
            let projectedGate = bankedGate ?? gProj(lastInput)
            let gate = MLXNN.softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output = output * MLX.expandedDimensions(gate, axis: gate.ndim)
            } else {
                output = output.reshaped(batch, 1, -1) * gate
                return oProj(output)
            }
        }
        return oProj(output.reshaped(batch, 1, -1))
    }

    func prefillMask(
        queryLength: Int,
        cache: Gemma4AttentionCache?,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let queryOffset = cache?.offset ?? 0
        let previousKeyLength = cache?.currentState()?.0.dim(2) ?? 0
        let keyLength = previousKeyLength + queryLength
        return attentionMask(
            queryLength: queryLength,
            queryOffsets: [queryOffset],
            keyLengths: [keyLength],
            keyLength: keyLength,
            dtype: dtype
        )
    }

    func prefillRoPEAngleAtlas(length: Int) -> MLXArray? {
        rope.angleAtlas(length: length)
    }

    private func attentionMask(
        queryLength: Int,
        queryOffsets: [Int],
        keyLengths: [Int],
        keyLength: Int,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(queryOffsets.count == keyLengths.count)
        guard queryLength > 1 || keyLengths.contains(where: { $0 != keyLength }) else {
            return .none
        }
        if queryOffsets.count == 1, keyLengths[0] == keyLength {
            let queryOffset = queryOffsets[0]
            let keyStart = max(0, queryOffset + queryLength - keyLength)
            let queryPositions = MLXArray(
                Int32(queryOffset)..<Int32(queryOffset + queryLength)
            ).reshaped(queryLength, 1)
            let keyPositions = MLXArray(
                Int32(keyStart)..<Int32(keyStart + keyLength)
            ).reshaped(1, keyLength)
            var allowed = keyPositions .<= queryPositions
            if let slidingWindow {
                allowed = allowed
                    .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
            }
            let typed = allowed.asType(dtype).reshaped(1, 1, queryLength, keyLength)
            let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
            let negative = zeros + MLXArray(-1e9).asType(dtype)
            return .array(MLX.where(
                typed .> MLXArray(0).asType(dtype),
                zeros,
                negative
            ))
        }

        let rowMasks = zip(queryOffsets, keyLengths).map { queryOffset, validKeyLength in
            let keyStart = max(0, queryOffset + queryLength - validKeyLength)
            let queryPositions = MLXArray(
                Int32(queryOffset)..<Int32(queryOffset + queryLength)
            ).reshaped(queryLength, 1)
            let keyIndices = MLXArray(Int32(0)..<Int32(keyLength)).reshaped(1, keyLength)
            let keyPositions = keyIndices + Int32(keyStart)
            var allowed = (keyIndices .< Int32(validKeyLength))
                .&& (keyPositions .<= queryPositions)
            if let slidingWindow {
                allowed = allowed
                    .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
            }
            return allowed
        }
        let allowed = stacked(rowMasks).reshaped(
            queryOffsets.count,
            1,
            queryLength,
            keyLength
        )
        let zeros = MLXArray.zeros(
            [queryOffsets.count, 1, queryLength, keyLength],
            dtype: dtype
        )
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(allowed, zeros, negative))
    }
}

final class LagunaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: LagunaAttention
    @ModuleInfo(key: "mlp") var mlp: LagunaFeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: LagunaConfig, layerIndex: Int) {
        self._selfAttention.wrappedValue = LagunaAttention(config: config, layerIndex: layerIndex)
        self._mlp.wrappedValue = config.isSparse(layerIndex: layerIndex)
            ? LagunaSparseMoE(config: config)
            : LagunaDenseMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize
            )
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        precomputedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        precomputedRoPEAtlas: MLXArray? = nil,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let attentionBranch = selfAttention(
            inputLayerNorm(x),
            cache: cache,
            precomputedMask: precomputedMask,
            precomputedRoPEAtlas: precomputedRoPEAtlas,
            useCustomKernels: useCustomKernels
        )
        let attended: MLXArray
        let normalized: MLXArray
        if useCustomKernels,
           LagunaGraphAccelerationPolicy.prefillFusedResidualRMSNormEnabled,
           let fused = LagunaFusedPrefill.residualRMSNorm(
               residual: x,
               branch: attentionBranch,
               weight: postAttentionLayerNorm.weight
           ) {
            attended = fused.summed
            normalized = fused.normalized
        } else {
            attended = x + attentionBranch
            normalized = postAttentionLayerNorm(attended)
        }
        if normalized.dim(0) == 1,
           normalized.dim(1) == 1,
           let sparse = mlp as? LagunaSparseMoE {
            return sparse(
                normalized,
                residual: attended,
                useCustomKernels: useCustomKernels
            )
        }
        if let sparse = mlp as? LagunaSparseMoE {
            return attended + sparse(
                normalized,
                residual: nil,
                useCustomKernels: useCustomKernels
            )
        }
        return attended + mlp(normalized)
    }

    /// Preserve every terminal-layer K/V row while carrying only the consumed
    /// final residual row through attention output and the MLP.
    func callLastPrefillRow(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?
    ) -> MLXArray {
        let normalized = inputLayerNorm(x)
        let attentionBranch = selfAttention.callLastPrefillRow(
            normalized,
            cache: cache
        )
        let attended = x[0..., (x.dim(1) - 1)..., 0...] + attentionBranch
        return attended + mlp(postAttentionLayerNorm(attended))
    }
}

struct LagunaLanguageModelOutput {
    let hidden: MLXArray
    let capturedHiddenStates: [Int: MLXArray]
}

final class LagunaLanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LagunaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private let config: LagunaConfig
    private var fullRoPEAngleAtlas: MLXArray?
    private var slidingRoPEAngleAtlas: MLXArray?

    init(config: LagunaConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            LagunaDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        forward(inputIDs, cache: cache).hidden
    }

    func forward(
        _ inputIDs: MLXArray,
        cache: [Gemma4AttentionCache]? = nil,
        captureLayerIndices: Set<Int> = [],
        lastPositionOnly: Bool = false,
        terminalPrefillRowEnabled: Bool? = nil,
        prefillAsyncLadderEnabled: Bool = true,
        useCustomKernels: Bool = true
    ) -> LagunaLanguageModelOutput {
        var hidden = embedTokens(inputIDs)
        var capturedHiddenStates: [Int: MLXArray] = [:]
        let sequenceLength = hidden.dim(1)
        let usesSharedMasks = LagunaGraphAccelerationPolicy.sharedAttentionMasksEnabled
            && hidden.dim(0) == 1
            && sequenceLength > 1
        let fullLayerIndex = config.layerTypes.firstIndex(of: "full_attention")
        let slidingLayerIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        let fullMask = usesSharedMasks ? fullLayerIndex.map { index in
            layers[index].selfAttention.prefillMask(
                queryLength: sequenceLength,
                cache: cache?[index],
                dtype: hidden.dtype
            )
        } : nil
        let slidingMask = usesSharedMasks ? slidingLayerIndex.map { index in
            layers[index].selfAttention.prefillMask(
                queryLength: sequenceLength,
                cache: cache?[index],
                dtype: hidden.dtype
            )
        } : nil
        let usesPrefillRoPEAtlas = useCustomKernels
            && LagunaGraphAccelerationPolicy.prefillQKNormRoPEEnabled
            && hidden.dim(0) == 1
            && sequenceLength > 1
        let fullAtlas: MLXArray? = usesPrefillRoPEAtlas ? fullLayerIndex.flatMap { index in
            let offset = cache?[index].offset ?? 0
            guard offset >= 0,
                  offset + sequenceLength <= LagunaFusedPrefill.ropeAngleAtlasLength else {
                return nil
            }
            return fullRoPEAngleAtlas
        } : nil
        let slidingAtlas: MLXArray? = usesPrefillRoPEAtlas ? slidingLayerIndex.flatMap { index in
            let offset = cache?[index].offset ?? 0
            guard offset >= 0,
                  offset + sequenceLength <= LagunaFusedPrefill.ropeAngleAtlasLength else {
                return nil
            }
            return slidingRoPEAngleAtlas
        } : nil
        for (index, layer) in layers.enumerated() {
            let mask = config.layerTypes[index] == "full_attention"
                ? fullMask
                : slidingMask
            let ropeAtlas = config.layerTypes[index] == "full_attention"
                ? fullAtlas
                : slidingAtlas
            let useTerminalPrefillRow =
                (
                    terminalPrefillRowEnabled
                        ?? LagunaGraphAccelerationPolicy.terminalPrefillRowEnabled
                )
                    && lastPositionOnly
                    && hidden.dim(0) == 1
                    && hidden.dim(1) > 1
                    && index == layers.count - 1
                    && !captureLayerIndices.contains(index)
            if useTerminalPrefillRow {
                hidden = layer.callLastPrefillRow(hidden, cache: cache?[index])
            } else {
                hidden = layer(
                    hidden,
                    cache: cache?[index],
                    precomputedMask: mask,
                    precomputedRoPEAtlas: ropeAtlas,
                    useCustomKernels: useCustomKernels
                )
            }
            if captureLayerIndices.contains(index) {
                capturedHiddenStates[index] = hidden
            }
            let ladderStride = prefillAsyncLadderEnabled
                ? LagunaGraphAccelerationPolicy.prefillAsyncLadderStride
                : 0
            if ladderStride > 0,
               sequenceLength > 1,
               (index + 1).isMultiple(of: ladderStride) {
                asyncEval(hidden)
            }
        }
        if lastPositionOnly, hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        return LagunaLanguageModelOutput(
            hidden: norm(hidden),
            capturedHiddenStates: capturedHiddenStates
        )
    }

    func makeCache() -> [Gemma4AttentionCache] {
        config.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                return Gemma4SlidingKVCache(maxSize: config.slidingWindow)
            }
            return Gemma4FullKVCache()
        }
    }

    func preparePrefillAcceleration() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if LagunaGraphAccelerationPolicy.prefillQKNormRoPEEnabled,
           let fullLayerIndex = config.layerTypes.firstIndex(of: "full_attention"),
           let slidingLayerIndex = config.layerTypes.firstIndex(of: "sliding_attention"),
           let fullAtlas = layers[fullLayerIndex].selfAttention.prefillRoPEAngleAtlas(
               length: LagunaFusedPrefill.ropeAngleAtlasLength
           ),
           let slidingAtlas = layers[slidingLayerIndex].selfAttention.prefillRoPEAngleAtlas(
               length: LagunaFusedPrefill.ropeAngleAtlasLength
           ) {
            self.fullRoPEAngleAtlas = fullAtlas
            self.slidingRoPEAngleAtlas = slidingAtlas
            arrays.append(contentsOf: [fullAtlas, slidingAtlas])
        }
        for layer in layers {
            guard let sparse = layer.mlp as? LagunaSparseMoE,
                  let warmUp = sparse.preparePrefillAcceleration() else {
                continue
            }
            arrays.append(warmUp)
            break
        }
        return arrays
    }

    func prepareRuntimeAcceleration() -> [MLXArray] {
        for layer in layers {
            if let sparse = layer.mlp as? LagunaSparseMoE {
                sparse.preparePrefillPairwiseScaleReuse()
            }
        }
        var arrays = preparePrefillAcceleration()
        for layer in layers {
            arrays.append(contentsOf: layer.selfAttention.prepareNativeAffineQKV())
            arrays.append(
                contentsOf: layer.selfAttention.prepareTerminalPrefillProjectionWeights()
            )
        }
        return arrays
    }

    func invalidateTextLoRAUnsafeAcceleration() {
        for layer in layers {
            layer.selfAttention.invalidateTextLoRAUnsafeAcceleration()
        }
    }
}

struct LagunaForwardOutput {
    let logits: MLXArray
    let capturedHiddenStates: [Int: MLXArray]
}

final class LagunaCausalLM: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") var model: LagunaLanguageModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: LagunaConfig

    init(config: LagunaConfig, quantizedSharedExperts: Bool = false) {
        self.config = config
        self._model.wrappedValue = LagunaLanguageModel(config: config)
        self._lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()

        if quantizedSharedExperts, let quantization = config.quantization {
            let mode = QuantizationMode(rawValue: quantization.mode) ?? .affine
            for layer in model.layers where layer.mlp is LagunaSparseMoE {
                MLXNN.quantize(model: layer) { path, _ in
                    guard path.contains("shared_expert") else {
                        return nil
                    }
                    return (
                        groupSize: quantization.groupSize,
                        bits: quantization.bits,
                        mode: mode
                    )
                }
            }
        }
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        forward(inputIDs, cache: cache).logits
    }

    /// Project only loss-bearing flattened token positions through the large
    /// vocabulary head while retaining the full hidden-state training graph.
    func trainingLogits(
        inputIDs: MLXArray,
        flatTargetPositions: MLXArray
    ) -> MLXArray {
        let hidden = model.forward(
            inputIDs,
            prefillAsyncLadderEnabled: false,
            useCustomKernels: false
        ).hidden
        let flattened = hidden.reshaped([-1, hidden.dim(-1)])
        let selected = take(
            flattened,
            flatTargetPositions.asType(.int32),
            axis: 0
        )
        return logits(from: selected)
    }

    /// Full-vocabulary fallback for training configurations that disable the
    /// gathered loss. Inference-only custom kernels do not define VJPs.
    func trainingForward(_ inputIDs: MLXArray) -> MLXArray {
        let hidden = model.forward(
            inputIDs,
            prefillAsyncLadderEnabled: false,
            useCustomKernels: false
        ).hidden
        return logits(from: hidden)
    }

    func forward(
        _ inputIDs: MLXArray,
        cache: [Gemma4AttentionCache]? = nil,
        captureLayerIndices: Set<Int> = [],
        lastPositionOnly: Bool = false,
        terminalPrefillRowEnabled: Bool? = nil
    ) -> LagunaForwardOutput {
        let output = model.forward(
            inputIDs,
            cache: cache,
            captureLayerIndices: captureLayerIndices,
            lastPositionOnly: lastPositionOnly,
            terminalPrefillRowEnabled: terminalPrefillRowEnabled
        )
        return LagunaForwardOutput(
            logits: lmHead?(output.hidden) ?? model.embedTokens.asLinear(output.hidden),
            capturedHiddenStates: output.capturedHiddenStates
        )
    }

    func lastPositionLogits(
        _ inputIDs: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) -> MLXArray {
        forward(inputIDs, cache: cache, lastPositionOnly: true).logits
    }

    func inputEmbeddings(for inputIDs: MLXArray) -> MLXArray {
        model.embedTokens(inputIDs)
    }

    func logits(from hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    func makeCache() -> [Gemma4AttentionCache] {
        model.makeCache()
    }

    func preparePrefillAcceleration() -> [MLXArray] {
        model.preparePrefillAcceleration()
    }

    func prepareRuntimeAcceleration() -> [MLXArray] {
        model.prepareRuntimeAcceleration()
    }

    func invalidateTextLoRAUnsafeAcceleration() {
        model.invalidateTextLoRAUnsafeAcceleration()
    }
}
