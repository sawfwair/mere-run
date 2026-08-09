import Foundation
import MLX
import MLXFast

struct DynamicSparseAttentionRequest: Sendable, Equatable {
    let prefixTokenCount: Int
    let thresholdStandardDeviations: Float
}

struct DynamicSparseAttentionPolicy: Sendable, Equatable {
    let thresholdStandardDeviations: Float
    let minimumSequenceLength: Int
    let denseLeadingStepFraction: Float
    let denseTrailingStepCount: Int
    let denseLeadingLayerCount: Int

    init(
        thresholdStandardDeviations: Float = 1,
        minimumSequenceLength: Int = 12_000,
        denseLeadingStepFraction: Float = 0.2,
        denseTrailingStepCount: Int = 1,
        denseLeadingLayerCount: Int = 2
    ) {
        precondition(thresholdStandardDeviations >= 0)
        precondition(minimumSequenceLength > 0)
        precondition((0...1).contains(denseLeadingStepFraction))
        precondition(denseTrailingStepCount >= 0)
        precondition(denseLeadingLayerCount >= 0)
        self.thresholdStandardDeviations = thresholdStandardDeviations
        self.minimumSequenceLength = minimumSequenceLength
        self.denseLeadingStepFraction = denseLeadingStepFraction
        self.denseTrailingStepCount = denseTrailingStepCount
        self.denseLeadingLayerCount = denseLeadingLayerCount
    }

    func request(
        stepIndex: Int,
        stepCount: Int,
        layerIndex: Int,
        sequenceLength: Int,
        prefixTokenCount: Int
    ) -> DynamicSparseAttentionRequest? {
        guard stepCount > 0,
              stepIndex >= 0,
              stepIndex < stepCount,
              layerIndex >= denseLeadingLayerCount,
              sequenceLength >= minimumSequenceLength,
              prefixTokenCount >= 0,
              prefixTokenCount < sequenceLength else { return nil }
        let denseLeadingStepCount = Int(
            ceil(Float(stepCount) * denseLeadingStepFraction)
        )
        guard stepIndex >= denseLeadingStepCount,
              stepIndex < stepCount - denseTrailingStepCount else { return nil }
        return .init(
            prefixTokenCount: prefixTokenCount,
            thresholdStandardDeviations: thresholdStandardDeviations
        )
    }
}

struct DynamicSparseAttentionGate: Sendable, Equatable {
    let maximumAbsoluteError: Float
    let meanAbsoluteError: Float
    let maximumRelativeError: Float
    let meanRelativeError: Float
    let relativeL2Error: Float
    let passed: Bool
}

/// Dynamic block-sparse attention for long, batch-one diffusion sequences on Apple GPUs.
///
/// The implementation independently applies the on-the-fly sparsification
/// ideas described by Sol-Attn. Every 64-token
/// block is represented by a key centroid and a summed value. Query-dependent
/// routing retains high-score blocks exactly; skipped blocks contribute through
/// those summaries in the same online-softmax accumulator. Prefix keys and
/// adjacent blocks are always exact, while any prefix queries remain on MLX's
/// dense fused SDPA path. Model integrations own admission, dense boundaries,
/// layout permutation, and numerical acceptance.
enum DynamicSparseAttention {
    static let blockSize = 64
    static let headDimension = 128

    struct RoutePlan {
        let keyCentroids: MLXArray
        let valueSums: MLXArray
        let routes: MLXArray
        let firstQueryBlock: Int
        let keyBlockCount: Int
    }

    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        request: DynamicSparseAttentionRequest,
        scale: Float,
        maximumQueryTokens: Int,
        maximumKernelsPerEvaluation: Int
    ) -> MLXArray? {
        guard supports(
            queries: queries,
            keys: keys,
            values: values,
            prefixTokenCount: request.prefixTokenCount
        ) else { return nil }
        precondition(maximumQueryTokens > 0)
        precondition(maximumKernelsPerEvaluation > 0)

        let sparseQueryStart = min(
            queries.dim(2),
            ((request.prefixTokenCount + blockSize - 1) / blockSize) * blockSize
        )
        guard sparseQueryStart < queries.dim(2) else {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
        }
        let plan = makeRoutePlan(
            queries: queries,
            keys: keys,
            values: values,
            queryStart: sparseQueryStart,
            thresholdStandardDeviations: request.thresholdStandardDeviations,
            scale: scale
        )
        let targetCount = queries.dim(2) - sparseQueryStart
        // Dense SDPA chunks for intermediate-memory control. This kernel has
        // no quadratic score allocation, so larger query submissions keep the
        // shared summaries resident and avoid CPU/GPU boundary churn.
        let sparseQueryChunk = max(
            16_384,
            (maximumQueryTokens / blockSize) * blockSize
        )
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(1 + (targetCount + sparseQueryChunk - 1) / sparseQueryChunk)
        var pending: [MLXArray] = []
        pending.reserveCapacity(maximumKernelsPerEvaluation)
        if sparseQueryStart > 0 {
            let densePrefix = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., 0..<sparseQueryStart, 0...],
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
            outputs.append(densePrefix)
            pending.append(densePrefix)
        }
        for localStart in stride(from: 0, to: targetCount, by: sparseQueryChunk) {
            let count = min(sparseQueryChunk, targetCount - localStart)
            let sparse = sparseKernelOutput(
                queries: queries,
                keys: keys,
                values: values,
                keyCentroids: plan.keyCentroids,
                valueSums: plan.valueSums,
                routes: plan.routes,
                queryStart: sparseQueryStart + localStart,
                queryCount: count,
                prefixTokenCount: request.prefixTokenCount,
                firstQueryBlock: plan.firstQueryBlock,
                keyBlockCount: plan.keyBlockCount,
                scale: scale
            )
            outputs.append(sparse)
            pending.append(sparse)
            if pending.count == maximumKernelsPerEvaluation {
                MLX.eval(pending)
                pending.removeAll(keepingCapacity: true)
            }
        }
        if !pending.isEmpty {
            MLX.eval(pending)
        }
        precondition(!outputs.isEmpty)
        return outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 2)
    }

    static func denseRouteGate(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        queryStart: Int,
        scale: Float,
        sampleQueryCount: Int = 8
    ) -> DynamicSparseAttentionGate? {
        guard supports(
            queries: queries,
            keys: keys,
            values: values,
            prefixTokenCount: queryStart
        ) else { return nil }
        let alignedQueryStart = min(
            queries.dim(2),
            ((queryStart + blockSize - 1) / blockSize) * blockSize
        )
        let queryCount = min(sampleQueryCount, queries.dim(2) - alignedQueryStart)
        guard queryCount > 0 else { return nil }
        let firstQueryBlock = alignedQueryStart / blockSize
        let lastQueryBlock = (alignedQueryStart + queryCount - 1) / blockSize
        let queryBlockCount = lastQueryBlock - firstQueryBlock + 1
        let keyBlockCount = (queries.dim(2) + blockSize - 1) / blockSize
        let routes = MLXArray.ones(
            [1, queries.dim(1), queryBlockCount, keyBlockCount],
            dtype: .uint8
        )
        let emptySummaries = MLXArray.zeros(
            [1, queries.dim(1), keyBlockCount, headDimension],
            dtype: .bfloat16
        )
        let candidate = sparseKernelOutput(
            queries: queries,
            keys: keys,
            values: values,
            keyCentroids: emptySummaries,
            valueSums: emptySummaries,
            routes: routes,
            queryStart: alignedQueryStart,
            queryCount: queryCount,
            prefixTokenCount: 0,
            firstQueryBlock: firstQueryBlock,
            keyBlockCount: keyBlockCount,
            scale: scale
        )
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries[
                0..., 0..., alignedQueryStart..<(alignedQueryStart + queryCount), 0...
            ],
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )
        let delta = candidate.asType(.float32) - reference.asType(.float32)
        let absolute = MLX.abs(delta)
        let metrics = MLX.stacked([
            MLX.max(absolute),
            MLX.mean(absolute),
            MLX.sqrt(
                MLX.sum(delta * delta)
                    / MLX.maximum(
                        MLX.sum(reference.asType(.float32) * reference.asType(.float32)),
                        MLXArray(Float(1e-12))
                    )
            ),
            MLX.max(MLX.abs(reference.asType(.float32))),
            MLX.mean(MLX.abs(reference.asType(.float32))),
        ])
        MLX.eval(metrics)
        let measured = metrics.asArray(Float.self)
        let maximumRelativeError = measured[0] / max(measured[3], 1e-12)
        let meanRelativeError = measured[1] / max(measured[4], 1e-12)
        // Absolute activation scales differ substantially across native model
        // families. Admit the dense-route implementation by scale-invariant
        // error while retaining tight BF16 conversion and aggregate bounds.
        let passed = maximumRelativeError <= 0.015
            && meanRelativeError <= 0.005
            && measured[2] <= 0.005
        return .init(
            maximumAbsoluteError: measured[0],
            meanAbsoluteError: measured[1],
            maximumRelativeError: maximumRelativeError,
            meanRelativeError: meanRelativeError,
            relativeL2Error: measured[2],
            passed: passed
        )
    }

    static func makeRoutePlan(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        queryStart: Int,
        thresholdStandardDeviations: Float,
        scale: Float
    ) -> RoutePlan {
        precondition(queryStart >= 0 && queryStart < queries.dim(2))
        precondition(thresholdStandardDeviations >= 0)
        let summaries = summarize(queries: queries, keys: keys, values: values)
        let firstQueryBlock = queryStart / blockSize
        let queryCentroids = summaries.queries[0..., 0..., firstQueryBlock..., 0...]
        let proxyScores = MLX.matmul(
            queryCentroids,
            summaries.keys.transposed(0, 1, 3, 2)
        ).asType(.float32) * scale
        let mean = proxyScores.mean(axis: -1, keepDims: true)
        let centered = proxyScores - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        let threshold = mean + thresholdStandardDeviations * MLX.sqrt(variance + 1e-6)
        let routes = (proxyScores .> threshold).asType(.uint8)
        return .init(
            keyCentroids: summaries.keys,
            valueSums: summaries.values,
            routes: routes,
            firstQueryBlock: firstQueryBlock,
            keyBlockCount: summaries.keys.dim(2)
        )
    }

    static func routesForTesting(
        queryCentroids: MLXArray,
        keyCentroids: MLXArray,
        thresholdStandardDeviations: Float,
        scale: Float = 1
    ) -> MLXArray {
        let proxyScores = MLX.matmul(
            queryCentroids,
            keyCentroids.transposed(0, 1, 3, 2)
        ).asType(.float32) * scale
        let mean = proxyScores.mean(axis: -1, keepDims: true)
        let centered = proxyScores - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        let threshold = mean + thresholdStandardDeviations * MLX.sqrt(variance + 1e-6)
        return (proxyScores .> threshold).asType(.uint8)
    }

    static func sparseOutputForTesting(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        routes: MLXArray,
        queryStart: Int,
        queryCount: Int,
        prefixTokenCount: Int,
        scale: Float
    ) -> MLXArray? {
        guard supports(
            queries: queries,
            keys: keys,
            values: values,
            prefixTokenCount: prefixTokenCount
        ) else { return nil }
        let summaries = summarize(queries: queries, keys: keys, values: values)
        return sparseKernelOutput(
            queries: queries,
            keys: keys,
            values: values,
            keyCentroids: summaries.keys,
            valueSums: summaries.values,
            routes: routes,
            queryStart: queryStart,
            queryCount: queryCount,
            prefixTokenCount: prefixTokenCount,
            firstQueryBlock: queryStart / blockSize,
            keyBlockCount: summaries.keys.dim(2),
            scale: scale
        )
    }

    private static func supports(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        prefixTokenCount: Int
    ) -> Bool {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        return Device.defaultDevice().deviceType == .gpu
            && [.bfloat16, .float32].contains(queries.dtype)
            && [.bfloat16, .float32].contains(keys.dtype)
            && [.bfloat16, .float32].contains(values.dtype)
            && queries.shape == keys.shape
            && queries.shape == values.shape
            && queries.ndim == 4
            && queries.dim(0) == 1
            && queries.dim(3) == headDimension
            && prefixTokenCount >= 0
            && prefixTokenCount < queries.dim(2)
        #else
        return false
        #endif
    }

    private static func summarize(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let tokenCount = queries.dim(2)
        let heads = queries.dim(1)
        let blockCount = (tokenCount + blockSize - 1) / blockSize
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let outputs = summaryKernel(
            [queries, keys, values],
            template: [
                ("TOKEN_COUNT", tokenCount),
                ("BLOCK_COUNT", blockCount),
            ],
            grid: (headDimension, blockCount, heads),
            threadGroup: (headDimension, 1, 1),
            outputShapes: [
                [1, heads, blockCount, headDimension],
                [1, heads, blockCount, headDimension],
                [1, heads, blockCount, headDimension],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1], outputs[2])
        #else
        preconditionFailure("Dynamic sparse summaries require Metal")
        #endif
    }

    private static func sparseKernelOutput(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        keyCentroids: MLXArray,
        valueSums: MLXArray,
        routes: MLXArray,
        queryStart: Int,
        queryCount: Int,
        prefixTokenCount: Int,
        firstQueryBlock: Int,
        keyBlockCount: Int,
        scale: Float
    ) -> MLXArray {
        precondition(queryStart >= 0 && queryCount > 0)
        precondition(queryStart + queryCount <= queries.dim(2))
        precondition(queryStart.isMultiple(of: blockSize))
        precondition(routes.dtype == .uint8)
        let heads = queries.dim(1)
        let scaleArray = MLXArray([scale])
        let outputDType: DType = queries.dtype == .float32
            || keys.dtype == .float32
            || values.dtype == .float32
            ? .float32
            : .bfloat16
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        return attentionKernel(
            [queries, keys, values, keyCentroids, valueSums, routes, scaleArray],
            template: [
                ("TOKEN_COUNT", queries.dim(2)),
                ("QUERY_START", queryStart),
                ("QUERY_COUNT", queryCount),
                ("PREFIX_TOKENS", prefixTokenCount),
                ("FIRST_QUERY_BLOCK", firstQueryBlock),
                ("KEY_BLOCK_COUNT", keyBlockCount),
                ("ROUTE_QUERY_BLOCKS", routes.dim(2)),
                ("OutputT", outputDType),
            ],
            grid: (32, ((queryCount + 31) / 32) * 4, heads),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, heads, queryCount, headDimension]],
            outputDTypes: [outputDType]
        )[0]
        #else
        preconditionFailure("Dynamic sparse attention requires Metal")
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let summaryKernel = MLXFast.metalKernel(
        name: "mere_dynamic_sparse_summaries_bf16_d128_b64_v1",
        inputNames: ["queries", "keys", "values"],
        outputNames: ["query_centroids", "key_centroids", "value_sums"],
        source: """
            constexpr uint block_size = 64;
            constexpr uint head_dimension = 128;

            uint dimension = thread_position_in_grid.x;
            uint block = thread_position_in_grid.y;
            uint head = thread_position_in_grid.z;
            if (dimension >= head_dimension || block >= BLOCK_COUNT) return;

            uint token_start = block * block_size;
            uint token_end = metal::min(token_start + block_size, uint(TOKEN_COUNT));
            float query_total = 0.0f;
            float key_total = 0.0f;
            float value_total = 0.0f;
            for (uint token = token_start; token < token_end; ++token) {
                uint offset = (head * TOKEN_COUNT + token) * head_dimension + dimension;
                query_total += float(queries[offset]);
                key_total += float(keys[offset]);
                value_total += float(values[offset]);
            }
            float inverse_count = 1.0f / float(token_end - token_start);
            uint output_offset = (head * BLOCK_COUNT + block) * head_dimension + dimension;
            query_centroids[output_offset] = bfloat(query_total * inverse_count);
            key_centroids[output_offset] = bfloat(key_total * inverse_count);
            value_sums[output_offset] = bfloat(value_total);
        """,
        ensureRowContiguous: true
    )

    private static let attentionKernel = MLXFast.metalKernel(
        name: "mere_dynamic_sparse_online_softmax_bf16_d128_b64_mma_v6",
        inputNames: [
            "queries", "keys", "values", "key_centroids", "value_sums",
            "routes", "scale_value",
        ],
        outputNames: ["output"],
        source: """
            constexpr uint block_size = 64;
            constexpr uint head_dimension = 128;
            constexpr uint matrix_size = 8;
            constexpr uint matrix_count = head_dimension / matrix_size;
            constexpr uint query_tile_rows = 32;
            constexpr uint simdgroup_count = 4;
            constexpr uint query_stride = head_dimension + 8;
            constexpr uint key_stride = matrix_size + 8;
            constexpr uint value_stride = head_dimension + 8;
            constexpr float log2e = 1.4426950408889634f;

            uint lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint thread_linear = thread_position_in_threadgroup.y * 32
                + thread_position_in_threadgroup.x;
            uint query_group = threadgroup_position_in_grid.y;
            uint head = threadgroup_position_in_grid.z;
            uint quad = lane / 4;
            uint matrix_row = (quad & 4) + ((lane / 2) % 4);
            uint matrix_column = (quad & 2) * 2 + (lane % 2) * 2;
            uint group_query_start = query_group * query_tile_rows;
            uint local_query = group_query_start
                + simd_group * matrix_size + matrix_row;
            bool query_valid = local_query < QUERY_COUNT;
            uint safe_local_query = metal::min(local_query, uint(QUERY_COUNT - 1));
            uint query_token = QUERY_START + safe_local_query;
            uint query_block = query_token / block_size;
            uint route_query_block = query_block - FIRST_QUERY_BLOCK;
            float attention_scale = float(scale_value[0]) * log2e;

            threadgroup bfloat query_shared[query_tile_rows * query_stride];
            threadgroup bfloat key_value_shared[head_dimension * key_stride];
            for (uint index = thread_linear;
                 index < query_tile_rows * head_dimension;
                 index += 32 * simdgroup_count) {
                uint row = index / head_dimension;
                uint dimension = index - row * head_dimension;
                uint candidate = group_query_start + row;
                bool valid = candidate < QUERY_COUNT;
                uint token = QUERY_START + metal::min(candidate, uint(QUERY_COUNT - 1));
                query_shared[row * query_stride + dimension] = valid
                    ? bfloat(queries[(head * TOKEN_COUNT + token) * head_dimension + dimension])
                    : bfloat(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            thread simdgroup_matrix<float, 8, 8> accumulated[matrix_count];
            for (uint frag = 0; frag < matrix_count; ++frag) {
                accumulated[frag].thread_elements()[0] = 0.0f;
                accumulated[frag].thread_elements()[1] = 0.0f;
            }

            float row_maximum = -INFINITY;
            float row_sum = 0.0f;
            for (uint key_block = 0; key_block < KEY_BLOCK_COUNT; ++key_block) {
                uint key_start = key_block * block_size;
                uint key_end = metal::min(key_start + block_size, uint(TOKEN_COUNT));
                uint route_offset =
                    (head * ROUTE_QUERY_BLOCKS + route_query_block) * KEY_BLOCK_COUNT + key_block;
                int block_distance = int(query_block) - int(key_block);
                bool neighboring = block_distance >= -1 && block_distance <= 1;
                bool prefix_sink = key_start < PREFIX_TOKENS;
                bool exact = routes[route_offset] != 0 || neighboring || prefix_sink;

                if (exact) {
                    for (uint key_tile = key_start;
                         key_tile < key_end;
                         key_tile += matrix_size) {
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint index = thread_linear;
                             index < matrix_size * head_dimension;
                             index += 32 * simdgroup_count) {
                            uint dimension = index / matrix_size;
                            uint key_column = index - dimension * matrix_size;
                            uint token = key_tile + key_column;
                            key_value_shared[dimension * key_stride + key_column] = token < key_end
                                ? bfloat(keys[(head * TOKEN_COUNT + token) * head_dimension + dimension])
                                : bfloat(0.0f);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);

                        thread simdgroup_matrix<float, 8, 8> scores;
                        scores.thread_elements()[0] = 0.0f;
                        scores.thread_elements()[1] = 0.0f;
                        for (uint frag = 0; frag < matrix_count; ++frag) {
                            thread simdgroup_matrix<bfloat, 8, 8> query_fragment;
                            thread simdgroup_matrix<bfloat, 8, 8> key_fragment;
                            uint query_dimension = frag * matrix_size + matrix_column;
                            uint query_row = simd_group * matrix_size + matrix_row;
                            query_fragment.thread_elements()[0] =
                                query_shared[query_row * query_stride + query_dimension];
                            query_fragment.thread_elements()[1] =
                                query_shared[query_row * query_stride + query_dimension + 1];
                            uint key_dimension = frag * matrix_size + matrix_row;
                            key_fragment.thread_elements()[0] =
                                key_value_shared[key_dimension * key_stride + matrix_column];
                            key_fragment.thread_elements()[1] =
                                key_value_shared[key_dimension * key_stride + matrix_column + 1];
                            thread simdgroup_matrix<float, 8, 8> next_scores;
                            simdgroup_multiply_accumulate(
                                next_scores,
                                query_fragment,
                                key_fragment,
                                scores
                            );
                            scores = next_scores;
                        }

                        float score0 = scores.thread_elements()[0] * attention_scale;
                        float score1 = scores.thread_elements()[1] * attention_scale;
                        if (key_tile + matrix_column >= key_end) score0 = -INFINITY;
                        if (key_tile + matrix_column + 1 >= key_end) score1 = -INFINITY;
                        float tile_maximum = metal::max(score0, score1);
                        tile_maximum = metal::max(
                            tile_maximum,
                            simd_shuffle_xor(tile_maximum, ushort(1))
                        );
                        tile_maximum = metal::max(
                            tile_maximum,
                            simd_shuffle_xor(tile_maximum, ushort(8))
                        );
                        float new_maximum = metal::max(row_maximum, tile_maximum);
                        float alpha = metal::fast::exp2(row_maximum - new_maximum);
                        float probability0 = metal::fast::exp2(score0 - new_maximum);
                        float probability1 = metal::fast::exp2(score1 - new_maximum);
                        float probability_sum = probability0 + probability1;
                        probability_sum += simd_shuffle_xor(probability_sum, ushort(1));
                        probability_sum += simd_shuffle_xor(probability_sum, ushort(8));
                        row_sum = row_sum * alpha + probability_sum;
                        row_maximum = new_maximum;

                        thread simdgroup_matrix<float, 8, 8> probabilities;
                        probabilities.thread_elements()[0] = probability0;
                        probabilities.thread_elements()[1] = probability1;
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint index = thread_linear;
                             index < matrix_size * head_dimension;
                             index += 32 * simdgroup_count) {
                            uint value_row = index / head_dimension;
                            uint dimension = index - value_row * head_dimension;
                            uint token = key_tile + value_row;
                            key_value_shared[value_row * value_stride + dimension] = token < key_end
                                ? bfloat(values[(head * TOKEN_COUNT + token) * head_dimension + dimension])
                                : bfloat(0.0f);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint frag = 0; frag < matrix_count; ++frag) {
                            accumulated[frag].thread_elements()[0] *= alpha;
                            accumulated[frag].thread_elements()[1] *= alpha;
                            thread simdgroup_matrix<bfloat, 8, 8> value_fragment;
                            uint output_dimension = frag * matrix_size + matrix_column;
                            value_fragment.thread_elements()[0] =
                                key_value_shared[matrix_row * value_stride + output_dimension];
                            value_fragment.thread_elements()[1] =
                                key_value_shared[matrix_row * value_stride + output_dimension + 1];
                            thread simdgroup_matrix<float, 8, 8> next_accumulated;
                            simdgroup_multiply_accumulate(
                                next_accumulated,
                                probabilities,
                                value_fragment,
                                accumulated[frag]
                            );
                            accumulated[frag] = next_accumulated;
                        }
                    }
                }
            }

            uint skipped_blocks[matrix_size];
            uint skipped_count = 0;
            for (uint scan = 0; scan <= KEY_BLOCK_COUNT; ++scan) {
                bool at_end = scan == KEY_BLOCK_COUNT;
                if (!at_end) {
                    uint key_start = scan * block_size;
                    uint route_offset =
                        (head * ROUTE_QUERY_BLOCKS + route_query_block) * KEY_BLOCK_COUNT + scan;
                    int block_distance = int(query_block) - int(scan);
                    bool neighboring = block_distance >= -1 && block_distance <= 1;
                    bool prefix_sink = key_start < PREFIX_TOKENS;
                    bool exact = routes[route_offset] != 0 || neighboring || prefix_sink;
                    if (!exact) {
                        skipped_blocks[skipped_count++] = scan;
                    }
                }
                bool batch_ready = skipped_count == matrix_size
                    || (at_end && skipped_count > 0);
                if (!batch_ready) continue;

                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint index = thread_linear;
                     index < matrix_size * head_dimension;
                     index += 32 * simdgroup_count) {
                    uint dimension = index / matrix_size;
                    uint summary_column = index - dimension * matrix_size;
                    bool valid = summary_column < skipped_count;
                    uint block = valid ? skipped_blocks[summary_column] : 0;
                    uint summary_offset =
                        (head * KEY_BLOCK_COUNT + block) * head_dimension + dimension;
                    key_value_shared[dimension * key_stride + summary_column] = valid
                        ? key_centroids[summary_offset]
                        : bfloat(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                thread simdgroup_matrix<float, 8, 8> proxy_scores;
                proxy_scores.thread_elements()[0] = 0.0f;
                proxy_scores.thread_elements()[1] = 0.0f;
                for (uint frag = 0; frag < matrix_count; ++frag) {
                    thread simdgroup_matrix<bfloat, 8, 8> query_fragment;
                    thread simdgroup_matrix<bfloat, 8, 8> centroid_fragment;
                    uint query_dimension = frag * matrix_size + matrix_column;
                    uint query_row = simd_group * matrix_size + matrix_row;
                    query_fragment.thread_elements()[0] =
                        query_shared[query_row * query_stride + query_dimension];
                    query_fragment.thread_elements()[1] =
                        query_shared[query_row * query_stride + query_dimension + 1];
                    uint centroid_dimension = frag * matrix_size + matrix_row;
                    centroid_fragment.thread_elements()[0] =
                        key_value_shared[centroid_dimension * key_stride + matrix_column];
                    centroid_fragment.thread_elements()[1] =
                        key_value_shared[centroid_dimension * key_stride + matrix_column + 1];
                    thread simdgroup_matrix<float, 8, 8> next_proxy_scores;
                    simdgroup_multiply_accumulate(
                        next_proxy_scores,
                        query_fragment,
                        centroid_fragment,
                        proxy_scores
                    );
                    proxy_scores = next_proxy_scores;
                }

                bool valid0 = matrix_column < skipped_count;
                bool valid1 = matrix_column + 1 < skipped_count;
                float score0 = valid0
                    ? proxy_scores.thread_elements()[0] * attention_scale
                    : -INFINITY;
                float score1 = valid1
                    ? proxy_scores.thread_elements()[1] * attention_scale
                    : -INFINITY;
                float batch_maximum = metal::max(score0, score1);
                batch_maximum = metal::max(
                    batch_maximum,
                    simd_shuffle_xor(batch_maximum, ushort(1))
                );
                batch_maximum = metal::max(
                    batch_maximum,
                    simd_shuffle_xor(batch_maximum, ushort(8))
                );
                float new_maximum = metal::max(row_maximum, batch_maximum);
                float alpha = metal::fast::exp2(row_maximum - new_maximum);
                float probability0 = metal::fast::exp2(score0 - new_maximum);
                float probability1 = metal::fast::exp2(score1 - new_maximum);
                uint block0 = valid0 ? skipped_blocks[matrix_column] : 0;
                uint block1 = valid1 ? skipped_blocks[matrix_column + 1] : 0;
                uint start0 = block0 * block_size;
                uint start1 = block1 * block_size;
                float length0 = valid0
                    ? float(metal::min(start0 + block_size, uint(TOKEN_COUNT)) - start0)
                    : 0.0f;
                float length1 = valid1
                    ? float(metal::min(start1 + block_size, uint(TOKEN_COUNT)) - start1)
                    : 0.0f;
                float weighted_sum = probability0 * length0 + probability1 * length1;
                weighted_sum += simd_shuffle_xor(weighted_sum, ushort(1));
                weighted_sum += simd_shuffle_xor(weighted_sum, ushort(8));

                thread simdgroup_matrix<float, 8, 8> probabilities;
                probabilities.thread_elements()[0] = probability0;
                probabilities.thread_elements()[1] = probability1;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint index = thread_linear;
                     index < matrix_size * head_dimension;
                     index += 32 * simdgroup_count) {
                    uint summary_row = index / head_dimension;
                    uint dimension = index - summary_row * head_dimension;
                    bool valid = summary_row < skipped_count;
                    uint block = valid ? skipped_blocks[summary_row] : 0;
                    uint summary_offset =
                        (head * KEY_BLOCK_COUNT + block) * head_dimension + dimension;
                    key_value_shared[summary_row * value_stride + dimension] = valid
                        ? value_sums[summary_offset]
                        : bfloat(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint frag = 0; frag < matrix_count; ++frag) {
                    accumulated[frag].thread_elements()[0] *= alpha;
                    accumulated[frag].thread_elements()[1] *= alpha;
                    thread simdgroup_matrix<bfloat, 8, 8> value_fragment;
                    uint output_dimension = frag * matrix_size + matrix_column;
                    value_fragment.thread_elements()[0] =
                        key_value_shared[matrix_row * value_stride + output_dimension];
                    value_fragment.thread_elements()[1] =
                        key_value_shared[matrix_row * value_stride + output_dimension + 1];
                    thread simdgroup_matrix<float, 8, 8> next_accumulated;
                    simdgroup_multiply_accumulate(
                        next_accumulated,
                        probabilities,
                        value_fragment,
                        accumulated[frag]
                    );
                    accumulated[frag] = next_accumulated;
                }
                row_sum = row_sum * alpha + weighted_sum;
                row_maximum = new_maximum;
                skipped_count = 0;
            }

            if (query_valid) {
                uint output_base = (head * QUERY_COUNT + local_query) * head_dimension;
                for (uint frag = 0; frag < matrix_count; ++frag) {
                    uint output_dimension = frag * matrix_size + matrix_column;
                    output[output_base + output_dimension] = OutputT(
                        accumulated[frag].thread_elements()[0] / row_sum
                    );
                    output[output_base + output_dimension + 1] = OutputT(
                        accumulated[frag].thread_elements()[1] / row_sum
                    );
                }
            }
        """,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )
    #endif
}
