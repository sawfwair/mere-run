import Foundation
import MLX
import MLXFast

struct MiniMaxH3FastVSAGeometry: Sendable, Equatable {
    static let tileSize = 64
    static let videoTileShape = (temporal: 4, height: 4, width: 4)

    let paddedToOriginal: [Int32]
    let originalToPadded: [Int32]
    let blockSizes: [Int32]
    let prefixTileCount: Int
    let videoTileCount: Int

    var tileCount: Int { blockSizes.count }
    var paddedTokenCount: Int { paddedToOriginal.count }

    init(layout: MiniMaxH3PackedLayout) {
        let prefixSegments = [
            layout.textRows.count,
            layout.conditionRows.count,
            layout.targetAudioRows.count,
        ].filter { $0 > 0 }
        let prefixCount = prefixSegments.reduce(0, +)
        precondition(prefixCount == layout.targetVideoRows.lowerBound)

        var tiles: [[Int32]] = []
        var prefixStart = 0
        for segment in prefixSegments {
            for start in stride(from: 0, to: segment, by: Self.tileSize) {
                let count = min(Self.tileSize, segment - start)
                tiles.append((0..<count).map { Int32(prefixStart + start + $0) })
            }
            prefixStart += segment
        }
        self.prefixTileCount = tiles.count

        let temporal = layout.videoLatentFrames
        let height = layout.latentHeight / 2
        let width = layout.latentWidth / 2
        let shape = Self.videoTileShape
        for tileT in stride(from: 0, to: temporal, by: shape.temporal) {
            for tileH in stride(from: 0, to: height, by: shape.height) {
                for tileW in stride(from: 0, to: width, by: shape.width) {
                    var tile: [Int32] = []
                    tile.reserveCapacity(Self.tileSize)
                    for t in tileT..<min(tileT + shape.temporal, temporal) {
                        for h in tileH..<min(tileH + shape.height, height) {
                            for w in tileW..<min(tileW + shape.width, width) {
                                tile.append(Int32(prefixCount + (t * height + h) * width + w))
                            }
                        }
                    }
                    tiles.append(tile)
                }
            }
        }
        self.videoTileCount = tiles.count - prefixTileCount
        precondition(videoTileCount > 0)

        let sentinel = Int32(layout.sequenceLength)
        var paddedToOriginal: [Int32] = []
        var originalToPadded = Array(repeating: Int32(-1), count: layout.sequenceLength)
        var blockSizes: [Int32] = []
        paddedToOriginal.reserveCapacity(tiles.count * Self.tileSize)
        blockSizes.reserveCapacity(tiles.count)
        for tile in tiles {
            precondition(!tile.isEmpty && tile.count <= Self.tileSize)
            blockSizes.append(Int32(tile.count))
            let paddedStart = paddedToOriginal.count
            for (offset, original) in tile.enumerated() {
                precondition(original >= 0 && Int(original) < layout.sequenceLength)
                precondition(originalToPadded[Int(original)] == -1)
                originalToPadded[Int(original)] = Int32(paddedStart + offset)
                paddedToOriginal.append(original)
            }
            paddedToOriginal += Array(repeating: sentinel, count: Self.tileSize - tile.count)
        }
        precondition(originalToPadded.allSatisfy { $0 >= 0 })
        precondition(blockSizes.reduce(0, { $0 + Int($1) }) == layout.sequenceLength)
        self.paddedToOriginal = paddedToOriginal
        self.originalToPadded = originalToPadded
        self.blockSizes = blockSizes
    }
}

struct MiniMaxH3FastVSAPreparedContext {
    let geometry: MiniMaxH3FastVSAGeometry
    let paddedIndices: MLXArray
    let originalIndices: MLXArray
    let blockSizes: MLXArray
    let poolingSizes: MLXArray
}

/// Metal implementation of FastVideo's released VSA-H3 tile-64 inference contract.
///
/// Prefix query tiles remain dense, every query retains all prefix key tiles,
/// and video keys are selected by per-head top-k pooled QK scores. The selected
/// blocks use exact token attention; the trained compression branch separately
/// supplies dense pooled-value context through the released gate projection.
enum MiniMaxH3FastVSA {
    static let sparsity: Float = 0.9
    static let headDimension = 128

    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        compressionGate: MLXArray,
        layout: MiniMaxH3PackedLayout,
        sparsity: Float = sparsity
    ) -> MLXArray? {
        call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: compressionGate,
            prepared: prepare(layout: layout),
            sparsity: sparsity
        )
    }

    static func prepare(layout: MiniMaxH3PackedLayout) -> MiniMaxH3FastVSAPreparedContext {
        let geometry = MiniMaxH3FastVSAGeometry(layout: layout)
        let paddedIndices = MLXArray(geometry.paddedToOriginal)
        let originalIndices = MLXArray(geometry.originalToPadded)
        let blockSizes = MLXArray(geometry.blockSizes)
        let poolingSizes = blockSizes.asType(.float32)
            .reshaped(1, 1, geometry.tileCount, 1)
        MLX.eval(paddedIndices, originalIndices, blockSizes, poolingSizes)
        return MiniMaxH3FastVSAPreparedContext(
            geometry: geometry,
            paddedIndices: paddedIndices,
            originalIndices: originalIndices,
            blockSizes: blockSizes,
            poolingSizes: poolingSizes
        )
    }

    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        compressionGate: MLXArray,
        prepared: MiniMaxH3FastVSAPreparedContext,
        sparsity: Float = sparsity
    ) -> MLXArray? {
        guard supports(queries: queries, keys: keys, values: values, gate: compressionGate),
              (0..<1).contains(sparsity) else { return nil }
        let geometry = prepared.geometry
        guard queries.dim(2) == geometry.originalToPadded.count else { return nil }

        // RoPE coefficients are FP32, so the normal H3 projection path can
        // promote Q/K even when the checkpoint and hidden state are BF16.
        // FastH3's released VSA kernels run attention in BF16; normalize every
        // projected branch at this boundary instead of rejecting the faithful
        // FP32 rotary result.
        let attentionQueries = queries.asType(.bfloat16)
        let attentionKeys = keys.asType(.bfloat16)
        let attentionValues = values.asType(.bfloat16)
        let attentionGate = compressionGate.asType(.bfloat16)

        let tiledQueries = tile(attentionQueries, paddedIndices: prepared.paddedIndices)
        let tiledKeys = tile(attentionKeys, paddedIndices: prepared.paddedIndices)
        let tiledValues = tile(attentionValues, paddedIndices: prepared.paddedIndices)
        let tiledGate = tile(attentionGate, paddedIndices: prepared.paddedIndices)

        let pooledQueries = pool(
            tiledQueries,
            sizes: prepared.poolingSizes,
            tileCount: geometry.tileCount
        )
        let pooledKeys = pool(tiledKeys, sizes: prepared.poolingSizes, tileCount: geometry.tileCount)
        let pooledValues = pool(tiledValues, sizes: prepared.poolingSizes, tileCount: geometry.tileCount)
        let scale = 1 / sqrt(Float(headDimension))
        let scores = MLX.matmul(
            pooledQueries,
            pooledKeys.transposed(0, 1, 3, 2)
        ) * scale
        let videoRoutes = selectedVideoRoutes(
            scores: scores,
            prefixTileCount: geometry.prefixTileCount,
            videoTileCount: geometry.videoTileCount,
            sparsity: sparsity
        )
        let sparse = sparseKernelOutput(
            queries: tiledQueries,
            keys: tiledKeys,
            values: tiledValues,
            videoRoutes: videoRoutes,
            blockSizes: prepared.blockSizes,
            prefixTileCount: geometry.prefixTileCount,
            scale: scale
        )

        let compressed = MLX.matmul(
            MLX.softmax(scores, axis: -1, precise: true),
            pooledValues
        ).asType(sparse.dtype)
        let tileCount = geometry.tileCount
        let heads = attentionQueries.dim(1)
        let sparseTiles = sparse.reshaped(1, heads, tileCount, MiniMaxH3FastVSAGeometry.tileSize, headDimension)
        let gateTiles = tiledGate.reshaped(
            1, heads, tileCount, MiniMaxH3FastVSAGeometry.tileSize, headDimension
        )
        let corrected = sparseTiles + compressed.expandedDimensions(axis: 3) * gateTiles
        let tiledOutput = corrected.reshaped(1, heads, geometry.paddedTokenCount, headDimension)
        return MLX.take(tiledOutput, prepared.originalIndices, axis: 2)
    }

    static func routesForTesting(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float = sparsity
    ) -> MLXArray {
        routes(
            scores: scores,
            prefixTileCount: prefixTileCount,
            videoTileCount: videoTileCount,
            sparsity: sparsity
        )
    }

    static func selectedVideoRoutesForTesting(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float = sparsity
    ) -> MLXArray {
        selectedVideoRoutes(
            scores: scores,
            prefixTileCount: prefixTileCount,
            videoTileCount: videoTileCount,
            sparsity: sparsity
        )
    }

    private static func tile(_ value: MLXArray, paddedIndices: MLXArray) -> MLXArray {
        let zero = MLXArray.zeros(
            [value.dim(0), value.dim(1), 1, value.dim(3)],
            dtype: value.dtype
        )
        return MLX.take(MLX.concatenated([value, zero], axis: 2), paddedIndices, axis: 2)
    }

    private static func pool(
        _ value: MLXArray,
        sizes: MLXArray,
        tileCount: Int
    ) -> MLXArray {
        value.asType(.float32)
            .reshaped(1, value.dim(1), tileCount, MiniMaxH3FastVSAGeometry.tileSize, headDimension)
            .sum(axis: 3) / sizes
    }

    private static func routes(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float
    ) -> MLXArray {
        precondition(scores.dim(2) == prefixTileCount + videoTileCount)
        precondition(scores.dim(3) == prefixTileCount + videoTileCount)
        let keepVideo = max(1, min(Int(ceil((1 - sparsity) * Float(videoTileCount))), videoTileCount))
        let videoScores = scores[.ellipsis, prefixTileCount...]
        let videoRoutes: MLXArray
        if keepVideo == videoTileCount {
            videoRoutes = MLXArray.ones(videoScores.shape, dtype: .uint8)
        } else {
            let selected = MLX.argPartition(
                -videoScores,
                kth: keepVideo - 1,
                axis: -1
            )[.ellipsis, 0..<keepVideo]
            videoRoutes = MLX.putAlong(
                MLXArray.zeros(videoScores.shape, dtype: .uint8),
                selected,
                values: MLXArray.ones(selected.shape, dtype: .uint8),
                axis: -1
            )
        }
        var result = prefixTileCount == 0
            ? videoRoutes
            : MLX.concatenated([
                MLXArray.ones(
                    [scores.dim(0), scores.dim(1), scores.dim(2), prefixTileCount],
                    dtype: .uint8
                ),
                videoRoutes,
            ], axis: -1)
        if prefixTileCount > 0 {
            result = MLX.concatenated([
                MLXArray.ones(
                    [scores.dim(0), scores.dim(1), prefixTileCount, scores.dim(3)],
                    dtype: .uint8
                ),
                result[0..., 0..., prefixTileCount..., 0...],
            ], axis: 2)
        }
        return result
    }

    private static func selectedVideoRoutes(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float
    ) -> MLXArray {
        precondition(scores.dim(2) == prefixTileCount + videoTileCount)
        precondition(scores.dim(3) == prefixTileCount + videoTileCount)
        let keepVideo = max(1, min(Int(ceil((1 - sparsity) * Float(videoTileCount))), videoTileCount))
        let videoScores = scores[.ellipsis, prefixTileCount...]
        if keepVideo == videoTileCount {
            let indices = (
                MLX.arange(videoTileCount, dtype: .int32) + MLXArray(Int32(prefixTileCount))
            ).reshaped(1, 1, 1, videoTileCount)
            return MLX.broadcast(indices, to: Array(scores.shape.dropLast()) + [videoTileCount])
        }
        return (
            MLX.argPartition(-videoScores, kth: keepVideo - 1, axis: -1)[.ellipsis, 0..<keepVideo]
                .asType(.int32) + MLXArray(Int32(prefixTileCount))
        )
    }

    private static func supports(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        gate: MLXArray
    ) -> Bool {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        return Device.defaultDevice().deviceType == .gpu
            && queries.shape == keys.shape
            && queries.shape == values.shape
            && queries.shape == gate.shape
            && queries.ndim == 4
            && queries.dim(0) == 1
            && queries.dim(3) == headDimension
        #else
        return false
        #endif
    }

    private static func sparseKernelOutput(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        videoRoutes: MLXArray,
        blockSizes: MLXArray,
        prefixTileCount: Int,
        scale: Float
    ) -> MLXArray {
        let tokenCount = queries.dim(2)
        let blockCount = blockSizes.dim(0)
        let keepVideo = videoRoutes.dim(3)
        precondition(tokenCount == blockCount * MiniMaxH3FastVSAGeometry.tileSize)
        precondition(videoRoutes.dim(2) == blockCount)
        precondition(prefixTileCount + keepVideo <= blockCount)
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        return attentionKernel(
            [queries, keys, values, videoRoutes, blockSizes, MLXArray([scale])],
            template: [
                ("TOKEN_COUNT", tokenCount),
                ("BLOCK_COUNT", blockCount),
                ("PREFIX_TILE_COUNT", prefixTileCount),
                ("KEEP_VIDEO", keepVideo),
            ],
            grid: (32, ((tokenCount + 31) / 32) * 4, queries.dim(1)),
            threadGroup: (32, 4, 1),
            outputShapes: [queries.shape],
            outputDTypes: [.bfloat16]
        )[0]
        #else
        preconditionFailure("FastH3 VSA requires Metal")
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let attentionKernel = MLXFast.metalKernel(
        name: "mere_fasth3_vsa64_online_softmax_bf16_d128_compact_v2",
        inputNames: ["queries", "keys", "values", "video_routes", "block_sizes", "scale_value"],
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
            uint local_query = group_query_start + simd_group * matrix_size + matrix_row;
            uint safe_query = metal::min(local_query, uint(TOKEN_COUNT - 1));
            uint query_block = safe_query / block_size;
            uint query_offset = safe_query - query_block * block_size;
            bool query_valid = local_query < TOKEN_COUNT
                && query_offset < uint(block_sizes[query_block]);
            float attention_scale = float(scale_value[0]) * log2e;

            threadgroup bfloat query_shared[query_tile_rows * query_stride];
            threadgroup bfloat key_value_shared[head_dimension * key_stride];
            for (uint index = thread_linear;
                 index < query_tile_rows * head_dimension;
                 index += 32 * simdgroup_count) {
                uint row = index / head_dimension;
                uint dimension = index - row * head_dimension;
                uint candidate = group_query_start + row;
                bool valid = candidate < TOKEN_COUNT;
                uint token = metal::min(candidate, uint(TOKEN_COUNT - 1));
                uint block = token / block_size;
                valid = valid && token - block * block_size < uint(block_sizes[block]);
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

            uint route_count = query_block < PREFIX_TILE_COUNT
                ? BLOCK_COUNT
                : PREFIX_TILE_COUNT + KEEP_VIDEO;
            for (uint route_index = 0; route_index < route_count; ++route_index) {
                uint key_block;
                if (query_block < PREFIX_TILE_COUNT || route_index < PREFIX_TILE_COUNT) {
                    key_block = route_index;
                } else {
                    uint route_offset = (head * BLOCK_COUNT + query_block) * KEEP_VIDEO
                        + route_index - PREFIX_TILE_COUNT;
                    key_block = uint(video_routes[route_offset]);
                }
                uint key_start = key_block * block_size;
                uint key_end = key_start + uint(block_sizes[key_block]);
                for (uint key_tile = key_start; key_tile < key_end; key_tile += matrix_size) {
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
                        simdgroup_multiply_accumulate(next_scores, query_fragment, key_fragment, scores);
                        scores = next_scores;
                    }

                    float score0 = scores.thread_elements()[0] * attention_scale;
                    float score1 = scores.thread_elements()[1] * attention_scale;
                    if (key_tile + matrix_column >= key_end) score0 = -INFINITY;
                    if (key_tile + matrix_column + 1 >= key_end) score1 = -INFINITY;
                    float tile_maximum = metal::max(score0, score1);
                    tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(1)));
                    tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(8)));
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
                            next_accumulated, probabilities, value_fragment, accumulated[frag]
                        );
                        accumulated[frag] = next_accumulated;
                    }
                }
            }

            if (query_valid) {
                uint output_base = (head * TOKEN_COUNT + local_query) * head_dimension;
                for (uint frag = 0; frag < matrix_count; ++frag) {
                    uint output_dimension = frag * matrix_size + matrix_column;
                    output[output_base + output_dimension] = bfloat(
                        accumulated[frag].thread_elements()[0] / row_sum
                    );
                    output[output_base + output_dimension + 1] = bfloat(
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
