import MLX
import MLXFast

extension MiniMaxH3FastVSA {
    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    static let attentionKernel = MLXFast.metalKernel(
        name: "mere_fasth3_vsa64_online_softmax_bf16_d128_compact_v3",
        inputNames: ["queries", "keys", "values", "video_routes", "block_sizes", "scale_value"],
        outputNames: ["output"],
        source: """
            constexpr uint block_size = 64;
            constexpr uint head_dimension = 128;
            constexpr uint matrix_size = 8;
            constexpr uint matrix_count = head_dimension / matrix_size;
            constexpr uint query_tile_rows = QUERY_TILE_ROWS;
            constexpr uint simdgroup_count = SIMDGROUP_COUNT;
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

    #if DEBUG
    /// Full 64-row query tile with 16-row K/V staging. Two score fragments are
    /// retained before each V load, halving barriers without the register and
    /// occupancy cost of retaining an entire 64-column score tile.
    static let attentionKV16Kernel = MLXFast.metalKernel(
        name: "mere_fasth3_vsa64_online_softmax_bf16_d128_kv16_v1",
        inputNames: ["queries", "keys", "values", "video_routes", "block_sizes", "scale_value"],
        outputNames: ["output"],
        source: """
            constexpr uint block_size = 64;
            constexpr uint head_dimension = 128;
            constexpr uint matrix_size = 8;
            constexpr uint matrix_count = head_dimension / matrix_size;
            constexpr uint key_tile_rows = 16;
            constexpr uint key_matrix_count = key_tile_rows / matrix_size;
            constexpr uint simdgroup_count = 8;
            constexpr float log2e = 1.4426950408889634f;

            uint lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint thread_linear = thread_position_in_threadgroup.y * 32
                + thread_position_in_threadgroup.x;
            uint query_block = threadgroup_position_in_grid.y;
            uint head = threadgroup_position_in_grid.z;
            uint quad = lane / 4;
            uint matrix_row = (quad & 4) + ((lane / 2) % 4);
            uint matrix_column = (quad & 2) * 2 + (lane % 2) * 2;
            uint group_query_start = query_block * block_size;
            uint local_query = group_query_start + simd_group * matrix_size + matrix_row;
            uint query_offset = simd_group * matrix_size + matrix_row;
            bool query_valid = query_offset < uint(block_sizes[query_block]);
            float attention_scale = float(scale_value[0]) * log2e;

            threadgroup bfloat query_shared[block_size * head_dimension];
            threadgroup bfloat key_value_shared[key_tile_rows * head_dimension];
            for (uint index = thread_linear;
                 index < block_size * head_dimension;
                 index += 32 * simdgroup_count) {
                uint row = index / head_dimension;
                uint dimension = index - row * head_dimension;
                uint token = group_query_start + row;
                bool valid = row < uint(block_sizes[query_block]);
                query_shared[index] = valid
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
                uint key_size = uint(block_sizes[key_block]);
                for (uint key_tile_offset = 0;
                     key_tile_offset < key_size;
                     key_tile_offset += key_tile_rows) {
                    uint key_tile_size = metal::min(key_tile_rows, key_size - key_tile_offset);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint index = thread_linear;
                         index < key_tile_rows * head_dimension;
                         index += 32 * simdgroup_count) {
                        uint dimension = index / key_tile_rows;
                        uint key_column = index - dimension * key_tile_rows;
                        key_value_shared[index] = key_column < key_tile_size
                            ? bfloat(keys[
                                (head * TOKEN_COUNT + key_start + key_tile_offset + key_column)
                                    * head_dimension + dimension
                            ])
                            : bfloat(0.0f);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    thread simdgroup_matrix<float, 8, 8> probabilities[key_matrix_count];
                    float tile_maximum = -INFINITY;
                    for (uint key_fragment_index = 0;
                         key_fragment_index < key_matrix_count;
                         ++key_fragment_index) {
                        thread simdgroup_matrix<float, 8, 8> scores;
                        scores.thread_elements()[0] = 0.0f;
                        scores.thread_elements()[1] = 0.0f;
                        for (uint frag = 0; frag < matrix_count; ++frag) {
                            thread simdgroup_matrix<bfloat, 8, 8> query_fragment;
                            thread simdgroup_matrix<bfloat, 8, 8> key_fragment;
                            uint query_dimension = frag * matrix_size + matrix_column;
                            uint query_row = simd_group * matrix_size + matrix_row;
                            query_fragment.thread_elements()[0] =
                                query_shared[query_row * head_dimension + query_dimension];
                            query_fragment.thread_elements()[1] =
                                query_shared[query_row * head_dimension + query_dimension + 1];
                            uint key_dimension = frag * matrix_size + matrix_row;
                            uint key_column = key_fragment_index * matrix_size + matrix_column;
                            key_fragment.thread_elements()[0] =
                                key_value_shared[key_dimension * key_tile_rows + key_column];
                            key_fragment.thread_elements()[1] =
                                key_value_shared[key_dimension * key_tile_rows + key_column + 1];
                            thread simdgroup_matrix<float, 8, 8> next_scores;
                            simdgroup_multiply_accumulate(
                                next_scores, query_fragment, key_fragment, scores
                            );
                            scores = next_scores;
                        }
                        float score0 = scores.thread_elements()[0] * attention_scale;
                        float score1 = scores.thread_elements()[1] * attention_scale;
                        uint key_column = key_fragment_index * matrix_size + matrix_column;
                        if (key_column >= key_tile_size) score0 = -INFINITY;
                        if (key_column + 1 >= key_tile_size) score1 = -INFINITY;
                        scores.thread_elements()[0] = score0;
                        scores.thread_elements()[1] = score1;
                        probabilities[key_fragment_index] = scores;
                        tile_maximum = metal::max(tile_maximum, metal::max(score0, score1));
                    }
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
                    float probability_sum = 0.0f;
                    for (uint key_fragment_index = 0;
                         key_fragment_index < key_matrix_count;
                         ++key_fragment_index) {
                        float probability0 = metal::fast::exp2(
                            probabilities[key_fragment_index].thread_elements()[0] - new_maximum
                        );
                        float probability1 = metal::fast::exp2(
                            probabilities[key_fragment_index].thread_elements()[1] - new_maximum
                        );
                        probabilities[key_fragment_index].thread_elements()[0] = probability0;
                        probabilities[key_fragment_index].thread_elements()[1] = probability1;
                        probability_sum += probability0 + probability1;
                    }
                    probability_sum += simd_shuffle_xor(probability_sum, ushort(1));
                    probability_sum += simd_shuffle_xor(probability_sum, ushort(8));
                    row_sum = row_sum * alpha + probability_sum;
                    row_maximum = new_maximum;
                    for (uint frag = 0; frag < matrix_count; ++frag) {
                        accumulated[frag].thread_elements()[0] *= alpha;
                        accumulated[frag].thread_elements()[1] *= alpha;
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint index = thread_linear;
                         index < key_tile_rows * head_dimension;
                         index += 32 * simdgroup_count) {
                        uint value_row = index / head_dimension;
                        uint dimension = index - value_row * head_dimension;
                        key_value_shared[index] = value_row < key_tile_size
                            ? bfloat(values[
                                (head * TOKEN_COUNT + key_start + key_tile_offset + value_row)
                                    * head_dimension + dimension
                            ])
                            : bfloat(0.0f);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint key_fragment_index = 0;
                         key_fragment_index < key_matrix_count;
                         ++key_fragment_index) {
                        for (uint frag = 0; frag < matrix_count; ++frag) {
                            thread simdgroup_matrix<bfloat, 8, 8> value_fragment;
                            uint value_row = key_fragment_index * matrix_size + matrix_row;
                            uint output_dimension = frag * matrix_size + matrix_column;
                            value_fragment.thread_elements()[0] =
                                key_value_shared[value_row * head_dimension + output_dimension];
                            value_fragment.thread_elements()[1] =
                                key_value_shared[value_row * head_dimension + output_dimension + 1];
                            thread simdgroup_matrix<float, 8, 8> next_accumulated;
                            simdgroup_multiply_accumulate(
                                next_accumulated,
                                probabilities[key_fragment_index],
                                value_fragment,
                                accumulated[frag]
                            );
                            accumulated[frag] = next_accumulated;
                        }
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
    #endif
}
