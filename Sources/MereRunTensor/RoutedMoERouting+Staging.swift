import Foundation
import MLX
import MLXFast

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
extension RoutedMoERouting {
    static let invertPermutationKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_invert_permutation",
        inputNames: ["order"],
        outputNames: ["inverse"],
        source: """
            const uint index = thread_position_in_grid.x;
            if (index < COUNT) {
                inverse[uint(order[index])] = IndexT(index);
            }
        """,
        ensureRowContiguous: true
    )

    static let rankedPrefillRowCopyKernel = MLXFast.metalKernel(
        name: "mere_laguna_ranked_prefill_row_copy_bf16_512x8_v1",
        inputNames: ["source", "order"],
        outputNames: ["sorted"],
        source: """
            constexpr uint hidden_vectors = 2048 / 8;
            constexpr uint experts_per_token = 8;

            uint vector_index = thread_position_in_grid.x;
            uint sorted_row = thread_position_in_grid.y;
            uint original_row = order[sorted_row];
            uint source_row = original_row / experts_per_token;
            const device uint4* source_vectors =
                reinterpret_cast<const device uint4*>(source);
            device uint4* sorted_vectors =
                reinterpret_cast<device uint4*>(sorted);
            sorted_vectors[sorted_row * hidden_vectors + vector_index] =
                source_vectors[source_row * hidden_vectors + vector_index];
        """,
        ensureRowContiguous: true
    )

    static let rankedPrefillRouteMetadataKernel = MLXFast.metalKernel(
        name: "mere_laguna_ranked_prefill_route_metadata_u32_4096_v1",
        inputNames: ["flat_indices", "order"],
        outputNames: ["sorted_indices", "inverse_order"],
        source: """
            uint sorted_row = thread_position_in_grid.x;
            uint original_row = order[sorted_row];
            sorted_indices[sorted_row] = flat_indices[original_row];
            inverse_order[original_row] = sorted_row;
        """,
        ensureRowContiguous: true
    )

    static let sortedExpertTileScheduleKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_sorted_expert_tile_schedule",
        inputNames: ["expert_indices"],
        outputNames: ["tile_starts", "tile_rows", "tile_experts"],
        source: """
            threadgroup uint expert_tile_prefix[256];

            const uint expert = thread_index_in_threadgroup;
            uint lower = 0;
            uint upper = ROUTE_COUNT;
            while (lower < upper) {
                const uint middle = lower + (upper - lower) / 2;
                if (uint(expert_indices[middle]) < expert) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            const uint route_start = lower;

            upper = ROUTE_COUNT;
            while (lower < upper) {
                const uint middle = lower + (upper - lower) / 2;
                if (uint(expert_indices[middle]) <= expert) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            const uint route_count = lower - route_start;
            const uint expert_tiles = (route_count + 15) / 16;
            expert_tile_prefix[expert] = expert_tiles;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint offset = 1;
                 offset < EXPERT_COUNT;
                 offset *= 2) {
                const uint prior =
                    expert >= offset
                        ? expert_tile_prefix[expert - offset]
                        : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                expert_tile_prefix[expert] += prior;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint tile_start =
                expert == 0 ? 0 : expert_tile_prefix[expert - 1];
            for (uint tile = 0; tile < expert_tiles; ++tile) {
                const uint destination = tile_start + tile;
                const uint consumed_rows = tile * 16;
                tile_starts[destination] =
                    IndexT(route_start + consumed_rows);
                tile_rows[destination] =
                    IndexT(min(16u, route_count - consumed_rows));
                tile_experts[destination] = IndexT(expert);
            }

            const uint total_tiles =
                expert_tile_prefix[EXPERT_COUNT - 1];
            for (uint tile = total_tiles + expert;
                 tile < TILE_COUNT;
                 tile += EXPERT_COUNT) {
                tile_starts[tile] = IndexT(ROUTE_COUNT);
                tile_rows[tile] = IndexT(0);
                tile_experts[tile] = IndexT(0);
            }
        """,
        ensureRowContiguous: true
    )

}

#endif
