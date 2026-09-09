import Foundation
import MLX
import MLXFast
import MLXNN

func ltxDiffVAENeighborhoodAttention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    kernel: (Int, Int, Int),
    scoreBudget: Int
) -> MLXArray {
    let batch = query.dim(0)
    let dims = [query.dim(1), query.dim(2), query.dim(3)]
    let kernels = [kernel.0, kernel.1, kernel.2]
    let bounds = zip(dims, kernels).map { ltxDiffVAEWindowBounds(length: $0, kernel: $1) }
    var tile = dims
    func scoreCost(_ candidate: [Int]) -> Int {
        let queries = candidate.reduce(1, *)
        let references = zip(zip(candidate, kernels), dims)
            .map { min($1, $0.0 + $0.1 - 1) }
            .reduce(1, *)
        return queries * references
    }
    while scoreCost(tile) > scoreBudget, tile.max()! > 1 {
        let axis = (0..<3).max { lhs, rhs in
            Double(tile[lhs]) / Double(kernels[lhs]) < Double(tile[rhs]) / Double(kernels[rhs])
        }!
        tile[axis] = max(1, (tile[axis] + 1) / 2)
    }

    var temporalParts: [MLXArray] = []
    for temporalStart in stride(from: 0, to: dims[0], by: tile[0]) {
        let temporalEnd = min(temporalStart + tile[0], dims[0])
        var heightParts: [MLXArray] = []
        for heightStart in stride(from: 0, to: dims[1], by: tile[1]) {
            let heightEnd = min(heightStart + tile[1], dims[1])
            var widthParts: [MLXArray] = []
            for widthStart in stride(from: 0, to: dims[2], by: tile[2]) {
                let widthEnd = min(widthStart + tile[2], dims[2])
                let queryRanges = [
                    temporalStart..<temporalEnd,
                    heightStart..<heightEnd,
                    widthStart..<widthEnd,
                ]
                let referenceRanges: [Range<Int>] = (0..<3).map { axis in
                    let lower = bounds[axis].starts[queryRanges[axis].lowerBound]
                    let upper = bounds[axis].ends[queryRanges[axis].upperBound - 1]
                    return lower..<upper
                }
                let queryTile = query[
                    0...,
                    queryRanges[0],
                    queryRanges[1],
                    queryRanges[2],
                    0...,
                    0...
                ]
                let keyTile = key[
                    0...,
                    referenceRanges[0],
                    referenceRanges[1],
                    referenceRanges[2],
                    0...,
                    0...
                ]
                let valueTile = value[
                    0...,
                    referenceRanges[0],
                    referenceRanges[1],
                    referenceRanges[2],
                    0...,
                    0...
                ]
                let queryCount = queryRanges.map(\.count).reduce(1, *)
                let referenceCount = referenceRanges.map(\.count).reduce(1, *)
                let heads = query.dim(4)
                let headDimension = query.dim(5)
                let flattenedQuery = queryTile
                    .transposed(0, 4, 1, 2, 3, 5)
                    .reshaped(batch, heads, queryCount, headDimension)
                let flattenedKey = keyTile
                    .transposed(0, 4, 1, 2, 3, 5)
                    .reshaped(batch, heads, referenceCount, headDimension)
                let flattenedValue = valueTile
                    .transposed(0, 4, 1, 2, 3, 5)
                    .reshaped(batch, heads, referenceCount, headDimension)
                let mask = ltxDiffVAENeighborhoodMask(
                    queryRanges: queryRanges,
                    referenceRanges: referenceRanges,
                    bounds: bounds,
                    dtype: query.dtype
                )
                let attended = MLXFast.scaledDotProductAttention(
                    queries: flattenedQuery,
                    keys: flattenedKey,
                    values: flattenedValue,
                    scale: 1,
                    mask: .array(mask)
                )
                widthParts.append(
                    attended
                        .reshaped(
                            batch,
                            heads,
                            queryRanges[0].count,
                            queryRanges[1].count,
                            queryRanges[2].count,
                            headDimension
                        )
                        .transposed(0, 2, 3, 4, 1, 5)
                )
            }
            heightParts.append(MLX.concatenated(widthParts, axis: 3))
        }
        temporalParts.append(MLX.concatenated(heightParts, axis: 2))
    }
    return MLX.concatenated(temporalParts, axis: 1)
}

enum LTXDiffVAEMetalNeighborhoodAttention {
    static func apply(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        kernel: (Int, Int, Int)
    ) -> MLXArray? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        guard ProcessInfo.processInfo.environment["MERERUN_LTX_DIFFVAE_ATTENTION"] != "mlx",
              Device.defaultDevice().deviceType == .gpu,
              query.shape == key.shape,
              query.shape == value.shape,
              query.ndim == 6,
              query.dim(5) == 64,
              query.dtype == key.dtype,
              query.dtype == value.dtype,
              query.dtype == .bfloat16 || query.dtype == .float16 || query.dtype == .float32,
              query.dim(1) >= kernel.0,
              query.dim(2) >= kernel.1,
              query.dim(3) >= kernel.2 else {
            return nil
        }
        let tokenCount = query.dim(1) * query.dim(2) * query.dim(3)
        let batchHeads = query.dim(0) * query.dim(4)
        return neighborhoodKernel(
            [query, key, value],
            template: [
                ("Frames", query.dim(1)),
                ("Height", query.dim(2)),
                ("Width", query.dim(3)),
                ("Heads", query.dim(4)),
                ("KernelT", kernel.0),
                ("KernelH", kernel.1),
                ("KernelW", kernel.2),
                ("OutputT", query.dtype),
            ],
            grid: (32, tokenCount, batchHeads),
            threadGroup: (32, 1, 1),
            outputShapes: [query.shape],
            outputDTypes: [query.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let neighborhoodKernel = MLXFast.metalKernel(
        name: "mere_ltx_diffvae_neighborhood_attention_3d_d64_v1",
        inputNames: ["queries", "keys", "values"],
        outputNames: ["output"],
        source: """
            constexpr uint HeadDimension = 64;
            uint dimension = thread_position_in_threadgroup.x;
            uint upper_dimension = dimension + 32;
            uint query_token = threadgroup_position_in_grid.y;
            uint batch_head = threadgroup_position_in_grid.z;
            uint head = batch_head % Heads;
            uint batch = batch_head / Heads;
            uint query_width = query_token % Width;
            uint query_plane = query_token / Width;
            uint query_height = query_plane % Height;
            uint query_frame = query_plane / Height;

            uint temporal_start = metal::min(
                query_frame > KernelT / 2 ? query_frame - KernelT / 2 : 0,
                uint(Frames - KernelT)
            );
            uint height_start = metal::min(
                query_height > KernelH / 2 ? query_height - KernelH / 2 : 0,
                uint(Height - KernelH)
            );
            uint width_start = metal::min(
                query_width > KernelW / 2 ? query_width - KernelW / 2 : 0,
                uint(Width - KernelW)
            );
            uint query_offset = (
                (batch * Frames * Height * Width + query_token) * Heads + head
            ) * HeadDimension;
            float query_value = float(queries[query_offset + dimension]);
            float upper_query_value = float(queries[query_offset + upper_dimension]);
            float row_maximum = -INFINITY;
            float row_sum = 0.0f;
            float accumulated = 0.0f;
            float upper_accumulated = 0.0f;

            for (uint temporal = 0; temporal < KernelT; ++temporal) {
                uint key_frame = temporal_start + temporal;
                for (uint y = 0; y < KernelH; ++y) {
                    uint key_height = height_start + y;
                    for (uint x = 0; x < KernelW; ++x) {
                        uint key_width = width_start + x;
                        uint key_token = (key_frame * Height + key_height) * Width + key_width;
                        uint key_offset = (
                            (batch * Frames * Height * Width + key_token) * Heads + head
                        ) * HeadDimension;
                        float score = simd_sum(
                            query_value * float(keys[key_offset + dimension])
                                + upper_query_value * float(keys[key_offset + upper_dimension])
                        );
                        float next_maximum = metal::max(row_maximum, score);
                        float previous_scale = metal::fast::exp(row_maximum - next_maximum);
                        float current_scale = metal::fast::exp(score - next_maximum);
                        accumulated = accumulated * previous_scale
                            + current_scale * float(values[key_offset + dimension]);
                        upper_accumulated = upper_accumulated * previous_scale
                            + current_scale * float(values[key_offset + upper_dimension]);
                        row_sum = row_sum * previous_scale + current_scale;
                        row_maximum = next_maximum;
                    }
                }
            }
            output[query_offset + dimension] = OutputT(accumulated / row_sum);
            output[query_offset + upper_dimension] = OutputT(upper_accumulated / row_sum);
        """,
        ensureRowContiguous: true
    )
    #endif
}
