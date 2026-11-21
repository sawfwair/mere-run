import MLX

#if os(macOS)
enum Inkling2HeadNormalization {
    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_head_normalization",
        inputNames: ["query", "key", "value", "relative", "q_scale", "r_scale", "thresholds", "epsilon", "tau"],
        outputNames: ["q_out", "k_out", "v_out", "r_out"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint token = threadgroup_position_in_grid.y;
            const uint lane = thread_index_in_simdgroup;
            const float position_scale = tau[token % query_shape[1]];
            if (row < HEADS) {
                const uint base = (token * HEADS + row) * WIDTH;
                inkling2_scale_head<T, WIDTH, true, false, true>(query + base, q_out + base, q_scale + row * WIDTH,
                    lane, thresholds[0], epsilon, position_scale);
            } else if (row < HEADS + KV_HEADS) {
                const uint base = (token * KV_HEADS + row - HEADS) * WIDTH;
                inkling2_scale_head<T, WIDTH, false, false, false>(key + base, k_out + base, q_scale,
                    lane, thresholds[0], epsilon, position_scale);
            } else if (row < HEADS + 2 * KV_HEADS) {
                const uint base = (token * KV_HEADS + row - HEADS - KV_HEADS) * WIDTH;
                inkling2_scale_head<T, WIDTH, false, true, false>(value + base, v_out + base, q_scale,
                    lane, thresholds[0], epsilon, position_scale);
            } else {
                const uint head = row - HEADS - 2 * KV_HEADS;
                const uint base = (token * HEADS + head) * REL_WIDTH;
                inkling2_scale_head<T, REL_WIDTH, true, false, true>(relative + base, r_out + base, r_scale + head * REL_WIDTH,
                    lane, thresholds[1], epsilon, position_scale);
            }
            """,
        header: """
            template <typename T, int Width, bool Learned, bool RMS, bool PositionScale, typename Scale>
            inline void inkling2_scale_head(const device T* input, device T* output, const device Scale* scale,
                                           uint lane, float threshold, float epsilon, float tau) {
                float squares = 0;
                for (uint d = lane; d < Width; d += 32) {
                    const float v = float(input[d]) * (Learned ? float(scale[d]) : 1.0f);
                    squares += v * v;
                }
                const float norm = simd_sum(squares);
                const float factor = RMS ? rsqrt(norm / float(Width) + epsilon)
                                         : min(threshold / (sqrt(norm) + 1e-6f), 1.0f);
                for (uint d = lane; d < Width; d += 32) {
                    const float v = float(input[d]) * (Learned ? float(scale[d]) : 1.0f);
                    const T normalized = T(v * factor);
                    output[d] = PositionScale ? T(float(normalized) * tau) : normalized;
                }
            }
            """
    )

    static func apply(
        query: MLXArray, key: MLXArray, value: MLXArray, relative: MLXArray,
        queryScale: MLXArray, relativeScale: MLXArray, thresholds: MLXArray, epsilon: Float, tau: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let heads = query.dim(2), kvHeads = key.dim(2)
        let inputs = [query, key, value, relative]
        let outputs = kernel(
            [query, key, value, relative, queryScale, relativeScale, thresholds, epsilon, tau],
            template: [("T", query.dtype), ("HEADS", heads), ("KV_HEADS", kvHeads),
                       ("WIDTH", query.dim(-1)), ("REL_WIDTH", relative.dim(-1))],
            grid: (32 * (2 * heads + 2 * kvHeads), query.dim(0) * query.dim(1), 1), threadGroup: (32, 1, 1),
            outputShapes: inputs.map(\.shape), outputDTypes: inputs.map(\.dtype)
        )
        return (outputs[0], outputs[1], outputs[2], outputs[3])
    }
}
#endif
