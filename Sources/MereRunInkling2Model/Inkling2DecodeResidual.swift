import MLX

#if os(macOS)
enum Inkling2DecodeResidual {
    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_decode_residual",
        inputNames: ["partial", "block0", "block1", "block2", "block3", "query"],
        outputNames: ["output"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint batch = threadgroup_position_in_grid.y;
            const uint tid = thread_index_in_threadgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint base = batch * WIDTH;
            threadgroup float logits[STATES];
            threadgroup float scales[STATES];

            for (uint block = group; block < STATES; block += 4) {
                float squares = 0, dot = 0;
                for (uint d = lane; d < WIDTH; d += 32) {
                    const float v = inkling2_residual_value<STATES>(block, base + d, partial, block0, block1, block2, block3);
                    squares += v * v;
                    dot += v * query[row * WIDTH + d];
                }
                const float norm = rsqrt(simd_sum(squares) / float(WIDTH) + 1e-6f);
                const float score = simd_sum(dot) * norm * rsqrt(float(WIDTH));
                if (lane == 0) {
                    logits[block] = score;
                    scales[block] = norm;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float maximum = logits[0];
                for (uint block = 1; block < STATES; ++block) maximum = max(maximum, logits[block]);
                float denominator = 0;
                for (uint block = 0; block < STATES; ++block) {
                    logits[block] = exp(logits[block] - maximum);
                    denominator += logits[block];
                }
                for (uint block = 0; block < STATES; ++block) scales[block] *= logits[block] / denominator;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = tid; d < WIDTH; d += 128) {
                float sum = 0;
                for (uint block = 0; block < STATES; ++block) {
                    const float v = inkling2_residual_value<STATES>(block, base + d, partial, block0, block1, block2, block3);
                    sum += v * scales[block];
                }
                output[(batch * QUERIES + row) * WIDTH + d] = sum;
            }
            """,
        header: """
            template <int States, typename P, typename B0, typename B1, typename B2, typename B3>
            inline float inkling2_residual_value(
                uint block, uint index, const device P* partial, const device B0* block0,
                const device B1* block1, const device B2* block2, const device B3* block3
            ) {
                if (block == States - 1) return float(partial[index]);
                switch (block) {
                    case 0: return float(block0[index]);
                    case 1: return float(block1[index]);
                    case 2: return float(block2[index]);
                    default: return float(block3[index]);
                }
            }
            """
    )

    static func apply(_ blocks: [MLXArray], partial: MLXArray, query: MLXArray) -> [MLXArray] {
        let batch = partial.dim(0), tokens = partial.dim(1), width = partial.dim(-1)
        let query = query.asType(.float32).reshaped(-1, width)
        let rows = query.dim(0)
        let inputs = [partial] + blocks + Array(repeating: partial, count: 4 - blocks.count) + [query]
        let result = kernel(inputs, template: [("STATES", blocks.count + 1), ("WIDTH", width), ("QUERIES", rows)],
                            grid: (128 * rows, batch * tokens, 1), threadGroup: (128, 1, 1),
                            outputShapes: [[batch, tokens, rows, width]], outputDTypes: [.float32])[0]
        return (0..<rows).map { result[.ellipsis, $0, 0...] }
    }
}
#endif
