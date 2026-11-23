import MLX

#if os(macOS)
enum Inkling2Router {
    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_route128",
        inputNames: ["logits", "bias", "route_scale"],
        outputNames: ["indices", "weights"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint index = thread_index_in_threadgroup;
            const uint token = threadgroup_position_in_grid.x;
            const float score = 1.0f / (1.0f + exp(-logits[token * 128 + index]));
            threadgroup float raw[128];
            threadgroup float scores[32];
            threadgroup uint experts[32];
            raw[index] = score;
            float corrected = score + bias[index];
            uint expert = index;
            mererun_inkling2_sort_routes(lane, corrected, expert);
            if (lane < 8) {
                scores[group * 8 + lane] = corrected;
                experts[group * 8 + lane] = expert;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0) {
                corrected = scores[lane];
                expert = experts[lane];
                mererun_inkling2_sort_routes(lane, corrected, expert);
                const float selected = raw[expert];
                float sum = 0;
                #pragma unroll
                for (uint slot = 0; slot < 8; ++slot) sum += simd_shuffle(selected, slot);
                if (lane < 8) {
                    indices[token * 8 + lane] = expert;
                    weights[token * 8 + lane] = selected / (sum + 1e-20f) * route_scale;
                }
            }
            """,
        header: """
            // The global top eight must be among the top eight of each SIMD group.
            inline void mererun_inkling2_sort_routes(uint lane, thread float& score, thread uint& expert) {
                #pragma unroll
                for (uint size = 2; size <= 32; size <<= 1) {
                    #pragma unroll
                    for (uint distance = size >> 1; distance > 0; distance >>= 1) {
                        const float other_score = simd_shuffle_xor(score, distance);
                        const uint other_expert = simd_shuffle_xor(expert, distance);
                        const bool better = other_score > score || (other_score == score && other_expert < expert);
                        const bool wants_better = ((lane & size) == 0) == ((lane & distance) == 0);
                        if (better == wants_better) {
                            score = other_score;
                            expert = other_expert;
                        }
                    }
                }
            }
            """
    )

    static func select(_ logits: MLXArray, bias: MLXArray, scale: Float) -> (MLXArray, MLXArray) {
        let tokens = logits.dim(0)
        let result = kernel([logits, bias, scale], grid: (128 * tokens, 1, 1), threadGroup: (128, 1, 1),
                            outputShapes: [[tokens, 8], [tokens, 8]], outputDTypes: [.uint32, .float32])
        return (result[0], result[1])
    }
}
#endif
