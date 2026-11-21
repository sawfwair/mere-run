import MLX

#if os(macOS)
enum Inkling2Router {
    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_route128",
        inputNames: ["logits", "bias", "route_scale"],
        outputNames: ["indices", "weights"],
        source: """
            const uint lane = thread_index_in_threadgroup;
            const uint token = threadgroup_position_in_grid.x;
            const float score = 1.0f / (1.0f + exp(-logits[token * 128 + lane]));
            threadgroup float raw[128];
            threadgroup float corrected[128];
            threadgroup uint expert[128];
            raw[lane] = score;
            corrected[lane] = score + bias[lane];
            expert[lane] = lane;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint size = 2; size <= 128; size <<= 1) {
                for (uint distance = size >> 1; distance > 0; distance >>= 1) {
                    const uint partner = lane ^ distance;
                    if (lane < partner) {
                        const float a = corrected[lane], b = corrected[partner];
                        const uint first = expert[lane], second = expert[partner];
                        const bool preferred = a > b || (a == b && first < second);
                        if (preferred != ((lane & size) == 0)) {
                            corrected[lane] = b;
                            corrected[partner] = a;
                            expert[lane] = second;
                            expert[partner] = first;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if (lane < 8) {
                float sum = 0;
                for (uint slot = 0; slot < 8; ++slot) sum += raw[expert[slot]];
                indices[token * 8 + lane] = expert[lane];
                weights[token * 8 + lane] = raw[expert[lane]] / (sum + 1e-20f) * route_scale;
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
