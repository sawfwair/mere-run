#if os(macOS)
import MLX
import MLXFast

extension PackedAffineQ4Prefill {
    // One SIMD group owns eight rows and 256 output columns. Compact columns
    // with an ambiguous result and recompute them eight at a time, using the
    // same FP32 8x8 MMA and BF16 weight rounding as the reference QMM.
    static let repairEarlyGDN = MLXFast.metalKernel(
        name: "mere_gdn_prefill_repair", inputNames: ["x", "w", "s", "b", "marked"], outputNames: ["y"],
        source: """
            using namespace mlx::steel;
            constexpr int span = 256;
            const int k = x_shape[1], n = w_shape[0], lane = thread_index_in_simdgroup;
            const int rowBase = int(threadgroup_position_in_grid.y) * 8;
            const int colBase = int(threadgroup_position_in_grid.x) * span;
            threadgroup ushort selected[span];
            int count = 0;
            for (int offset = lane; offset < span; offset += 32) {
                uint bits[8];
                bool needsRepair = false;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    bits[i] = marked[(rowBase + i) * n + colBase + offset];
                    needsRepair |= (bits[i] >> 16) != 0;
                }
                const int index = simd_prefix_exclusive_sum(int(needsRepair));
                if (needsRepair) {
                    selected[count + index] = ushort(offset);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                        y[(rowBase + i) * n + colBase + offset] = as_type<bfloat>(ushort(bits[i]));
                }
                count += simd_sum(int(needsRepair));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            using Fragment = BaseMMAFrag<float, 8, 8>;
            const short2 coord = Fragment::get_coord(lane);
            for (int base = 0; base < count; base += 8) {
                const int remaining = count - base;
                const int col0 = colBase + selected[base + min(int(coord.x), remaining - 1)];
                const int col1 = colBase + selected[base + min(int(coord.x) + 1, remaining - 1)];
                Fragment::frag_type acc = 0;
                for (int group = 0; group < k / 64; ++group) {
                    const int meta0 = col0 * (k / 64) + group, meta1 = col1 * (k / 64) + group;
                    const float2 scale(s[meta0], s[meta1]), bias(b[meta0], b[meta1]);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const uint2 words(w[col0 * (k / 8) + group * 8 + j], w[col1 * (k / 8) + group * 8 + j]);
                        const float2 codes = float2((words >> (4 * coord.y)) & uint2(15));
                        float2 weights = float2(bfloat2(fma(codes, scale, bias)));
                        const int input = (rowBase + coord.y) * k + group * 64 + j * 8 + coord.x;
                        float2 values = float2(*reinterpret_cast<const device bfloat2*>(x + input));
                        Fragment::mma(acc, values, weights, acc);
                    }
                }
                if (coord.x < remaining) y[(rowBase + coord.y) * n + col0] = bfloat(acc[0]);
                if (coord.x + 1 < remaining) y[(rowBase + coord.y) * n + col1] = bfloat(acc[1]);
            }
            """, header: "// MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS\n", ensureRowContiguous: true
    )
}
#endif
