#if os(macOS)
import Foundation
import MereRunEnvironment
import MLX
import MLXFast

extension M5TiledAffineQ4 {
    @TaskLocal static var prefillFFNRowReuse =
        MereRunEnvironmentSwitch.isEnabled("MERERUN_Q38_PREFILL_FFN_ROW_REUSE", default: false)

    static func prefillFFNRowTiles(width: Int, input: Int, output: Int) -> Int {
        guard prefillFFNRowReuse, width >= 512 else { return 1 }
        switch (input, output) {
        case (5_120, 17_408) where width.isMultiple(of: 32): return 2
        case (17_408, 5_120) where width.isMultiple(of: 64): return 4
        default: return 1
        }
    }

    // Each SIMD group unpacks the same 32 weight columns once and applies them
    // to two or four independent 16-row inputs before advancing in K.
    static let prefillFFNProjection = MLXFast.metalKernel(
        name: "mere_m5_group_affine_prefill_ffn_rows",
        inputNames: ["w0", "s0", "b0", "w1", "s1", "b1", "x", "sums"], outputNames: ["y"],
        source: """
            const int k = K, kg = K / 64, n = N;
            const int rowBase = int(threadgroup_position_in_grid.y) * 16 * RowTiles, rows = sums_shape[1];
            constexpr int activeRows = 16 * RowTiles;
            \(groupProjectionBindings)
            float bankValues[RowTiles][16] = {}, bankCorrections[RowTiles][16] = {};
            MereGroupLoad next = mere_group_fetch(words, scales, biases, sums + rowBase, first, nb, mb, rows);
            for (int group = first; group < last; ++group) {
                MereGroupLoad current = next;
                \(unpackGroupCodes)
                if (group + 1 < last) next = mere_group_fetch(words, scales, biases, sums + rowBase, group + 1, nb, mb, rows);
                #pragma unroll
                for (int bank = 0; bank < RowTiles; ++bank) {
                    if (bank > 0) {
                        current.x0 = sums[group * rows + rowBase + bank * 16 + mb];
                        current.x1 = sums[group * rows + rowBase + bank * 16 + mb + 8];
                    }
                    auto input = a.slice(group * 64, bank * 16);
                    operation.run(input, codes, product);
                    thread float* values = bankValues[bank];
                    thread float* corrections = bankCorrections[bank];
                    \(accumulateGroupProducts)
                }
            }
            threadgroup float partial[(Splits > 1 ? Splits - 1 : 1) * 16 * 32];
            #pragma unroll
            for (int bank = 0; bank < RowTiles; ++bank) {
                thread float* values = bankValues[bank];
                \(reduceGroupSplits)
                if (split == 0) for (int i = 0; i < 16; ++i) {
                    const int row = rowBase + bank * 16 + mb + 8 * ((i >> 2) & 1);
                    const int col = outColumn + nb + 16 * (i >> 3) + (i & 3);
                    y[ulong(row) * stride + col] = bfloat(values[i]);
                }
                if (Splits > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """, header: groupHeader, ensureRowContiguous: true
    )
}
#endif
