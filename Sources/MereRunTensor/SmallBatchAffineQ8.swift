#if os(macOS)
import MLX
import MLXFast

/// Reuse affine-Q8 weights across a complete small verification block. In
/// particular, eight input rows share one read instead of two four-row tiles.
/// Each input retains the native serial Q8 FP32 accumulation order.
package enum SmallBatchAffineQ8 {
    private static let kernel = MLXFast.metalKernel(
        name: "mere_affine8_g64_shared_rows_v1",
        inputNames: ["w", "scales", "biases", "x"], outputNames: ["y"],
        source: """
            const int k_size = x_shape[x_ndim - 1];
            const int n_size = w_shape[0];
            const int out_row = int(threadgroup_position_in_grid.y) * (2 * R)
                + int(simdgroup_index_in_threadgroup) * R;
            const int lane = int(thread_index_in_simdgroup);
            typedef vec<float, M> VF;
            VF acc[R];
            for (int r = 0; r < R; r++) { acc[r] = VF(0.0f); }
            for (int k = 0; k < k_size; k += 256) {
                uint2 packed[R];
                float scale[R], bias[R];
                for (int r = 0; r < R; r++) {
                    const int row = out_row + r;
                    const device uchar* bytes = reinterpret_cast<const device uchar*>(w)
                        + row * k_size + k + lane * 8;
                    packed[r] = *reinterpret_cast<const device uint2*>(bytes);
                    const int group = row * (k_size / 64) + k / 64 + lane / 8;
                    scale[r] = float(scales[group]);
                    bias[r] = float(biases[group]);
                }
                VF sum = VF(0.0f);
                VF partial[R];
                for (int r = 0; r < R; r++) { partial[r] = VF(0.0f); }
                for (int i = 0; i < 8; i++) {
                    VF a;
                    for (int m = 0; m < M; m++) {
                        a[m] = float(x[m * k_size + k + lane * 8 + i]);
                    }
                    sum += a;
                    for (int r = 0; r < R; r++) {
                        const uint word = i < 4 ? packed[r].x : packed[r].y;
                        partial[r] += a * float((word >> ((i % 4) * 8)) & 255);
                    }
                }
                for (int r = 0; r < R; r++) {
                    acc[r] += scale[r] * partial[r] + sum * bias[r];
                }
            }
            for (int r = 0; r < R; r++) {
                for (int m = 0; m < M; m++) {
                    const float value = simd_sum(acc[r][m]);
                    if (lane == 0) { y[m * n_size + out_row + r] = bfloat16_t(value); }
                }
            }
            """,
        ensureRowContiguous: true
    )

    package static func matmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        groupSize: Int, bits: Int, mode: QuantizationMode, rowsPerSIMD: Int
    ) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu,
              bits == 8, groupSize == 64, mode == .affine,
              [1, 2, 4].contains(rowsPerSIMD), x.ndim >= 2, weight.ndim == 2,
              x.dtype == .bfloat16, weight.dtype == .uint32,
              scales.dtype == .bfloat16, biases.dtype == .bfloat16 else { return nil }
        let input = x.dim(-1), output = weight.dim(0)
        // Metal's packed-byte offsets below use signed 32-bit indexing.
        // Larger projections retain native matmul rather than overflowing it.
        guard input > 0, output > 0, output <= Int(Int32.max) / input,
              scales.shape == [output, input / 64], biases.shape == scales.shape else { return nil }
        let rows = x.size / input
        guard (2...8).contains(rows), x.dim(-2) == rows,
              input.isMultiple(of: 256), weight.dim(1) == input / 4,
              output.isMultiple(of: 2 * rowsPerSIMD) else { return nil }
        var shape = x.shape
        shape[shape.count - 1] = output
        return kernel(
            [weight, scales, biases, x], template: [("M", rows), ("R", rowsPerSIMD)],
            grid: (32, output / rowsPerSIMD, 1), threadGroup: (32, 2, 1),
            outputShapes: [shape], outputDTypes: [.bfloat16]
        )[0]
    }
}
#endif
