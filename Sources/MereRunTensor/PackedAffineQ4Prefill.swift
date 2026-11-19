#if os(macOS)
import MLX
import MLXFast

/// Long-row affine Q4 experiments. The SIMD path retains native QMM's
/// BF16 staging and ordered eight-term FP32 matrix accumulations.
enum PackedAffineQ4Prefill {
    enum Engine: String { case simd, tensor, gdnTensor, gdnRepaired }

    // Fragment coordinates and Q4 staging follow Apple's MLX steel QMM
    // helpers under MIT. See THIRD_PARTY_NOTICES.md.
    private static let simd = MLXFast.metalKernel(
        name: "mere_packed_q4_prefill_simd_v3", inputNames: ["x", "w", "s", "b"], outputNames: ["y"],
        source: """
            constexpr int BK = 32, LD = 40;
            threadgroup bfloat16_t xs[BM * LD];
            threadgroup bfloat16_t ws[BN * LD];
            qmm_t_impl<bfloat16_t, 64, 4, false, bfloat16_t, BM, BK, BN>(
                w, s, b, x, y, xs, ws, x_shape[1], w_shape[0], x_shape[0], x_shape[1],
                threadgroup_position_in_grid, thread_index_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            """, header: "// MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS\n", ensureRowContiguous: true
    )

    private static let tensor = makeTensorKernel(recordRounding: false)
    private static let roundingCandidates = makeTensorKernel(recordRounding: true)

    // The low word holds the rounded result. The high word selects outputs
    // near a BF16 midpoint relative to the largest observed partial sum.
    // The margin is empirical and is qualified separately for the GDN shape.
    private static let roundingStore = """
        const float value = acc[i], magnitude = abs(value);
        const float midpoint = as_type<float>((as_type<uint>(magnitude) & 0xffff0000u) | 0x8000u);
        const bool repair = !isfinite(value) || abs(magnitude - midpoint) <= peak[i] * 0x1p-17f;
        y[offset] = uint(as_type<ushort>(bfloat(value))) | (uint(repair) << 16);
        """

    private static func makeTensorKernel(recordRounding: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: recordRounding ? "mere_gdn_prefill_rounding" : "mere_packed_q4_prefill_tensor_v1",
            inputNames: ["x", "w", "s", "b"], outputNames: ["y"],
            source: """
            constexpr int Rows = BM / 2, Columns = BN / 2;
            threadgroup bfloat stages[BN * Step];
            const int tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
            const int k = x_shape[1], m = x_shape[0], n = w_shape[0];
            const int mr = threadgroup_position_in_grid.y * BM + (sg / 2) * Rows;
            const int nc = threadgroup_position_in_grid.x * BN;
            constexpr auto descriptor = matmul2d_descriptor(
                Rows, Columns, Step, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto a = tensor(reinterpret_cast<device bfloat*>(const_cast<device bfloat16_t*>(x)),
                            dextents<int, 2>{k, m}, array<int, 2>{1, k});
            tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> buffer(
                stages, dextents<int, 2>{Step, BN}, array<int, 2>{1, Step});
            auto a0 = a.template slice<Step, Rows>(0, mr);
            auto b0 = buffer.template slice<Step, Columns>(0, (sg % 2) * Columns);
            auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
            for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
            \(recordRounding ? "float peak[Rows * Columns / 32] = {};" : "")
            for (int step = 0; step < k; step += Step) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = tid; i < BN * (Step / 8); i += 128) {
                    const int row = i / (Step / 8), col = (i % (Step / 8)) * 8;
                    if (nc + row < n) {
                        const uint code = w[(nc + row) * (k / 8) + (step + col) / 8];
                        const float scale = float(s[(nc + row) * (k / 64) + step / 64]);
                        const float bias = float(b[(nc + row) * (k / 64) + step / 64]);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            stages[row * Step + col + j] = bfloat(scale * float((code >> (4 * j)) & 15u) + bias);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) stages[row * Step + col + j] = bfloat(0);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto ai = a.template slice<Step, Rows>(step, mr);
                operation.run(ai, b0, acc);
                \(recordRounding ? "for (ushort i = 0; i < acc.get_capacity(); ++i) peak[i] = max(peak[i], abs(acc[i]));" : "")
            }
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto index = acc.get_multidimensional_index(i);
                const int col = nc + (sg % 2) * Columns + index[0];
                if (col < n) {
                    const int offset = (mr + index[1]) * n + col;
                    \(recordRounding ? roundingStore : "y[offset] = bfloat16_t(acc[i]);")
                }
            }
            """, header: """
            #include <metal_tensor>
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
            using namespace metal;
            using namespace mpp::tensor_ops;
            #pragma clang fp reassociate(off)
            """ + "\n", ensureRowContiguous: true
        )
    }

    static func matmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        engine: Engine, tileM: Int = 64, tileN: Int = 64
    ) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu,
              x.dtype == .bfloat16, weight.dtype == .uint32, scales.dtype == .bfloat16, biases.dtype == .bfloat16,
              x.ndim >= 2, weight.ndim == 2 else { return nil }
        let k = x.dim(-1), n = weight.dim(0), rows = x.size / k
        guard k.isMultiple(of: 64), weight.dim(1) == k / 8,
              scales.shape == [n, k / 64], biases.shape == scales.shape else { return nil }
        precondition([32, 64].contains(tileM) && [32, 64].contains(tileN))
        if engine != .simd, !M5AffineQ4Verification.isSupported { return nil }
        var input = x.reshaped(rows, k)
        let paddedRows = ((rows + tileM - 1) / tileM) * tileM
        if engine != .simd, paddedRows != rows {
            input = concatenated([input, MLXArray.zeros([paddedRows - rows, k], dtype: .bfloat16)], axis: 0)
        }
        if engine == .gdnRepaired {
            guard tileM == 64, tileN == 64, k == 5_120, n == 16_384 else { return nil }
            let marked = roundingCandidates([input, weight, scales, biases],
                template: [("BM", 64), ("BN", 64), ("Step", 64)],
                grid: (n / 64 * 128, paddedRows / 64, 1), threadGroup: (128, 1, 1),
                outputShapes: [[paddedRows, n]], outputDTypes: [.uint32])[0]
            let result = repairEarlyGDN([input, weight, scales, biases, marked],
                grid: (n / 256 * 32, paddedRows / 8, 1), threadGroup: (32, 1, 1),
                outputShapes: [[paddedRows, n]], outputDTypes: [.bfloat16])[0]
            return result[0..<rows].reshaped(Array(x.shape.dropLast()) + [n])
        }
        let count = engine == .simd ? rows : paddedRows
        let kernel = engine == .simd ? simd : tensor
        let result = kernel([input, weight, scales, biases], template: [("BM", tileM), ("BN", tileN), ("Step", engine == .gdnTensor ? 64 : 32)],
            grid: (((n + tileN - 1) / tileN) * 128, (count + tileM - 1) / tileM, 1), threadGroup: (128, 1, 1),
            outputShapes: [[count, n]], outputDTypes: [.bfloat16])[0]
        var shape = x.shape
        shape[shape.count - 1] = n
        return result[0..<rows].reshaped(shape)
    }
}
#endif
