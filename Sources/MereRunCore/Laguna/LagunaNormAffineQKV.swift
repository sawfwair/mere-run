import MLX
import MLXFast

/// Single-token Laguna XS input RMSNorm plus group-32 affine Q/K/V(/gate)
/// projection. The kernel reproduces MLX's precise RMSNorm reduction and BF16
/// rounding boundary, then keeps each projection row's affine-QMV arithmetic
/// and reduction order unchanged. Prefetching immutable weight bytes above the
/// RMS prologue overlaps memory latency without changing consumption order.
enum LagunaNormAffineQKV {
    private static let hiddenSize = 2_048
    private static let headDimension = 128
    private static let keyValueHeads = 8
    private static let prefetchDepth = 4

    static func call(
        residual: MLXArray,
        normWeight: MLXArray,
        codes: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        heads: Int,
        gateRows: Int
    ) -> MLXArray? {
        let rows = (heads + 2 * keyValueHeads) * headDimension + gateRows
        guard let kernel = kernels[rows],
              residual.dtype == .bfloat16,
              residual.shape == [1, 1, hiddenSize],
              normWeight.dtype == .bfloat16,
              normWeight.shape == [hiddenSize],
              codes.dtype == .uint32,
              codes.shape == [rows, hiddenSize / 4],
              scales.dtype == .bfloat16,
              scales.shape == [rows, hiddenSize / 32],
              biases.dtype == .bfloat16,
              biases.shape == [rows, hiddenSize / 32] else {
            return nil
        }

        return kernel(
            [residual, normWeight, codes, scales, biases],
            grid: ((rows / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, rows]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let kernels: [Int: MLXFast.MLXFastKernel] = {
        var result: [Int: MLXFast.MLXFastKernel] = [:]
        for heads in [48, 64] {
            for gateRows in [0, heads] {
                let rows = (heads + 2 * keyValueHeads) * headDimension + gateRows
                result[rows] = MLXFast.metalKernel(
                    name: "mere_laguna_norm_affine_qkv_i8g32_r\(rows)_pf4_v1",
                    inputNames: [
                        "residual", "norm_weight", "weight_codes",
                        "weight_scales", "weight_biases",
                    ],
                    outputNames: ["projected"],
                    source: source(rows: rows),
                    ensureRowContiguous: true
                )
            }
        }
        return result
    }()
    #else
    private static let kernels: [Int: MLXFast.MLXFastKernel] = [:]
    #endif

    private static func source(rows: Int) -> String {
        """
        constexpr uint axis_size = \(hiddenSize);
        constexpr uint out_vec_size = \(rows);
        constexpr uint n_reads = 4;
        constexpr uint norm_threads = axis_size / n_reads;
        constexpr uint real_threads = 64;
        constexpr uint virtual_per_thread = norm_threads / real_threads;
        constexpr uint simd_size = 32;
        constexpr float norm_eps = 1.0e-6f;
        constexpr uint values_per_thread = 8;
        constexpr uint block_size = 256;
        constexpr uint results_per_simdgroup = 4;
        constexpr uint num_simdgroups = 2;
        constexpr uint group_size = 32;
        constexpr uint scale_step_per_thread = group_size / values_per_thread;
        constexpr uint in_vec_size_g = axis_size / group_size;
        constexpr uint pf_depth = \(prefetchDepth);

        uint tile = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];

        uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
            simd_gid * results_per_simdgroup;
        const device uint8_t* ws = (const device uint8_t*)weight_codes +
            out_row * axis_size + simd_lid * values_per_thread;
        const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;

        uint8_t pf_w[pf_depth][results_per_simdgroup][values_per_thread];
        float pf_s[pf_depth][results_per_simdgroup];
        float pf_b[pf_depth][results_per_simdgroup];
        for (uint d = 0; d < pf_depth; ++d) {
            for (uint row = 0; row < results_per_simdgroup; ++row) {
                const device uint8_t* wl =
                    ws + d * block_size + row * axis_size;
                for (uint i = 0; i < values_per_thread; ++i) {
                    pf_w[d][row][i] = wl[i];
                }
                pf_s[d][row] = float(sc[
                    d * (block_size / group_size) + row * in_vec_size_g]);
                pf_b[d][row] = float(bs[
                    d * (block_size / group_size) + row * in_vec_size_g]);
            }
        }

        if (lid < simd_size) {
            local_sums[lid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < virtual_per_thread; ++j) {
            uint base = (lid + j * real_threads) * n_reads;
            float accum = 0.0f;
            for (uint i = 0; i < n_reads; ++i) {
                float value = float(residual[base + i]);
                accum += value * value;
            }
            accum = simd_sum(accum);
            if (simd_lid == 0) {
                local_sums[simd_gid + num_simdgroups * j] = accum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            float total = simd_sum(local_sums[simd_lid]);
            if (simd_lid == 0) {
                local_inv_mean[0] =
                    metal::precise::rsqrt(total / float(axis_size) + norm_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inv_mean = local_inv_mean[0];
        thread float x_thread[values_per_thread];
        thread float result[results_per_simdgroup] = {
            0.0f, 0.0f, 0.0f, 0.0f
        };

        uint column = simd_lid * values_per_thread;
        for (uint d = 0; d < pf_depth; ++d) {
            float sum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                float value = float(bfloat(
                    norm_weight[column + i] *
                    bfloat(float(residual[column + i]) * inv_mean)));
                sum += value;
                x_thread[i] = value;
            }
            for (uint row = 0; row < results_per_simdgroup; ++row) {
                float accum = 0.0f;
                for (uint i = 0; i < values_per_thread; ++i) {
                    accum += x_thread[i] * pf_w[d][row][i];
                }
                result[row] += pf_s[d][row] * accum + sum * pf_b[d][row];
            }
            ws += block_size;
            sc += block_size / group_size;
            bs += block_size / group_size;
            column += block_size;
        }

        for (uint k = pf_depth * block_size; k < axis_size; k += block_size) {
            float sum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                float value = float(bfloat(
                    norm_weight[column + i] *
                    bfloat(float(residual[column + i]) * inv_mean)));
                sum += value;
                x_thread[i] = value;
            }
            for (uint row = 0; row < results_per_simdgroup; ++row) {
                const device uint8_t* wl = ws + row * axis_size;
                float scale = float(sc[row * in_vec_size_g]);
                float bias = float(bs[row * in_vec_size_g]);
                float accum = 0.0f;
                for (uint i = 0; i < values_per_thread; ++i) {
                    accum += x_thread[i] * wl[i];
                }
                result[row] += scale * accum + sum * bias;
            }
            ws += block_size;
            sc += block_size / group_size;
            bs += block_size / group_size;
            column += block_size;
        }

        for (uint row = 0; row < results_per_simdgroup; ++row) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) {
                projected[out_row + row] = bfloat(result[row]);
            }
        }
        """
    }
}
