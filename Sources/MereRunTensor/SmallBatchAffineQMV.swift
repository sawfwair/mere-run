#if os(macOS)
import Foundation
import MLX
import MLXFast

/// A serial-arithmetic-preserving affine-Q4 matrix/vector kernel for the
/// two-through-thirty-two-row verification blocks used by Qwen MTP.
///
/// MLX's native wide QMV deliberately reassociates the reduction to reuse
/// weights across rows. This kernel retains the serial QMV's per-row reduction
/// order while sharing the weight reads, which keeps speculative verification
/// on the same greedy trajectory as one-token decoding.
enum SmallBatchAffineQMV {
    /// Wide Flash-Next verification only needs serial QMV arithmetic on these
    /// checkpoint geometries to preserve the greedy trajectory. The omitted
    /// 4x10240 injection and 2560x640 shared-down tails stay on MLX's wide
    /// kernel; serializing either costs throughput without changing parity.
    private static let q38WideExactShapes: Set<String> = [
        "48x2560", "320x10240", "512x2560", "640x2560",
        "2560x2560", "2560x6144", "6144x2560", "10240x320",
        "10240x2560", "12288x2560", "248320x2560",
    ]
    private static let exactWideEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MERERUN_Q38_EXACT_WIDE_Q4"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return value != "0" && value != "false" && value != "off"
    }()
    private static let exactWideShapes: Set<String>? = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_Q38_EXACT_WIDE_Q4_SHAPES"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else { return nil }
        return Set(raw.split(separator: ",").map(String.init))
    }()

    private static let header = """
        template <int NA>
        inline void mere_affine4_qmv_wide(
            const device uint32_t* w,
            const device bfloat16_t* scales,
            const device bfloat16_t* biases,
            const device bfloat16_t* x,
            device bfloat16_t* y,
            const int in_vec_size,
            const int out_vec_size,
            int first_m,
            int out_row,
            uint simd_lid
        ) {
            typedef vec<float, NA> VF;
            constexpr int rows_per_simd = 4;
            constexpr int values_per_thread = 16;
            constexpr int block_size = values_per_thread * 32;
            constexpr int bytes_per_lane = 8;
            const int in_vec_size_w = in_vec_size / 2;
            const int in_vec_size_g = in_vec_size / 64;

            VF acc[rows_per_simd];
            for (int r = 0; r < rows_per_simd; r++) {
                acc[r] = VF(0.0f);
            }

            for (int k = 0; k < in_vec_size; k += block_size) {
                thread uint16_t packed[rows_per_simd][4];
                thread float scale_local[rows_per_simd];
                thread float bias_local[rows_per_simd];
                for (int r = 0; r < rows_per_simd; r++) {
                    const int row = out_row + r;
                    const device uint16_t* ws =
                        reinterpret_cast<const device uint16_t*>(
                            reinterpret_cast<const device uint8_t*>(w) +
                            row * in_vec_size_w + k / 2 +
                            simd_lid * bytes_per_lane);
                    for (int i = 0; i < 4; i++) {
                        packed[r][i] = ws[i];
                    }
                    const int group_index =
                        row * in_vec_size_g + k / 64 + int(simd_lid) / 4;
                    scale_local[r] = scales[group_index];
                    bias_local[r] = biases[group_index];
                }

                VF sums = VF(0.0f);
                VF partial[rows_per_simd];
                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] = VF(0.0f);
                }
                for (int i = 0; i < 4; i++) {
                    VF a0, a1, a2, a3;
                    for (int m = 0; m < NA; m++) {
                        const device bfloat16_t* xm =
                            x + (first_m + m) * in_vec_size + k +
                            simd_lid * values_per_thread + 4 * i;
                        const vec<bfloat16_t, 4> xv =
                            *reinterpret_cast<const device vec<bfloat16_t, 4>*>(xm);
                        a0[m] = static_cast<float>(xv[0]);
                        a1[m] = static_cast<float>(xv[1]);
                        a2[m] = static_cast<float>(xv[2]);
                        a3[m] = static_cast<float>(xv[3]);
                        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                    }
                    for (int r = 0; r < rows_per_simd; r++) {
                        partial[r] +=
                            a0 * (packed[r][i] & 0x000f) +
                            a1 * ((packed[r][i] >> 4) & 0x000f) +
                            a2 * ((packed[r][i] >> 8) & 0x000f) +
                            a3 * ((packed[r][i] >> 12) & 0x000f);
                    }
                }
                for (int r = 0; r < rows_per_simd; r++) {
                    acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
                }
            }

            for (int r = 0; r < rows_per_simd; r++) {
                for (int m = 0; m < NA; m++) {
                    const float reduced = simd_sum(acc[r][m]);
                    if (simd_lid == 0) {
                        y[(first_m + m) * out_vec_size + out_row + r] =
                            static_cast<bfloat16_t>(reduced);
                    }
                }
            }
        }

        template <int M, int IPG>
        inline void mere_affine4_qmv_m(
            const device uint32_t* w,
            const device bfloat16_t* scales,
            const device bfloat16_t* biases,
            const device bfloat16_t* x,
            device bfloat16_t* y,
            const int in_vec_size,
            const int out_vec_size,
            int group_x,
            int out_row,
            uint simd_lid
        ) {
            static_assert(M % IPG != 1, "one-input tail groups are not used");
            constexpr int tail = M % IPG;
            const int first_m = group_x * IPG;
            if (first_m >= M) {
                return;
            }
            if (tail == 0 || M - first_m >= IPG) {
                mere_affine4_qmv_wide<IPG>(
                    w, scales, biases, x, y, in_vec_size, out_vec_size,
                    first_m, out_row, simd_lid);
            } else {
                mere_affine4_qmv_wide<(tail >= 2 ? tail : 2)>(
                    w, scales, biases, x, y, in_vec_size, out_vec_size,
                    first_m, out_row, simd_lid);
            }
        }
        """

    private static let exactRowsKernel = MLXFast.metalKernel(
        name: "mere_affine4_g64_exact_rows_qmv_v1",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: """
            qmv_impl<bfloat16_t, 64, 4>(
                w, scales, biases, x, y,
                x_shape[x_ndim - 1], w_shape[0],
                threadgroup_position_in_grid,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: "// MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS\n",
        ensureRowContiguous: true
    )

    private static let source = """
        const int m = x_shape[x_ndim - 2];
        const int k = x_shape[x_ndim - 1];
        const int n = w_shape[0];
        const uint3 tid = threadgroup_position_in_grid;
        const uint lid = thread_index_in_simdgroup;
        const uint sgid = simdgroup_index_in_threadgroup;
        const int out_row = int(tid.y) * 8 + int(sgid) * 4;
        const int group_x = int(tid.x);
        switch (m) {
            case 2: mere_affine4_qmv_m<2, 2>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 3: mere_affine4_qmv_m<3, 3>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 4: mere_affine4_qmv_m<4, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 5: mere_affine4_qmv_m<5, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 6: mere_affine4_qmv_m<6, 3>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 7: mere_affine4_qmv_m<7, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 8: mere_affine4_qmv_m<8, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 9: mere_affine4_qmv_m<9, 3>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 10: mere_affine4_qmv_m<10, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 11: mere_affine4_qmv_m<11, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 12: mere_affine4_qmv_m<12, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 13: mere_affine4_qmv_m<13, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 14: mere_affine4_qmv_m<14, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 15: mere_affine4_qmv_m<15, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 16: mere_affine4_qmv_m<16, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 17: mere_affine4_qmv_m<17, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 18: mere_affine4_qmv_m<18, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 19: mere_affine4_qmv_m<19, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 20: mere_affine4_qmv_m<20, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 21: mere_affine4_qmv_m<21, 3>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 22: mere_affine4_qmv_m<22, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 23: mere_affine4_qmv_m<23, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 24: mere_affine4_qmv_m<24, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 25: mere_affine4_qmv_m<25, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 26: mere_affine4_qmv_m<26, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 27: mere_affine4_qmv_m<27, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 28: mere_affine4_qmv_m<28, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 29: mere_affine4_qmv_m<29, 3>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 30: mere_affine4_qmv_m<30, 5>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 31: mere_affine4_qmv_m<31, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            case 32: mere_affine4_qmv_m<32, 4>(w, scales, biases, x, y, k, n, group_x, out_row, lid); break;
            default: break;
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "mere_affine4_g64_small_batch_qmv_v1",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: source,
        header: header,
        ensureRowContiguous: true
    )

    static func matmul(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> MLXArray? {
        guard exactWideEnabled,
              Device.defaultDevice().deviceType == .gpu,
              bits == 4, groupSize == 64, mode == .affine else { return nil }
        guard x.dtype == .bfloat16,
              weight.dtype == .uint32,
              scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              x.ndim >= 2,
              weight.ndim == 2 else {
            return nil
        }

        let inputSize = x.dim(-1)
        let outputSize = weight.dim(0)
        let width = x.size / inputSize
        let shape = "\(outputSize)x\(inputSize)"
        let selected = exactWideShapes?.contains(shape)
            ?? (width <= 9 || q38WideExactShapes.contains(shape))
        guard selected,
              (2...32).contains(width),
              x.dim(-2) == width,
              weight.dim(1) == inputSize / 8 else {
            return nil
        }

        // Flash-Next's 320-wide hyper-connection projections, 640-wide
        // shared-expert outputs, and four-output injection gates do not fit
        // the aligned kernel below. Native wide QMV changes their reduction
        // order too. Keep those small verification blocks on native one-row
        // QMV until a matching weight-reusing kernel covers the tail shapes.
        guard outputSize % 8 == 0, outputSize >= 8 else {
            return MLX.concatenated((0..<width).map { row in
                MLX.quantizedMM(
                    x[.ellipsis, row..<(row + 1), 0...], weight,
                    scales: scales, biases: biases, transpose: true,
                    groupSize: groupSize, bits: bits, mode: mode
                )
            }, axis: -2)
        }

        if !inputSize.isMultiple(of: 512) {
            var outputShape = x.shape
            outputShape[outputShape.count - 1] = outputSize
            return exactRowsKernel(
                [weight, scales, biases, x],
                grid: (width * 32, (outputSize / 8) * 2, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
        }

        let inputsPerGroup: Int
        switch width {
        case 2: inputsPerGroup = 2
        case 3: inputsPerGroup = 3
        case 4: inputsPerGroup = 4
        case 5: inputsPerGroup = 5
        case 6: inputsPerGroup = 3
        case 7: inputsPerGroup = 4
        case 8: inputsPerGroup = 4
        case 9: inputsPerGroup = 3
        case 10: inputsPerGroup = 5
        case 11, 12, 14, 16, 18, 19, 22, 23, 24, 26, 27, 28, 31, 32:
            inputsPerGroup = 4
        case 13, 15, 17, 20, 25, 30: inputsPerGroup = 5
        case 21, 29: inputsPerGroup = 3
        default: return nil
        }

        var outputShape = x.shape
        outputShape[outputShape.count - 1] = outputSize
        let activeGroups = (width + inputsPerGroup - 1) / inputsPerGroup
        return kernel(
            [weight, scales, biases, x],
            grid: (activeGroups * 32, (outputSize / 8) * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
#endif
