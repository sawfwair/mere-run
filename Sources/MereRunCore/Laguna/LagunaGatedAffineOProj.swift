import MLX
import MLXFast

/// Decode-only Laguna XS attention tail. The kernel consumes the raw BF16
/// per-head gate logits and the ungated attention row, reproduces MLX's stable
/// FP32 softplus and BF16 rounding boundary, then performs the accepted
/// group-32 affine INT8 output projection in the same dispatch.
enum LagunaGatedAffineOProj {
    private static let hiddenSize = 2_048
    private static let headDimension = 128

    static func call(
        attentionOutput: MLXArray,
        gateLogits: MLXArray,
        codes: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        heads: Int
    ) -> MLXArray? {
        let inputWidth = heads * headDimension
        guard let kernel = kernels[heads],
              attentionOutput.dtype == .bfloat16,
              attentionOutput.shape == [1, 1, inputWidth],
              gateLogits.dtype == .bfloat16,
              gateLogits.shape == [1, 1, heads],
              codes.dtype == .uint32,
              codes.shape == [hiddenSize, inputWidth / 4],
              scales.dtype == .bfloat16,
              scales.shape == [hiddenSize, inputWidth / 32],
              biases.dtype == .bfloat16,
              biases.shape == [hiddenSize, inputWidth / 32] else {
            return nil
        }

        return kernel(
            [attentionOutput, gateLogits, codes, scales, biases],
            grid: ((hiddenSize / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, hiddenSize]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let kernels: [Int: MLXFast.MLXFastKernel] = {
        var result: [Int: MLXFast.MLXFastKernel] = [:]
        for heads in [48, 64] {
            result[heads] = MLXFast.metalKernel(
                name: "mere_laguna_gated_affine_oproj_i8g32_h\(heads)_v1",
                inputNames: [
                    "attention_output", "gate_logits", "weight_codes",
                    "weight_scales", "weight_biases",
                ],
                outputNames: ["projected"],
                source: source(heads: heads),
                ensureRowContiguous: true
            )
        }
        return result
    }()
    #else
    private static let kernels: [Int: MLXFast.MLXFastKernel] = [:]
    #endif

    private static func source(heads: Int) -> String {
        """
        constexpr uint in_vec_size = \(heads * headDimension);
        constexpr uint out_vec_size = \(hiddenSize);
        constexpr uint gate_heads = \(heads);
        constexpr uint head_shift = 7;
        constexpr uint values_per_thread = 8;
        constexpr uint block_size = 256;
        constexpr uint results_per_simdgroup = 4;
        constexpr uint num_simdgroups = 2;
        constexpr uint group_size = 32;
        constexpr uint scale_step_per_thread = group_size / values_per_thread;
        constexpr uint in_vec_size_g = in_vec_size / group_size;

        uint tile = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        threadgroup float gate_table[gate_heads];
        if (lid < gate_heads) {
            float logit = float(gate_logits[lid]);
            float gate;
            if (metal::isnan(logit)) {
                gate = NAN;
            } else {
                float maxval = metal::max(logit, 0.0f);
                float minval = metal::min(logit, 0.0f);
                gate = (metal::isinf(minval) || metal::isinf(maxval))
                    ? maxval
                    : maxval + log1p(metal::exp(minval - maxval));
            }
            gate_table[lid] = float(bfloat(gate));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
            simd_gid * results_per_simdgroup;
        const device uint8_t* ws = (const device uint8_t*)weight_codes +
            out_row * in_vec_size + simd_lid * values_per_thread;
        const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        const device bfloat* xp = attention_output + simd_lid * values_per_thread;

        thread float x_thread[values_per_thread];
        thread float result[results_per_simdgroup] = {
            0.0f, 0.0f, 0.0f, 0.0f
        };

        uint column = simd_lid * values_per_thread;
        for (uint k = 0; k < in_vec_size; k += block_size) {
            float gate = gate_table[column >> head_shift];
            float sum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                float value = float(bfloat(float(xp[i]) * gate));
                sum += value;
                x_thread[i] = value;
            }

            for (uint row = 0; row < results_per_simdgroup; ++row) {
                const device uint8_t* wl = ws + row * in_vec_size;
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
            xp += block_size;
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
