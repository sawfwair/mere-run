#if os(macOS)
import MLX
import MLXFast

extension M5AffineQ4Verification {
    // The stable sigmoid expression follows MLX's unary_ops.h (Apple, MIT).
    // Preserve BF16 operation boundaries from Q35CompiledOperations.swiglu.
    private static let swigluHeader = metalHeader + """
        inline bfloat mere_m5_sigmoid(bfloat x) {
            auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
        }
        """

    private static let swigluKernel = MLXFast.metalKernel(
        name: "mere_m5_affine4_g64_swiglu_v1",
        inputNames: ["gw", "gs", "gb", "uw", "us", "ub", "x"],
        outputNames: ["y"],
        source: """
            threadgroup half stages[8192];
            const int k = x_shape[x_ndim - 1], n = gw_shape[0];
            const uint column = threadgroup_position_in_grid.x * 64 + simdgroup_index_in_threadgroup * 32;
            const uint lane = thread_index_in_simdgroup;
            threadgroup half* stage = stages + simdgroup_index_in_threadgroup * 4096;
            auto gate = mere_m5_affine4_accumulate<1>(
                gw, reinterpret_cast<const device bfloat*>(gs), reinterpret_cast<const device bfloat*>(gb),
                reinterpret_cast<const device bfloat*>(x), k, column, 0, lane, stage);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto up = mere_m5_affine4_accumulate<1>(
                uw, reinterpret_cast<const device bfloat*>(us), reinterpret_cast<const device bfloat*>(ub),
                reinterpret_cast<const device bfloat*>(x), k, column, 0, lane, stage);
            for (ushort i = 0; i < gate.get_capacity(); ++i) {
                if (!gate.is_valid_element(i)) continue;
                const auto index = gate.get_multidimensional_index(i);
                const bfloat g = bfloat(gate[i]), u = bfloat(up[i]);
                const bfloat activated = g * mere_m5_sigmoid(g);
                y[index[1] * n + column + index[0]] = activated * u;
            }
            """,
        header: swigluHeader, ensureRowContiguous: true
    )

    static func swiglu(
        _ x: MLXArray,
        gateWeight: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upWeight: MLXArray, upScales: MLXArray, upBiases: MLXArray
    ) -> MLXArray? {
        guard upWeight.shape == gateWeight.shape, upWeight.dtype == .uint32,
              upScales.shape == gateScales.shape, upScales.dtype == .bfloat16,
              upBiases.shape == gateBiases.shape, upBiases.dtype == .bfloat16,
              let layout = inputLayout(
                  x, weight: gateWeight, scales: gateScales, biases: gateBiases, splits: 1
              ) else { return nil }
        let output = swigluKernel(
            [gateWeight, gateScales, gateBiases, upWeight, upScales, upBiases, layout.padded],
            grid: (layout.outputSize, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[8, layout.outputSize]], outputDTypes: [.bfloat16]
        )[0]
        return output[0..<layout.width, 0...].reshaped(layout.outputShape)
    }
}
#endif
