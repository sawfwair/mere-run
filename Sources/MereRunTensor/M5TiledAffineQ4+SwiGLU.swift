#if os(macOS)
import MLX
import MLXFast

extension M5TiledAffineQ4 {
    private static let swigluKernel = MLXFast.metalKernel(
        name: "mere_m5_tiled_affine4_swiglu_v1",
        inputNames: ["gw", "gs", "gb", "uw", "us", "ub", "x"], outputNames: ["y"],
        source: """
            threadgroup half stage[2048];
            const int k = x_shape[x_ndim - 1], steps = k / 32, n = gw_shape[0] * 32;
            const uint block = threadgroup_position_in_grid.x, lane = thread_index_in_simdgroup;
            const device uint4* gateCodes = reinterpret_cast<const device uint4*>(gw);
            const device uint4* upCodes = reinterpret_cast<const device uint4*>(uw);
            const device bfloat* gateScales = reinterpret_cast<const device bfloat*>(gs);
            const device bfloat* gateBiases = reinterpret_cast<const device bfloat*>(gb);
            const device bfloat* upScales = reinterpret_cast<const device bfloat*>(us);
            const device bfloat* upBiases = reinterpret_cast<const device bfloat*>(ub);
            const device bfloat* input = reinterpret_cast<const device bfloat*>(x);
            auto a = tensor(const_cast<device bfloat*>(input), dextents<int, 2>{k, 8}, array<int, 2>{1, k});
            tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
                stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
            tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
                stage + 1024, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
            auto bg = bt0.slice<32, 32>(0, 0), bu = bt1.slice<32, 32>(0, 0);
            constexpr auto descriptor = matmul2d_descriptor(
                8, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto gate = mere_tiled_accumulator(input, k, stage);
            auto up = mere_tiled_accumulator(input, k, stage + 1024);
            uint4 gp = gateCodes[ulong(block) * steps * 32 + lane];
            uint4 upayload = upCodes[ulong(block) * steps * 32 + lane];
            const ulong first = ulong(block) * (steps / 2) * 32 + lane;
            float gscale = float(gateScales[first]), gbias = float(gateBiases[first]);
            float uscale = float(upScales[first]), ubias = float(upBiases[first]);
            for (int step = 0; step < steps; ++step) {
                mere_tiled_dequant(gp, gscale, gbias, lane, stage);
                mere_tiled_dequant(upayload, uscale, ubias, lane, stage + 1024);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (step + 1 < steps) {
                    const ulong next = (ulong(block) * steps + step + 1) * 32 + lane;
                    gp = gateCodes[next]; upayload = upCodes[next];
                    if ((step + 1) / 2 != step / 2) {
                        const ulong meta = (ulong(block) * (steps / 2) + (step + 1) / 2) * 32 + lane;
                        gscale = float(gateScales[meta]); gbias = float(gateBiases[meta]);
                        uscale = float(upScales[meta]); ubias = float(upBiases[meta]);
                    }
                }
                auto av = a.slice<32, 8>(step * 32, 0);
                operation.run(av, bg, gate);
                operation.run(av, bu, up);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (ushort i = 0; i < gate.get_capacity(); ++i) {
                if (!gate.is_valid_element(i)) continue;
                const auto index = gate.get_multidimensional_index(i);
                const bfloat g = bfloat(gate[i]), u = bfloat(up[i]);
                const bfloat activated = g * mere_tiled_sigmoid(g);
                y[index[1] * n + block * 32 + index[0]] = activated * u;
            }
            """, header: header, ensureRowContiguous: true
    )

    static func swiglu(_ x: MLXArray, gate: Weights, up: Weights) -> MLXArray? {
        guard gate.originalWeight.shape == up.originalWeight.shape,
              let layout = M5AffineQ4Verification.inputLayout(
                  x, weight: gate.originalWeight, scales: gate.originalScales,
                  biases: gate.originalBiases, splits: 1
              ) else { return nil }
        let output = swigluKernel(
            [gate.codes, gate.scales, gate.biases, up.codes, up.scales, up.biases, layout.padded],
            grid: (layout.outputSize, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[8, layout.outputSize]], outputDTypes: [.bfloat16]
        )[0]
        return output[0..<layout.width, 0...].reshaped(layout.outputShape)
    }
}
#endif
