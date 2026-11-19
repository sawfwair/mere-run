#if os(macOS)
import MLX
import MLXFast

extension M5TiledAffineQ4 {
    // Keep the staged path's FP16 dequantization, K=32 steps, FP32 accumulator,
    // and ordered split reduction. Only the matrix operand's storage changes.
    private static let registerHeader = header + """

        template <class Tile>
        inline void mere_register_tile(thread Tile& tile, const device uint4* codes,
                                       const device bfloat* scales, const device bfloat* biases,
                                       uint block, int step, int steps) {
            #pragma unroll
            for (ushort i = 0; i < tile.get_capacity(); ++i) {
                const auto index = tile.get_multidimensional_index(i);
                const uint column = index[0], row = index[1];
                const uint4 packed = codes[(ulong(block) * steps + step) * 32 + row];
                const uint code = (packed[column / 8] >> (4 * (column % 8))) & 15u;
                const ulong meta = (ulong(block) * (steps / 2) + step / 2) * 32 + row;
                tile[i] = half(float(scales[meta]) * float(code) + float(biases[meta]));
            }
        }
        """

    static let registerProjectionKernel = MLXFast.metalKernel(
        name: "mere_m5_register_affine4_projection_v1",
        inputNames: ["w", "s", "b", "x"], outputNames: ["y"],
        source: """
            const int k = x_shape[x_ndim - 1], steps = k / 32, n = w_shape[0] * 32;
            const uint split = threadgroup_position_in_grid.y;
            const int first = int(split) * (steps / Splits), end = first + steps / Splits;
            const uint block = threadgroup_position_in_grid.x * Groups + simdgroup_index_in_threadgroup;
            if (block >= uint(w_shape[0])) return;
            const device bfloat* input = reinterpret_cast<const device bfloat*>(x);
            auto a = tensor(const_cast<device bfloat*>(input), dextents<int, 2>{k, Rows}, array<int, 2>{1, k});
            constexpr auto descriptor = matmul2d_descriptor(
                Rows, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto weight = operation.template get_right_input_cooperative_tensor<bfloat, half, float>();
            auto av = a.template slice<32, Rows>(0, 0);
            auto acc = operation.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(av)>, metal::remove_addrspace_t<decltype(weight)>, float>();
            for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
            for (int step = first; step < end; ++step) {
                mere_register_tile(weight, reinterpret_cast<const device uint4*>(w),
                    reinterpret_cast<const device bfloat*>(s), reinterpret_cast<const device bfloat*>(b),
                    block, step, steps);
                auto ax = a.template slice<32, Rows>(step * 32, 0);
                operation.run(ax, weight, acc);
            }
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto index = acc.get_multidimensional_index(i);
                mere_tiled_store(y, (ulong(split) * Rows + index[1]) * n + block * 32 + index[0], acc[i]);
            }
            """, header: registerHeader, ensureRowContiguous: true
    )

    static let registerSwiGLUKernel = MLXFast.metalKernel(
        name: "mere_m5_register_affine4_swiglu_v1",
        inputNames: ["gw", "gs", "gb", "uw", "us", "ub", "x"], outputNames: ["y"],
        source: """
            const int k = x_shape[x_ndim - 1], steps = k / 32, n = gw_shape[0] * 32;
            const uint block = threadgroup_position_in_grid.x * Groups + simdgroup_index_in_threadgroup;
            if (block >= uint(gw_shape[0])) return;
            const device bfloat* input = reinterpret_cast<const device bfloat*>(x);
            auto a = tensor(const_cast<device bfloat*>(input), dextents<int, 2>{k, Rows}, array<int, 2>{1, k});
            constexpr auto descriptor = matmul2d_descriptor(
                Rows, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto weight = operation.template get_right_input_cooperative_tensor<bfloat, half, float>();
            auto av = a.template slice<32, Rows>(0, 0);
            auto gate = operation.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(av)>, metal::remove_addrspace_t<decltype(weight)>, float>();
            auto up = operation.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(av)>, metal::remove_addrspace_t<decltype(weight)>, float>();
            for (ushort i = 0; i < gate.get_capacity(); ++i) { gate[i] = 0.0f; up[i] = 0.0f; }
            for (int step = 0; step < steps; ++step) {
                auto ax = a.template slice<32, Rows>(step * 32, 0);
                mere_register_tile(weight, reinterpret_cast<const device uint4*>(gw),
                    reinterpret_cast<const device bfloat*>(gs), reinterpret_cast<const device bfloat*>(gb),
                    block, step, steps);
                operation.run(ax, weight, gate);
                mere_register_tile(weight, reinterpret_cast<const device uint4*>(uw),
                    reinterpret_cast<const device bfloat*>(us), reinterpret_cast<const device bfloat*>(ub),
                    block, step, steps);
                operation.run(ax, weight, up);
            }
            for (ushort i = 0; i < gate.get_capacity(); ++i) {
                if (!gate.is_valid_element(i)) continue;
                const auto index = gate.get_multidimensional_index(i);
                const bfloat g = bfloat(gate[i]), u = bfloat(up[i]);
                const bfloat activated = g * mere_tiled_sigmoid(g);
                y[index[1] * n + block * 32 + index[0]] = activated * u;
            }
            """, header: registerHeader, ensureRowContiguous: true
    )
}
#endif
