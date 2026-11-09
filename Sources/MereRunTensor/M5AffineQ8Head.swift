#if os(macOS)
import Metal
import MLX
import MLXFast

/// Experimental M5 projection from original affine-Q8 buffers. BF16 staging
/// and matrix accumulation differ from native serial QMV arithmetic.
package enum M5AffineQ8Head {
    package static var isSupported: Bool {
        Device.defaultDevice().deviceType == .gpu && hardwareSupported
    }

    private static let hardwareSupported: Bool = {
        if #available(macOS 26.2, *) {
            return MTLCreateSystemDefaultDevice()?.supportsFamily(.apple10) == true
        }
        return false
    }()

    // Operand construction adapts the Qwen experiment and Splash 1.6.0's
    // gguf_staged_tile.h (Apache-2.0); see THIRD_PARTY_NOTICES.md.
    private static let kernel = MLXFast.metalKernel(
        name: "mere_m5_affine8_bf16_head_v1",
        inputNames: ["w", "scales", "biases", "x"], outputNames: ["partial"],
        source: """
            constexpr int Rows = 8, Columns = 32, Step = 64;
            const int k_size = x_shape[x_ndim - 1], n_size = w_shape[0];
            const uint lane = thread_index_in_simdgroup;
            const uint column_origin = threadgroup_position_in_grid.x * 64
                + simdgroup_index_in_threadgroup * Columns;
            const uint split = threadgroup_position_in_grid.y;
            constexpr auto descriptor = matmul2d_descriptor(
                Rows, Columns, Step, false, true, false,
                matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto a = tensor(reinterpret_cast<device bfloat*>(const_cast<device bfloat16_t*>(x)),
                dextents<int, 2>{k_size, Rows}, array<int, 2>{1, k_size});
            threadgroup bfloat storage[8192];
            threadgroup bfloat* stages = storage + simdgroup_index_in_threadgroup * 4096;
            tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> stage0(
                stages, dextents<int, 2>{Step, Columns}, array<int, 2>{1, Step});
            tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> stage1(
                stages + Step * Columns, dextents<int, 2>{Step, Columns}, array<int, 2>{1, Step});
            auto b0 = stage0.slice<Step, Columns>(0, 0);
            auto b1 = stage1.slice<Step, Columns>(0, 0);
            auto a0 = a.template slice<Step, Rows>(0, 0);
            auto accumulator = operation.template get_destination_cooperative_tensor<
                decltype(a0), decltype(b0), float>();
            for (ushort i = 0; i < accumulator.get_capacity(); ++i) accumulator[i] = 0.0f;
            const int first = int(split) * (k_size / Step / Splits);
            const int end = first + k_size / Step / Splits;
            uint packed[16]; float scale[16], bias[16];
            for (int i = 0; i < 16; ++i) {
                const uint word = lane + 32 * i;
                const uint column = column_origin + word / 16;
                packed[i] = w[ulong(column) * (k_size / 4) + first * 16 + word % 16];
                const ulong group = ulong(column) * (k_size / Step) + first;
                scale[i] = float(scales[group]); bias[i] = float(biases[group]);
            }
            for (int step = first; step < end; ++step) {
                threadgroup bfloat* stage = stages + ((step - first) & 1) * Step * Columns;
                for (int i = 0; i < 16; ++i) {
                    const uint word = lane + 32 * i;
                    for (int j = 0; j < 4; ++j) {
                        stage[word * 4 + j] = bfloat(
                            scale[i] * float((packed[i] >> (8 * j)) & 255u) + bias[i]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (step + 1 < end) {
                    for (int i = 0; i < 16; ++i) {
                        const uint word = lane + 32 * i;
                        const uint column = column_origin + word / 16;
                        packed[i] = w[ulong(column) * (k_size / 4) + (step + 1) * 16 + word % 16];
                        const ulong group = ulong(column) * (k_size / Step) + step + 1;
                        scale[i] = float(scales[group]); bias[i] = float(biases[group]);
                    }
                }
                auto operands = a.template slice<Step, Rows>(step * Step, 0);
                if ((step - first) & 1) operation.run(operands, b1, accumulator);
                else operation.run(operands, b0, accumulator);
            }
            for (ushort i = 0; i < accumulator.get_capacity(); ++i) {
                if (!accumulator.is_valid_element(i)) continue;
                const auto index = accumulator.get_multidimensional_index(i);
                partial[(ulong(split) * Rows + index[1]) * n_size + column_origin + index[0]]
                    = Out(accumulator[i]);
            }
            """,
        header: """
            #include <metal_stdlib>
            #include <metal_tensor>
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
            using namespace metal;
            using namespace mpp::tensor_ops;
            #pragma clang fp reassociate(off)

            """,
        ensureRowContiguous: true
    )

    package static func matmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, splits: Int
    ) -> MLXArray? {
        guard isSupported, x.dtype == .bfloat16, weight.dtype == .uint32,
              scales.dtype == .bfloat16, biases.dtype == .bfloat16,
              x.ndim >= 2, weight.ndim == 2, [1, 2, 4, 8].contains(splits) else { return nil }
        let input = x.dim(-1), output = weight.dim(0)
        guard input > 0, output > 0, input.isMultiple(of: 64 * splits), output.isMultiple(of: 64),
              weight.dim(1) == input / 4, scales.shape == [output, input / 64], biases.shape == scales.shape else { return nil }
        let rows = x.size / input
        guard (1...8).contains(rows), x.dim(-2) == rows else { return nil }
        let matrix = x.reshaped(rows, input)
        let padded = rows == 8 ? matrix : MLX.concatenated([
            matrix, MLXArray.zeros([8 - rows, input], dtype: .bfloat16)
        ], axis: 0)
        let partial = kernel([weight, scales, biases, padded],
            template: [("Splits", splits), ("Out", splits == 1 ? DType.bfloat16 : DType.float32)],
            grid: (output, splits, 1), threadGroup: (64, 1, 1),
            outputShapes: [[splits, 8, output]], outputDTypes: [splits == 1 ? .bfloat16 : .float32])[0]
        let result = splits == 1 ? partial[0, 0..., 0...] : partial.sum(axis: 0).asType(.bfloat16)
        var shape = x.shape
        shape[shape.count - 1] = output
        return result[0..<rows, 0...].reshaped(shape)
    }
}
#endif
