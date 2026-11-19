#if os(macOS)
import Foundation
import MLX
import MLXFast

/// Tile-major inference buffers. Original parameters remain available to prefill.
enum M5TiledAffineQ4 {
    private static let configuredLayout = ProcessInfo.processInfo.environment["MERERUN_Q38_M5_GROUP_AFFINE"]
    private static let nativeExecution = ProcessInfo.processInfo.environment["MERERUN_Q38_SWIFT_METAL"] == "1"
    @TaskLocal static var groupAffineEnabledOverride: Bool?
    @TaskLocal static var groupAffinePrefillEnabledOverride: Bool?
    static var groupAffineEnabled: Bool {
        groupAffineEnabledOverride ?? (nativeExecution || (configuredLayout.map { ["1", "decode", "all"].contains($0) } ?? QwenDFlashDefaults.active))
    }
    static var groupAffinePrefillEnabled: Bool {
        groupAffinePrefillEnabledOverride ?? (configuredLayout.map { $0 == "all" } ?? QwenDFlashDefaults.active)
    }
    @TaskLocal static var registerOperandsEnabled =
        ProcessInfo.processInfo.environment["MERERUN_Q38_M5_REGISTER_OPERANDS"] == "1"
    private static let configuredHeadTileGroups: Int? = ProcessInfo.processInfo.environment["MERERUN_Q38_M5_HEAD_TILE_GROUPS"]
        .flatMap(Int.init).flatMap { [1, 2, 4, 8].contains($0) ? $0 : nil }

    struct Weights {
        let originalWeight: MLXArray
        let originalScales: MLXArray
        let originalBiases: MLXArray
        let groupAffineLayout: Bool
        let codes: MLXArray
        let scales: MLXArray
        let biases: MLXArray

        init?(weight: MLXArray, scales: MLXArray, biases: MLXArray) {
            guard M5AffineQ4Verification.isSupported, weight.dtype == .uint32,
                  scales.dtype == .bfloat16, biases.dtype == .bfloat16,
                  weight.ndim == 2, weight.dim(0).isMultiple(of: 64),
                  weight.dim(1).isMultiple(of: 8),
                  scales.shape == [weight.dim(0), weight.dim(1) / 8], biases.shape == scales.shape else { return nil }
            let n = weight.dim(0), k = weight.dim(1) * 8
            self.originalWeight = weight
            self.originalScales = scales
            self.originalBiases = biases
            self.groupAffineLayout = groupAffineEnabled
            self.codes = groupAffineEnabled ? groupPackedCodes(weight)
                : weight.reshaped(n / 32, 32, k / 32, 4).transposed(0, 2, 1, 3).contiguous()
            self.scales = scales.reshaped(n / 32, 32, k / 64).transposed(0, 2, 1).contiguous()
            self.biases = biases.reshaped(n / 32, 32, k / 64).transposed(0, 2, 1).contiguous()
            MLX.eval(self.codes, self.scales, self.biases)
        }

        func matches(weight: MLXArray, scales: MLXArray, biases: MLXArray) -> Bool {
            originalWeight === weight && originalScales === scales && originalBiases === biases
                && groupAffineLayout == groupAffineEnabled
        }
    }

    // Tile-major column payloads, 32-channel stages, and paired per-step matmuls
    // follow Splash 1.6.0's Apache-2.0 staged decoder. See THIRD_PARTY_NOTICES.md.
    static let header = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;
        using namespace mpp::tensor_ops;
        #pragma clang fp reassociate(off)

        template <class Out>
        inline void mere_tiled_store(device Out* output, ulong index, float value) {
            output[index] = Out(value);
        }

        inline void mere_tiled_dequant(uint4 packed, float scale, float bias,
                                       uint lane, threadgroup half* stage) {
            #pragma unroll
            for (ushort word = 0; word < 4; ++word) {
                #pragma unroll
                for (ushort pair = 0; pair < 4; ++pair) {
                    const uint v = packed[word] >> (pair * 8);
                    const float2 q = float2(v & 15u, (v >> 4) & 15u);
                    reinterpret_cast<threadgroup half2*>(stage)[lane * 16 + word * 4 + pair] =
                        half2(scale * q + bias);
                }
            }
        }

        template <int Rows>
        inline auto mere_tiled_accumulator(const device bfloat* input, int k,
                                           threadgroup half* stage) {
            constexpr auto descriptor = matmul2d_descriptor(
                Rows, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto a = tensor(const_cast<device bfloat*>(input), dextents<int, 2>{k, Rows}, array<int, 2>{1, k});
            tensor<threadgroup half, dextents<int, 2>, tensor_inline> b(
                stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
            auto av = a.template slice<32, Rows>(0, 0);
            auto bv = b.slice<32, 32>(0, 0);
            auto acc = operation.template get_destination_cooperative_tensor<decltype(av), decltype(bv), float>();
            for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
            return acc;
        }

        // Same stable BF16 sigmoid expression as MLX unary_ops.h (Apple, MIT).
        inline bfloat mere_tiled_sigmoid(bfloat x) {
            auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
        }
        """

    private static let projectionKernel = MLXFast.metalKernel(
        name: "mere_m5_tiled_affine4_projection_v3",
        inputNames: ["w", "s", "b", "x"], outputNames: ["y"],
        source: """
            threadgroup half stages[Groups * 2048];
            const int k = x_shape[x_ndim - 1], steps = k / 32, n = w_shape[0] * 32;
            const uint split = threadgroup_position_in_grid.y;
            const int first = int(split) * (steps / Splits), end = first + steps / Splits;
            const uint block = threadgroup_position_in_grid.x * Groups + simdgroup_index_in_threadgroup;
            if (block >= uint(w_shape[0])) return;
            const uint lane = thread_index_in_simdgroup;
            const device uint4* codes = reinterpret_cast<const device uint4*>(w);
            const device bfloat* scales = reinterpret_cast<const device bfloat*>(s);
            const device bfloat* biases = reinterpret_cast<const device bfloat*>(b);
            const device bfloat* input = reinterpret_cast<const device bfloat*>(x);
            threadgroup half* stage = stages + simdgroup_index_in_threadgroup * 2048;
            auto a = tensor(const_cast<device bfloat*>(input), dextents<int, 2>{k, Rows}, array<int, 2>{1, k});
            tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
                stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
            tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
                stage + 1024, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
            auto b0 = bt0.slice<32, 32>(0, 0), b1 = bt1.slice<32, 32>(0, 0);
            constexpr auto descriptor = matmul2d_descriptor(
                Rows, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto acc = mere_tiled_accumulator<Rows>(input, k, stage);
            uint4 packed = codes[(ulong(block) * steps + first) * 32 + lane];
            const ulong firstMeta = (ulong(block) * (steps / 2) + first / 2) * 32 + lane;
            float scale = float(scales[firstMeta]), bias = float(biases[firstMeta]);
            for (int step = first; step < end; ++step) {
                mere_tiled_dequant(packed, scale, bias, lane, stage + ((step - first) & 1) * 1024);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (step + 1 < end) {
                    packed = codes[(ulong(block) * steps + step + 1) * 32 + lane];
                    if ((step + 1) / 2 != step / 2) {
                        const ulong meta = (ulong(block) * (steps / 2) + (step + 1) / 2) * 32 + lane;
                        scale = float(scales[meta]); bias = float(biases[meta]);
                    }
                }
                auto av = a.template slice<32, Rows>(step * 32, 0);
                if ((step - first) & 1) operation.run(av, b1, acc);
                else operation.run(av, b0, acc);
            }
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto index = acc.get_multidimensional_index(i);
                mere_tiled_store(y, (ulong(split) * Rows + index[1]) * n + block * 32 + index[0], acc[i]);
            }
            """, header: header, ensureRowContiguous: true
    )

    private static let reductionKernel = MLXFast.metalKernel(
        name: "mere_m5_tiled_affine4_reduce_v2", inputNames: ["partial"], outputNames: ["y"],
        source: """
            const uint index = thread_position_in_grid.x, count = partial_shape[1] * partial_shape[2];
            if (index >= count) return;
            float total = partial[index];
            for (uint split = 1; split < Splits; ++split) total += partial[split * count + index];
            y[index] = bfloat16_t(total);
            """, ensureRowContiguous: true
    )

    /// Fixed by geometry for both serial and verification calls. An unsplit
    /// 5120-output projection launches only 80 two-SIMD-group threadgroups.
    static func partitionCount(inputSize: Int, outputSize: Int) -> Int {
        switch (inputSize, outputSize) {
        case (17_408, 5_120), (6_144, 5_120): return 2
        default: return 1
        }
    }

    static func matmul(
        _ x: MLXArray, weights: Weights, splits requestedSplits: Int? = nil, tileGroups: Int? = nil,
        arithmetic: GroupAffineArithmetic = .standard
    ) -> MLXArray? {
        let splits = requestedSplits ?? (groupAffineEnabled
            ? groupAffinePartitionCount(inputSize: x.dim(-1), outputSize: weights.originalWeight.dim(0), rows: x.size / x.dim(-1))
            : partitionCount(inputSize: x.dim(-1), outputSize: weights.originalWeight.dim(0)))
        let headGroups = x.dim(-1) == 5_120 && weights.originalWeight.dim(0) == 248_320 ? configuredHeadTileGroups : nil
        let groups = tileGroups ?? headGroups ?? 2
        precondition([1, 2, 4, 8].contains(groups))
        guard weights.groupAffineLayout == groupAffineEnabled else { return nil }
        if groupAffineEnabled { return groupAffineProjection(x, weights: weights, splits: splits, arithmetic: arithmetic) }
        guard let layout = M5AffineQ4Verification.inputLayout(
            x, weight: weights.originalWeight, scales: weights.originalScales,
            biases: weights.originalBiases, splits: splits, maximumRows: 16
        ) else { return nil }
        let kernel = registerOperandsEnabled ? registerProjectionKernel : projectionKernel
        let partials = kernel(
            [weights.codes, weights.scales, weights.biases, layout.padded],
            template: [("Splits", splits), ("Groups", groups), ("Rows", layout.tileRows)],
            grid: (((layout.outputSize + groups * 32 - 1) / (groups * 32)) * groups * 32, splits, 1),
            threadGroup: (groups * 32, 1, 1),
            outputShapes: [[splits, layout.tileRows, layout.outputSize]], outputDTypes: [splits == 1 ? .bfloat16 : .float32]
        )[0]
        let output = splits == 1 ? partials[0, 0..., 0...] : reductionKernel(
            [partials], template: [("Splits", splits)],
            grid: (layout.tileRows * layout.outputSize, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[layout.tileRows, layout.outputSize]], outputDTypes: [.bfloat16]
        )[0]
        return output[0..<layout.width, 0...].reshaped(layout.outputShape)
    }
}
#endif
