#if os(macOS)
import MLX
import MLXFast

extension M5TiledAffineQ4 {
    static func groupAffinePartitionCount(inputSize: Int, outputSize: Int, rows: Int) -> Int {
        switch (inputSize, outputSize) {
        case (5_120, 17_408): return rows > 16 ? 1 : 2
        case (17_408, 5_120), (6_144, 5_120): return rows > 16 ? 2 : 4
        case (5_120, 16_384): return 2
        case (5_120, 14_336): return 4
        default: return 1
        }
    }
    // TensorFold 1.2.7's register-code operand and group-affine algorithm,
    // Apache-2.0. API-derived code packing and separate tile-major metadata.
    // This is a new numerical reference, not FP16-dequantized weight staging.
    static let groupHeader = header + """
        struct MereGroupLoad {
            uint4 w0, w1;
            ushort4 s0, b0, s1, b1;
            float x0, x1;
        };

        inline MereGroupLoad mere_group_fetch(
            const device uint4* words, const device ushort4* scales,
            const device ushort4* biases, const device float* sums,
            int group, int column, int row, int stride) {
            const int meta = group * 8 + column / 4;
            return {words[ulong(group) * 64], words[ulong(group) * 64 + 32],
                    scales[meta], biases[meta], scales[meta + 4], biases[meta + 4],
                    sums[group * stride + row], sums[group * stride + row + 8]};
        }

        template <int Arithmetic>
        inline void mere_group_accumulate(thread float& value, thread float& correction,
                                          float dot, float sum, float scale, float bias) {
            if constexpr (Arithmetic == 0) {
                value = fma(scale, dot, fma(bias, sum, value));
            } else {
                // The tensor operand contains q - 8, so move 8 * scale into the bias.
                const float term = fma(fma(8.0f, scale, bias), sum, scale * dot);
                if constexpr (Arithmetic == 2) {
                    const float adjusted = term - correction;
                    const float next = value + adjusted;
                    correction = (next - value) - adjusted;
                    value = next;
                } else {
                    value += term;
                }
            }
        }
        """

    private static let groupPack = MLXFast.metalKernel(
        name: "mere_m5_group_affine_pack_v1", inputNames: ["w"], outputNames: ["packed"],
        source: """
            const int lane = int(thread_index_in_simdgroup);
            const int block = int(threadgroup_position_in_grid.x), group = int(threadgroup_position_in_grid.y);
            const int wordsPerRow = w_shape[1], kg = wordsPerRow / 8;
            constexpr auto descriptor = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
            matmul2d<descriptor, execution_simdgroups<1>> operation;
            auto operand = operation.template get_right_input_cooperative_tensor<bfloat, half, float>();
            uint words[8];
            for (int i = 0; i < 8; ++i) words[i] = 0;
            for (int i = 0; i < operand.get_capacity(); ++i) {
                const auto coordinate = operand.get_multidimensional_index(i);
                const int k = coordinate[0], n = block * 32 + coordinate[1];
                const uint source = w[ulong(n) * wordsPerRow + group * 8 + k / 8];
                const uint code = (source >> (4 * (k % 8))) & 15u;
                // Inverse of the paired-half unpack used by the register kernel.
                const int pair = i / 2, within = pair % 16;
                const int word = 2 * ((within % 8) / 2) + pair / 16;
                const int nibble = 4 * (i % 2) + (within & 1) + (within >= 8 ? 2 : 0);
                words[word] |= code << (4 * nibble);
            }
            device uint4* out = reinterpret_cast<device uint4*>(packed);
            const ulong index = (ulong(block) * kg + group) * 64 + lane;
            out[index] = uint4(words[0], words[1], words[2], words[3]);
            out[index + 32] = uint4(words[4], words[5], words[6], words[7]);
            """, header: groupHeader, ensureRowContiguous: true
    )

    static func groupPackedCodes(_ weight: MLXArray) -> MLXArray {
        let tiles = weight.dim(0) / 32, groups = weight.dim(1) / 8
        return groupPack([weight], grid: (tiles * 32, groups, 1), threadGroup: (32, 1, 1),
            outputShapes: [[tiles, groups, 64, 4]], outputDTypes: [.uint32])[0]
    }

    private static let groupSums = MLXFast.metalKernel(
        name: "mere_m5_group_affine_sums_v1", inputNames: ["x"], outputNames: ["sums"],
        source: groupSumsSource, header: groupHeader, ensureRowContiguous: true
    )

    static let groupSumsSource = """
            const uint at = thread_position_in_grid.x;
            const int k = K, rows = x_shape[0], row = int(at) % BankRows, group = int(at) / BankRows;
            if (group >= k / 64) return;
            float total = 0.0f;
            if (row < rows) for (int i = 0; i < 64; ++i) total += float(x[row * k + group * 64 + i]);
            sums[at] = total;
            """

    private static let groupProjection = MLXFast.metalKernel(
        name: "mere_m5_group_affine_projection_v1",
        inputNames: ["w0", "s0", "b0", "w1", "s1", "b1", "x", "sums"], outputNames: ["y"],
        source: groupProjectionSource, header: groupHeader, ensureRowContiguous: true
    )

    static let groupProjectionBindings = """
        const int tile = int(threadgroup_position_in_grid.x);
        const bool second = Stacked && tile >= N / 32;
        const int block = second ? tile - N / 32 : tile;
        const int outColumn = block * 32 + (second ? n : 0);
        const int stride = Stacked ? 2 * n : n;
        const int lane = int(thread_index_in_simdgroup), split = int(simdgroup_index_in_threadgroup);
        const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1);
        const int mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
        const int first = split * kg / Splits, last = (split + 1) * kg / Splits;
        const device uint4* words = reinterpret_cast<const device uint4*>(second ? w1 : w0)
            + ulong(block) * kg * 64 + lane;
        const device ushort4* scales = reinterpret_cast<const device ushort4*>(second ? s1 : s0)
            + ulong(block) * kg * 8;
        const device ushort4* biases = reinterpret_cast<const device ushort4*>(second ? b1 : b0)
            + ulong(block) * kg * 8;
        constexpr auto descriptor = matmul2d_descriptor(
            16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
        matmul2d<descriptor, execution_simdgroups<1>> operation;
        const device bfloat* inputValues = reinterpret_cast<const device bfloat*>(x) + ulong(rowBase) * k;
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a(
            const_cast<device bfloat*>(inputValues), dextents<int32_t, 2>(k, activeRows));
        auto ax = a.slice(0, 0);
        auto codes = operation.template get_right_input_cooperative_tensor<bfloat, half, float>();
        auto product = operation.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(ax)>, metal::remove_addrspace_t<decltype(codes)>, float>();
        thread half2* pairs = reinterpret_cast<thread half2*>(&codes[0]);
        constexpr half center = Arithmetic == 0 ? 1024.0h : 1032.0h;
        """

    static let unpackGroupCodes = """
        for (int word = 0; word < 8; ++word) {
            const uint packed = word < 4 ? current.w0[word] : current.w1[word - 4];
            const uint low = packed & 0x0F0F0F0Fu, high = (packed >> 4) & 0x0F0F0F0Fu;
            const int at = 16 * (word & 1) + 2 * (word >> 1);
            pairs[at] = as_type<half2>((low & 0x00FF00FFu) | 0x64006400u) - half2(center);
            pairs[at + 1] = as_type<half2>((high & 0x00FF00FFu) | 0x64006400u) - half2(center);
            pairs[at + 8] = as_type<half2>(((low >> 8) & 0x00FF00FFu) | 0x64006400u) - half2(center);
            pairs[at + 9] = as_type<half2>(((high >> 8) & 0x00FF00FFu) | 0x64006400u) - half2(center);
        }
        """

    static let accumulateGroupProducts = """
        for (int j = 0; j < 4; ++j) {
            const float scale0 = as_type<float>(uint(current.s0[j]) << 16);
            const float bias0 = as_type<float>(uint(current.b0[j]) << 16);
            const float scale1 = as_type<float>(uint(current.s1[j]) << 16);
            const float bias1 = as_type<float>(uint(current.b1[j]) << 16);
            mere_group_accumulate<Arithmetic>(values[j], corrections[j], product[j], current.x0, scale0, bias0);
            mere_group_accumulate<Arithmetic>(values[4 + j], corrections[4 + j], product[4 + j], current.x1, scale0, bias0);
            mere_group_accumulate<Arithmetic>(values[8 + j], corrections[8 + j], product[8 + j], current.x0, scale1, bias1);
            mere_group_accumulate<Arithmetic>(values[12 + j], corrections[12 + j], product[12 + j], current.x1, scale1, bias1);
        }
        """

    static let reduceGroupSplits = """
        if (Splits > 1) {
            if (split > 0) for (int i = 0; i < 16; ++i) partial[((split - 1) * 16 + i) * 32 + lane] = values[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (split == 0) for (int other = 1; other < Splits; ++other)
                for (int i = 0; i < 16; ++i) values[i] += partial[((other - 1) * 16 + i) * 32 + lane];
        }
        """

    static let groupProjectionAccumulator = """
        const int k = K, kg = K / 64, n = N;
        const int rowBase = int(threadgroup_position_in_grid.y) * 16, rows = sums_shape[1];
        const int activeRows = min(16, int(x_shape[0]) - rowBase);
        \(groupProjectionBindings)
        float values[16], corrections[16];
        for (int i = 0; i < 16; ++i) values[i] = corrections[i] = 0.0f;
        MereGroupLoad next = mere_group_fetch(words, scales, biases, sums + rowBase, first, nb, mb, rows);
        for (int group = first; group < last; ++group) {
            const MereGroupLoad current = next;
            \(unpackGroupCodes)
            if (group + 1 < last) next = mere_group_fetch(words, scales, biases, sums + rowBase, group + 1, nb, mb, rows);
            auto input = a.slice(group * 64, 0);
            operation.run(input, codes, product);
            \(accumulateGroupProducts)
        }
        threadgroup float partial[(Splits > 1 ? Splits - 1 : 1) * 16 * 32];
        \(reduceGroupSplits)
        """

    static let groupStore = """
        if (split == 0) for (int i = 0; i < 16; ++i) {
            const int row = mb + 8 * ((i >> 2) & 1);
            const int column = outColumn + nb + 16 * (i >> 3) + (i & 3);
            y[ulong(rowBase + row) * stride + column] = bfloat(values[i]);
        }
        """

    static let groupProjectionSource = groupProjectionAccumulator + "\n" + groupStore

    private static let groupActivation = MLXFast.metalKernel(
        name: "mere_m5_group_affine_activation_v1", inputNames: ["gu"], outputNames: ["y"],
        source: groupActivationSource, header: groupHeader, ensureRowContiguous: true
    )

    static let groupActivationSource = """
            const uint index = thread_position_in_grid.x, n = gu_shape[1] / 2;
            if (index >= gu_shape[0] * n) return;
            const uint row = index / n, column = index % n;
            const bfloat gate = gu[row * 2 * n + column], up = gu[row * 2 * n + n + column];
            const bfloat activated = gate * mere_tiled_sigmoid(gate);
            y[index] = activated * up;
            """

    private struct GroupLayout {
        let x: MLXArray
        let sums: MLXArray
        let bankRows: Int
        let width: Int
        let n: Int
        let shape: [Int]
    }

    private static func groupLayout(_ input: MLXArray, weights: Weights, splits: Int) -> GroupLayout? {
        guard M5AffineQ4Verification.isSupported, input.dtype == .bfloat16, input.ndim >= 2 else { return nil }
        let k = input.dim(-1), width = input.size / k, n = weights.originalWeight.dim(0)
        guard k == weights.originalWeight.dim(1) * 8, input.dim(-2) == width, width > 0,
              width <= 16 || groupAffinePrefillEnabled,
              [1, 2, 4, 8].contains(splits), k.isMultiple(of: 64 * splits), k / splits >= 512 else { return nil }
        let bankRows = ((width + 15) / 16) * 16
        let x = input.reshaped(width, k)
        let sums = groupSums([x], template: [("K", k), ("BankRows", bankRows)],
            grid: (k / 64 * bankRows, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[k / 64, bankRows]], outputDTypes: [.float32])[0]
        var shape = input.shape
        shape[shape.count - 1] = n
        return GroupLayout(x: x, sums: sums, bankRows: bankRows, width: width, n: n, shape: shape)
    }

    static func groupAffineProjection(
        _ x: MLXArray, weights: Weights, splits: Int, arithmetic: GroupAffineArithmetic
    ) -> MLXArray? {
        guard let layout = groupLayout(x, weights: weights, splits: splits) else { return nil }
        let rows = layout.bankRows
        let rowTiles = prefillFFNRowTiles(width: layout.width, input: layout.x.dim(1), output: layout.n)
        let kernel = rowTiles == 1 ? groupProjection : prefillFFNProjection
        let output = kernel(
            [weights.codes, weights.scales, weights.biases, weights.codes, weights.scales, weights.biases, layout.x, layout.sums],
            template: [("Splits", splits), ("Stacked", 0), ("K", layout.x.dim(1)), ("N", layout.n), ("Arithmetic", arithmetic.rawValue)] + (rowTiles == 1 ? [] : [("RowTiles", rowTiles)]),
            grid: (layout.n * splits, rows / (16 * rowTiles), 1), threadGroup: (splits * 32, 1, 1),
            outputShapes: [[rows, layout.n]], outputDTypes: [.bfloat16])[0]
        return output[0..<layout.width, 0...].reshaped(layout.shape)
    }

    static func groupAffineSwiGLU(_ x: MLXArray, gate: Weights, up: Weights) -> MLXArray? {
        guard gate.originalWeight.shape == up.originalWeight.shape,
              let layout = groupLayout(x, weights: gate, splits: 1) else { return nil }
        let rows = layout.bankRows
        let splits = groupAffinePartitionCount(inputSize: x.dim(-1), outputSize: layout.n, rows: layout.width)
        let rowTiles = prefillFFNRowTiles(width: layout.width, input: layout.x.dim(1), output: layout.n)
        let kernel = rowTiles == 1 ? groupProjection : prefillFFNProjection
        let projected = kernel(
            [gate.codes, gate.scales, gate.biases, up.codes, up.scales, up.biases, layout.x, layout.sums],
            template: [("Splits", splits), ("Stacked", 1), ("K", layout.x.dim(1)), ("N", layout.n), ("Arithmetic", 0)] + (rowTiles == 1 ? [] : [("RowTiles", rowTiles)]),
            grid: (layout.n * 2 * splits, rows / (16 * rowTiles), 1), threadGroup: (splits * 32, 1, 1),
            outputShapes: [[rows, layout.n * 2]], outputDTypes: [.bfloat16])[0]
        let output = groupActivation([projected], grid: (rows * layout.n, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, layout.n]], outputDTypes: [.bfloat16])[0]
        return output[0..<layout.width, 0...].reshaped(layout.shape)
    }
}
#endif
