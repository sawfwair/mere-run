#if os(macOS)
@available(macOS 26.0, *)
extension M5NativeFFN {
    // Match MLX's MIT-licensed bf16_math.h overloads used by the shared
    // sigmoid: abs and precise exp both round their results back to BF16.
    private static let bfloatMath = """
        #include <metal_stdlib>
        namespace metal {
            METAL_FUNC bfloat abs(bfloat x) {
                return bfloat(__metal_fabs(float(x), __METAL_MAYBE_FAST_MATH__));
            }
            METAL_FUNC bfloat exp(bfloat x) {
                return bfloat(__metal_exp(float(x), __METAL_MAYBE_FAST_MATH__));
            }
            namespace precise {
                METAL_FUNC bfloat exp(bfloat x) {
                    return bfloat(__metal_exp(float(x), __METAL_PRECISE_MATH__));
                }
            }
        }
        """

    static func sums(name: String, width: Int) -> String {
        """
        kernel void \(name)(const device bfloat* x [[buffer(0)]], device float* sums [[buffer(1)]],
            constant int& rowCount [[buffer(2)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
            constexpr int K = \(width);
            const int BankRows = ((rowCount + 15) / 16) * 16;
            const int x_shape[] = {rowCount, K};
            \(M5TiledAffineQ4.groupSumsSource)
        }
        """
    }

    static func projection(
        name: String, width: Int, columns: Int, splits: Int, stacked: Bool, arithmetic: GroupAffineArithmetic = .standard,
        skippingOutputRows: Range<Int>? = nil
    ) -> String {
        // Packed weights are contiguous in 32-column tiles. Skip the omitted tiles in
        // each weight bank while keeping the destination rows densely packed.
        let offsetWeights = skippingOutputRows.map { gap in
            """
            if (threadgroup_position_in_grid.x >= \(gap.lowerBound / 32)) {
                w0 += \(gap.count * width / 8);
                s0 += \(gap.count * width / 64);
                b0 += \(gap.count * width / 64);
            }
            """
        } ?? ""
        return """
        kernel void \(name)(
            const device uint* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],
            const device bfloat* b0 [[buffer(2)]], const device uint* w1 [[buffer(3)]],
            const device bfloat* s1 [[buffer(4)]], const device bfloat* b1 [[buffer(5)]],
            const device bfloat* x [[buffer(6)]], const device float* sums [[buffer(7)]],
            device bfloat* y [[buffer(8)]], constant int& rowCount [[buffer(9)]],
            uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
            uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
            uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
            constexpr int K = \(width), N = \(columns), Splits = \(splits), Stacked = \(stacked ? 1 : 0);
            constexpr int Arithmetic = \(arithmetic.rawValue);
            const int x_shape[] = {rowCount, K}, sums_shape[] = {K / 64, ((rowCount + 15) / 16) * 16};
            \(offsetWeights)
            \(M5TiledAffineQ4.groupProjectionSource)
        }
        """
    }

    static var source: String {
        shaderHeader
            + sums(name: "mere_native_ffn_input_sums", width: 5_120)
            + projection(name: "mere_native_ffn_gate_up", width: 5_120, columns: 17_408, splits: 2, stacked: true)
            + """
            kernel void mere_native_ffn_activation(const device bfloat* gu [[buffer(0)]],
                device bfloat* y [[buffer(1)]], constant int& rowCount [[buffer(2)]],
                uint3 thread_position_in_grid [[thread_position_in_grid]]) {
                const int gu_shape[] = {rowCount, 34816};
                \(M5TiledAffineQ4.groupActivationSource)
            }
            """
            + """
            kernel void mere_native_ffn_activation_with_sums(const device bfloat* gu [[buffer(0)]],
                device bfloat* y [[buffer(1)]], constant int& rows [[buffer(2)]], device float* sums [[buffer(3)]],
                uint2 tile [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
                constexpr uint n = 17408;
                const uint row = tile.y, column = tile.x * 64 + lid;
                const uint sumIndex = tile.x * (((rows + 15) / 16) * 16) + row;
                if (row >= uint(rows)) {
                    if (lid == 0) sums[sumIndex] = 0.0f;
                    return;
                }
                threadgroup bfloat values[64];
                const bfloat gate = gu[row * 2 * n + column];
                const bfloat activated = gate * mere_tiled_sigmoid(gate);
                const bfloat result = activated * gu[row * 2 * n + n + column];
                y[row * n + column] = result;
                values[lid] = result;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid == 0) {
                    float total = 0.0f;
                    for (uint i = 0; i < 64; ++i) total += float(values[i]);
                    sums[sumIndex] = total;
                }
            }
            """
            + sums(name: "mere_native_ffn_activation_sums", width: 17_408)
            + projection(name: "mere_native_ffn_down", width: 17_408, columns: 5_120, splits: 4, stacked: false)
    }

    package static var shaderHeader: String { bfloatMath + "\n" + M5TiledAffineQ4.groupHeader }
}
#endif
