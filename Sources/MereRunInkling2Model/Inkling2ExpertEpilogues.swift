import MLX

#if os(macOS)
enum Inkling2ExpertEpilogues {
    private static let useScaleTable = GPU.deviceInfo().architecture == "applegpu_g17s"

    // Match the transposed packed gathers' FP32 dot-product order.
    private static let packedHeader = """
        // MLX_INCLUDE_FP_QUANTIZED_HEADERS
        // FP16 bit patterns from the MLX E4M3 scale decoder.
        constant ushort inkling_scale_table[256] = {
            0x0000, 0x1800, 0x1c00, 0x1e00, 0x2000, 0x2100, 0x2200, 0x2300,
            0x2400, 0x2480, 0x2500, 0x2580, 0x2600, 0x2680, 0x2700, 0x2780,
            0x2800, 0x2880, 0x2900, 0x2980, 0x2a00, 0x2a80, 0x2b00, 0x2b80,
            0x2c00, 0x2c80, 0x2d00, 0x2d80, 0x2e00, 0x2e80, 0x2f00, 0x2f80,
            0x3000, 0x3080, 0x3100, 0x3180, 0x3200, 0x3280, 0x3300, 0x3380,
            0x3400, 0x3480, 0x3500, 0x3580, 0x3600, 0x3680, 0x3700, 0x3780,
            0x3800, 0x3880, 0x3900, 0x3980, 0x3a00, 0x3a80, 0x3b00, 0x3b80,
            0x3c00, 0x3c80, 0x3d00, 0x3d80, 0x3e00, 0x3e80, 0x3f00, 0x3f80,
            0x4000, 0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380,
            0x4400, 0x4480, 0x4500, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780,
            0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80,
            0x4c00, 0x4c80, 0x4d00, 0x4d80, 0x4e00, 0x4e80, 0x4f00, 0x4f80,
            0x5000, 0x5080, 0x5100, 0x5180, 0x5200, 0x5280, 0x5300, 0x5380,
            0x5400, 0x5480, 0x5500, 0x5580, 0x5600, 0x5680, 0x5700, 0x5780,
            0x5800, 0x5880, 0x5900, 0x5980, 0x5a00, 0x5a80, 0x5b00, 0x5b80,
            0x5c00, 0x5c80, 0x5d00, 0x5d80, 0x5e00, 0x5e80, 0x5f00, 0x7f80,
            0x8000, 0x9800, 0x9c00, 0x9e00, 0xa000, 0xa100, 0xa200, 0xa300,
            0xa400, 0xa480, 0xa500, 0xa580, 0xa600, 0xa680, 0xa700, 0xa780,
            0xa800, 0xa880, 0xa900, 0xa980, 0xaa00, 0xaa80, 0xab00, 0xab80,
            0xac00, 0xac80, 0xad00, 0xad80, 0xae00, 0xae80, 0xaf00, 0xaf80,
            0xb000, 0xb080, 0xb100, 0xb180, 0xb200, 0xb280, 0xb300, 0xb380,
            0xb400, 0xb480, 0xb500, 0xb580, 0xb600, 0xb680, 0xb700, 0xb780,
            0xb800, 0xb880, 0xb900, 0xb980, 0xba00, 0xba80, 0xbb00, 0xbb80,
            0xbc00, 0xbc80, 0xbd00, 0xbd80, 0xbe00, 0xbe80, 0xbf00, 0xbf80,
            0xc000, 0xc080, 0xc100, 0xc180, 0xc200, 0xc280, 0xc300, 0xc380,
            0xc400, 0xc480, 0xc500, 0xc580, 0xc600, 0xc680, 0xc700, 0xc780,
            0xc800, 0xc880, 0xc900, 0xc980, 0xca00, 0xca80, 0xcb00, 0xcb80,
            0xcc00, 0xcc80, 0xcd00, 0xcd80, 0xce00, 0xce80, 0xcf00, 0xcf80,
            0xd000, 0xd080, 0xd100, 0xd180, 0xd200, 0xd280, 0xd300, 0xd380,
            0xd400, 0xd480, 0xd500, 0xd580, 0xd600, 0xd680, 0xd700, 0xd780,
            0xd800, 0xd880, 0xd900, 0xd980, 0xda00, 0xda80, 0xdb00, 0xdb80,
            0xdc00, 0xdc80, 0xdd00, 0xdd80, 0xde00, 0xde80, 0xdf00, 0xff80
        };
        template <typename T, int K, int PACK, int BITS, int GROUP, bool LOOKUP>
        inline void inkling2_packed_dot4(const device T* input, const device uint* weights,
                                        const device uchar* scales, uint lane, thread float* acc) {
            for (uint col = lane * PACK; col < K; col += 32 * PACK) {
                float x_thread[PACK];
                load_vector<T, float, PACK>(input + col, x_thread);
                #pragma unroll
                for (uint row = 0; row < 4; ++row) {
                    const device uint* words = weights + row * (K * BITS / 32) + col * BITS / 32;
                    const device uchar* code = reinterpret_cast<const device uchar*>(words);
                    const uchar encoded = scales[row * (K / GROUP) + col / GROUP];
                    float scale;
                    if constexpr (LOOKUP) scale = float(as_type<half>(inkling_scale_table[encoded]));
                    else scale = dequantize_scale<float, GROUP>(encoded);
                    acc[row] += qdot<float, PACK, BITS>(code, x_thread, scale);
                }
            }
            #pragma unroll
            for (uint row = 0; row < 4; ++row) acc[row] = simd_sum(acc[row]);
        }

        """

    private static let fusedGateKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_packed_gate_swiglu",
        inputNames: ["x", "packed", "blocks", "indices", "outer", "gate_cap", "up_cap"],
        outputNames: ["hidden"],
        source: """
            const uint route = threadgroup_position_in_grid.y;
            const uint expert = uint(indices[route]);
            const uint lane = thread_index_in_simdgroup;
            const uint first = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup * 4;
            const device T* input = x + (route / ROUTES) * K;
            const device uint* weights = packed + (expert * N + first) * (K * BITS / 32);
            const device uchar* scales = blocks + (expert * N + first) * (K / GROUP);
            float acc[4] = {0};
            inkling2_packed_dot4<T, K, PACK, BITS, GROUP, LOOKUP>(input, weights, scales, lane, acc);
            if (lane == 0) {
                const float scale = BITS == 4 ? outer[expert] : 1.0f;
                #pragma unroll
                for (uint pair = 0; pair < 2; ++pair) {
                    // Preserve the BF16 projection boundary before activation.
                    const float gate = float(T(acc[2 * pair] * scale));
                    const float up = float(T(acc[2 * pair + 1] * scale));
                    const float g = gate_cap * tanh(gate / gate_cap);
                    const float u = up_cap * tanh(up / up_cap);
                    const float sigmoid = 1.0f / (1.0f + exp(-gate));
                    hidden[route * (N / 2) + first / 2 + pair] = T(g * sigmoid * u);
                }
            }
            """,
        header: packedHeader
    )

    static func fusedGate(_ input: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray, outer: MLXArray?,
                          gateCap: Float, upCap: Float) -> MLXArray {
        let k = input.dim(-1), n = packed.dim(-2)
        return fusedGateKernel([input, packed, scales, indices, outer ?? input, gateCap, upCap],
                               template: projectionTemplate(input.dtype, k: k, n: n,
                                                            routes: indices.dim(-1), hasOuterScale: outer != nil),
                               grid: (n / 8 * 64, indices.size, 1), threadGroup: (64, 1, 1),
                               outputShapes: [indices.shape + [1, n / 2]], outputDTypes: [input.dtype])[0]
    }

    private static let fusedDownKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_packed_down_weighted_reduce",
        inputNames: ["x", "packed", "blocks", "indices", "outer", "routing"],
        outputNames: ["output"],
        source: """
            const uint token = threadgroup_position_in_grid.y;
            const uint choice = simdgroup_index_in_threadgroup;
            const uint route = token * ROUTES + choice;
            const uint expert = uint(indices[route]);
            const uint lane = thread_index_in_simdgroup;
            const uint first = threadgroup_position_in_grid.x * 4;
            const device T* input = x + route * K;
            const device uint* weights = packed + (expert * N + first) * (K * BITS / 32);
            const device uchar* scales = blocks + (expert * N + first) * (K / GROUP);
            threadgroup float contributions[ROUTES][4];
            float acc[4] = {0};
            inkling2_packed_dot4<T, K, PACK, BITS, GROUP, LOOKUP>(input, weights, scales, lane, acc);
            if (lane == 0) {
                const float scale = BITS == 4 ? outer[expert] : 1.0f;
                for (uint row = 0; row < 4; ++row) {
                    contributions[choice][row] = float(T(acc[row] * scale));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint row = thread_index_in_threadgroup;
            if (row < 4) {
                float result = 0;
                // Sum rounded projections in the original routing order.
                for (uint choice = 0; choice < ROUTES; ++choice) {
                    result += contributions[choice][row] * float(routing[token * ROUTES + choice]);
                }
                output[token * N + first + row] = T(result);
            }
            """,
        header: packedHeader
    )

    static func fusedDown(_ input: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray,
                          outer: MLXArray?, routing: MLXArray) -> MLXArray {
        let k = input.dim(-1), n = packed.dim(-2), routes = indices.dim(-1)
        return fusedDownKernel([input, packed, scales, indices, outer ?? input, routing],
                               template: projectionTemplate(input.dtype, k: k, n: n, routes: routes,
                                                            hasOuterScale: outer != nil),
                               grid: (n / 4 * 32, indices.size, 1), threadGroup: (32, routes, 1),
                               outputShapes: [[indices.dim(0), n]], outputDTypes: [input.dtype])[0]
    }

    private static func projectionTemplate(_ dtype: DType, k: Int, n: Int, routes: Int,
                                           hasOuterScale: Bool) -> [(String, any KernelTemplateArg)] {
        let bits = hasOuterScale ? 4 : 8
        let valuesPerWord = 32 / bits
        let valuesPerLane = k.isMultiple(of: 64 * valuesPerWord) ? 2 * valuesPerWord : valuesPerWord
        return [("T", dtype), ("K", k), ("N", n), ("ROUTES", routes),
                ("PACK", valuesPerLane), ("BITS", bits), ("GROUP", hasOuterScale ? 16 : 32), ("LOOKUP", hasOuterScale && useScaleTable)]
    }

    private static let activateKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_expert_activation",
        inputNames: ["projected", "indices", "scales", "gate_cap", "up_cap"],
        outputNames: ["output"],
        source: """
            const uint index = thread_position_in_grid.x;
            const uint route = index / WIDTH;
            const float scale = SCALED ? float(scales[indices[route]]) : 1.0f;
            const float gate = float(T(float(projected[2 * index]) * scale));
            const float up = float(T(float(projected[2 * index + 1]) * scale));
            const float g = gate_cap * tanh(gate / gate_cap);
            const float u = up_cap * tanh(up / up_cap);
            const float sigmoid = 1.0f / (1.0f + exp(-gate));
            output[index] = T(g * sigmoid * u);
            """
    )

    private static let reduceKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_expert_reduce",
        inputNames: ["projected", "indices", "scales", "routing"],
        outputNames: ["output"],
        source: """
            const uint index = thread_position_in_grid.x;
            const uint token = index / WIDTH;
            const uint channel = index % WIDTH;
            float result = 0;
            for (uint choice = 0; choice < ROUTES; ++choice) {
                const uint route = token * ROUTES + choice;
                const float scale = SCALED ? float(scales[indices[route]]) : 1.0f;
                const T value = T(float(projected[route * WIDTH + channel]) * scale);
                result += float(value) * float(routing[route]);
            }
            output[index] = T(result);
            """
    )

    private static let invertOrderKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_prefill_invert_expert_order",
        inputNames: ["order"], outputNames: ["inverse"],
        source: """
            const uint slot = thread_position_in_grid.x;
            inverse[order[slot]] = slot;
            """
    )

    private static let reduceReorderedKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_prefill_reduce_reordered_experts",
        inputNames: ["projected", "indices", "inverse", "scales", "routing"], outputNames: ["output"],
        source: """
            const uint index = thread_position_in_grid.x;
            const uint token = index / WIDTH;
            const uint channel = index % WIDTH;
            float result = 0;
            for (uint choice = 0; choice < ROUTES; ++choice) {
                const uint route = token * ROUTES + choice;
                const float scale = SCALED ? float(scales[indices[route]]) : 1.0f;
                const T value = T(float(projected[inverse[route] * WIDTH + channel]) * scale);
                result += float(value) * float(routing[route]);
            }
            output[index] = T(result);
            """
    )

    static func reduceReorderedPrefill(
        _ projected: MLXArray, indices: MLXArray, order: MLXArray, scales: MLXArray?, routing: MLXArray, dtype: DType
    ) -> MLXArray {
        let width = projected.dim(-1)
        let tokens = indices.dim(0)
        let count = tokens * width
        let inverse = invertOrderKernel([order], grid: (order.size, 1, 1), threadGroup: (min(order.size, 256), 1, 1),
                                        outputShapes: [order.shape], outputDTypes: [.uint32])[0]
        return reduceReorderedKernel(
            [projected, indices, inverse, scales ?? projected, routing],
            template: [("T", dtype), ("WIDTH", width), ("ROUTES", indices.dim(1)), ("SCALED", scales != nil)],
            grid: (count, 1, 1), threadGroup: (min(count, 256), 1, 1),
            outputShapes: [[tokens, width]], outputDTypes: [dtype]
        )[0]
    }

    static func activate(
        _ projected: MLXArray, indices: MLXArray, scales: MLXArray?, dtype: DType, gateCap: Float, upCap: Float
    ) -> MLXArray {
        let width = projected.dim(-1) / 2
        let count = indices.size * width
        return activateKernel([projected, indices, scales ?? projected, gateCap, upCap],
                              template: [("T", dtype), ("WIDTH", width), ("SCALED", scales != nil)],
                              grid: (count, 1, 1), threadGroup: (min(count, 256), 1, 1),
                              outputShapes: [indices.shape + [1, width]], outputDTypes: [dtype])[0]
    }

    static func reduce(
        _ projected: MLXArray, indices: MLXArray, scales: MLXArray?, routing: MLXArray, dtype: DType
    ) -> MLXArray {
        let width = projected.dim(-1), tokens = indices.dim(0), count = tokens * width
        return reduceKernel([projected, indices, scales ?? projected, routing],
                            template: [("T", dtype), ("WIDTH", width), ("ROUTES", indices.dim(1)), ("SCALED", scales != nil)],
                            grid: (count, 1, 1), threadGroup: (min(count, 256), 1, 1),
                            outputShapes: [[tokens, width]], outputDTypes: [dtype])[0]
    }
}
#endif
