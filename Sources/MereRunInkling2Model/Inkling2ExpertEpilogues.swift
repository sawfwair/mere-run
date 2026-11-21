import MLX

#if os(macOS)
enum Inkling2ExpertEpilogues {
    // Match the transposed NVFP4 gather's FP32 dot-product order.
    private static let nvfp4Header = """
        // MLX_INCLUDE_FP_QUANTIZED_HEADERS
        template <typename T, int K, int PACK>
        inline void inkling2_nvfp4_dot4(const device T* input, const device uint* weights,
                                        const device uchar* scales, uint lane, thread float* acc) {
            for (uint col = lane * PACK; col < K; col += 32 * PACK) {
                float x_thread[PACK];
                load_vector<T, float, PACK>(input + col, x_thread);
                #pragma unroll
                for (uint row = 0; row < 4; ++row) {
                    const device uchar* code = reinterpret_cast<const device uchar*>(weights + row * (K / 8) + col / 8);
                    const float scale = dequantize_scale<float, 16>(scales[row * (K / 16) + col / 16]);
                    acc[row] += qdot<float, PACK, 4>(code, x_thread, scale);
                }
            }
            #pragma unroll
            for (uint row = 0; row < 4; ++row) acc[row] = simd_sum(acc[row]);
        }

        """

    private static let fusedGateKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_nvfp4_gate_swiglu",
        inputNames: ["x", "packed", "blocks", "indices", "outer", "gate_cap", "up_cap"],
        outputNames: ["hidden"],
        source: """
            const uint route = threadgroup_position_in_grid.y;
            const uint expert = uint(indices[route]);
            const uint lane = thread_index_in_simdgroup;
            const uint first = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup * 4;
            const device T* input = x + (route / ROUTES) * K;
            const device uint* weights = packed + (expert * N + first) * (K / 8);
            const device uchar* scales = blocks + (expert * N + first) * (K / 16);
            float acc[4] = {0};
            inkling2_nvfp4_dot4<T, K, PACK>(input, weights, scales, lane, acc);
            if (lane == 0) {
                const float scale = outer[expert];
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
        header: nvfp4Header
    )

    static func fusedGate(_ input: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray, outer: MLXArray,
                          gateCap: Float, upCap: Float) -> MLXArray {
        let k = input.dim(-1), n = packed.dim(-2)
        return fusedGateKernel([input, packed, scales, indices, outer, gateCap, upCap],
                               template: [("T", input.dtype), ("K", k), ("N", n),
                                          ("ROUTES", indices.dim(-1)), ("PACK", k.isMultiple(of: 512) ? 16 : 8)],
                               grid: (n / 8 * 64, indices.size, 1), threadGroup: (64, 1, 1),
                               outputShapes: [indices.shape + [1, n / 2]], outputDTypes: [input.dtype])[0]
    }

    private static let fusedDownKernel = MLXFast.metalKernel(
        name: "mererun_inkling2_nvfp4_down_weighted_reduce",
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
            const device uint* weights = packed + (expert * N + first) * (K / 8);
            const device uchar* scales = blocks + (expert * N + first) * (K / 16);
            threadgroup float contributions[ROUTES][4];
            float acc[4] = {0};
            inkling2_nvfp4_dot4<T, K, PACK>(input, weights, scales, lane, acc);
            if (lane == 0) {
                const float scale = outer[expert];
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
        header: nvfp4Header
    )

    static func fusedDown(_ input: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray,
                          outer: MLXArray, routing: MLXArray) -> MLXArray {
        let k = input.dim(-1), n = packed.dim(-2), routes = indices.dim(-1)
        return fusedDownKernel([input, packed, scales, indices, outer, routing],
                               template: [("T", input.dtype), ("K", k), ("N", n),
                                          ("ROUTES", routes), ("PACK", k.isMultiple(of: 512) ? 16 : 8)],
                               grid: (n / 4 * 32, indices.size, 1), threadGroup: (32, routes, 1),
                               outputShapes: [[indices.dim(0), n]], outputDTypes: [input.dtype])[0]
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
