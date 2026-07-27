import Foundation
import MLX
import MLXFast

/// Fused gate/up gather-GEMV primitives for small-route quantized MoE decode.
///
/// Compatible models keep their native down projection and all prefill paths.
/// Each helper returns `nil` outside its measured quantization and alignment
/// boundary so callers fall back to MLX's portable gather implementation.
enum RoutedMoERouting {
    static func parseBoolean(_ raw: String?, default defaultValue: Bool) -> Bool {
        guard let raw else { return defaultValue }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return defaultValue
        }
    }

    /// Computes an unsorted NVFP4 gate/up gather-GEMV and SwiGLU activation.
    ///
    /// One threadgroup runs the gate and up projections concurrently for one
    /// expert route and one eight-column output tile. The input token is read
    /// directly from `x`, avoiding the repeated top-k route tensor.
    static func fusedGatherNVFP4SwiGLU(
        _ x: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        expertIndices: MLXArray,
        topK: Int,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        guard Device.defaultDevice().deviceType == .gpu,
              x.dtype == .bfloat16 || x.dtype == .float16,
              gateWeight.dtype == .uint32,
              upWeight.dtype == .uint32,
              gateScales.dtype == .uint8,
              upScales.dtype == .uint8,
              expertIndices.dtype == .int32 || expertIndices.dtype == .uint32,
              x.ndim == 3,
              gateWeight.shape == upWeight.shape,
              gateScales.shape == upScales.shape,
              gateWeight.dim(0) == gateScales.dim(0),
              gateWeight.dim(1) == gateScales.dim(1),
              topK > 0,
              expertIndices.size == x.dim(0) * x.dim(1) * topK,
              groupSize == 16,
              bits == 4 else {
            return nil
        }

        let routeCount = expertIndices.size
        let outputDimensions = gateWeight.dim(1)
        let inputDimensions = x.dim(2)
        guard routeCount > 0,
              outputDimensions.isMultiple(of: outputTileWidth),
              inputDimensions.isMultiple(of: inputBlockWidth),
              gateWeight.dim(2) == inputDimensions / 8,
              gateScales.dim(2) == inputDimensions / groupSize else {
            return nil
        }

        return fusedGatherNVFP4SwiGLUKernel(
            [x, gateWeight, gateScales, upWeight, upScales, expertIndices],
            template: [
                ("DataT", x.dtype),
                ("GROUP_SIZE", groupSize),
                ("BITS", bits),
                ("TOP_K", topK),
                ("ROUTE_COUNT", routeCount),
                ("OUTPUT_DIMENSIONS", outputDimensions),
                ("INPUT_DIMENSIONS", inputDimensions),
            ],
            grid: (
                simdWidth,
                (outputDimensions / outputTileWidth) * parallelSIMDGroups,
                routeCount
            ),
            threadGroup: (simdWidth, parallelSIMDGroups, 1),
            outputShapes: [[routeCount, 1, outputDimensions]],
            outputDTypes: [x.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// Computes an unsorted affine-8 gate/up gather-GEMV and SwiGLU
    /// activation for compatible small-route expert layers.
    static func fusedGatherAffine8SwiGLU(
        _ x: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        gateBiases: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        upBiases: MLXArray,
        expertIndices: MLXArray,
        topK: Int,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        guard Device.defaultDevice().deviceType == .gpu,
              x.dtype == .bfloat16 || x.dtype == .float16,
              gateWeight.dtype == .uint32,
              upWeight.dtype == .uint32,
              gateScales.dtype == x.dtype,
              gateBiases.dtype == x.dtype,
              upScales.dtype == x.dtype,
              upBiases.dtype == x.dtype,
              expertIndices.dtype == .int32 || expertIndices.dtype == .uint32,
              x.ndim == 3,
              gateWeight.shape == upWeight.shape,
              gateScales.shape == upScales.shape,
              gateBiases.shape == upBiases.shape,
              gateScales.shape == gateBiases.shape,
              gateWeight.dim(0) == gateScales.dim(0),
              gateWeight.dim(1) == gateScales.dim(1),
              topK > 0,
              expertIndices.size == x.dim(0) * x.dim(1) * topK,
              groupSize == 64,
              bits == 8 else {
            return nil
        }

        let routeCount = expertIndices.size
        let outputDimensions = gateWeight.dim(1)
        let inputDimensions = x.dim(2)
        guard routeCount > 0,
              outputDimensions.isMultiple(of: outputTileWidth),
              inputDimensions.isMultiple(of: inputBlockWidth),
              gateWeight.dim(2) == inputDimensions / 4,
              gateScales.dim(2) == inputDimensions / groupSize else {
            return nil
        }

        return fusedGatherAffine8SwiGLUKernel(
            [
                x,
                gateWeight,
                gateScales,
                gateBiases,
                upWeight,
                upScales,
                upBiases,
                expertIndices,
            ],
            template: [
                ("DataT", x.dtype),
                ("GROUP_SIZE", groupSize),
                ("BITS", bits),
                ("TOP_K", topK),
                ("ROUTE_COUNT", routeCount),
                ("OUTPUT_DIMENSIONS", outputDimensions),
                ("INPUT_DIMENSIONS", inputDimensions),
            ],
            grid: (
                simdWidth,
                (outputDimensions / outputTileWidth) * parallelSIMDGroups,
                routeCount
            ),
            threadGroup: (simdWidth, parallelSIMDGroups, 1),
            outputShapes: [[routeCount, 1, outputDimensions]],
            outputDTypes: [x.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let outputTileWidth = 8
    private static let inputBlockWidth = 512
    private static let simdWidth = 32
    private static let parallelSIMDGroups = 4

    private static let fusedGatherNVFP4SwiGLUKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_gather_nvfp4_swiglu",
        inputNames: [
            "x",
            "gate_weight",
            "gate_scales",
            "up_weight",
            "up_scales",
            "expert_indices",
        ],
        outputNames: ["output"],
        source: """
            constexpr int packs_per_thread = 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int outputs_per_threadgroup = 8;
            constexpr int pack_factor = get_pack_factor<32, BITS>();
            constexpr int bytes_per_pack = get_bytes_per_pack<32>();
            constexpr int values_per_thread = pack_factor * packs_per_thread;
            constexpr int block_size = values_per_thread * SIMD_SIZE;
            constexpr int scale_step_per_thread =
                GROUP_SIZE / values_per_thread;
            constexpr int packed_input =
                INPUT_DIMENSIONS * bytes_per_pack / pack_factor;
            constexpr int scale_input = INPUT_DIMENSIONS / GROUP_SIZE;
            constexpr size_t weight_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * packed_input;
            constexpr size_t scale_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * scale_input;

            threadgroup DataT projection_values[
                2 * outputs_per_threadgroup
            ];

            const uint3 tile = threadgroup_position_in_grid;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint simd_lane = thread_index_in_simdgroup;
            const uint thread_index = thread_index_in_threadgroup;
            const bool gate_projection = simd_group < 2;
            const uint local_simd_group = simd_group % 2;
            const uint route = tile.z;
            if (route >= ROUTE_COUNT) {
                return;
            }

            const uint expert = uint(expert_indices[route]);
            const int output_row =
                int(tile.y) * outputs_per_threadgroup
                    + int(local_simd_group) * results_per_simdgroup;
            const device uint8_t* selected_weight =
                reinterpret_cast<const device uint8_t*>(
                    gate_projection ? gate_weight : up_weight);
            const device uint8_t* selected_scales =
                gate_projection ? gate_scales : up_scales;
            selected_weight +=
                size_t(expert) * weight_expert_stride
                    + size_t(output_row) * packed_input
                    + simd_lane * packs_per_thread * bytes_per_pack;
            selected_scales +=
                size_t(expert) * scale_expert_stride
                    + size_t(output_row) * scale_input
                    + simd_lane / scale_step_per_thread;
            const device DataT* input =
                x
                    + size_t(route / TOP_K) * INPUT_DIMENSIONS
                    + simd_lane * values_per_thread;

            float input_values[values_per_thread];
            float results[results_per_simdgroup] = {0};
            for (int k = 0; k < INPUT_DIMENSIONS; k += block_size) {
                load_vector<
                    DataT,
                    float,
                    values_per_thread>(input, input_values);

                STEEL_PRAGMA_UNROLL
                for (int row = 0; row < results_per_simdgroup; ++row) {
                    const device uint8_t* row_weight =
                        selected_weight + size_t(row) * packed_input;
                    const device uint8_t* row_scales =
                        selected_scales + size_t(row) * scale_input;
                    const float scale =
                        dequantize_scale<float, GROUP_SIZE>(row_scales[0]);
                    results[row] += qdot<
                        float,
                        values_per_thread,
                        BITS>(row_weight, input_values, scale);
                }

                selected_weight +=
                    block_size * bytes_per_pack / pack_factor;
                selected_scales += block_size / GROUP_SIZE;
                input += block_size;
            }

            STEEL_PRAGMA_UNROLL
            for (int row = 0; row < results_per_simdgroup; ++row) {
                results[row] = simd_sum(results[row]);
                if (simd_lane == 0) {
                    const uint projection_offset =
                        gate_projection ? 0 : outputs_per_threadgroup;
                    projection_values[
                        projection_offset
                            + local_simd_group * results_per_simdgroup
                            + row
                    ] = DataT(results[row]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (thread_index < outputs_per_threadgroup) {
                const DataT gate_value = projection_values[thread_index];
                const DataT up_value =
                    projection_values[
                        outputs_per_threadgroup + thread_index
                    ];
                const DataT sigmoid_base =
                    DataT(1)
                        / (
                            DataT(1)
                                + metal::exp(metal::abs(gate_value))
                        );
                const DataT sigmoid_value =
                    gate_value < DataT(0)
                        ? sigmoid_base
                        : DataT(1) - sigmoid_base;
                output[
                    size_t(route) * OUTPUT_DIMENSIONS
                        + size_t(tile.y) * outputs_per_threadgroup
                        + thread_index
                ] = DataT(DataT(gate_value * sigmoid_value) * up_value);
            }
        """,
        header: "// MLX_INCLUDE_FP_QUANTIZED_HEADERS\n",
        ensureRowContiguous: true
    )

    private static let fusedGatherAffine8SwiGLUKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_gather_affine8_swiglu",
        inputNames: [
            "x",
            "gate_weight",
            "gate_scales",
            "gate_biases",
            "up_weight",
            "up_scales",
            "up_biases",
            "expert_indices",
        ],
        outputNames: ["output"],
        source: """
            constexpr int packs_per_thread = BITS <= 2 ? 1 : 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int outputs_per_threadgroup = 8;
            constexpr int pack_factor = get_pack_factor<BITS, 32>();
            constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
            constexpr int values_per_thread = pack_factor * packs_per_thread;
            constexpr int block_size = values_per_thread * SIMD_SIZE;
            constexpr int scale_step_per_thread =
                GROUP_SIZE / values_per_thread;
            constexpr int packed_input =
                INPUT_DIMENSIONS * bytes_per_pack / pack_factor;
            constexpr int scale_input = INPUT_DIMENSIONS / GROUP_SIZE;
            constexpr size_t weight_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * packed_input;
            constexpr size_t scale_expert_stride =
                size_t(OUTPUT_DIMENSIONS) * scale_input;

            threadgroup DataT projection_values[
                2 * outputs_per_threadgroup
            ];

            const uint3 tile = threadgroup_position_in_grid;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint simd_lane = thread_index_in_simdgroup;
            const uint thread_index = thread_index_in_threadgroup;
            const bool gate_projection = simd_group < 2;
            const uint local_simd_group = simd_group % 2;
            const uint route = tile.z;
            if (route >= ROUTE_COUNT) {
                return;
            }

            const uint expert = uint(expert_indices[route]);
            const int output_row =
                int(tile.y) * outputs_per_threadgroup
                    + int(local_simd_group) * results_per_simdgroup;
            const device uint8_t* selected_weight =
                reinterpret_cast<const device uint8_t*>(
                    gate_projection ? gate_weight : up_weight);
            const device DataT* selected_scales =
                gate_projection ? gate_scales : up_scales;
            const device DataT* selected_biases =
                gate_projection ? gate_biases : up_biases;
            selected_weight +=
                size_t(expert) * weight_expert_stride
                    + size_t(output_row) * packed_input
                    + simd_lane * packs_per_thread * bytes_per_pack;
            selected_scales +=
                size_t(expert) * scale_expert_stride
                    + size_t(output_row) * scale_input
                    + simd_lane / scale_step_per_thread;
            selected_biases +=
                size_t(expert) * scale_expert_stride
                    + size_t(output_row) * scale_input
                    + simd_lane / scale_step_per_thread;
            const device DataT* input =
                x
                    + size_t(route / TOP_K) * INPUT_DIMENSIONS
                    + simd_lane * values_per_thread;

            float input_values[values_per_thread];
            float results[results_per_simdgroup] = {0};
            for (int k = 0; k < INPUT_DIMENSIONS; k += block_size) {
                const float input_sum = load_vector<
                    DataT,
                    float,
                    values_per_thread,
                    BITS>(input, input_values);

                STEEL_PRAGMA_UNROLL
                for (int row = 0; row < results_per_simdgroup; ++row) {
                    const device uint8_t* row_weight =
                        selected_weight + size_t(row) * packed_input;
                    const device DataT* row_scales =
                        selected_scales + size_t(row) * scale_input;
                    const device DataT* row_biases =
                        selected_biases + size_t(row) * scale_input;
                    results[row] += qdot<
                        float,
                        values_per_thread,
                        BITS>(
                            row_weight,
                            input_values,
                            float(row_scales[0]),
                            float(row_biases[0]),
                            input_sum);
                }

                selected_weight +=
                    block_size * bytes_per_pack / pack_factor;
                selected_scales += block_size / GROUP_SIZE;
                selected_biases += block_size / GROUP_SIZE;
                input += block_size;
            }

            STEEL_PRAGMA_UNROLL
            for (int row = 0; row < results_per_simdgroup; ++row) {
                results[row] = simd_sum(results[row]);
                if (simd_lane == 0) {
                    const uint projection_offset =
                        gate_projection ? 0 : outputs_per_threadgroup;
                    projection_values[
                        projection_offset
                            + local_simd_group * results_per_simdgroup
                            + row
                    ] = DataT(results[row]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (thread_index < outputs_per_threadgroup) {
                const DataT gate_value = projection_values[thread_index];
                const DataT up_value =
                    projection_values[
                        outputs_per_threadgroup + thread_index
                    ];
                const DataT sigmoid_base =
                    DataT(1)
                        / (
                            DataT(1)
                                + metal::exp(metal::abs(gate_value))
                        );
                const DataT sigmoid_value =
                    gate_value < DataT(0)
                        ? sigmoid_base
                        : DataT(1) - sigmoid_base;
                output[
                    size_t(route) * OUTPUT_DIMENSIONS
                        + size_t(tile.y) * outputs_per_threadgroup
                        + thread_index
                ] = DataT(DataT(gate_value * sigmoid_value) * up_value);
            }
        """,
        header: "// MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS\n",
        ensureRowContiguous: true
    )
    #endif
}
