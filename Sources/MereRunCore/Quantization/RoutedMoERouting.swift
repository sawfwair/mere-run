import Foundation
import MLX
import MLXFast

/// Guarded gate/up projection primitives for routed quantized MoE layers.
///
/// Compatible models keep their native down projection. Laguna's measured
/// prefill layout uses an expert-aligned matrix kernel; small decode layouts
/// use gather-GEMV kernels. Each helper returns `nil` outside its measured
/// quantization and alignment boundary so callers fall back to MLX.
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

    private static var supportsExpertAlignedNVFP4Metal: Bool {
        #if os(macOS)
        let architecture = GPU.deviceInfo().architecture
        return Device.defaultDevice().deviceType == .gpu
            && ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26
            && (architecture == "applegpu_g16s" || architecture == "applegpu_g17s")
        #else
        return false
        #endif
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

    /// Inverts a one-dimensional permutation without paying for a second sort.
    static func invertPermutation(_ order: MLXArray) -> MLXArray? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        guard Device.defaultDevice().deviceType == .gpu,
              order.ndim == 1,
              order.size > 0,
              order.dtype == .uint32 || order.dtype == .int32 else {
            return nil
        }
        return invertPermutationKernel(
            [order],
            template: [("IndexT", order.dtype), ("COUNT", order.size)],
            grid: (order.size, 1, 1),
            threadGroup: (min(order.size, 256), 1, 1),
            outputShapes: [order.shape],
            outputDTypes: [order.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// Replaces the ranked Laguna prefill route gathers with fixed-shape
    /// byte copies and constructs both metadata permutations directly.
    /// Every other model, shape, dtype, and route count falls back to MLX.
    static func stageRankedLagunaPrefillRoute(
        _ input: MLXArray,
        flatIndices: MLXArray,
        order: MLXArray,
        topK: Int
    ) -> (
        sortedInput: MLXArray,
        sortedIndices: MLXArray,
        inverseOrder: MLXArray
    )? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let tokenCount = 512
        let hiddenSize = 2_048
        let rankedTopK = 8
        let routeCount = tokenCount * rankedTopK
        guard Device.defaultDevice().deviceType == .gpu,
              input.dtype == .bfloat16,
              input.shape == [tokenCount, hiddenSize],
              topK == rankedTopK,
              flatIndices.dtype == .uint32,
              flatIndices.shape == [routeCount],
              order.dtype == .uint32,
              order.shape == [routeCount] else {
            return nil
        }

        let sortedInput = rankedPrefillRowCopyKernel(
            [input, order],
            grid: (hiddenSize / 8, routeCount, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[routeCount, 1, hiddenSize]],
            outputDTypes: [.bfloat16]
        )[0]
        let metadata = rankedPrefillRouteMetadataKernel(
            [flatIndices, order],
            grid: (routeCount, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[routeCount], [routeCount]],
            outputDTypes: [.uint32, .uint32]
        )
        return (sortedInput, metadata[0], metadata[1])
        #else
        return nil
        #endif
    }

    /// Runs the two sorted NVFP4 expert projections in one classic matrix
    /// dispatch and applies SwiGLU before the intermediate leaves the kernel.
    static func fusedSortedNVFP4SwiGLU(
        _ sortedInput: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        sortedExpertIndices: MLXArray,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        #if os(macOS)
        guard supportsExpertAlignedNVFP4Metal,
              sortedInput.dtype == .bfloat16,
              gateWeight.dtype == .uint32,
              upWeight.dtype == .uint32,
              gateScales.dtype == .uint8,
              upScales.dtype == .uint8,
              sortedExpertIndices.dtype == .int32
                || sortedExpertIndices.dtype == .uint32,
              sortedInput.ndim == 3,
              sortedInput.dim(1) == 1,
              gateWeight.shape == upWeight.shape,
              gateScales.shape == upScales.shape,
              gateWeight.dim(0) == gateScales.dim(0),
              gateWeight.dim(0) > 0,
              gateWeight.dim(0) <= 256,
              gateWeight.dim(1) == gateScales.dim(1),
              sortedInput.dim(0) == sortedExpertIndices.size,
              groupSize == 16,
              bits == 4 else {
            return nil
        }

        let routeCount = sortedInput.dim(0)
        let outputDimensions = gateWeight.dim(1)
        let inputDimensions = sortedInput.dim(2)
        guard routeCount >= 64,
              outputDimensions.isMultiple(of: 64),
              inputDimensions.isMultiple(of: 64),
              gateWeight.dim(2) == inputDimensions / 8,
              gateScales.dim(2) == inputDimensions / groupSize else {
            return nil
        }

        let outputTiles = outputDimensions / 32
        let maximumRouteTiles =
            (routeCount + 15) / 16
                + min(gateWeight.dim(0), routeCount)
                - 1
        let schedule = sortedExpertTileScheduleKernel(
            [sortedExpertIndices],
            template: [
                ("IndexT", sortedExpertIndices.dtype),
                ("ROUTE_COUNT", routeCount),
                ("EXPERT_COUNT", gateWeight.dim(0)),
                ("TILE_COUNT", maximumRouteTiles),
            ],
            grid: (gateWeight.dim(0), 1, 1),
            threadGroup: (gateWeight.dim(0), 1, 1),
            outputShapes: [
                [maximumRouteTiles],
                [maximumRouteTiles],
                [maximumRouteTiles],
            ],
            outputDTypes: [
                sortedExpertIndices.dtype,
                sortedExpertIndices.dtype,
                sortedExpertIndices.dtype,
            ]
        )
        return fusedSortedNVFP4SwiGLUKernel(
            [
                sortedInput,
                gateWeight,
                gateScales,
                upWeight,
                upScales,
                schedule[0],
                schedule[1],
                schedule[2],
            ],
            template: [
                ("DataT", sortedInput.dtype),
                ("ROUTE_COUNT", routeCount),
                ("OUTPUT_DIMENSIONS", outputDimensions),
                ("INPUT_DIMENSIONS", inputDimensions),
            ],
            grid: (outputTiles * 32, maximumRouteTiles * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[routeCount, 1, outputDimensions]],
            outputDTypes: [sortedInput.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// Runs one sorted NVFP4 expert projection with the same expert-aligned
    /// tile schedule used by the fused gate/up prefill path. Laguna uses this
    /// for the routed down projection after SwiGLU, avoiding the generic
    /// gather-QMM run loop while preserving each output row's MMA order.
    static func sortedNVFP4Projection(
        _ sortedInput: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        sortedExpertIndices: MLXArray,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        #if os(macOS)
        guard supportsExpertAlignedNVFP4Metal,
              sortedInput.dtype == .bfloat16,
              weight.dtype == .uint32,
              scales.dtype == .uint8,
              sortedExpertIndices.dtype == .int32
                || sortedExpertIndices.dtype == .uint32,
              sortedInput.ndim == 3,
              sortedInput.dim(1) == 1,
              weight.dim(0) == scales.dim(0),
              weight.dim(0) > 0,
              weight.dim(0) <= 256,
              weight.dim(1) == scales.dim(1),
              sortedInput.dim(0) == sortedExpertIndices.size,
              groupSize == 16,
              bits == 4 else {
            return nil
        }

        let routeCount = sortedInput.dim(0)
        let outputDimensions = weight.dim(1)
        let inputDimensions = sortedInput.dim(2)
        guard routeCount >= 64,
              outputDimensions.isMultiple(of: 64),
              inputDimensions.isMultiple(of: 64),
              weight.dim(2) == inputDimensions / 8,
              scales.dim(2) == inputDimensions / groupSize else {
            return nil
        }

        let maximumRouteTiles =
            (routeCount + 15) / 16
                + min(weight.dim(0), routeCount)
                - 1
        let schedule = sortedExpertTileScheduleKernel(
            [sortedExpertIndices],
            template: [
                ("IndexT", sortedExpertIndices.dtype),
                ("ROUTE_COUNT", routeCount),
                ("EXPERT_COUNT", weight.dim(0)),
                ("TILE_COUNT", maximumRouteTiles),
            ],
            grid: (weight.dim(0), 1, 1),
            threadGroup: (weight.dim(0), 1, 1),
            outputShapes: [
                [maximumRouteTiles],
                [maximumRouteTiles],
                [maximumRouteTiles],
            ],
            outputDTypes: [
                sortedExpertIndices.dtype,
                sortedExpertIndices.dtype,
                sortedExpertIndices.dtype,
            ]
        )
        return sortedNVFP4ProjectionKernel(
            [
                sortedInput,
                weight,
                scales,
                schedule[0],
                schedule[1],
                schedule[2],
                Int32(routeCount),
            ],
            template: [
                ("DataT", sortedInput.dtype),
                ("OUTPUT_DIMENSIONS", outputDimensions),
                ("INPUT_DIMENSIONS", inputDimensions),
            ],
            grid: ((outputDimensions / 32) * 32, maximumRouteTiles * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[routeCount, 1, outputDimensions]],
            outputDTypes: [sortedInput.dtype]
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

    private static let invertPermutationKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_invert_permutation",
        inputNames: ["order"],
        outputNames: ["inverse"],
        source: """
            const uint index = thread_position_in_grid.x;
            if (index < COUNT) {
                inverse[uint(order[index])] = IndexT(index);
            }
        """,
        ensureRowContiguous: true
    )

    private static let rankedPrefillRowCopyKernel = MLXFast.metalKernel(
        name: "mere_laguna_ranked_prefill_row_copy_bf16_512x8_v1",
        inputNames: ["source", "order"],
        outputNames: ["sorted"],
        source: """
            constexpr uint hidden_vectors = 2048 / 8;
            constexpr uint experts_per_token = 8;

            uint vector_index = thread_position_in_grid.x;
            uint sorted_row = thread_position_in_grid.y;
            uint original_row = order[sorted_row];
            uint source_row = original_row / experts_per_token;
            const device uint4* source_vectors =
                reinterpret_cast<const device uint4*>(source);
            device uint4* sorted_vectors =
                reinterpret_cast<device uint4*>(sorted);
            sorted_vectors[sorted_row * hidden_vectors + vector_index] =
                source_vectors[source_row * hidden_vectors + vector_index];
        """,
        ensureRowContiguous: true
    )

    private static let rankedPrefillRouteMetadataKernel = MLXFast.metalKernel(
        name: "mere_laguna_ranked_prefill_route_metadata_u32_4096_v1",
        inputNames: ["flat_indices", "order"],
        outputNames: ["sorted_indices", "inverse_order"],
        source: """
            uint sorted_row = thread_position_in_grid.x;
            uint original_row = order[sorted_row];
            sorted_indices[sorted_row] = flat_indices[original_row];
            inverse_order[original_row] = sorted_row;
        """,
        ensureRowContiguous: true
    )

    private static let sortedExpertTileScheduleKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_sorted_expert_tile_schedule",
        inputNames: ["expert_indices"],
        outputNames: ["tile_starts", "tile_rows", "tile_experts"],
        source: """
            threadgroup uint expert_tile_prefix[256];

            const uint expert = thread_index_in_threadgroup;
            uint lower = 0;
            uint upper = ROUTE_COUNT;
            while (lower < upper) {
                const uint middle = lower + (upper - lower) / 2;
                if (uint(expert_indices[middle]) < expert) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            const uint route_start = lower;

            upper = ROUTE_COUNT;
            while (lower < upper) {
                const uint middle = lower + (upper - lower) / 2;
                if (uint(expert_indices[middle]) <= expert) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            const uint route_count = lower - route_start;
            const uint expert_tiles = (route_count + 15) / 16;
            expert_tile_prefix[expert] = expert_tiles;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint offset = 1;
                 offset < EXPERT_COUNT;
                 offset *= 2) {
                const uint prior =
                    expert >= offset
                        ? expert_tile_prefix[expert - offset]
                        : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                expert_tile_prefix[expert] += prior;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint tile_start =
                expert == 0 ? 0 : expert_tile_prefix[expert - 1];
            for (uint tile = 0; tile < expert_tiles; ++tile) {
                const uint destination = tile_start + tile;
                const uint consumed_rows = tile * 16;
                tile_starts[destination] =
                    IndexT(route_start + consumed_rows);
                tile_rows[destination] =
                    IndexT(min(16u, route_count - consumed_rows));
                tile_experts[destination] = IndexT(expert);
            }

            const uint total_tiles =
                expert_tile_prefix[EXPERT_COUNT - 1];
            for (uint tile = total_tiles + expert;
                 tile < TILE_COUNT;
                 tile += EXPERT_COUNT) {
                tile_starts[tile] = IndexT(ROUTE_COUNT);
                tile_rows[tile] = IndexT(0);
                tile_experts[tile] = IndexT(0);
            }
        """,
        ensureRowContiguous: true
    )

    private static let fusedSortedNVFP4SwiGLUKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_sorted_nvfp4_swiglu",
        inputNames: [
            "x",
            "gate_weight",
            "gate_scales",
            "up_weight",
            "up_scales",
            "tile_starts",
            "scheduled_tile_rows",
            "tile_experts",
        ],
        outputNames: ["output"],
        source: """
            constexpr int BM = 16;
            constexpr int BN = 32;
            constexpr int BK = 32;
            constexpr int WM = 1;
            constexpr int WN = 2;
            constexpr int GROUP_SIZE = 16;
            constexpr int BITS = 4;
            constexpr int PACK_FACTOR = get_pack_factor<8, BITS>();
            constexpr int BYTES_PER_PACK = get_bytes_per_pack();
            constexpr int BK_PADDED = BK + 16 / sizeof(DataT);
            constexpr int K_WEIGHT =
                INPUT_DIMENSIONS * BYTES_PER_PACK / PACK_FACTOR;
            constexpr int K_SCALE = INPUT_DIMENSIONS / GROUP_SIZE;
            constexpr int K_ITERATIONS = INPUT_DIMENSIONS / BK;
            constexpr size_t WEIGHT_EXPERT_STRIDE =
                size_t(OUTPUT_DIMENSIONS) * K_WEIGHT;
            constexpr size_t SCALE_EXPERT_STRIDE =
                size_t(OUTPUT_DIMENSIONS) * K_SCALE;

            using mma_t = mlx::steel::BlockMMA<
                DataT,
                DataT,
                BM,
                BN,
                BK,
                WM,
                WN,
                false,
                true,
                BK_PADDED,
                BK_PADDED>;
            using input_loader_t = mlx::steel::BlockLoader<
                DataT,
                BM,
                BK,
                BK_PADDED,
                1,
                WM * WN * SIMD_SIZE>;
            using weight_loader_t = QuantizedBlockLoader<
                DataT,
                BN,
                BK,
                BK_PADDED,
                true,
                WM * WN * SIMD_SIZE,
                GROUP_SIZE,
                BITS>;

            threadgroup DataT input_tile[BM * BK_PADDED];
            threadgroup DataT gate_weight_tile[BN * BK_PADDED];
            threadgroup DataT up_weight_tile[BN * BK_PADDED];

            const uint3 tile = threadgroup_position_in_grid;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint simd_lane = thread_index_in_simdgroup;
            const int output_row = int(tile_starts[tile.y]);
            if (output_row >= ROUTE_COUNT) {
                return;
            }
            const int output_column = int(tile.x) * BN;
            const short tile_rows =
                short(scheduled_tile_rows[tile.y]);
            const uint expert = uint(tile_experts[tile.y]);

            const device DataT* tile_input =
                x + size_t(output_row) * INPUT_DIMENSIONS;
            device DataT* tile_output =
                output
                    + size_t(output_row) * OUTPUT_DIMENSIONS
                    + output_column;
            const device uint8_t* gate_weight_bytes =
                reinterpret_cast<const device uint8_t*>(gate_weight)
                    + size_t(output_column) * K_WEIGHT;
            const device uint8_t* up_weight_bytes =
                reinterpret_cast<const device uint8_t*>(up_weight)
                    + size_t(output_column) * K_WEIGHT;
            const device uint8_t* gate_scale_bytes =
                gate_scales + size_t(output_column) * K_SCALE;
            const device uint8_t* up_scale_bytes =
                up_scales + size_t(output_column) * K_SCALE;

            thread mma_t gate_mma(simd_group, simd_lane);
            thread mma_t up_mma(simd_group, simd_lane);
            thread input_loader_t input_loader(
                tile_input,
                INPUT_DIMENSIONS,
                input_tile,
                simd_group,
                simd_lane);
            thread weight_loader_t gate_loader(
                gate_weight_bytes
                    + size_t(expert) * WEIGHT_EXPERT_STRIDE,
                gate_scale_bytes
                    + size_t(expert) * SCALE_EXPERT_STRIDE,
                INPUT_DIMENSIONS,
                gate_weight_tile,
                simd_group,
                simd_lane);
            thread weight_loader_t up_loader(
                up_weight_bytes
                    + size_t(expert) * WEIGHT_EXPERT_STRIDE,
                up_scale_bytes
                    + size_t(expert) * SCALE_EXPERT_STRIDE,
                INPUT_DIMENSIONS,
                up_weight_tile,
                simd_group,
                simd_lane);

            for (int k = 0; k < K_ITERATIONS; ++k) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tile_rows == BM) {
                    input_loader.load_unsafe();
                } else {
                    input_loader.load_safe(short2(BK, tile_rows));
                }
                gate_loader.load_unsafe();
                up_loader.load_unsafe();
                threadgroup_barrier(mem_flags::mem_threadgroup);
                gate_mma.mma(input_tile, gate_weight_tile);
                up_mma.mma(input_tile, up_weight_tile);

                input_loader.next();
                gate_loader.next();
                up_loader.next();
            }

            STEEL_PRAGMA_UNROLL
            for (short element = 0;
                 element < decltype(gate_mma.Ctile)::kElemsPerTile;
                 ++element) {
                const DataT gate_value =
                    DataT(gate_mma.Ctile.elems()[element]);
                const DataT up_value =
                    DataT(up_mma.Ctile.elems()[element]);
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
                gate_mma.Ctile.elems()[element] = float(
                    DataT(
                        DataT(gate_value * sigmoid_value)
                            * up_value
                    )
                );
            }

            if (tile_rows == BM) {
                gate_mma.store_result(
                    tile_output,
                    OUTPUT_DIMENSIONS);
            } else {
                gate_mma.store_result_safe(
                    tile_output,
                    OUTPUT_DIMENSIONS,
                    short2(BN, tile_rows));
            }
        """,
        header: "// MLX_INCLUDE_FP_QUANTIZED_HEADERS\n",
        ensureRowContiguous: true
    )

    private static let sortedNVFP4ProjectionKernel = MLXFast.metalKernel(
        name: "mere_routed_moe_sorted_nvfp4_projection",
        inputNames: [
            "x",
            "weight",
            "scales",
            "tile_starts",
            "scheduled_tile_rows",
            "tile_experts",
            "route_count",
        ],
        outputNames: ["output"],
        source: """
            constexpr int BM = 16;
            constexpr int BN = 32;
            constexpr int BK = 32;
            constexpr int WM = 1;
            constexpr int WN = 2;
            constexpr int GROUP_SIZE = 16;
            constexpr int BITS = 4;
            constexpr int PACK_FACTOR = get_pack_factor<8, BITS>();
            constexpr int BYTES_PER_PACK = get_bytes_per_pack();
            constexpr int BK_PADDED = BK + 16 / sizeof(DataT);
            constexpr int K_WEIGHT =
                INPUT_DIMENSIONS * BYTES_PER_PACK / PACK_FACTOR;
            constexpr int K_SCALE = INPUT_DIMENSIONS / GROUP_SIZE;
            constexpr int K_ITERATIONS = INPUT_DIMENSIONS / BK;
            constexpr size_t WEIGHT_EXPERT_STRIDE =
                size_t(OUTPUT_DIMENSIONS) * K_WEIGHT;
            constexpr size_t SCALE_EXPERT_STRIDE =
                size_t(OUTPUT_DIMENSIONS) * K_SCALE;

            using mma_t = mlx::steel::BlockMMA<
                DataT,
                DataT,
                BM,
                BN,
                BK,
                WM,
                WN,
                false,
                true,
                BK_PADDED,
                BK_PADDED>;
            using input_loader_t = mlx::steel::BlockLoader<
                DataT,
                BM,
                BK,
                BK_PADDED,
                1,
                WM * WN * SIMD_SIZE>;
            using weight_loader_t = QuantizedBlockLoader<
                DataT,
                BN,
                BK,
                BK_PADDED,
                true,
                WM * WN * SIMD_SIZE,
                GROUP_SIZE,
                BITS>;

            threadgroup DataT input_tile[BM * BK_PADDED];
            threadgroup DataT weight_tile[BN * BK_PADDED];

            const uint3 tile = threadgroup_position_in_grid;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint simd_lane = thread_index_in_simdgroup;
            const int output_row = int(tile_starts[tile.y]);
            if (output_row >= int(route_count)) {
                return;
            }
            const int output_column = int(tile.x) * BN;
            const short tile_rows = short(scheduled_tile_rows[tile.y]);
            const uint expert = uint(tile_experts[tile.y]);

            const device DataT* tile_input =
                x + size_t(output_row) * INPUT_DIMENSIONS;
            device DataT* tile_output =
                output
                    + size_t(output_row) * OUTPUT_DIMENSIONS
                    + output_column;
            const device uint8_t* weight_bytes =
                reinterpret_cast<const device uint8_t*>(weight)
                    + size_t(output_column) * K_WEIGHT
                    + size_t(expert) * WEIGHT_EXPERT_STRIDE;
            const device uint8_t* scale_bytes =
                scales
                    + size_t(output_column) * K_SCALE
                    + size_t(expert) * SCALE_EXPERT_STRIDE;

            thread mma_t mma(simd_group, simd_lane);
            thread input_loader_t input_loader(
                tile_input,
                INPUT_DIMENSIONS,
                input_tile,
                simd_group,
                simd_lane);
            thread weight_loader_t weight_loader(
                weight_bytes,
                scale_bytes,
                INPUT_DIMENSIONS,
                weight_tile,
                simd_group,
                simd_lane);

            for (int k = 0; k < K_ITERATIONS; ++k) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tile_rows == BM) {
                    input_loader.load_unsafe();
                } else {
                    input_loader.load_safe(short2(BK, tile_rows));
                }
                weight_loader.load_unsafe();
                threadgroup_barrier(mem_flags::mem_threadgroup);
                mma.mma(input_tile, weight_tile);

                input_loader.next();
                weight_loader.next();
            }

            if (tile_rows == BM) {
                mma.store_result(tile_output, OUTPUT_DIMENSIONS);
            } else {
                mma.store_result_safe(
                    tile_output,
                    OUTPUT_DIMENSIONS,
                    short2(BN, tile_rows));
            }
        """,
        header: "// MLX_INCLUDE_FP_QUANTIZED_HEADERS\n",
        ensureRowContiguous: true
    )

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
