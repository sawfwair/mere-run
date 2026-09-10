import Foundation
import MLX
import MLXFast

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
extension RoutedMoERouting {
    /// Drop-in replacement for MLX's `QuantizedBlockLoader` at the exact tiled
    /// geometry shared by Laguna's routed gate, up, and down prefill kernels.
    /// Two adjacent SIMD lanes own the two group-16 halves of one 32-weight
    /// span. Once a loaded scale plane has passed
    /// `lagunaNVFP4AdjacentScalePairsCertified`, the even lane loads their
    /// shared uint8 scale and broadcasts it to the odd lane. The first logical
    /// pair of every output tile is conservatively read by both lanes, which
    /// preserves the only pair the checkpoint quantizer may leave unequal.
    ///
    /// No values are recomputed or converted differently: `fp4nv_scale_x16384`,
    /// code loads, decode order, threadgroup stores, and MMA order are copied
    /// directly from the stock loader. The false template arm below continues
    /// to instantiate MLX's original loader as the exact kill-switch path.
    static let pairwiseNVFP4BlockLoaderHeader = """
        // MLX_INCLUDE_FP_QUANTIZED_HEADERS

        template <
            typename T,
            short BROWS,
            short BCOLS,
            short dst_ld,
            short reduction_dim,
            short tgp_size,
            short group_size,
            short bits>
        struct MereLagunaPairwiseNVFP4BlockLoader {
            MLX_MTL_CONST short pack_factor = get_pack_factor<8, bits>();
            MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();
            MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
            MLX_MTL_CONST short n_reads =
                (BCOLS_PACKED * BROWS) / tgp_size;

            static_assert(BROWS == 32, "Laguna pairwise loader requires BROWS=32");
            static_assert(BCOLS == 32, "Laguna pairwise loader requires BCOLS=32");
            static_assert(reduction_dim == 1, "Laguna pairwise loader requires reduction_dim=1");
            static_assert(tgp_size == 64, "Laguna pairwise loader requires 64 threads");
            static_assert(group_size == 16, "Laguna pairwise loader requires group-16");
            static_assert(bits == 4, "Laguna pairwise loader requires NVFP4");
            static_assert(pack_factor == 2, "Laguna pairwise loader requires pack factor 2");
            static_assert(bytes_per_pack == 1, "Laguna pairwise loader requires byte packs");
            static_assert(n_reads == 8, "Laguna pairwise loader requires eight reads per lane");

            const int src_ld;
            const int tile_stride;
            const short thread_idx;
            const short bi;
            const short bj;

            threadgroup T* dst;
            const device uint8_t* src;
            const device uint8_t* scales;
            bool first_k;

            MereLagunaPairwiseNVFP4BlockLoader(
                const device uint8_t* src_,
                const device uint8_t* scales_,
                const int src_ld_,
                threadgroup T* dst_,
                ushort simd_group_id [[simdgroup_index_in_threadgroup]],
                ushort simd_lane_id [[thread_index_in_simdgroup]])
                : src_ld(src_ld_),
                  tile_stride(BCOLS_PACKED * bytes_per_pack),
                  thread_idx(simd_group_id * 32 + simd_lane_id),
                  bi(n_reads * thread_idx / BCOLS_PACKED),
                  bj((n_reads * thread_idx) % BCOLS_PACKED),
                  dst(dst_ + bi * dst_ld + bj * pack_factor),
                  src(src_ + bi * src_ld * bytes_per_pack / pack_factor
                      + bj * bytes_per_pack),
                  scales(scales_ + bi * src_ld / group_size
                      + (bj * pack_factor) / group_size),
                  first_k(true) {}

            void load_unsafe() const {
                const bool even_group = (thread_idx & 1) == 0;
                float scale = 0.0f;
                if (even_group) {
                    scale = fp4nv_scale_x16384(*scales);
                }
                const float paired_scale =
                    simd_shuffle_xor(scale, ushort(1));
                if (!even_group) {
                    scale = paired_scale;
                }
                // Preserve the odd byte of the first logical pair. Doing this
                // for every output tile is conservative and makes the loader
                // exact without specializing on an expert/tile coordinate.
                if (first_k && bi == 0 && !even_group) {
                    scale = fp4nv_scale_x16384(*scales);
                }

                // The normal odd lane reuses the same converted float bits as
                // the even lane instead of converting their certified-alias
                // byte again. This is the production form of the officially
                // promoted M5 prefill scale-conversion hoist.
                for (int i = 0; i < n_reads / 4; ++i) {
                    T values[8];
                    fp4nv_decode8<T>(
                        fp4nv_pack4(src + i * 4), scale, values);
                    for (int j = 0; j < 8; ++j) {
                        dst[i * 8 + j] = values[j];
                    }
                }
            }

            void next() {
                src += tile_stride;
                scales += BCOLS / group_size;
                first_k = false;
            }
        };
        """

    static let fusedSortedNVFP4SwiGLUKernel = MLXFast.metalKernel(
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
            using weight_loader_t = metal::conditional_t<
                PAIRWISE_SCALE_REUSE,
                MereLagunaPairwiseNVFP4BlockLoader<
                    DataT,
                    BN,
                    BK,
                    BK_PADDED,
                    true,
                    WM * WN * SIMD_SIZE,
                    GROUP_SIZE,
                    BITS>,
                QuantizedBlockLoader<
                    DataT,
                    BN,
                    BK,
                    BK_PADDED,
                    true,
                    WM * WN * SIMD_SIZE,
                    GROUP_SIZE,
                    BITS>>;

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
        header: pairwiseNVFP4BlockLoaderHeader,
        ensureRowContiguous: true
    )

    static let sortedNVFP4ProjectionKernel = MLXFast.metalKernel(
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
            using weight_loader_t = metal::conditional_t<
                PAIRWISE_SCALE_REUSE,
                MereLagunaPairwiseNVFP4BlockLoader<
                    DataT,
                    BN,
                    BK,
                    BK_PADDED,
                    true,
                    WM * WN * SIMD_SIZE,
                    GROUP_SIZE,
                    BITS>,
                QuantizedBlockLoader<
                    DataT,
                    BN,
                    BK,
                    BK_PADDED,
                    true,
                    WM * WN * SIMD_SIZE,
                    GROUP_SIZE,
                    BITS>>;

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
        header: pairwiseNVFP4BlockLoaderHeader,
        ensureRowContiguous: true
    )

}

#endif
