import Foundation
import MLX
import MLXFast

/// Experimental BF16 projection primitive for the H3 kernel lab.
///
/// The MPP shader structure and H3-specific tile choices are adapted from
/// WeeTodd-Nodes commit e5b0e014db1abe4c86fedc195d12dfcd18562042 and
/// translated to Swift/MLX. This primitive remains outside production model
/// dispatch until the repository's exactness and clean-host benchmark gates
/// qualify it on supported Apple GPUs.
enum MiniMaxH3MPPProjection {
    struct Tile: Equatable, Sendable {
        let rows: Int
        let columns: Int
        let simdgroups: Int

        init(rows: Int, columns: Int, simdgroups: Int) {
            precondition(rows > 0)
            precondition(columns > 0)
            precondition(simdgroups > 0)
            self.rows = rows
            self.columns = columns
            self.simdgroups = simdgroups
        }
    }

    static let standardTile = Tile(rows: 32, columns: 64, simdgroups: 2)
    static let feedForwardOutputTile = Tile(rows: 64, columns: 128, simdgroups: 8)

    static func tile(inputDimension: Int, outputDimension: Int) -> Tile {
        if inputDimension == 14_336, outputDimension == 5_376 {
            return feedForwardOutputTile
        }
        return standardTile
    }

    static var isAvailable: Bool {
        #if os(macOS)
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return Device.defaultDevice().deviceType == .gpu
            && version.majorVersion >= 26
        #else
        return false
        #endif
    }

    /// Computes `source @ weight.T` for contiguous BF16 H3 projection tensors.
    ///
    /// Returning `nil` is the complete capability and shape fallback contract;
    /// callers retain standard MLX matmul as the source of truth.
    static func project(
        source: MLXArray,
        weight: MLXArray,
        tile requestedTile: Tile? = nil
    ) -> MLXArray? {
        #if os(macOS)
        guard isAvailable,
              source.dtype == .bfloat16,
              weight.dtype == .bfloat16,
              source.ndim >= 2,
              weight.ndim == 2,
              source.dim(-1) == weight.dim(1) else {
            return nil
        }

        let inputDimension = source.dim(-1)
        let outputDimension = weight.dim(0)
        let rows = source.size / inputDimension
        guard rows > 0 else { return nil }

        let tile = requestedTile ?? tile(
            inputDimension: inputDimension,
            outputDimension: outputDimension
        )
        let threadCount = 32 * tile.simdgroups
        let outputShape = Array(source.shape.dropLast()) + [outputDimension]
        return kernel(
            [source, weight],
            template: [
                ("ROWS", rows),
                ("OUTPUT_DIM", outputDimension),
                ("INPUT_DIM", inputDimension),
                ("TILE_M", tile.rows),
                ("TILE_N", tile.columns),
                ("SIMDGROUPS", tile.simdgroups),
            ],
            grid: (
                divideRoundUp(outputDimension, by: tile.columns) * threadCount,
                divideRoundUp(rows, by: tile.rows),
                1
            ),
            threadGroup: (threadCount, 1, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
        #else
        return nil
        #endif
    }

    private static func divideRoundUp(_ value: Int, by divisor: Int) -> Int {
        (value + divisor - 1) / divisor
    }

    #if os(macOS)
    private static let kernel = MLXFast.metalKernel(
        name: "mere_h3_mpp_bf16_nt_matmul_v1",
        inputNames: ["source", "weight"],
        outputNames: ["output"],
        source: """
            auto matrix_a = tensor(
                (device bfloat*)source,
                dextents<int, 2>{INPUT_DIM, ROWS},
                array<int, 2>{1, INPUT_DIM});
            auto matrix_b = tensor(
                (device bfloat*)weight,
                dextents<int, 2>{INPUT_DIM, OUTPUT_DIM},
                array<int, 2>{1, INPUT_DIM});
            auto matrix_c = tensor(
                (device bfloat*)output,
                dextents<int, 2>{OUTPUT_DIM, ROWS},
                array<int, 2>{1, OUTPUT_DIM});
            constexpr auto descriptor = matmul2d_descriptor(
                TILE_M,
                TILE_N,
                static_cast<int>(dynamic_extent),
                false,
                true,
                false);
            matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> operation;
            auto tile_a = matrix_a.slice(
                0,
                threadgroup_position_in_grid.y * TILE_M);
            auto tile_b = matrix_b.slice(
                0,
                threadgroup_position_in_grid.x * TILE_N);
            auto tile_c = matrix_c.slice(
                threadgroup_position_in_grid.x * TILE_N,
                threadgroup_position_in_grid.y * TILE_M);
            auto result = operation.template get_destination_cooperative_tensor<
                decltype(tile_a), decltype(tile_b), bfloat>();
            #pragma unroll
            for (ushort index = 0; index < result.get_capacity(); ++index) {
                result[index] = bfloat(0.0f);
            }
            operation.run(tile_a, tile_b, result);
            result.store(tile_c);
        """,
        header: """
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
            using namespace metal;
            using namespace mpp::tensor_ops;
        """,
        ensureRowContiguous: true
    )
    #endif
}
