import Foundation
import MLX
import MLXFast

/// Guarded gate/up projection primitives for routed quantized MoE layers.
///
/// Compatible models keep their native down projection. Laguna's measured
/// prefill layout uses an expert-aligned matrix kernel; small decode layouts
/// use gather-GEMV kernels. Each helper returns `nil` outside its measured
/// quantization and alignment boundary so callers fall back to MLX.
package enum RoutedMoERouting {
    package static func parseBoolean(_ raw: String?, default defaultValue: Bool) -> Bool {
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

    static var supportsExpertAlignedNVFP4Metal: Bool {
        #if os(macOS)
        let architecture = GPU.deviceInfo().architecture
        return Device.defaultDevice().deviceType == .gpu
            && ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26
            && (architecture == "applegpu_g16s" || architecture == "applegpu_g17s")
        #else
        return false
        #endif
    }

    static let lagunaNVFP4QDotFastEnabled = parseBoolean(
        ProcessInfo.processInfo.environment["MERERUN_LAGUNA_NVFP4_QDOT_FAST"],
        default: true
    )

    /// Computes an unsorted NVFP4 gate/up gather-GEMV and SwiGLU activation.
    ///
    /// One threadgroup runs the gate and up projections concurrently for one
    /// expert route and one output tile. The M5 tile halves each SIMD group's
    /// live result accumulators while preserving each output row's reduction
    /// order. The input token is read directly from `x`, avoiding the repeated
    /// top-k route tensor.
    package static func fusedGatherNVFP4SwiGLU(
        _ x: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        expertIndices: MLXArray,
        topK: Int,
        groupSize: Int,
        bits: Int,
        rowsPerSIMDGroup: Int = 4
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
              bits == 4,
              rowsPerSIMDGroup == 1
                || rowsPerSIMDGroup == 2
                || rowsPerSIMDGroup == 4 else {
            return nil
        }

        let routeCount = expertIndices.size
        let outputDimensions = gateWeight.dim(1)
        let inputDimensions = x.dim(2)
        let outputTileWidth = 2 * rowsPerSIMDGroup
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
                ("RESULTS_PER_SIMDGROUP", rowsPerSIMDGroup),
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

    /// Fuses Laguna XS single-token routed and shared NVFP4 down projections,
    /// the ordered BF16 route reduction, the fixed 2.5 routed scale, and the
    /// decoder residual add. The one-row SIMD retile is the paired-M5-ranked
    /// layout; all non-XS shapes and quantization layouts fall back to MLX.
    package static func fusedLagunaXSRoutedSharedDownResidual(
        routedActivated: MLXArray,
        routedDownWeight: MLXArray,
        routedDownScales: MLXArray,
        indices: MLXArray,
        routerWeights: MLXArray,
        sharedActivated: MLXArray,
        sharedDownWeight: MLXArray,
        sharedDownScales: MLXArray,
        residual: MLXArray
    ) -> MLXArray? {
        #if os(macOS)
        guard supportsExpertAlignedNVFP4Metal,
              routedActivated.dtype == .bfloat16,
              routedActivated.shape == [8, 1, 512],
              routedDownWeight.dtype == .uint32,
              routedDownWeight.shape == [256, 2_048, 64],
              routedDownScales.dtype == .uint8,
              routedDownScales.shape == [256, 2_048, 32],
              indices.dtype == .uint32,
              indices.shape == [1, 1, 8],
              routerWeights.dtype == .bfloat16,
              routerWeights.shape == [1, 1, 8],
              sharedActivated.dtype == .bfloat16,
              sharedActivated.shape == [1, 1, 512],
              sharedDownWeight.dtype == .uint32,
              sharedDownWeight.shape == [2_048, 64],
              sharedDownScales.dtype == .uint8,
              sharedDownScales.shape == [2_048, 32],
              residual.dtype == .bfloat16,
              residual.shape == [1, 1, 2_048] else {
            return nil
        }
        return lagunaXSRoutedSharedDownResidualKernel(
            [
                routedActivated,
                routedDownWeight,
                routedDownScales,
                indices,
                routerWeights,
                sharedActivated,
                sharedDownWeight,
                sharedDownScales,
                residual,
            ],
            grid: (2_048 * 288, 1, 1),
            threadGroup: (288, 1, 1),
            outputShapes: [[1, 1, 2_048]],
            outputDTypes: [.bfloat16]
        )[0]
        #else
        return nil
        #endif
    }

    /// Inverts a one-dimensional permutation without paying for a second sort.
    package static func invertPermutation(_ order: MLXArray) -> MLXArray? {
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
    package static func stageRankedLagunaPrefillRoute(
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
    package static func fusedSortedNVFP4SwiGLU(
        _ sortedInput: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        sortedExpertIndices: MLXArray,
        groupSize: Int,
        bits: Int,
        pairwiseScaleReuse: Bool = false
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
                ("PAIRWISE_SCALE_REUSE", pairwiseScaleReuse),
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
    package static func sortedNVFP4Projection(
        _ sortedInput: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        sortedExpertIndices: MLXArray,
        groupSize: Int,
        bits: Int,
        pairwiseScaleReuse: Bool = false
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
                ("PAIRWISE_SCALE_REUSE", pairwiseScaleReuse),
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
    package static func fusedGatherAffine8SwiGLU(
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
    static let outputTileWidth = 8
    static let inputBlockWidth = 512
    static let simdWidth = 32
    static let parallelSIMDGroups = 4

    #endif
}
