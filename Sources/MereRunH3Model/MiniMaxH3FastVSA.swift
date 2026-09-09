import Foundation
import MLX
import MLXFast

package struct MiniMaxH3FastVSAGeometry: Sendable, Equatable {
    package static let tileSize = 64
    package static let videoTileShape = (temporal: 4, height: 4, width: 4)

    package let paddedToOriginal: [Int32]
    package let originalToPadded: [Int32]
    package let blockSizes: [Int32]
    package let prefixTileCount: Int
    package let videoTileCount: Int

    package var tileCount: Int { blockSizes.count }
    package var paddedTokenCount: Int { paddedToOriginal.count }

    package init(layout: MiniMaxH3PackedLayout) {
        let prefixSegments = [
            layout.textRows.count,
            layout.conditionRows.count,
            layout.targetAudioRows.count,
        ].filter { $0 > 0 }
        let prefixCount = prefixSegments.reduce(0, +)
        precondition(prefixCount == layout.targetVideoRows.lowerBound)

        var tiles: [[Int32]] = []
        var prefixStart = 0
        for segment in prefixSegments {
            for start in stride(from: 0, to: segment, by: Self.tileSize) {
                let count = min(Self.tileSize, segment - start)
                tiles.append((0..<count).map { Int32(prefixStart + start + $0) })
            }
            prefixStart += segment
        }
        self.prefixTileCount = tiles.count

        let temporal = layout.videoLatentFrames
        let height = layout.latentHeight / 2
        let width = layout.latentWidth / 2
        let shape = Self.videoTileShape
        for tileT in stride(from: 0, to: temporal, by: shape.temporal) {
            for tileH in stride(from: 0, to: height, by: shape.height) {
                for tileW in stride(from: 0, to: width, by: shape.width) {
                    var tile: [Int32] = []
                    tile.reserveCapacity(Self.tileSize)
                    for t in tileT..<min(tileT + shape.temporal, temporal) {
                        for h in tileH..<min(tileH + shape.height, height) {
                            for w in tileW..<min(tileW + shape.width, width) {
                                tile.append(Int32(prefixCount + (t * height + h) * width + w))
                            }
                        }
                    }
                    tiles.append(tile)
                }
            }
        }
        self.videoTileCount = tiles.count - prefixTileCount
        precondition(videoTileCount > 0)

        let sentinel = Int32(layout.sequenceLength)
        var paddedToOriginal: [Int32] = []
        var originalToPadded = Array(repeating: Int32(-1), count: layout.sequenceLength)
        var blockSizes: [Int32] = []
        paddedToOriginal.reserveCapacity(tiles.count * Self.tileSize)
        blockSizes.reserveCapacity(tiles.count)
        for tile in tiles {
            precondition(!tile.isEmpty && tile.count <= Self.tileSize)
            blockSizes.append(Int32(tile.count))
            let paddedStart = paddedToOriginal.count
            for (offset, original) in tile.enumerated() {
                precondition(original >= 0 && Int(original) < layout.sequenceLength)
                precondition(originalToPadded[Int(original)] == -1)
                originalToPadded[Int(original)] = Int32(paddedStart + offset)
                paddedToOriginal.append(original)
            }
            paddedToOriginal += Array(repeating: sentinel, count: Self.tileSize - tile.count)
        }
        precondition(originalToPadded.allSatisfy { $0 >= 0 })
        precondition(blockSizes.reduce(0, { $0 + Int($1) }) == layout.sequenceLength)
        self.paddedToOriginal = paddedToOriginal
        self.originalToPadded = originalToPadded
        self.blockSizes = blockSizes
    }
}

package struct MiniMaxH3FastVSAPreparedContext {
    package let geometry: MiniMaxH3FastVSAGeometry
    package let paddedIndices: MLXArray
    package let originalIndices: MLXArray
    package let blockSizes: MLXArray
    package let poolingSizes: MLXArray
}

package enum MiniMaxH3FastVSAKernelMode: String, Sendable {
    case halfTile = "half-tile"
    case fullTile = "full-tile"
    #if DEBUG
    // Benchmark-only rejected candidate; never selected by the runtime environment.
    case fullTileKV16 = "full-tile-kv16"
    #endif

    package var queryTileRows: Int {
        switch self {
        case .halfTile: 32
        case .fullTile: MiniMaxH3FastVSAGeometry.tileSize
        #if DEBUG
        case .fullTileKV16: MiniMaxH3FastVSAGeometry.tileSize
        #endif
        }
    }

    package var simdgroupCount: Int { queryTileRows / 8 }

    package static let runtimeDefault: Self = {
        let requested = Self(
            rawValue: ProcessInfo.processInfo.environment["MERERUN_H3_FASTVSA_KERNEL"] ?? ""
        )
        return requested == .halfTile ? .halfTile : .fullTile
    }()
}

/// Metal implementation of FastVideo's released VSA-H3 tile-64 inference contract.
///
/// Prefix query tiles remain dense, every query retains all prefix key tiles,
/// and video keys are selected by per-head top-k pooled QK scores. The selected
/// blocks use exact token attention; the trained compression branch separately
/// supplies dense pooled-value context through the released gate projection.
package enum MiniMaxH3FastVSA {
    package static let sparsity: Float = 0.9
    package static let headDimension = 128

    package static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        compressionGate: MLXArray,
        layout: MiniMaxH3PackedLayout,
        sparsity: Float = sparsity,
        kernelMode: MiniMaxH3FastVSAKernelMode = .fullTile
    ) -> MLXArray? {
        call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: compressionGate,
            prepared: prepare(layout: layout),
            sparsity: sparsity,
            kernelMode: kernelMode
        )
    }

    package static func prepare(layout: MiniMaxH3PackedLayout) -> MiniMaxH3FastVSAPreparedContext {
        let geometry = MiniMaxH3FastVSAGeometry(layout: layout)
        let paddedIndices = MLXArray(geometry.paddedToOriginal)
        let originalIndices = MLXArray(geometry.originalToPadded)
        let blockSizes = MLXArray(geometry.blockSizes)
        let poolingSizes = blockSizes.asType(.float32)
            .reshaped(1, 1, geometry.tileCount, 1)
        MLX.eval(paddedIndices, originalIndices, blockSizes, poolingSizes)
        return MiniMaxH3FastVSAPreparedContext(
            geometry: geometry,
            paddedIndices: paddedIndices,
            originalIndices: originalIndices,
            blockSizes: blockSizes,
            poolingSizes: poolingSizes
        )
    }

    package static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        compressionGate: MLXArray,
        prepared: MiniMaxH3FastVSAPreparedContext,
        sparsity: Float = sparsity,
        kernelMode: MiniMaxH3FastVSAKernelMode = .fullTile
    ) -> MLXArray? {
        guard supports(queries: queries, keys: keys, values: values, gate: compressionGate),
              (0..<1).contains(sparsity) else { return nil }
        let geometry = prepared.geometry
        guard queries.dim(2) == geometry.originalToPadded.count else { return nil }

        // RoPE coefficients are FP32, so the normal H3 projection path can
        // promote Q/K even when the checkpoint and hidden state are BF16.
        // FastH3's released VSA kernels run attention in BF16; normalize every
        // projected branch at this boundary instead of rejecting the faithful
        // FP32 rotary result.
        let attentionQueries = queries.asType(.bfloat16)
        let attentionKeys = keys.asType(.bfloat16)
        let attentionValues = values.asType(.bfloat16)
        let attentionGate = compressionGate.asType(.bfloat16)

        let tiledQueries = tile(attentionQueries, paddedIndices: prepared.paddedIndices)
        let tiledKeys = tile(attentionKeys, paddedIndices: prepared.paddedIndices)
        let tiledValues = tile(attentionValues, paddedIndices: prepared.paddedIndices)
        let tiledGate = tile(attentionGate, paddedIndices: prepared.paddedIndices)

        let pooledQueries = pool(
            tiledQueries,
            sizes: prepared.poolingSizes,
            tileCount: geometry.tileCount
        )
        let pooledKeys = pool(tiledKeys, sizes: prepared.poolingSizes, tileCount: geometry.tileCount)
        let pooledValues = pool(tiledValues, sizes: prepared.poolingSizes, tileCount: geometry.tileCount)
        let scale = 1 / sqrt(Float(headDimension))
        let scores = MLX.matmul(
            pooledQueries,
            pooledKeys.transposed(0, 1, 3, 2)
        ) * scale
        let videoRoutes = selectedVideoRoutes(
            scores: scores,
            prefixTileCount: geometry.prefixTileCount,
            videoTileCount: geometry.videoTileCount,
            sparsity: sparsity
        )
        let sparse = sparseKernelOutput(
            queries: tiledQueries,
            keys: tiledKeys,
            values: tiledValues,
            videoRoutes: videoRoutes,
            blockSizes: prepared.blockSizes,
            prefixTileCount: geometry.prefixTileCount,
            scale: scale,
            kernelMode: kernelMode
        )

        let compressed = MLX.matmul(
            MLX.softmax(scores, axis: -1, precise: true),
            pooledValues
        ).asType(sparse.dtype)
        let tileCount = geometry.tileCount
        let heads = attentionQueries.dim(1)
        let sparseTiles = sparse.reshaped(1, heads, tileCount, MiniMaxH3FastVSAGeometry.tileSize, headDimension)
        let gateTiles = tiledGate.reshaped(
            1, heads, tileCount, MiniMaxH3FastVSAGeometry.tileSize, headDimension
        )
        let corrected = sparseTiles + compressed.expandedDimensions(axis: 3) * gateTiles
        let tiledOutput = corrected.reshaped(1, heads, geometry.paddedTokenCount, headDimension)
        return MLX.take(tiledOutput, prepared.originalIndices, axis: 2)
    }

    package static func routesForTesting(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float = sparsity
    ) -> MLXArray {
        routes(
            scores: scores,
            prefixTileCount: prefixTileCount,
            videoTileCount: videoTileCount,
            sparsity: sparsity
        )
    }

    package static func selectedVideoRoutesForTesting(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float = sparsity
    ) -> MLXArray {
        selectedVideoRoutes(
            scores: scores,
            prefixTileCount: prefixTileCount,
            videoTileCount: videoTileCount,
            sparsity: sparsity
        )
    }

    package static func sparseOutputForTesting(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        videoRoutes: MLXArray,
        blockSizes: MLXArray,
        prefixTileCount: Int,
        kernelMode: MiniMaxH3FastVSAKernelMode
    ) -> MLXArray {
        sparseKernelOutput(
            queries: queries,
            keys: keys,
            values: values,
            videoRoutes: videoRoutes,
            blockSizes: blockSizes,
            prefixTileCount: prefixTileCount,
            scale: 1 / sqrt(Float(headDimension)),
            kernelMode: kernelMode
        )
    }

    static func tile(_ value: MLXArray, paddedIndices: MLXArray) -> MLXArray {
        let zero = MLXArray.zeros(
            [value.dim(0), value.dim(1), 1, value.dim(3)],
            dtype: value.dtype
        )
        return MLX.take(MLX.concatenated([value, zero], axis: 2), paddedIndices, axis: 2)
    }

    static func pool(
        _ value: MLXArray,
        sizes: MLXArray,
        tileCount: Int
    ) -> MLXArray {
        value.asType(.float32)
            .reshaped(1, value.dim(1), tileCount, MiniMaxH3FastVSAGeometry.tileSize, headDimension)
            .sum(axis: 3) / sizes
    }

    static func routes(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float
    ) -> MLXArray {
        precondition(scores.dim(2) == prefixTileCount + videoTileCount)
        precondition(scores.dim(3) == prefixTileCount + videoTileCount)
        let keepVideo = max(1, min(Int(ceil((1 - sparsity) * Float(videoTileCount))), videoTileCount))
        let videoScores = scores[.ellipsis, prefixTileCount...]
        let videoRoutes: MLXArray
        if keepVideo == videoTileCount {
            videoRoutes = MLXArray.ones(videoScores.shape, dtype: .uint8)
        } else {
            let selected = MLX.argPartition(
                -videoScores,
                kth: keepVideo - 1,
                axis: -1
            )[.ellipsis, 0..<keepVideo]
            videoRoutes = MLX.putAlong(
                MLXArray.zeros(videoScores.shape, dtype: .uint8),
                selected,
                values: MLXArray.ones(selected.shape, dtype: .uint8),
                axis: -1
            )
        }
        var result = prefixTileCount == 0
            ? videoRoutes
            : MLX.concatenated([
                MLXArray.ones(
                    [scores.dim(0), scores.dim(1), scores.dim(2), prefixTileCount],
                    dtype: .uint8
                ),
                videoRoutes,
            ], axis: -1)
        if prefixTileCount > 0 {
            result = MLX.concatenated([
                MLXArray.ones(
                    [scores.dim(0), scores.dim(1), prefixTileCount, scores.dim(3)],
                    dtype: .uint8
                ),
                result[0..., 0..., prefixTileCount..., 0...],
            ], axis: 2)
        }
        return result
    }

    static func selectedVideoRoutes(
        scores: MLXArray,
        prefixTileCount: Int,
        videoTileCount: Int,
        sparsity: Float
    ) -> MLXArray {
        precondition(scores.dim(2) == prefixTileCount + videoTileCount)
        precondition(scores.dim(3) == prefixTileCount + videoTileCount)
        let keepVideo = max(1, min(Int(ceil((1 - sparsity) * Float(videoTileCount))), videoTileCount))
        let videoScores = scores[.ellipsis, prefixTileCount...]
        if keepVideo == videoTileCount {
            let indices = (
                MLX.arange(videoTileCount, dtype: .int32) + MLXArray(Int32(prefixTileCount))
            ).reshaped(1, 1, 1, videoTileCount)
            return MLX.broadcast(indices, to: Array(scores.shape.dropLast()) + [videoTileCount])
        }
        return (
            MLX.argPartition(-videoScores, kth: keepVideo - 1, axis: -1)[.ellipsis, 0..<keepVideo]
                .asType(.int32) + MLXArray(Int32(prefixTileCount))
        )
    }

    static func supports(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        gate: MLXArray
    ) -> Bool {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        return Device.defaultDevice().deviceType == .gpu
            && queries.shape == keys.shape
            && queries.shape == values.shape
            && queries.shape == gate.shape
            && queries.ndim == 4
            && queries.dim(0) == 1
            && queries.dim(3) == headDimension
        #else
        return false
        #endif
    }

    static func sparseKernelOutput(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        videoRoutes: MLXArray,
        blockSizes: MLXArray,
        prefixTileCount: Int,
        scale: Float,
        kernelMode: MiniMaxH3FastVSAKernelMode
    ) -> MLXArray {
        let tokenCount = queries.dim(2)
        let blockCount = blockSizes.dim(0)
        let keepVideo = videoRoutes.dim(3)
        precondition(tokenCount == blockCount * MiniMaxH3FastVSAGeometry.tileSize)
        precondition(videoRoutes.dim(2) == blockCount)
        precondition(prefixTileCount + keepVideo <= blockCount)
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let queryTileRows = kernelMode.queryTileRows
        let simdgroupCount = kernelMode.simdgroupCount
        let inputs = [queries, keys, values, videoRoutes, blockSizes, MLXArray([scale])]
        let template: [(String, any KernelTemplateArg)] = [
            ("TOKEN_COUNT", tokenCount),
            ("BLOCK_COUNT", blockCount),
            ("PREFIX_TILE_COUNT", prefixTileCount),
            ("KEEP_VIDEO", keepVideo),
            ("QUERY_TILE_ROWS", queryTileRows),
            ("SIMDGROUP_COUNT", simdgroupCount),
        ]
        let grid = (
            32,
            ((tokenCount + queryTileRows - 1) / queryTileRows) * simdgroupCount,
            queries.dim(1)
        )
        #if DEBUG
        if kernelMode == .fullTileKV16 {
            return attentionKV16Kernel(
                inputs,
                template: template,
                grid: grid,
                threadGroup: (32, simdgroupCount, 1),
                outputShapes: [queries.shape],
                outputDTypes: [.bfloat16]
            )[0]
        }
        #endif
        return attentionKernel(
            inputs,
            template: template,
            grid: grid,
            threadGroup: (32, simdgroupCount, 1),
            outputShapes: [queries.shape],
            outputDTypes: [.bfloat16]
        )[0]
        #else
        preconditionFailure("FastH3 VSA requires Metal")
        #endif
    }
}
