import Foundation
import MLX
import MLXFast
import MLXNN

public final class MiniMaxH3VideoVAE: Module {
    package static let defaultSpatialTileSize = 256
    package static let minimumSpatialTileOverlap = 64

    package var spatialTileSize = defaultSpatialTileSize
    package var usesCompiledTileDecoder = true
    package var evaluatesTemporalChunksIndividually = true

    @ModuleInfo(key: "encoder") public var encoder: MiniMaxH3VideoEncoder
    @ModuleInfo(key: "quant_conv") package var quantConvolution: Conv3d
    @ModuleInfo(key: "post_quant_conv") package var postQuantConvolution: Conv3d
    @ModuleInfo(key: "decoder") public var decoder: MiniMaxH3VideoDecoder
    @ParameterInfo(key: "latents_mean") package var latentMean: MLXArray
    @ParameterInfo(key: "latents_std") package var latentStandardDeviation: MLXArray

    public override init() {
        _encoder.wrappedValue = MiniMaxH3VideoEncoder()
        _quantConvolution.wrappedValue = Conv3d(
            inputChannels: 48,
            outputChannels: 48,
            kernelSize: .init(1),
            bias: true
        )
        _postQuantConvolution.wrappedValue = Conv3d(
            inputChannels: 24,
            outputChannels: 24,
            kernelSize: .init(1),
            bias: true
        )
        _decoder.wrappedValue = MiniMaxH3VideoDecoder()
        _latentMean.wrappedValue = MLXArray.zeros([24])
        _latentStandardDeviation.wrappedValue = MLXArray.ones([24])
        super.init()
    }


    /// Encodes one prepared RGB keyframe `[1, H, W, 3]` with H3's fixed
    /// posterior seed and returns normalized `[1, 24, 1, H/16, W/16]` latents.
    public func encodeKeyframe(_ rgb: MLXArray) -> MLXArray {
        precondition(rgb.ndim == 4 && rgb.dim(0) == 1 && rgb.dim(3) == 3)
        let meanRGB = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 3)
        let stdRGB = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 3)
        let normalized = ((rgb - meanRGB) / stdRGB).expandedDimensions(axis: 1)
        let moments = encodeClip(normalized)
        return sampleAndNormalize(moments)
    }

    /// Encodes prepared `[1, T, H, W, 3]` RGB reference video. Frames are
    /// repeated to 17-frame chunks and H3's three trailing latent tokens are
    /// dropped after concatenation, matching the released Ref2VA pipeline.
    public func encodeReferenceVideo(_ rgb: MLXArray) -> MLXArray {
        precondition(rgb.ndim == 5 && rgb.dim(0) == 1 && rgb.dim(4) == 3 && rgb.dim(1) > 0)
        let meanRGB = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 1, 3)
        let stdRGB = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 1, 3)
        var normalized = (rgb - meanRGB) / stdRGB
        let remainder = normalized.dim(1) % MiniMaxH3Geometry.videoFramesPerChunk
        if remainder != 0 {
            let count = MiniMaxH3Geometry.videoFramesPerChunk - remainder
            let last = normalized[0..., (normalized.dim(1) - 1)..., 0..., 0..., 0...]
            normalized = MLX.concatenated(
                [normalized, MLX.tiled(last, repetitions: [1, count, 1, 1, 1])],
                axis: 1
            )
        }
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: normalized.dim(1), by: MiniMaxH3Geometry.videoFramesPerChunk) {
            chunks.append(encodeClip(
                normalized[0..., start..<(start + MiniMaxH3Geometry.videoFramesPerChunk), 0..., 0..., 0...]
            ))
        }
        let moments = MLX.concatenated(chunks, axis: 1)
        precondition(moments.dim(1) > 3)
        return sampleAndNormalize(moments[0..., 0..<(moments.dim(1) - 3), 0..., 0..., 0...])
    }

    private func sampleAndNormalize(_ moments: MLXArray) -> MLXArray {
        let parts = MLX.split(moments, parts: 2, axis: -1)
        MLXRandom.seed(42)
        let logVariance = MLX.clip(parts[1], min: -30, max: 20)
        let sample = parts[0] + MLX.exp(0.5 * logVariance) * MLXRandom.normal(parts[0].shape)
        let channelFirst = sample.asType(.float16).asType(.float32).transposed(0, 4, 1, 2, 3)
        return (channelFirst - latentMean.reshaped(1, 24, 1, 1, 1))
            / latentStandardDeviation.reshaped(1, 24, 1, 1, 1)
    }

    private func encodeClip(_ video: MLXArray) -> MLXArray {
        let pixelHeight = video.dim(2)
        let pixelWidth = video.dim(3)
        let y = Self.tilePlan(length: pixelHeight, tileSize: spatialTileSize)
        let x = Self.tilePlan(length: pixelWidth, tileSize: spatialTileSize)
        var tiles: [MLXArray] = []
        for (yIndex, yStart) in y.starts.enumerated() {
            for (xIndex, xStart) in x.starts.enumerated() {
                tiles.append(video[
                    0..., 0...,
                    yStart..<(yStart + y.lengths[yIndex]),
                    xStart..<(xStart + x.lengths[xIndex]),
                    0...
                ])
            }
        }
        let encodedTiles = MLX.split(
            quantConvolution(encoder(MLX.concatenated(tiles, axis: 0))),
            parts: tiles.count,
            axis: 0
        )
        var rows: [[MLXArray]] = []
        rows.reserveCapacity(y.starts.count)
        for rowIndex in y.starts.indices {
            let start = rowIndex * x.starts.count
            rows.append(Array(encodedTiles[start..<(start + x.starts.count)]))
        }

        let yOverlaps = y.overlaps.map { $0 / 16 }
        let xOverlaps = x.overlaps.map { $0 / 16 }
        var stitchedRows: [MLXArray] = []
        for rowIndex in rows.indices {
            var pieces: [MLXArray] = []
            for columnIndex in rows[rowIndex].indices {
                var tile = rows[rowIndex][columnIndex]
                if rowIndex > 0 {
                    tile = Self.blend(rows[rowIndex - 1][columnIndex], tile, extent: yOverlaps[rowIndex - 1], axis: 2)
                }
                if columnIndex > 0 {
                    tile = Self.blend(rows[rowIndex][columnIndex - 1], tile, extent: xOverlaps[columnIndex - 1], axis: 3)
                }
                if rowIndex + 1 < rows.count {
                    tile = tile[0..., 0..., 0..<(tile.dim(2) - yOverlaps[rowIndex]), 0..., 0...]
                }
                if columnIndex + 1 < rows[rowIndex].count {
                    tile = tile[0..., 0..., 0..., 0..<(tile.dim(3) - xOverlaps[columnIndex]), 0...]
                }
                pieces.append(tile)
            }
            stitchedRows.append(MLX.concatenated(pieces, axis: 3))
        }
        return MLX.concatenated(stitchedRows, axis: 2)
    }

    /// Converts normalized H3 latents to `[B, T, H, W, 3]` RGB in `[0, 1]`.
    public func decode(_ normalizedLatents: MLXArray) -> MLXArray {
        precondition(normalizedLatents.dim(2) >= 7, "H3 video decode requires at least seven latent frames")
        let mean = latentMean.reshaped(1, 24, 1, 1, 1)
        let standardDeviation = latentStandardDeviation.reshaped(1, 24, 1, 1, 1)
        var denormalized = normalizedLatents * standardDeviation + mean
        let tokensPerChunk = 5
        let tokenOverlap = 2
        let tokenDrop = 3
        let pixelFramesPerChunk = 20
        let framePrePadding = 3
        let frameOverlap = 5
        let padTokens = (-(denormalized.dim(2) + tokenDrop)).quotientAndRemainder(dividingBy: tokensPerChunk).remainder
        let resolvedPadTokens = padTokens == 0 ? 0 : padTokens + tokensPerChunk
        let originalTokenCount = denormalized.dim(2)
        if resolvedPadTokens > 0 {
            let last = denormalized[0..., 0..., (originalTokenCount - 1)..., 0..., 0...]
            denormalized = MLX.concatenated(
                [denormalized, MLX.tiled(last, repetitions: [1, 1, resolvedPadTokens, 1, 1])],
                axis: 2
            )
        }
        let chunkCount = (originalTokenCount + tokenDrop + resolvedPadTokens) / tokensPerChunk - 1
        var decodedChunks: [MLXArray] = []
        var overlap: MLXArray?
        let decodeTile: (MLXArray) -> MLXArray = { (channelLast: MLXArray) -> MLXArray in
            let projected = self.postQuantConvolution(channelLast).transposed(0, 4, 1, 2, 3)
            return self.decoder(projected)
        }
        let tileDecoder: (MLXArray) -> MLXArray = usesCompiledTileDecoder
            ? MLX.compile(decodeTile)
            : decodeTile
        for index in 0..<chunkCount {
            let start = index * tokensPerChunk
            let clip = decodeClip(
                denormalized[0..., 0..., start..<(start + tokensPerChunk + tokenOverlap), 0..., 0...],
                tileDecoder: tileDecoder
            )
            if evaluatesTemporalChunksIndividually {
                MLX.eval(clip)
            }
            for part in 0..<2 {
                let frameStart = part * pixelFramesPerChunk
                let frameEnd = min(frameStart + pixelFramesPerChunk, clip.dim(2))
                guard frameEnd > frameStart + framePrePadding else { continue }
                var chunk = clip[0..., 0..., (frameStart + framePrePadding)..<frameEnd, 0..., 0...]
                if part == 0 {
                    if let overlap { chunk = Self.blend(overlap, chunk, extent: frameOverlap, axis: 2) }
                    decodedChunks.append(chunk)
                } else {
                    overlap = chunk
                }
            }
        }
        if let overlap { decodedChunks.append(overlap) }
        var imageNet = MLX.concatenated(decodedChunks, axis: 2)
        if resolvedPadTokens > 0 {
            let intraTail = 1
            var padFrames = 0
            for index in 0..<resolvedPadTokens {
                padFrames += (originalTokenCount + index).isMultiple(of: tokensPerChunk) ? intraTail : 4
            }
            imageNet = imageNet[0..., 0..., 0..<(imageNet.dim(2) - padFrames), 0..., 0...]
        }
        imageNet = imageNet.transposed(0, 2, 3, 4, 1)
        let meanRGB = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 1, 3)
        let stdRGB = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 1, 3)
        return MLX.clip(imageNet * stdRGB + meanRGB, min: 0, max: 1)
    }

    private func decodeClip(
        _ latent: MLXArray,
        tileDecoder: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let pixelHeight = latent.dim(3) * 16
        let pixelWidth = latent.dim(4) * 16
        let y = Self.tilePlan(length: pixelHeight, tileSize: spatialTileSize)
        let x = Self.tilePlan(length: pixelWidth, tileSize: spatialTileSize)
        var tiles: [MLXArray] = []
        for (yIndex, yStart) in y.starts.enumerated() {
            for (xIndex, xStart) in x.starts.enumerated() {
                let tile = latent[
                    0..., 0..., 0...,
                    (yStart / 16)..<((yStart + y.lengths[yIndex]) / 16),
                    (xStart / 16)..<((xStart + x.lengths[xIndex]) / 16)
                ]
                tiles.append(tile.transposed(0, 2, 3, 4, 1))
            }
        }
        let decodedTiles = MLX.split(
            tileDecoder(MLX.concatenated(tiles, axis: 0)),
            parts: tiles.count,
            axis: 0
        )
        var rows: [[MLXArray]] = []
        rows.reserveCapacity(y.starts.count)
        for rowIndex in y.starts.indices {
            let start = rowIndex * x.starts.count
            rows.append(Array(decodedTiles[start..<(start + x.starts.count)]))
        }

        var stitchedRows: [MLXArray] = []
        for rowIndex in rows.indices {
            var pieces: [MLXArray] = []
            for columnIndex in rows[rowIndex].indices {
                var tile = rows[rowIndex][columnIndex]
                if rowIndex > 0 {
                    tile = Self.blend(rows[rowIndex - 1][columnIndex], tile, extent: y.overlaps[rowIndex - 1], axis: 3)
                }
                if columnIndex > 0 {
                    tile = Self.blend(rows[rowIndex][columnIndex - 1], tile, extent: x.overlaps[columnIndex - 1], axis: 4)
                }
                if rowIndex + 1 < rows.count {
                    tile = tile[0..., 0..., 0..., 0..<(tile.dim(3) - y.overlaps[rowIndex]), 0...]
                }
                if columnIndex + 1 < rows[rowIndex].count {
                    tile = tile[0..., 0..., 0..., 0..., 0..<(tile.dim(4) - x.overlaps[columnIndex])]
                }
                pieces.append(tile)
            }
            stitchedRows.append(MLX.concatenated(pieces, axis: 4))
        }
        return MLX.concatenated(stitchedRows, axis: 3)
    }

    package static func tilePlan(
        length: Int,
        tileSize: Int
    ) -> (starts: [Int], lengths: [Int], overlaps: [Int]) {
        precondition(tileSize >= minimumSpatialTileOverlap && tileSize.isMultiple(of: 16))
        let minimumOverlap = minimumSpatialTileOverlap
        guard length > tileSize else { return ([0], [length], []) }
        var count = Int(ceil(Double(length) / Double(tileSize)))
        while tileSize * count - minimumOverlap * (count - 1) < length { count += 1 }
        var overlaps = Array(repeating: minimumOverlap, count: count - 1)
        let remaining = tileSize * count - overlaps.reduce(0, +) - length
        for index in 0..<(remaining / 16) { overlaps[index % overlaps.count] += 16 }
        var starts = [0]
        for index in overlaps.indices { starts.append(starts.last! + tileSize - overlaps[index]) }
        return (starts, Array(repeating: tileSize, count: count), overlaps)
    }

    private static func blend(_ previous: MLXArray, _ current: MLXArray, extent: Int, axis: Int) -> MLXArray {
        let length = min(previous.dim(axis), current.dim(axis), extent)
        guard length > 0 else { return current }
        var shape = Array(repeating: 1, count: current.ndim)
        shape[axis] = length
        let position = MLXArray(0..<length).asType(.float32) / Float(length)
        let weightCurrent = position.reshaped(shape).asType(current.dtype)
        let weightPrevious = 1 - weightCurrent
        let previousIndices = MLXArray(((previous.dim(axis) - length)..<previous.dim(axis)).map(Int32.init))
        let currentIndices = MLXArray((0..<length).map(Int32.init))
        let blended = MLX.take(previous, previousIndices, axis: axis) * weightPrevious
            + MLX.take(current, currentIndices, axis: axis) * weightCurrent
        guard length < current.dim(axis) else { return blended }
        let restIndices = MLXArray((length..<current.dim(axis)).map(Int32.init))
        return MLX.concatenated([blended, MLX.take(current, restIndices, axis: axis)], axis: axis)
    }
}
