import Foundation
import MLX
import MLXFast
import MLXNN

package struct LTXIntervals {
    package let starts: [Int]
    package let ends: [Int]
    package let leftRamps: [Int]
    package let rightRamps: [Int]

    package init(starts: [Int], ends: [Int], leftRamps: [Int], rightRamps: [Int]) {
        self.starts = starts
        self.ends = ends
        self.leftRamps = leftRamps
        self.rightRamps = rightRamps
    }
}

package struct LTXDecodeTilingConfig {
    package let spatialTileSizeInPixels: Int?
    package let spatialTileOverlapInPixels: Int
    package let temporalTileSizeInFrames: Int?
    package let temporalTileOverlapInFrames: Int
}

package let ltxDecoderWorkspaceBytesPerOutputPixel = 1_280.0
package let ltxMinimumTemporalDecodeTileFrames = 32

package func selectDecodeTilingConfig(
    width: Int,
    height: Int,
    numFrames: Int,
    fps: Double,
    decodeBudgetGiB: Double? = nil,
    spatialTileSizeInPixels: Int? = nil,
    spatialTileOverlapInPixels: Int = 0
) -> LTXDecodeTilingConfig? {
    let framePixels = Double(max(1, width)) * Double(max(1, height))
    let budgetGiB = decodeBudgetGiB ?? decodeTilingBudgetGiB()
    let budgetBytes = max(1.0, budgetGiB) * 1024.0 * 1024.0 * 1024.0

    let decodeFramePixels: Double
    if let spatialTileSizeInPixels {
        decodeFramePixels = Double(min(max(1, width), spatialTileSizeInPixels))
            * Double(min(max(1, height), spatialTileSizeInPixels))
    } else {
        decodeFramePixels = framePixels
    }

    // The late 3D convolutions dominate decode memory. Counting only their output
    // tensors misses the input transform and Metal workspace by roughly two orders
    // of magnitude. This bound covers the BF16 3x3x3 working surfaces at the widest
    // decoded stages and intentionally rounds above the measured native high-water.
    let bytesPerDecodeFrame = decodeFramePixels * ltxDecoderWorkspaceBytesPerOutputPixel
    let estimatedWorkspaceBytes = bytesPerDecodeFrame * Double(max(1, numFrames))
    let tileFrames: Int?
    if estimatedWorkspaceBytes > budgetBytes {
        let budgetedFrames = max(1, Int(budgetBytes / max(bytesPerDecodeFrame, 1.0)))
        let alignedFrames = (budgetedFrames / 8) * 8
        tileFrames = max(ltxMinimumTemporalDecodeTileFrames, alignedFrames)
    } else {
        tileFrames = nil
    }
    guard tileFrames != nil || spatialTileSizeInPixels != nil else {
        return nil
    }

    let overlapFrames: Int
    if let tileFrames {
        let oneSecondFrames = max(8, (Int(max(1, fps).rounded()) / 8) * 8)
        overlapFrames = min(oneSecondFrames, max(8, (tileFrames / 32) * 8))
    } else {
        overlapFrames = 0
    }

    return LTXDecodeTilingConfig(
        spatialTileSizeInPixels: spatialTileSizeInPixels,
        spatialTileOverlapInPixels: spatialTileOverlapInPixels,
        temporalTileSizeInFrames: tileFrames,
        temporalTileOverlapInFrames: min(
            max(0, overlapFrames),
            max(0, (tileFrames ?? 8) - 8)
        )
    )
}

package func decodeTilingBudgetGiB() -> Double {
    let environment = ProcessInfo.processInfo.environment
    for key in ["LTX2_VAE_DECODE_BUDGET_GB", "MERERUN_VIDEO_LTX_VAE_DECODE_BUDGET_GB"] {
        if let value = environment[key], let parsed = Double(value), parsed.isFinite, parsed > 0 {
            return parsed
        }
    }
    return 8.0
}

package func computeTrapezoidalMask1D(
    length: Int,
    rampLeft: Int,
    rampRight: Int,
    leftStartsFromZero: Bool
) -> [Float] {
    precondition(length > 0, "Mask length must be positive.")
    let left = max(0, min(rampLeft, length))
    let right = max(0, min(rampRight, length))

    var mask = [Float](repeating: 1.0, count: length)

    if left > 0 {
        let intervalLength = leftStartsFromZero ? (left + 1) : (left + 2)
        var fade = [Float](repeating: 0, count: left)
        if leftStartsFromZero {
            for i in 0..<left {
                fade[i] = Float(i) / Float(intervalLength - 1)
            }
        } else {
            for i in 0..<left {
                fade[i] = Float(i + 1) / Float(intervalLength - 1)
            }
        }
        for i in 0..<left {
            mask[i] *= fade[i]
        }
    }

    if right > 0 {
        for i in 0..<right {
            let fade = Float(right - i) / Float(right + 1)
            mask[length - right + i] *= fade
        }
    }

    for i in 0..<mask.count {
        mask[i] = min(1.0, max(0.0, mask[i]))
    }
    return mask
}

package func splitInSpatial(size: Int, overlap: Int, dimensionSize: Int) -> LTXIntervals {
    if dimensionSize <= size {
        return LTXIntervals(starts: [0], ends: [dimensionSize], leftRamps: [0], rightRamps: [0])
    }

    let amount = (dimensionSize + size - 2 * overlap - 1) / (size - overlap)
    let starts = (0..<amount).map { $0 * (size - overlap) }
    var ends = starts.map { $0 + size }
    if !ends.isEmpty {
        ends[ends.count - 1] = dimensionSize
    }
    let leftRamps = [0] + Array(repeating: overlap, count: max(0, amount - 1))
    let rightRamps = Array(repeating: overlap, count: max(0, amount - 1)) + [0]
    return LTXIntervals(starts: starts, ends: ends, leftRamps: leftRamps, rightRamps: rightRamps)
}

package func splitInTemporal(size: Int, overlap: Int, dimensionSize: Int) -> LTXIntervals {
    if dimensionSize <= size {
        return LTXIntervals(starts: [0], ends: [dimensionSize], leftRamps: [0], rightRamps: [0])
    }

    let intervals = splitInSpatial(size: size, overlap: overlap, dimensionSize: dimensionSize)
    var starts = intervals.starts
    var leftRamps = intervals.leftRamps

    if starts.count > 1 {
        for i in 1..<starts.count {
            starts[i] -= 1
            leftRamps[i] += 1
        }
    }

    return LTXIntervals(
        starts: starts,
        ends: intervals.ends,
        leftRamps: leftRamps,
        rightRamps: intervals.rightRamps
    )
}

package func mapSpatialOutputSlice(
    begin: Int,
    end: Int,
    leftRamp: Int,
    rightRamp: Int,
    scale: Int
) -> (start: Int, length: Int, mask: [Float]) {
    let start = begin * scale
    let stop = end * scale
    let length = max(1, stop - start)
    let mask = computeTrapezoidalMask1D(
        length: length,
        rampLeft: leftRamp * scale,
        rampRight: rightRamp * scale,
        leftStartsFromZero: false
    )
    return (start, length, mask)
}

package func mapTemporalOutputSlice(
    begin: Int,
    end: Int,
    leftRamp: Int,
    rightRamp: Int,
    scale: Int
) -> (start: Int, length: Int, mask: [Float]) {
    let start = begin * scale
    let stop = 1 + (end - 1) * scale
    let length = max(1, stop - start)
    let leftScaled = leftRamp > 0 ? (1 + (leftRamp - 1) * scale) : 0
    let rightScaled = rightRamp * scale
    let mask = computeTrapezoidalMask1D(
        length: length,
        rampLeft: leftScaled,
        rampRight: rightScaled,
        leftStartsFromZero: true
    )
    return (start, length, mask)
}

package func accumulateLTXDecodedTile(
    output: MLXArray,
    weights: MLXArray,
    tileDecoded: MLXArray,
    outputFrameStart: Int,
    outputHeightStart: Int,
    outputWidthStart: Int,
    temporalMask: [Float],
    heightMask: [Float],
    widthMask: [Float]
) {
    let tileFrames = min(tileDecoded.dim(2), min(temporalMask.count, output.dim(2) - outputFrameStart))
    let tileHeight = min(tileDecoded.dim(3), min(heightMask.count, output.dim(3) - outputHeightStart))
    let tileWidth = min(tileDecoded.dim(4), min(widthMask.count, output.dim(4) - outputWidthStart))
    guard tileFrames > 0, tileHeight > 0, tileWidth > 0 else { return }

    let temporal = MLXArray(Array(temporalMask.prefix(tileFrames)))
        .asType(output.dtype)
        .reshaped(1, 1, tileFrames, 1, 1)
    let vertical = MLXArray(Array(heightMask.prefix(tileHeight)))
        .asType(output.dtype)
        .reshaped(1, 1, 1, tileHeight, 1)
    let horizontal = MLXArray(Array(widthMask.prefix(tileWidth)))
        .asType(output.dtype)
        .reshaped(1, 1, 1, 1, tileWidth)
    let blend = temporal * vertical * horizontal
    let frameRange = outputFrameStart..<(outputFrameStart + tileFrames)
    let heightRange = outputHeightStart..<(outputHeightStart + tileHeight)
    let widthRange = outputWidthStart..<(outputWidthStart + tileWidth)
    let tile = tileDecoded[0..., 0..<3, 0..<tileFrames, 0..<tileHeight, 0..<tileWidth]
        .asType(output.dtype)

    output[0..., 0..., frameRange, heightRange, widthRange] =
        output[0..., 0..., frameRange, heightRange, widthRange] + (tile * blend)
    weights[0..., 0..., frameRange, heightRange, widthRange] =
        weights[0..., 0..., frameRange, heightRange, widthRange] + blend
}

package func finalizeLTXDecodedTiles(output: MLXArray, weights: MLXArray) -> MLXArray {
    var video = finalizeLTXDecodedTilesRaw(output: output, weights: weights)[0]
    video = video.transposed(1, 2, 3, 0)
    let zero = MLXArray(Float(0)).asType(video.dtype)
    let one = MLXArray(Float(1)).asType(video.dtype)
    video = MLX.clip(
        (video + one) / MLXArray(Float(2)).asType(video.dtype),
        min: zero,
        max: one
    )
    return (video * MLXArray(Float(255)).asType(video.dtype)).asType(.uint8)
}

package func finalizeLTXDecodedTilesRaw(output: MLXArray, weights: MLXArray) -> MLXArray {
    let epsilon: Float = weights.dtype == .float16 ? 1e-4 : 1e-8
    let denominator = MLX.maximum(weights, MLXArray(epsilon).asType(weights.dtype))
    return output / denominator
}

package func decodeWithTiling(
    decoder: LTXVideoDecoder,
    latents: MLXArray,
    spatialTileSizeInPixels: Int?,
    spatialOverlapInPixels: Int,
    temporalTileSizeInFrames: Int?,
    temporalOverlapInFrames: Int,
    spatialScale: Int,
    temporalScale: Int
) -> MLXArray {
    let decoded = decodeWithTilingRaw(
        decoder: decoder,
        latents: latents,
        spatialTileSizeInPixels: spatialTileSizeInPixels,
        spatialOverlapInPixels: spatialOverlapInPixels,
        temporalTileSizeInFrames: temporalTileSizeInFrames,
        temporalOverlapInFrames: temporalOverlapInFrames,
        spatialScale: spatialScale,
        temporalScale: temporalScale
    )
    var video = decoded[0].transposed(1, 2, 3, 0)
    let zero = MLXArray(Float(0)).asType(video.dtype)
    let one = MLXArray(Float(1)).asType(video.dtype)
    video = MLX.clip(
        (video + one) / MLXArray(Float(2)).asType(video.dtype),
        min: zero,
        max: one
    )
    let frames = (video * MLXArray(Float(255)).asType(video.dtype)).asType(.uint8)
    MLX.eval(frames)
    return frames
}

package func decodeWithTilingRaw(
    decoder: LTXVideoDecoder,
    latents: MLXArray,
    spatialTileSizeInPixels: Int?,
    spatialOverlapInPixels: Int,
    temporalTileSizeInFrames: Int?,
    temporalOverlapInFrames: Int,
    spatialScale: Int,
    temporalScale: Int
) -> MLXArray {
    precondition(latents.ndim == 5, "Expected latent tensor [B, C, F, H, W]")
    let batch = latents.dim(0)
    precondition(batch == 1, "Tiled decode currently supports batch=1.")

    let latentFrames = latents.dim(2)
    let latentH = latents.dim(3)
    let latentW = latents.dim(4)

    let outFrames = 1 + (latentFrames - 1) * temporalScale
    let outH = latentH * spatialScale
    let outW = latentW * spatialScale

    let temporalIntervals: LTXIntervals
    if let temporalTileSizeInFrames {
        let tileSize = max(1, temporalTileSizeInFrames / temporalScale)
        let overlap = max(0, temporalOverlapInFrames / temporalScale)
        temporalIntervals = splitInTemporal(size: tileSize, overlap: overlap, dimensionSize: latentFrames)
    } else {
        temporalIntervals = LTXIntervals(starts: [0], ends: [latentFrames], leftRamps: [0], rightRamps: [0])
    }

    let heightIntervals: LTXIntervals
    let widthIntervals: LTXIntervals
    if let spatialTileSizeInPixels {
        let tileSize = max(1, spatialTileSizeInPixels / spatialScale)
        let overlap = max(0, spatialOverlapInPixels / spatialScale)
        heightIntervals = splitInSpatial(size: tileSize, overlap: overlap, dimensionSize: latentH)
        widthIntervals = splitInSpatial(size: tileSize, overlap: overlap, dimensionSize: latentW)
    } else {
        heightIntervals = LTXIntervals(starts: [0], ends: [latentH], leftRamps: [0], rightRamps: [0])
        widthIntervals = LTXIntervals(starts: [0], ends: [latentW], leftRamps: [0], rightRamps: [0])
    }

    let totalRGB = 3 * outFrames * outH * outW
    let accumulatorDType: DType = totalRGB >= 128_000_000 ? .float16 : .float32
    let output = MLX.zeros([1, 3, outFrames, outH, outW], dtype: accumulatorDType)
    let weights = MLX.zeros([1, 1, outFrames, outH, outW], dtype: accumulatorDType)

    for tIndex in 0..<temporalIntervals.starts.count {
        let tStart = temporalIntervals.starts[tIndex]
        let tEnd = temporalIntervals.ends[tIndex]
        let temporalOutput = mapTemporalOutputSlice(
            begin: tStart,
            end: tEnd,
            leftRamp: temporalIntervals.leftRamps[tIndex],
            rightRamp: temporalIntervals.rightRamps[tIndex],
            scale: temporalScale
        )
        let outTStart = temporalOutput.start
        let expectedOutT = temporalOutput.length
        let tMask = temporalOutput.mask

        for hIndex in 0..<heightIntervals.starts.count {
            let hStart = heightIntervals.starts[hIndex]
            let hEnd = heightIntervals.ends[hIndex]
            let heightOutput = mapSpatialOutputSlice(
                begin: hStart,
                end: hEnd,
                leftRamp: heightIntervals.leftRamps[hIndex],
                rightRamp: heightIntervals.rightRamps[hIndex],
                scale: spatialScale
            )
            let outHStart = heightOutput.start
            let expectedOutH = heightOutput.length
            let hMask = heightOutput.mask

            for wIndex in 0..<widthIntervals.starts.count {
                let wStart = widthIntervals.starts[wIndex]
                let wEnd = widthIntervals.ends[wIndex]
                let widthOutput = mapSpatialOutputSlice(
                    begin: wStart,
                    end: wEnd,
                    leftRamp: widthIntervals.leftRamps[wIndex],
                    rightRamp: widthIntervals.rightRamps[wIndex],
                    scale: spatialScale
                )
                let outWStart = widthOutput.start
                let expectedOutW = widthOutput.length
                let wMask = widthOutput.mask

                let tileLatents = latents[0..., 0..., tStart..<tEnd, hStart..<hEnd, wStart..<wEnd]
                let tileDecoded = decoder.decode(sample: tileLatents, timestep: nil).asType(.float32)
                accumulateLTXDecodedTile(
                    output: output,
                    weights: weights,
                    tileDecoded: tileDecoded,
                    outputFrameStart: outTStart,
                    outputHeightStart: outHStart,
                    outputWidthStart: outWStart,
                    temporalMask: Array(tMask.prefix(expectedOutT)),
                    heightMask: Array(hMask.prefix(expectedOutH)),
                    widthMask: Array(wMask.prefix(expectedOutW))
                )
                MLX.eval(output, weights)

                Memory.clearCache()
            }
        }
    }

    let decoded = finalizeLTXDecodedTilesRaw(output: output, weights: weights)
    MLX.eval(decoded)
    return decoded
}
