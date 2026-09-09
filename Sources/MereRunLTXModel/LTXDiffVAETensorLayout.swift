import Foundation
import MLX
import MLXFast
import MLXNN

func ltxDiffVAENeighborhoodMask(
    queryRanges: [Range<Int>],
    referenceRanges: [Range<Int>],
    bounds: [(starts: [Int], ends: [Int])],
    dtype: DType
) -> MLXArray {
    var visibility: MLXArray?
    for axis in 0..<3 {
        let queryIndices = MLXArray(queryRanges[axis].map(Int32.init)).reshaped(
            axis == 0 ? queryRanges[axis].count : 1,
            axis == 1 ? queryRanges[axis].count : 1,
            axis == 2 ? queryRanges[axis].count : 1,
            1,
            1,
            1
        )
        let keyIndices = MLXArray(referenceRanges[axis].map(Int32.init)).reshaped(
            1,
            1,
            1,
            axis == 0 ? referenceRanges[axis].count : 1,
            axis == 1 ? referenceRanges[axis].count : 1,
            axis == 2 ? referenceRanges[axis].count : 1
        )
        let startValues = queryRanges[axis].map { Int32(bounds[axis].starts[$0]) }
        let endValues = queryRanges[axis].map { Int32(bounds[axis].ends[$0]) }
        let starts = MLXArray(startValues).reshaped(queryIndices.shape)
        let ends = MLXArray(endValues).reshaped(queryIndices.shape)
        let current = (keyIndices .>= starts) .&& (keyIndices .< ends)
        visibility = visibility.map { $0 .&& current } ?? current
    }
    let queryCount = queryRanges.map(\.count).reduce(1, *)
    let referenceCount = referenceRanges.map(\.count).reduce(1, *)
    let visible = visibility!.reshaped(queryCount, referenceCount)
    let zero = MLX.zeros([queryCount, referenceCount], dtype: dtype)
    let negativeInfinity = MLX.full(
        [queryCount, referenceCount],
        values: MLXArray(-Float.infinity).asType(dtype)
    )
    return MLX.where(visible, zero, negativeInfinity).reshaped(1, 1, queryCount, referenceCount)
}

func ltxDiffVAEAbsoluteRoPE(_ input: MLXArray) -> MLXArray {
    let split = [16, 24, 24]
    let axes = [1, 2, 3]
    var chunks: [MLXArray] = []
    var offset = 0
    for index in 0..<3 {
        let dimension = split[index]
        let positions = MLXArray(0..<input.dim(axes[index])).asType(.float32)
        let exponents = MLXArray(Array(stride(from: 0, to: dimension, by: 2))).asType(.float32)
            / MLXArray(Float(dimension))
        let inverse = exp(-MLXArray(Float(Foundation.log(Double(10_000)))) * exponents)
        var angleShape = [1, 1, 1, 1, 1, dimension / 2]
        angleShape[axes[index]] = positions.dim(0)
        let angles = (positions.reshaped(-1, 1) * inverse.reshaped(1, -1)).reshaped(angleShape)
        let source = input[0..., 0..., 0..., 0..., 0..., offset..<(offset + dimension)]
        let pairs = source.reshaped(source.dim(0), source.dim(1), source.dim(2), source.dim(3), source.dim(4), dimension / 2, 2)
        let even = pairs[0..., 0..., 0..., 0..., 0..., 0..., 0].asType(.float32)
        let odd = pairs[0..., 0..., 0..., 0..., 0..., 0..., 1].asType(.float32)
        let cosine = MLX.cos(angles)
        let sine = MLX.sin(angles)
        chunks.append(
            MLX.stacked(
                [even * cosine - odd * sine, even * sine + odd * cosine],
                axis: -1
            ).reshaped(source.shape).asType(input.dtype)
        )
        offset += dimension
    }
    return MLX.concatenated(chunks, axis: -1)
}

struct LTXDiffVAEResizedLatent {
    let array: MLXArray
    let heightBefore: Int
    let widthBefore: Int
}

func ltxDiffVAEResizeLatentToMinimum(
    _ input: MLXArray,
    minimum: (Int, Int, Int)
) -> LTXDiffVAEResizedLatent {
    var output = input
    if output.dim(2) < minimum.0 {
        let last = output[0..., 0..., (output.dim(2) - 1)..<output.dim(2), 0..., 0...]
        output = MLX.concatenated(
            [output, MLX.repeated(last, count: minimum.0 - output.dim(2), axis: 2)],
            axis: 2
        )
    }
    var heightBefore = 0
    if output.dim(3) < minimum.1 {
        let needed = minimum.1 - output.dim(3)
        heightBefore = needed / 2
        let first = output[0..., 0..., 0..., 0..<1, 0...]
        let last = output[0..., 0..., 0..., (output.dim(3) - 1)..<output.dim(3), 0...]
        output = MLX.concatenated([
            MLX.repeated(first, count: heightBefore, axis: 3),
            output,
            MLX.repeated(last, count: needed - heightBefore, axis: 3),
        ], axis: 3)
    }
    var widthBefore = 0
    if output.dim(4) < minimum.2 {
        let needed = minimum.2 - output.dim(4)
        widthBefore = needed / 2
        let first = output[0..., 0..., 0..., 0..., 0..<1]
        let last = output[0..., 0..., 0..., 0..., (output.dim(4) - 1)..<output.dim(4)]
        output = MLX.concatenated([
            MLX.repeated(first, count: widthBefore, axis: 4),
            output,
            MLX.repeated(last, count: needed - widthBefore, axis: 4),
        ], axis: 4)
    }
    return LTXDiffVAEResizedLatent(
        array: output,
        heightBefore: heightBefore,
        widthBefore: widthBefore
    )
}

func ltxDiffVAEMinimumLatentShape(
    stageKernels: [(Int, Int, Int)],
    stageStrides: [(Int, Int, Int)],
    stage5Kernel: (Int, Int, Int)
) -> (Int, Int, Int) {
    var minimum = [1, 1, 1]
    var cumulative = [1, 1, 1]
    for index in 0..<stageStrides.count {
        let kernel = [stageKernels[index].0, stageKernels[index].1, stageKernels[index].2]
        for axis in 0..<3 {
            minimum[axis] = max(minimum[axis], (kernel[axis] + cumulative[axis] - 1) / cumulative[axis])
        }
        let stride = [stageStrides[index].0, stageStrides[index].1, stageStrides[index].2]
        for axis in 0..<3 { cumulative[axis] *= stride[axis] }
    }
    let finalKernel = [stage5Kernel.0, stage5Kernel.1, stage5Kernel.2]
    for axis in 0..<3 {
        minimum[axis] = max(minimum[axis], (finalKernel[axis] + cumulative[axis] - 1) / cumulative[axis])
    }
    return (minimum[0], minimum[1], minimum[2])
}

func ltxDiffVAEPatchifyPixels(_ input: MLXArray) -> MLXArray {
    let batch = input.dim(0)
    let channels = input.dim(1)
    let frames = input.dim(2)
    let height = input.dim(3)
    let width = input.dim(4)
    return input
        .reshaped(batch, channels, frames, height / 4, 4, width / 4, 4)
        .transposed(0, 2, 3, 5, 1, 6, 4)
        .reshaped(batch, frames, height / 4, width / 4, channels * 16)
}

func ltxDiffVAEUnpatchifyPixels(_ input: MLXArray) -> MLXArray {
    let batch = input.dim(0)
    let channels = input.dim(1) / 16
    let frames = input.dim(2)
    let height = input.dim(3)
    let width = input.dim(4)
    return input
        .reshaped(batch, channels, 4, 4, frames, height, width)
        .transposed(0, 1, 4, 5, 3, 6, 2)
        .reshaped(batch, channels, frames, height * 4, width * 4)
}

func ltxDiffVAETimestepEmbedding(_ timestep: MLXArray) -> MLXArray {
    let halfDimension = 128
    let exponent = -Foundation.log(Double(10_000)) * MLXArray(0..<halfDimension).asType(.float32)
        / MLXArray(Float(halfDimension))
    let frequencies = exp(exponent)
    let angles = timestep.asType(.float32).reshaped(-1, 1) * frequencies.reshaped(1, -1)
    return MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
}

func ltxDiffVAESiLU(_ input: MLXArray) -> MLXArray {
    input * MLX.sigmoid(input)
}
