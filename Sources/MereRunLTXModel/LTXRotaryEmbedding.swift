import Foundation
import MLX
import MLXFast
import MLXNN

package func rmsNormNoWeight(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    let normalized = x32 * rsqrt(variance + MLXArray(eps))
    return normalized.asType(dtype)
}

package func applySplitRoPEHeads(
    _ x: MLXArray,
    cosFreq: MLXArray,
    sinFreq: MLXArray
) -> MLXArray {
    let dtype = x.dtype
    let cos = cosFreq.asType(dtype)
    let sin = sinFreq.asType(dtype)

    let halfDim = x.dim(3) / 2
    let x1 = x[0..., 0..., 0..., 0..<halfDim]
    let x2 = x[0..., 0..., 0..., halfDim...]

    let out1 = x1 * cos - sin * x2
    let out2 = x1 * sin + x2 * cos

    return MLX.concatenated([out1, out2], axis: 3)
}

package func precomputeSplitRope(
    positions: MLXArray,
    dim: Int,
    theta: Float,
    maxPos: [Int],
    numHeads: Int
) -> (cos: MLXArray, sin: MLXArray) {
    let batch = positions.dim(0)
    let positionDims = positions.dim(1)
    let tokenCount = positions.dim(2)

    let nElem = 2 * positionDims
    let indexCount = max(1, dim / nElem)
    let expectedFreqs = dim / 2
    let currentFreqs = positionDims * indexCount
    let padSize = max(0, expectedFreqs - currentFreqs)
    let halfHead = expectedFreqs / numHeads

    let grid = indexCount == 1
        ? MLXArray([Float(1)])
        : linspace(Float(0), Float(1), count: indexCount).asType(.float32)
    let indices = MLX.pow(MLXArray(theta), grid) * MLXArray(Float.pi / 2)

    let position32 = positions.asType(.float32)
    let middle = (
        position32[0..., 0..., 0..., 0]
            + position32[0..., 0..., 0..., 1]
    ) * MLXArray(Float(0.5))
    let tokenMajor = middle.transposed(0, 2, 1)
    let maxima = MLXArray(maxPos.map(Float.init)).reshaped(1, 1, positionDims)
    let fractional = tokenMajor / maxima
    var freqs = (
        indices.reshaped(1, 1, 1, indexCount)
            * (fractional.expandedDimensions(axis: 3) * MLXArray(Float(2)) - MLXArray(Float(1)))
    ).transposed(0, 1, 3, 2).reshaped(batch, tokenCount, currentFreqs)
    if padSize > 0 {
        freqs = MLX.concatenated([
            MLX.zeros([batch, tokenCount, padSize], dtype: .float32),
            freqs,
        ], axis: 2)
    }

    let cos = MLX.cos(freqs).reshaped(batch, tokenCount, numHeads, halfHead).transposed(0, 2, 1, 3)
    let sin = MLX.sin(freqs).reshaped(batch, tokenCount, numHeads, halfHead).transposed(0, 2, 1, 3)
    return (cos, sin)
}
