import Foundation
import MLX
import MLXFast
import MLXNN

package func makeLTXVideoKeyframesMask(
    batchSize: Int,
    tokenCount: Int,
    tokensPerFirstFrame: Int,
    dtype: DType
) -> MLXArray {
    precondition(batchSize > 0, "batchSize must be positive")
    precondition(tokenCount > 0, "tokenCount must be positive")
    precondition(
        tokensPerFirstFrame > 0 && tokensPerFirstFrame <= tokenCount,
        "tokensPerFirstFrame must fit in tokenCount"
    )
    let firstFrame = MLX.ones([batchSize, tokensPerFirstFrame, 1], dtype: dtype)
    guard tokensPerFirstFrame < tokenCount else { return firstFrame }
    let remainder = MLX.zeros(
        [batchSize, tokenCount - tokensPerFirstFrame, 1],
        dtype: dtype
    )
    return MLX.concatenated([firstFrame, remainder], axis: 1)
}

package func createPositionGrid(
    batchSize: Int,
    numFrames: Int,
    height: Int,
    width: Int,
    temporalScale: Int,
    spatialScale: Int,
    fps: Float,
    causalFix: Bool
) -> MLXArray {
    let tokenCount = numFrames * height * width
    var data = [Float](repeating: 0, count: batchSize * 3 * tokenCount * 2)

    for b in 0..<batchSize {
        var token = 0
        for t in 0..<numFrames {
            for h in 0..<height {
                for w in 0..<width {
                    let pixelT0 = Float(t * temporalScale)
                    let pixelT1 = Float((t + 1) * temporalScale)
                    let pixelH0 = Float(h * spatialScale)
                    let pixelH1 = Float((h + 1) * spatialScale)
                    let pixelW0 = Float(w * spatialScale)
                    let pixelW1 = Float((w + 1) * spatialScale)

                    let base = ((b * 3 * tokenCount) + token) * 2

                    var t0 = pixelT0
                    var t1 = pixelT1
                    if causalFix {
                        let shift = Float(1 - temporalScale)
                        t0 = max(0, t0 + shift)
                        t1 = max(0, t1 + shift)
                    }
                    t0 /= fps
                    t1 /= fps

                    data[base] = t0
                    data[base + 1] = t1

                    let hBase = ((b * 3 * tokenCount) + tokenCount + token) * 2
                    data[hBase] = pixelH0
                    data[hBase + 1] = pixelH1

                    let wBase = ((b * 3 * tokenCount) + (2 * tokenCount) + token) * 2
                    data[wBase] = pixelW0
                    data[wBase + 1] = pixelW1

                    token += 1
                }
            }
        }
    }

    return MLXArray(data).reshaped(batchSize, 3, tokenCount, 2)
}

package func getTimestepEmbedding(
    timesteps: MLXArray,
    embeddingDim: Int,
    flipSinToCos: Bool,
    downscaleFreqShift: Float,
    scale: Float,
    maxPeriod: Float
) -> MLXArray {
    let halfDim = embeddingDim / 2
    let exponent = -Foundation.log(maxPeriod) * MLXArray(0..<halfDim).asType(.float32)
        / MLXArray(Float(halfDim) - downscaleFreqShift)
    let emb = exp(exponent)
    let timestepExpanded = timesteps.asType(.float32).reshaped(-1, 1)
    let args = timestepExpanded * emb.reshaped(1, halfDim) * MLXArray(scale)

    let sinPart = MLX.sin(args)
    let cosPart = MLX.cos(args)
    var output = flipSinToCos
        ? MLX.concatenated([cosPart, sinPart], axis: -1)
        : MLX.concatenated([sinPart, cosPart], axis: -1)

    if embeddingDim % 2 == 1 {
        output = padded(output, widths: [[0, 0], [0, 1]])
    }
    return output
}
