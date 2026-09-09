import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func ltxAncestralEulerStep(
    sample: MLXArray,
    denoised: MLXArray,
    sigma: Float,
    nextSigma: Float,
    noise: MLXArray,
    eta: Float = 1
) -> MLXArray {
    guard nextSigma > 0 else {
        return denoised.asType(sample.dtype)
    }

    let downstepRatio = 1 + (nextSigma / sigma - 1) * eta
    let sigmaDown = nextSigma * downstepRatio
    let sigmaDownRatio = sigmaDown / sigma
    let alphaNext = 1.0 - nextSigma
    let alphaDown = 1.0 - sigmaDown
    let alphaRatio = alphaNext / alphaDown
    let variance = max(0, nextSigma * nextSigma - sigmaDown * sigmaDown * alphaRatio * alphaRatio)
    let renoiseCoefficient = sqrt(variance)

    let sample32 = sample.asType(.float32)
    let denoised32 = denoised.asType(.float32)
    var next = MLXArray(sigmaDownRatio) * sample32
        + MLXArray(1.0 - sigmaDownRatio) * denoised32
    next = MLXArray(alphaRatio) * next + MLXArray(renoiseCoefficient) * noise.asType(.float32)
    return next.asType(sample.dtype)
}

func denoiseLTX25VideoTokenLoop(
    videoState: LTX25VideoTokenState,
    videoRope: LTXRope,
    videoContext: MLXArray,
    transformer: LTXUnifiedAVTransformerV2,
    sigmas: [Float],
    ancestralNoiseSeed: Int,
    ancestralEta: Float
) -> LTX25VideoTokenState {
    var current = videoState
    let dtype = videoState.latent.dtype
    MLXRandom.seed(UInt64(bitPattern: Int64(ancestralNoiseSeed)))

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let timesteps = current.denoiseMask.squeezed(axis: -1)
            * MLXArray(sigma).asType(dtype)
        let globalTimestep = MLX.full(
            [current.targetShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )
        let velocity = transformer.forwardVideoOnly(
            videoLatent: current.latent,
            videoKeyframesMask: current.keyframesMask,
            videoAttentionMask: current.attentionMask,
            timestep: globalTimestep,
            videoTimesteps: timesteps,
            videoContext: videoContext,
            videoRope: videoRope
        )
        var denoised = toDenoised(noisy: current.latent, velocity: velocity, sigma: sigma)
        let one = MLXArray(1).asType(dtype)
        denoised = denoised * current.denoiseMask + current.cleanLatent * (one - current.denoiseMask)
        if nextSigma > 0 {
            current.latent = ltxAncestralEulerStep(
                sample: current.latent,
                denoised: denoised,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(current.latent.shape).asType(dtype),
                eta: ancestralEta
            )
            current.latent = current.latent * current.denoiseMask
                + current.cleanLatent * (one - current.denoiseMask)
        } else {
            current.latent = denoised
        }
        MLX.eval(current.latent)
    }
    return current
}

func splitLTXByCount(
    numTiles requestedTiles: Int,
    overlap requestedOverlap: Int,
    dimensionSize: Int
) -> LTXIntervals {
    guard requestedTiles > 1, dimensionSize > 1 else {
        return LTXIntervals(
            starts: [0],
            ends: [dimensionSize],
            leftRamps: [0],
            rightRamps: [0]
        )
    }
    let numTiles = min(requestedTiles, dimensionSize)
    let overlap = min(requestedOverlap, max(0, dimensionSize - numTiles))
    let total = dimensionSize + overlap * (numTiles - 1)
    let tileSize = total / numTiles
    guard tileSize > overlap else {
        return LTXIntervals(
            starts: [0],
            ends: [dimensionSize],
            leftRamps: [0],
            rightRamps: [0]
        )
    }
    let remainder = total % numTiles
    let base = splitInSpatial(
        size: tileSize,
        overlap: overlap,
        dimensionSize: dimensionSize - remainder
    )
    var starts: [Int] = []
    var ends: [Int] = []
    var leftRamps: [Int] = []
    var rightRamps: [Int] = []
    for index in base.starts.indices {
        let shift = min(index, remainder)
        let grow = index < remainder ? 1 : 0
        starts.append(base.starts[index] + shift)
        ends.append(base.ends[index] + shift + grow)
        leftRamps.append(base.leftRamps[index])
        rightRamps.append(base.rightRamps[index])
    }
    return LTXIntervals(
        starts: starts,
        ends: ends,
        leftRamps: leftRamps,
        rightRamps: rightRamps
    )
}
