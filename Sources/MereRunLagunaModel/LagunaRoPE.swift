import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package final class LagunaRoPE: Module, OffsetLayer {
    let dimensions: Int
    let traditional: Bool
    let magnitudeScale: Float
    let base: Float?
    let frequencies: MLXArray?
    package let prefillFusionKind: LagunaFusedPrefill.QKNormRoPEKind?

    package init(headDim: Int, parameters: LagunaRopeParameters) {
        let resolvedDimensions = max(2, Int(Float(headDim) * parameters.partialRotaryFactor))
        self.dimensions = resolvedDimensions
        self.traditional = false

        if parameters.ropeType == "yarn" {
            let factor = parameters.factor ?? 1
            let originalContext = parameters.originalMaxPositionEmbeddings ?? 4_096
            let betaFast = parameters.betaFast ?? 32
            let betaSlow = parameters.betaSlow ?? 1

            func correctionDimension(rotations: Float) -> Float {
                let numerator = Float(resolvedDimensions)
                    * log(Float(originalContext) / (rotations * 2 * Float.pi))
                return numerator / (2 * log(parameters.ropeTheta))
            }

            let low = max(Int(floor(correctionDimension(rotations: betaFast))), 0)
            let high = min(Int(ceil(correctionDimension(rotations: betaSlow))), resolvedDimensions - 1)
            let denominator: Float = low == high ? 0.001 : Float(high - low)
            let half = max(1, resolvedDimensions / 2)
            let halfIndices = MLXArray((0..<half).map(Float.init))
            let evenIndices = MLXArray(
                Array(stride(from: 0, to: resolvedDimensions, by: 2)).map(Float.init)
            )
            let frequencyExtra = MLX.pow(
                MLXArray(parameters.ropeTheta),
                evenIndices / Float(resolvedDimensions)
            )
            let frequencyInterpolated = MLXArray(factor) * frequencyExtra
            let ramp = MLX.clip(
                (halfIndices - Float(low)) / denominator,
                min: 0,
                max: 1
            )
            let frequencyMask = MLXArray(1) - ramp
            self.frequencies = (frequencyInterpolated * frequencyExtra)
                / (
                    frequencyInterpolated * frequencyMask
                        + frequencyExtra * (MLXArray(1) - frequencyMask)
                )
            // Match mlx-swift-lm's YarnRoPE runtime contract. The public
            // checkpoint carries attention_factor=1.0 as metadata, while the
            // runtime derives mscale from factor (and defaults mscale=1,
            // mscale_all_dim=0). For factor 32 the applied multiplier is
            // 1 + 0.1 * log(32), not the literal metadata field.
            self.magnitudeScale = factor <= 1 ? 1 : 0.1 * log(factor) + 1
            self.base = nil
        } else {
            self.frequencies = nil
            self.magnitudeScale = 1
            self.base = parameters.ropeTheta
        }
        if headDim == 128,
           resolvedDimensions == 64,
           parameters.ropeType == "yarn",
           parameters.factor == 32,
           parameters.originalMaxPositionEmbeddings == 8_192,
           parameters.betaFast == 64,
           parameters.betaSlow == 1,
           parameters.ropeTheta == 500_000 {
            self.prefillFusionKind = .fullYaRN
        } else if headDim == 128,
                  resolvedDimensions == 128,
                  parameters.ropeType == "default",
                  parameters.ropeTheta == 10_000 {
            self.prefillFusionKind = .sliding
        } else {
            self.prefillFusionKind = nil
        }
        super.init()
    }

    package func angleAtlas(length: Int) -> MLXArray? {
        guard prefillFusionKind != nil, length > 0 else {
            return nil
        }
        let seedMagnitude = magnitudeScale == 1 ? 1 : 1 / magnitudeScale
        let seed = MLXArray(
            Array(repeating: seedMagnitude, count: dimensions / 2)
                + Array(repeating: Float(0), count: dimensions / 2),
            [1, 1, 1, dimensions]
        )
        return callAsFunction(
            broadcast(seed, to: [1, 1, length, dimensions]),
            offset: 0
        )
    }

    package func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let input: MLXArray
        if magnitudeScale == 1 {
            input = x
        } else if dimensions < x.dim(-1) {
            input = concatenated(
                [
                    x[.ellipsis, ..<dimensions] * MLXArray(magnitudeScale).asType(x.dtype),
                    x[.ellipsis, dimensions...],
                ],
                axis: -1
            )
        } else {
            input = x * MLXArray(magnitudeScale).asType(x.dtype)
        }

        return MLXFast.RoPE(
            input,
            dimensions: dimensions,
            traditional: traditional,
            base: base,
            scale: 1,
            offset: offset,
            freqs: frequencies
        )
    }

    package func callAsFunction(_ x: MLXArray, offsets: [Int]) -> MLXArray {
        precondition(
            x.dim(0) == offsets.count,
            "Laguna RoPE requires one position offset per batch row."
        )
        guard let first = offsets.first else {
            return x
        }
        if offsets.allSatisfy({ $0 == first }) {
            return callAsFunction(x, offset: first)
        }
        return concatenated(
            offsets.enumerated().map { index, offset in
                callAsFunction(x[index..<(index + 1), 0..., 0..., 0...], offset: offset)
            },
            axis: 0
        )
    }
}
