import MLX
import MLXNN

public final class SharpGaussianComposer {
    public let params: SharpComposerParameters

    public init(params: SharpComposerParameters = SharpComposerParameters()) {
        self.params = params
    }

    public func callAsFunction(
        delta: MLXArray,
        baseValues: SharpGaussianBaseValues,
        globalScale: MLXArray? = nil,
        flattenOutput: Bool = true
    ) -> SharpGaussians3D {
        var workingDelta = delta

        let actualScaleFactor = baseValues.meanXNDC.dim(4) / delta.dim(4)
        if params.scaleFactor != 1 && actualScaleFactor != 1 {
            workingDelta = Self.upsampleDelta(workingDelta, scaleFactor: params.scaleFactor)
        }

        let meanVectors = forwardMean(baseValues: baseValues, delta: workingDelta)

        let baseScales: MLXArray
        if params.baseScaleOnPredictedMean {
            let predictedZ = meanVectors[0..., 2..<3, 0..., 0..., 0...]
            baseScales = baseValues.scales * baseValues.meanInverseZNDC * predictedZ
        } else {
            baseScales = baseValues.scales
        }

        let singularValues = scaleActivation(
            base: baseScales,
            learnedDelta: workingDelta[0..., 3..<6, 0..., 0..., 0...]
        )
        let quaternions = baseValues.quaternions +
            (params.deltaFactor.quaternion * workingDelta[0..., 6..<10, 0..., 0..., 0...])
        let colors = colorActivation(
            base: baseValues.colors,
            learnedDelta: workingDelta[0..., 10..<13, 0..., 0..., 0...]
        )
        let opacities = opacityActivation(
            base: baseValues.opacities,
            learnedDelta: workingDelta[0..., 13, 0..., 0..., 0...]
        )

        var flatMean = meanVectors
        var flatScales = singularValues
        var flatQuaternions = quaternions
        var flatColors = colors
        var flatOpacities = opacities

        if flattenOutput {
            let batch = meanVectors.dim(0)
            let layers = meanVectors.dim(2)
            let height = meanVectors.dim(3)
            let width = meanVectors.dim(4)
            let count = layers * height * width

            flatMean = meanVectors.transposed(0, 2, 3, 4, 1).reshaped(batch, count, 3)
            flatScales = singularValues.transposed(0, 2, 3, 4, 1).reshaped(batch, count, 3)
            flatQuaternions = quaternions.transposed(0, 2, 3, 4, 1).reshaped(batch, count, 4)
            flatColors = colors.transposed(0, 2, 3, 4, 1).reshaped(batch, count, 3)
            flatOpacities = opacities.flattened(start: 1, end: 3)
        }

        if let globalScale {
            let scale = globalScale.reshaped(globalScale.dim(0), 1, 1)
            flatMean = flatMean * scale
            flatScales = flatScales * scale
        }

        return SharpGaussians3D(
            meanVectors: flatMean,
            singularValues: flatScales,
            quaternions: flatQuaternions,
            colors: flatColors,
            opacities: flatOpacities
        )
    }
}

extension SharpGaussianComposer {
    private static func upsampleDelta(_ delta: MLXArray, scaleFactor: Int) -> MLXArray {
        guard scaleFactor > 1 else { return delta }

        let batch = delta.dim(0)
        let channels = delta.dim(1)
        let layers = delta.dim(2)
        let height = delta.dim(3)
        let width = delta.dim(4)

        var merged = delta.reshaped(batch, channels * layers, height, width)
        merged = merged.expandedDimensions(axis: 3)
        merged = MLX.repeated(merged, count: scaleFactor, axis: 3)
        merged = merged.reshaped(batch, channels * layers, height * scaleFactor, width)
        merged = merged.expandedDimensions(axis: 4)
        merged = MLX.repeated(merged, count: scaleFactor, axis: 4)
        merged = merged.reshaped(batch, channels * layers, height * scaleFactor, width * scaleFactor)

        return merged.reshaped(batch, channels, layers, height * scaleFactor, width * scaleFactor)
    }

    private func forwardMean(baseValues: SharpGaussianBaseValues, delta: MLXArray) -> MLXArray {
        let dtype = baseValues.meanXNDC.dtype

        let deltaFactor = MLXArray(
            [params.deltaFactor.xy, params.deltaFactor.xy, params.deltaFactor.z],
            [1, 3, 1, 1, 1]
        ).asType(dtype)

        let meanXMask = MLXArray([Float(1), Float(0), Float(0)], [1, 3, 1, 1, 1]).asType(dtype)
        let meanYMask = MLXArray([Float(0), Float(1), Float(0)], [1, 3, 1, 1, 1]).asType(dtype)
        let meanZMask = MLXArray([Float(0), Float(0), Float(1)], [1, 3, 1, 1, 1]).asType(dtype)

        let base =
            baseValues.meanXNDC * meanXMask +
            baseValues.meanYNDC * meanYMask +
            baseValues.meanInverseZNDC * meanZMask

        let learnedDelta = deltaFactor * delta[0..., 0..<3, 0..., 0..., 0...]
        return meanActivation(base: base, learnedDelta: learnedDelta)
    }

    private func meanActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        let xx = base[0..., 0..<1, 0..., 0..., 0...] + learnedDelta[0..., 0..<1, 0..., 0..., 0...]
        let yy = base[0..., 1..<2, 0..., 0..., 0...] + learnedDelta[0..., 1..<2, 0..., 0..., 0...]
        let a = base[0..., 2..<3, 0..., 0..., 0...]
        let b = learnedDelta[0..., 2..<3, 0..., 0..., 0...]

        let inverseZZ = softplus(SharpMath.inverseSoftplus(a) + b)
        let zz = 1.0 / (inverseZZ + 1e-3)

        return MLX.concatenated([zz * xx, zz * yy, zz], axis: 1)
    }

    private func scaleActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        let constants = scaleActivationConstants(maxScale: params.maxScale, minScale: params.minScale)
        let a = MLXArray(constants.a).asType(learnedDelta.dtype)
        let b = MLXArray(constants.b).asType(learnedDelta.dtype)
        let minScale = MLXArray(params.minScale).asType(learnedDelta.dtype)
        let maxScale = MLXArray(params.maxScale).asType(learnedDelta.dtype)
        let deltaScale = MLXArray(params.deltaFactor.scale).asType(learnedDelta.dtype)

        let scaleFactor = (maxScale - minScale) * sigmoid((a * deltaScale * learnedDelta) + b) + minScale
        return base * scaleFactor
    }

    private func colorActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        var adjustedBase = base
        switch params.colorActivationType {
        case .sigmoid:
            adjustedBase = MLX.clip(adjustedBase, min: 0.01, max: 0.99)
        case .exp, .softplus:
            adjustedBase = MLX.clip(adjustedBase, min: 0.01)
        case .linear, .reluWithPushback, .hardSigmoidWithPushback:
            break
        }

        let inv = SharpMath.activationInverse(params.colorActivationType, adjustedBase)
        let deltaColor = MLXArray(params.deltaFactor.color).asType(learnedDelta.dtype)
        var colors = SharpMath.activationForward(params.colorActivationType, inv + (deltaColor * learnedDelta))
        if params.colorSpace == .linearRGB {
            colors = SharpColorSpaceOps.sRGBToLinearRGB(colors)
        }
        return colors
    }

    private func opacityActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        let inv = SharpMath.activationInverse(params.opacityActivationType, base)
        let deltaOpacity = MLXArray(params.deltaFactor.opacity).asType(learnedDelta.dtype)
        return SharpMath.activationForward(params.opacityActivationType, inv + (deltaOpacity * learnedDelta))
    }

    private func scaleActivationConstants(maxScale: Float, minScale: Float) -> (a: Float, b: Float) {
        let one: Float = 1.0
        let constantA = (maxScale - minScale) / (one - minScale) / (maxScale - one)
        let t = (one - minScale) / (maxScale - minScale)
        let constantB = SharpMath.inverseSigmoid(MLXArray(t)).item(Float.self)
        return (constantA, constantB)
    }
}
