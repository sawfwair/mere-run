import Foundation
import MLX

/// Native diffusion samplers exposed by the official LTX-2 pipeline.
public enum LTXSamplerMode: String, Sendable, CaseIterable {
    case euler
    case res2s
    case eulerAncestral = "euler-ancestral"
    case cfgPlusPlus = "cfg-plus-plus"
    case gradientEstimatingEuler = "gradient-estimating-euler"
}

/// Controls shared by the native LTX-2 sampling loops.
public struct LTXSamplerConfiguration: Sendable, Hashable {
    public var mode: LTXSamplerMode
    public var eta: Float
    public var noiseSeedOffset: Int
    public var substepNoiseSeedOffset: Int
    public var res2sBongMath: Bool
    public var res2sBongMathMaxIterations: Int
    public var gradientEstimationGamma: Float

    public init(
        mode: LTXSamplerMode = .euler,
        eta: Float = 0.5,
        noiseSeedOffset: Int = 10_000,
        substepNoiseSeedOffset: Int = 20_000,
        res2sBongMath: Bool = true,
        res2sBongMathMaxIterations: Int = 100,
        gradientEstimationGamma: Float = 2
    ) {
        self.mode = mode
        self.eta = eta
        self.noiseSeedOffset = noiseSeedOffset
        self.substepNoiseSeedOffset = substepNoiseSeedOffset
        self.res2sBongMath = res2sBongMath
        self.res2sBongMathMaxIterations = res2sBongMathMaxIterations
        self.gradientEstimationGamma = gradientEstimationGamma
    }

    /// Official LTX-2.3/2.5 HQ recipe. Upstream deliberately keeps its
    /// full-step and midpoint SDE streams independent of the initial-latent seed.
    public static let hq = LTXSamplerConfiguration(
        mode: .res2s,
        eta: 0.5,
        noiseSeedOffset: -1,
        substepNoiseSeedOffset: 9_999
    )
}

public enum LTXGenerationPreset: String, Sendable, CaseIterable {
    case standard
    case hq
}

/// Official full-checkpoint diffusion topology.
public enum LTXGenerationPipeline: String, Sendable, CaseIterable {
    case twoStage = "two-stage"
    case keyframeInterpolation = "keyframe-interpolation"
    case devOneStage = "dev-one-stage"
}

public struct LTXRes2sCoefficients: Sendable, Hashable {
    public let a21: Double
    public let b1: Double
    public let b2: Double
}

/// Exact scalar coefficient helpers from the official Res2s implementation.
public enum LTXRes2s {
    public static func phi(order: Int, negativeStep: Double) -> Double {
        precondition(order > 0, "The phi-function order must be positive.")
        if abs(negativeStep) < 1e-10 {
            return 1 / Double(factorial(order))
        }
        var remainder = 0.0
        for index in 0..<order {
            remainder += pow(negativeStep, Double(index)) / Double(factorial(index))
        }
        return (exp(negativeStep) - remainder) / pow(negativeStep, Double(order))
    }

    public static func coefficients(step: Double, midpoint: Double = 0.5) -> LTXRes2sCoefficients {
        precondition(midpoint > 0, "The Res2s midpoint must be positive.")
        let a21 = midpoint * phi(order: 1, negativeStep: -step * midpoint)
        let b2 = phi(order: 2, negativeStep: -step) / midpoint
        let b1 = phi(order: 1, negativeStep: -step) - b2
        return LTXRes2sCoefficients(a21: a21, b1: b1, b2: b2)
    }

    private static func factorial(_ value: Int) -> Int {
        guard value > 1 else { return 1 }
        return (2...value).reduce(1, *)
    }
}

func ltxEulerStep(
    sample: MLXArray,
    denoised: MLXArray,
    sigma: Float,
    nextSigma: Float
) -> MLXArray {
    let sample32 = sample.asType(.float32)
    let denoised32 = denoised.asType(.float32)
    let velocity = (sample32 - denoised32) / MLXArray(sigma)
    return (sample32 + velocity * MLXArray(nextSigma - sigma)).asType(sample.dtype)
}

func ltxCfgPlusPlusStep(
    sample: MLXArray,
    denoised: MLXArray,
    unconditionalDenoised: MLXArray,
    sigma: Float,
    nextSigma: Float,
    eta: Float,
    noise: MLXArray?
) -> MLXArray {
    let epsilon = Float.ulpOfOne
    let alpha = max(epsilon, 1 - sigma)
    let nextAlpha = max(epsilon, 1 - nextSigma)
    let sample32 = sample.asType(.float32)
    let direction = (
        sample32 - MLXArray(alpha) * unconditionalDenoised.asType(.float32)
    ) / MLXArray(sigma)

    let sigmaFrom = sigma / alpha
    let sigmaTo = nextSigma / nextAlpha
    let variance = sigmaTo * sigmaTo
        * max(0, sigmaFrom * sigmaFrom - sigmaTo * sigmaTo)
        / max(epsilon, sigmaFrom * sigmaFrom)
    let sigmaUp = min(sigmaTo, eta * sqrt(variance))
    let sigmaDown = sqrt(max(0, sigmaTo * sigmaTo - sigmaUp * sigmaUp)) * nextAlpha
    var next = MLXArray(nextAlpha) * denoised.asType(.float32)
        + MLXArray(sigmaDown) * direction
    if let noise, eta > 0 {
        next = next + MLXArray(nextAlpha * sigmaUp) * noise.asType(.float32)
    }
    return next.asType(sample.dtype)
}

func ltxRes2sSDEStep(
    sample: MLXArray,
    denoised: MLXArray,
    sigma: Float,
    nextSigma: Float,
    eta: Float,
    noise: MLXArray
) -> MLXArray {
    guard nextSigma > 0 else { return denoised.asType(sample.dtype) }
    let sigmaUp = min(nextSigma * 0.9999, nextSigma * eta)
    let sigmaResidual = sqrt(max(0, nextSigma * nextSigma - sigmaUp * sigmaUp))
    let alphaRatio = 1 - nextSigma + sigmaResidual
    let sigmaDown = sigmaResidual / alphaRatio
    let epsilon = (
        sample.asType(.float32) - denoised.asType(.float32)
    ) / MLXArray(sigma - nextSigma)
    let denoisedNext = sample.asType(.float32) - MLXArray(sigma) * epsilon
    return (
        MLXArray(alphaRatio) * (denoisedNext + MLXArray(sigmaDown) * epsilon)
            + MLXArray(sigmaUp) * noise.asType(.float32)
    ).asType(sample.dtype)
}
