import Foundation
import MLX

public struct HiDreamO1Scheduler {
    public static let devTimesteps: [Int] = [
        999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764,
        737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
    ]

    public let timesteps: MLXArray
    public let sigmas: MLXArray
    public let timestepValues: [Float]
    public let sigmaValues: [Float]
    public let noiseScales: [Float]
    public let variant: MereRunModelManifest.Variant

    public init(
        steps: Int,
        variant: MereRunModelManifest.Variant,
        shift: Float,
        noiseScaleStart: Float = 7.5,
        noiseScaleEnd: Float = 7.5
    ) {
        let resolvedSteps = max(1, steps)
        let timestepValues: [Float]
        let shiftedSigmas: [Float]
        if variant == .distilled {
            let devSteps = Array(Self.devTimesteps.prefix(resolvedSteps)).map(Float.init)
            timestepValues = devSteps
            shiftedSigmas = devSteps.map { $0 / 1_000.0 }
        } else {
            let trainingSigmaMax = Self.shiftedSigma(0.999, shift: shift)
            let rawSigmas = (0..<resolvedSteps).map { index in
                trainingSigmaMax * (1.0 - Float(index) / Float(resolvedSteps))
            }
            shiftedSigmas = rawSigmas.map { Self.shiftedSigma($0, shift: shift) }
            timestepValues = shiftedSigmas.map { floor($0 * 1_000.0) }
        }

        let terminalSigmas = shiftedSigmas + [0.0]

        if resolvedSteps > 1 {
            self.noiseScales = (0..<resolvedSteps).map { index in
                noiseScaleStart + (noiseScaleEnd - noiseScaleStart) * Float(index) / Float(resolvedSteps - 1)
            }
        } else {
            self.noiseScales = [noiseScaleStart]
        }
        self.variant = variant
        self.timestepValues = timestepValues
        self.sigmaValues = terminalSigmas
        self.timesteps = MLXArray(timestepValues).asType(.float32)
        self.sigmas = MLXArray(terminalSigmas).asType(.float32)
    }

    public var usesFlashStep: Bool {
        variant == .distilled
    }

    public func sigma(at index: Int) -> MLXArray {
        sigmas[index]
    }

    public func timestep(at index: Int) -> MLXArray {
        timesteps[index]
    }

    private static func shiftedSigma(_ sigma: Float, shift: Float) -> Float {
        shift * sigma / (1.0 + (shift - 1.0) * sigma)
    }
}

struct HiDreamO1UniPCState {
    private let scheduler: HiDreamO1Scheduler
    private let solverOrder = 2
    private var modelOutputs: [MLXArray?]
    private var lastSample: MLXArray?
    private var lowerOrderNumbers = 0
    private var lastOrder = 1

    init(scheduler: HiDreamO1Scheduler) {
        self.scheduler = scheduler
        self.modelOutputs = Array(repeating: nil, count: solverOrder)
    }

    mutating func step(modelOutput: MLXArray, sample: MLXArray, stepIndex: Int) -> MLXArray {
        let converted = convertModelOutput(modelOutput: modelOutput, sample: sample, stepIndex: stepIndex)
        var workingSample = sample

        if stepIndex > 0, let previousSample = lastSample, let previousOutput = modelOutputs.last ?? nil {
            workingSample = correct(
                thisModelOutput: converted,
                previousModelOutput: previousOutput,
                lastSample: previousSample,
                thisSample: workingSample,
                stepIndex: stepIndex,
                order: lastOrder
            )
        }

        for index in 0..<(solverOrder - 1) {
            modelOutputs[index] = modelOutputs[index + 1]
        }
        modelOutputs[solverOrder - 1] = converted

        let remainingSteps = scheduler.timestepValues.count - stepIndex
        let thisOrder = max(1, min(solverOrder, remainingSteps, lowerOrderNumbers + 1))
        lastOrder = thisOrder
        lastSample = workingSample

        let previous = predict(sample: workingSample, stepIndex: stepIndex, order: thisOrder)
        if lowerOrderNumbers < solverOrder {
            lowerOrderNumbers += 1
        }
        return previous
    }

    private func convertModelOutput(modelOutput: MLXArray, sample: MLXArray, stepIndex: Int) -> MLXArray {
        sample - modelOutput * MLXArray(scheduler.sigmaValues[stepIndex])
    }

    private func predict(sample: MLXArray, stepIndex: Int, order: Int) -> MLXArray {
        guard let m0 = modelOutputs.last ?? nil else {
            return sample
        }
        let sigmaT = scheduler.sigmaValues[stepIndex + 1]
        let sigmaS0 = scheduler.sigmaValues[stepIndex]
        let alphaT = 1.0 - sigmaT
        let alphaS0 = 1.0 - sigmaS0
        let h = lambda(alpha: alphaT, sigma: sigmaT) - lambda(alpha: alphaS0, sigma: sigmaS0)
        let terms = bhTerms(h: h)
        let base = sample * MLXArray(sigmaT / sigmaS0) - m0 * MLXArray(alphaT * terms.hPhi1)

        guard order > 1,
              let previous = modelOutputs.first ?? nil else {
            return base
        }

        let sigmaSI = scheduler.sigmaValues[max(0, stepIndex - 1)]
        let lambdaSI = lambda(alpha: 1.0 - sigmaSI, sigma: sigmaSI)
        let lambdaS0 = lambda(alpha: alphaS0, sigma: sigmaS0)
        let rk = (lambdaSI - lambdaS0) / h
        let d1 = (previous - m0) / MLXArray(rk)
        return base - d1 * MLXArray(alphaT * terms.bH * 0.5)
    }

    private func correct(
        thisModelOutput: MLXArray,
        previousModelOutput m0: MLXArray,
        lastSample: MLXArray,
        thisSample _: MLXArray,
        stepIndex: Int,
        order: Int
    ) -> MLXArray {
        let sigmaT = scheduler.sigmaValues[stepIndex]
        let sigmaS0 = scheduler.sigmaValues[stepIndex - 1]
        let alphaT = 1.0 - sigmaT
        let alphaS0 = 1.0 - sigmaS0
        let h = lambda(alpha: alphaT, sigma: sigmaT) - lambda(alpha: alphaS0, sigma: sigmaS0)
        let terms = bhTerms(h: h)
        let base = lastSample * MLXArray(sigmaT / sigmaS0) - m0 * MLXArray(alphaT * terms.hPhi1)
        let d1Current = thisModelOutput - m0

        guard order > 1,
              let olderModelOutput = modelOutputs.first ?? nil else {
            return base - d1Current * MLXArray(alphaT * terms.bH * 0.5)
        }

        let sigmaSI = scheduler.sigmaValues[max(0, stepIndex - 2)]
        let lambdaSI = lambda(alpha: 1.0 - sigmaSI, sigma: sigmaSI)
        let lambdaS0 = lambda(alpha: alphaS0, sigma: sigmaS0)
        let rk = (lambdaSI - lambdaS0) / h
        let d1Previous = (olderModelOutput - m0) / MLXArray(rk)
        let rhos = correctorRhos(h: h, rk: rk, terms: terms)
        let correction = d1Previous * MLXArray(rhos.previous) + d1Current * MLXArray(rhos.current)
        return base - correction * MLXArray(alphaT * terms.bH)
    }

    private func lambda(alpha: Float, sigma: Float) -> Float {
        log(alpha) - log(sigma)
    }

    private func bhTerms(h: Float) -> (hPhi1: Float, bH: Float, hPhi2: Float) {
        let hh = -h
        let hPhi1 = expm1(hh)
        let bH = hPhi1
        let hPhi2 = hPhi1 / hh - 1.0
        return (hPhi1, bH, hPhi2)
    }

    private func correctorRhos(
        h: Float,
        rk: Float,
        terms: (hPhi1: Float, bH: Float, hPhi2: Float)
    ) -> (previous: Float, current: Float) {
        let hh = -h
        var hPhiK = terms.hPhi2
        var factorial: Float = 1.0
        var b: [Float] = []
        for _ in 1...2 {
            b.append(hPhiK * factorial / terms.bH)
            factorial *= Float(b.count + 2)
            hPhiK = hPhiK / hh - 1.0 / factorial
        }
        let determinant = 1.0 - rk
        guard abs(determinant) > 1e-6 else {
            return (0.5, 0.5)
        }
        return (
            previous: (b[0] - b[1]) / determinant,
            current: (b[1] - rk * b[0]) / determinant
        )
    }
}
