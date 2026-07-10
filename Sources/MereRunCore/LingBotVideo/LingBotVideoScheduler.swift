// Solver ported from Robbyant LingBot-Video's Apache-2.0 Flow-UniPC scheduler.
import Foundation
import MLX

public final class LingBotVideoFlowUniPCScheduler {
    public enum SchedulerError: LocalizedError {
        case invalidStepCount(Int)
        case invalidRefinerThreshold(Float)
        case invalidTailStepCount(Int)
        case invalidSigmaSchedule
        case exhausted

        public var errorDescription: String? {
            switch self {
            case .invalidStepCount(let count):
                return "LingBot-Video inference steps must be >= 1 (got \(count))."
            case .invalidRefinerThreshold(let threshold):
                return "LingBot-Video refiner threshold must be in (0, 1] (got \(threshold))."
            case .invalidTailStepCount(let count):
                return "LingBot-Video refiner sigma tail step count must be >= 0 (got \(count))."
            case .invalidSigmaSchedule:
                return "LingBot-Video refiner sigma schedule must be finite, strictly descending, and contained in (0, 1]."
            case .exhausted:
                return "LingBot-Video scheduler received more model outputs than configured timesteps."
            }
        }
    }

    public private(set) var sigmas: [Double] = []
    public private(set) var timesteps: [Float] = []

    private let solverOrder = 2
    private var stepIndex = 0
    private var modelOutputs: [MLXArray?] = [nil, nil]
    private var lowerOrderCount = 0
    private var lastSample: MLXArray?
    private var previousOrder = 1

    public init() {}

    public func setTimesteps(stepCount: Int, shift: Float) throws {
        guard stepCount >= 1 else {
            throw SchedulerError.invalidStepCount(stepCount)
        }

        let sigmaMax = 0.999
        let sigmaMin = 0.0
        let shiftValue = Double(shift)
        var schedule: [Double] = []
        schedule.reserveCapacity(stepCount + 1)
        for index in 0..<stepCount {
            let fraction = Double(index) / Double(stepCount)
            let base = sigmaMax + (sigmaMin - sigmaMax) * fraction
            schedule.append(shiftValue * base / (1 + (shiftValue - 1) * base))
        }
        schedule.append(0)

        self.sigmas = schedule
        self.timesteps = schedule.dropLast().map { Float(Int64($0 * 1000)) }
        resetSolverState()
    }

    public func setTimesteps(sigmas: [Double]) throws {
        guard !sigmas.isEmpty,
              sigmas.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }),
              zip(sigmas, sigmas.dropFirst()).allSatisfy({ $0 > $1 })
        else {
            throw SchedulerError.invalidSigmaSchedule
        }

        self.sigmas = sigmas + [0]
        self.timesteps = sigmas.map { Float(Int64($0 * 1000)) }
        resetSolverState()
    }

    public static func refinerSigmas(
        stepCount: Int,
        shift: Float,
        threshold: Float,
        tailStepCount: Int,
        sigmaMax: Double = 0.999,
        sigmaMin: Double = 0
    ) throws -> [Double] {
        guard stepCount >= 1 else {
            throw SchedulerError.invalidStepCount(stepCount)
        }
        guard threshold > 0, threshold <= 1 else {
            throw SchedulerError.invalidRefinerThreshold(threshold)
        }
        guard tailStepCount >= 0 else {
            throw SchedulerError.invalidTailStepCount(tailStepCount)
        }

        let shiftValue = Double(shift)
        let thresholdValue = Double(threshold)
        let epsilon = 1e-6
        var schedule = (0..<stepCount).map { index in
            let fraction = Double(index) / Double(stepCount)
            let base = sigmaMax + (sigmaMin - sigmaMax) * fraction
            return shiftValue * base / (1 + (shiftValue - 1) * base)
        }.filter { $0 <= thresholdValue + epsilon }

        if schedule.isEmpty || abs(schedule[0] - thresholdValue) > epsilon {
            schedule.insert(thresholdValue, at: 0)
        }
        if tailStepCount > 0, let start = schedule.last {
            let stop = min(sigmaMin, start)
            for index in 1...tailStepCount {
                let fraction = Double(index) / Double(tailStepCount + 1)
                schedule.append(start + (stop - start) * fraction)
            }
        }

        guard schedule.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }),
              zip(schedule, schedule.dropFirst()).allSatisfy({ $0 > $1 })
        else {
            throw SchedulerError.invalidSigmaSchedule
        }
        return schedule
    }

    private func resetSolverState() {
        self.stepIndex = 0
        self.modelOutputs = [nil, nil]
        self.lowerOrderCount = 0
        self.lastSample = nil
        self.previousOrder = 1
    }

    public func step(modelOutput: MLXArray, sample: MLXArray) throws -> MLXArray {
        guard stepIndex < timesteps.count else {
            throw SchedulerError.exhausted
        }

        let sigma = sigmas[stepIndex]
        let converted = sample - MLXArray(Float(sigma)).asType(sample.dtype) * modelOutput
        var correctedSample = sample
        if stepIndex > 0, let lastSample {
            correctedSample = correct(
                modelOutput: converted,
                lastSample: lastSample,
                currentSample: sample,
                order: previousOrder
            )
        }

        modelOutputs[0] = modelOutputs[1]
        modelOutputs[1] = converted

        let remainingSteps = timesteps.count - stepIndex
        let requestedOrder = min(solverOrder, remainingSteps)
        let order = min(requestedOrder, lowerOrderCount + 1)
        previousOrder = order
        self.lastSample = correctedSample

        let previousSample = predict(sample: correctedSample, order: order)
        lowerOrderCount = min(solverOrder, lowerOrderCount + 1)
        stepIndex += 1
        return previousSample.asType(sample.dtype)
    }

    private func predict(sample: MLXArray, order: Int) -> MLXArray {
        guard let currentModel = modelOutputs[1] else {
            preconditionFailure("Flow-UniPC predictor requires the current model output.")
        }

        let sigmaSource = sigmas[stepIndex]
        let sigmaTarget = sigmas[stepIndex + 1]
        if sigmaTarget == 0 {
            return currentModel
        }

        let alphaSource = 1 - sigmaSource
        let alphaTarget = 1 - sigmaTarget
        let lambdaSource = Foundation.log(alphaSource) - Foundation.log(sigmaSource)
        let lambdaTarget = Foundation.log(alphaTarget) - Foundation.log(sigmaTarget)
        let h = lambdaTarget - lambdaSource
        let hh = -h
        let phi1 = Foundation.expm1(hh)
        let bH = phi1

        var result = sample * Float(sigmaTarget / sigmaSource)
            - currentModel * Float(alphaTarget * phi1)

        if order == 2, stepIndex > 0, let previousModel = modelOutputs[0] {
            let sigmaPrevious = sigmas[stepIndex - 1]
            let alphaPrevious = 1 - sigmaPrevious
            let lambdaPrevious = Foundation.log(alphaPrevious) - Foundation.log(sigmaPrevious)
            let rk = (lambdaPrevious - lambdaSource) / h
            let firstDerivative = (previousModel - currentModel) / Float(rk)
            result = result - firstDerivative * Float(alphaTarget * bH * 0.5)
        }
        return result
    }

    private func correct(
        modelOutput: MLXArray,
        lastSample: MLXArray,
        currentSample: MLXArray,
        order: Int
    ) -> MLXArray {
        guard let previousModel = modelOutputs[1] else {
            return currentSample
        }

        let sigmaTarget = sigmas[stepIndex]
        let sigmaSource = sigmas[stepIndex - 1]
        let alphaTarget = 1 - sigmaTarget
        let alphaSource = 1 - sigmaSource
        let lambdaTarget = Foundation.log(alphaTarget) - Foundation.log(sigmaTarget)
        let lambdaSource = Foundation.log(alphaSource) - Foundation.log(sigmaSource)
        let h = lambdaTarget - lambdaSource
        let hh = -h
        let phi1 = Foundation.expm1(hh)
        let bH = phi1

        var correction = (modelOutput - previousModel) * 0.5
        if order == 2, stepIndex > 1, let olderModel = modelOutputs[0] {
            let sigmaOlder = sigmas[stepIndex - 2]
            let alphaOlder = 1 - sigmaOlder
            let lambdaOlder = Foundation.log(alphaOlder) - Foundation.log(sigmaOlder)
            let rk = (lambdaOlder - lambdaSource) / h

            let phi2 = phi1 / hh - 1
            let b1 = phi2 / bH
            let phi3 = phi2 / hh - 0.5
            let b2 = phi3 * 2 / bH
            let rho0 = (b2 - b1) / (rk - 1)
            let rho1 = b1 - rho0
            let firstDerivative = (olderModel - previousModel) / Float(rk)
            correction = firstDerivative * Float(rho0)
                + (modelOutput - previousModel) * Float(rho1)
        }

        let predicted = lastSample * Float(sigmaTarget / sigmaSource)
            - previousModel * Float(alphaTarget * phi1)
        return predicted - correction * Float(alphaTarget * bH)
    }
}
