import Foundation
import MLX

/// FlowMatchEulerDiscreteScheduler for FLUX.2 Klein
/// Supports both distilled (4-step) and base (50-step) inference modes.
public struct Flux2EulerScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int

    private static let shiftTerminal: Float = 0.02

    /// Initialize the scheduler for FLUX.2 Klein
    /// - Parameters:
    ///   - numInferenceSteps: Number of denoising steps (typically 4 for distilled, 50 for base)
    ///   - numTrainTimesteps: Training timesteps (1000)
    ///   - imageSeqLen: Number of latent tokens (height * width after patchify)
    ///   - isDistilled: If true, uses terminal stretch + mu=1.0 (distilled models).
    ///                  If false, uses linear 1→0 schedule with dynamic mu (base models).
    ///   - usesEmpiricalMu: Use the published FLUX.2-dev step- and sequence-aware shift.
    ///   - sigmaShift: Optional scalar FlowMatch shift override.
    ///   - customSigmas: Optional pre-shifted sigma value for each denoising step.
    public init(
        numInferenceSteps: Int = 4,
        numTrainTimesteps: Int = 1000,
        imageSeqLen: Int,
        isDistilled: Bool = true,
        usesEmpiricalMu: Bool = false,
        sigmaShift: Float? = nil,
        customSigmas: [Float]? = nil
    ) {
        self.numInferenceSteps = max(numInferenceSteps, 1)
        let steps = self.numInferenceSteps
        let numTrainTimestepsF = Float(numTrainTimesteps)

        let sigmasFinal: [Float]

        if let customSigmas {
            sigmasFinal = customSigmas
        } else if isDistilled {
            // Distilled mode: terminal stretch to 0.02, mu=1.0 (mflux behavior)
            let sigmaMin = 1.0 / numTrainTimestepsF
            let sigmaMax: Float = 1.0

            if steps == 1 {
                // Single step: use terminal sigma directly (skip shift/stretch which would divide by zero)
                sigmasFinal = [1.0 - Self.shiftTerminal]  // 0.98 - starts near full noise
            } else {
                var sigmasLinear = [Float]()
                for i in 0..<steps {
                    let timestep = sigmaMax * numTrainTimestepsF - Float(i) * (sigmaMax - sigmaMin) * numTrainTimestepsF / Float(steps - 1)
                    sigmasLinear.append(timestep / numTrainTimestepsF)
                }

                let sigmasShifted = sigmasLinear.map { Self.timeShiftExponentialScalar(mu: 1.0, sigmaPower: 1.0, t: $0) }
                sigmasFinal = Self.stretchToTerminal(sigmasShifted)
            }
        } else {
            // Base model mode: linear 1→0 with dynamic mu based on resolution
            // Matches ai-toolkit: timesteps = linspace(1, 0, steps+1), then time_shift
            var sigmasLinear = [Float]()
            for i in 0..<steps {
                sigmasLinear.append(1.0 - Float(i) / Float(steps))
            }

            let mu = usesEmpiricalMu
                ? Self.computeEmpiricalMu(imageSeqLen: imageSeqLen, numSteps: steps)
                : Self.computeDynamicMu(imageSeqLen: imageSeqLen)
            sigmasFinal = sigmasLinear.map { Self.timeShiftExponentialScalar(mu: mu, sigmaPower: 1.0, t: $0) }
            // NO terminal stretch for base model - schedule goes to 0
        }

        let shiftedSigmas = customSigmas == nil
            ? sigmaShift.map { Self.applyScalarSigmaShift(sigmas: sigmasFinal, shift: $0) } ?? sigmasFinal
            : sigmasFinal

        // Convert to timesteps
        let timestepsArr = shiftedSigmas.map { $0 * numTrainTimestepsF }

        // Append terminal 0.0 sigma
        let sigmasWithZero = shiftedSigmas + [0.0]

        self.sigmas = MLXArray(sigmasWithZero).asType(.float32)
        self.timesteps = MLXArray(timestepsArr).asType(.float32)
    }

    public static func validateCustomSigmas(_ sigmas: [Float], expectedSteps: Int) throws {
        guard expectedSteps > 0, sigmas.count == expectedSteps else {
            throw Flux2Error.invalidSigmaSchedule(
                "Expected \(expectedSteps) sigma values, found \(sigmas.count)."
            )
        }
        guard sigmas.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }) else {
            throw Flux2Error.invalidSigmaSchedule(
                "Sigma values must be finite and greater than 0 through 1."
            )
        }
        for index in 1..<sigmas.count where sigmas[index] >= sigmas[index - 1] {
            throw Flux2Error.invalidSigmaSchedule("Sigma values must be strictly descending.")
        }
    }

    /// Compute dynamic mu based on image sequence length (ai-toolkit formula)
    /// Linear interpolation: y1=0.5 at x1=256, y2=1.15 at x2=4096
    private static func computeDynamicMu(imageSeqLen: Int) -> Float {
        let m: Float = (1.15 - 0.5) / Float(4096 - 256)
        let b: Float = 0.5 - m * 256
        return m * Float(imageSeqLen) + b
    }

    /// Published FLUX.2-dev schedule fit used by the reference Diffusers pipeline.
    static func computeEmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
        let a1: Float = 8.73809524e-05
        let b1: Float = 1.89833333
        let a2: Float = 0.00016927
        let b2: Float = 0.45666666
        let sequenceLength = Float(imageSeqLen)

        if imageSeqLen > 4_300 {
            return a2 * sequenceLength + b2
        }

        let muAt200Steps = a2 * sequenceLength + b2
        let muAt10Steps = a1 * sequenceLength + b1
        let slope = (muAt200Steps - muAt10Steps) / 190
        let intercept = muAt200Steps - 200 * slope
        return slope * Float(numSteps) + intercept
    }

    /// Stretch sigmas so final sigma lands at shiftTerminal (0.02)
    private static func stretchToTerminal(_ sigmas: [Float]) -> [Float] {
        let oneMinusSigmas = sigmas.map { 1.0 - $0 }
        let scaleFactor = oneMinusSigmas.last! / (1.0 - shiftTerminal)
        return oneMinusSigmas.map { 1.0 - ($0 / scaleFactor) }
    }

    /// Scalar exponential time shift
    private static func timeShiftExponentialScalar(mu: Float, sigmaPower: Float, t: Float) -> Float {
        let expMu = exp(mu)
        return expMu / (expMu + pow(1.0 / t - 1.0, sigmaPower))
    }

    /// Apply exponential time shift (array version)
    static func timeShiftExponential(mu: Float, sigmas: MLXArray) -> MLXArray {
        let expMu = exp(mu)
        let one = MLXArray(Float(1.0))
        let expMuArr = MLXArray(expMu)
        return expMuArr / (expMuArr + (one / sigmas - one))
    }

    private static func applyScalarSigmaShift(sigmas: [Float], shift: Float) -> [Float] {
        guard shift > 0 else { return sigmas }
        return sigmas.map { sigma in
            shift * sigma / (1.0 + (shift - 1.0) * sigma)
        }
    }

    /// Perform one Euler step
    /// - Parameters:
    ///   - modelOutput: The velocity prediction from the transformer
    ///   - timestepIndex: Current step index (0 to numInferenceSteps-1)
    ///   - sample: Current latent sample
    /// - Returns: Updated latent sample
    public func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        let sigma = sigmas[timestepIndex]
        let sigmaNext = sigmas[timestepIndex + 1]
        let dt = sigmaNext - sigma

        // Euler step: x_{t+1} = x_t + dt * v
        // dt is negative (sigmas decrease), so this removes noise
        return sample + modelOutput.asType(sample.dtype) * dt.asType(sample.dtype)
    }

    /// Get the sigma value at a given step
    public func sigma(at index: Int) -> MLXArray {
        sigmas[index]
    }

    /// Get the timestep value at a given step (for passing to model, divide by 1000)
    public func timestep(at index: Int) -> MLXArray {
        timesteps[index]
    }
}
