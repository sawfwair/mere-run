import Foundation
import MLX

/// FlowMatchEulerDiscreteScheduler for Qwen-Image-Edit
/// Based on flow-matching as used in Stable Diffusion 3 and Qwen-Image models.
public struct FlowMatchEulerScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int

    /// Initialize the scheduler with a given config
    /// - Parameters:
    ///   - numInferenceSteps: Number of denoising steps (typically 50 for quality, 4-8 for speed)
    ///   - numTrainTimesteps: Training timesteps (typically 1000)
    ///   - shift: Sigma shift factor (typically 1.0 or from config)
    ///   - useDynamicShifting: Whether to apply resolution-dependent shifting
    ///   - imageSeqLen: Sequence length for dynamic shifting (width * height / patch_size^2)
    ///   - baseImageSeqLen: Base sequence length for dynamic shifting
    ///   - maxImageSeqLen: Max sequence length for dynamic shifting
    ///   - baseShift: Base shift for dynamic shifting
    ///   - maxShift: Max shift for dynamic shifting
    public init(
        numInferenceSteps: Int = 50,
        numTrainTimesteps: Int = 1000,
        shift: Float = 1.0,
        useDynamicShifting: Bool = false,
        imageSeqLen: Int? = nil,
        baseImageSeqLen: Int = 256,
        maxImageSeqLen: Int = 4096,
        baseShift: Float = 0.5,
        maxShift: Float = 1.15
    ) {
        self.numInferenceSteps = max(numInferenceSteps, 1)
        let steps = self.numInferenceSteps
        let numTrainTimestepsF = Float(numTrainTimesteps)

        // Diffusers FlowMatchEulerDiscreteScheduler uses:
        // - unshifted sigmas linearly spaced from 1.0 down to (1/num_train_timesteps),
        //   with length == num_inference_steps
        // - (optionally) time-shifts those sigmas
        // - then appends a terminal 0.0 so we have num_inference_steps + 1 sigmas
        let sigmaMin = 1.0 / numTrainTimestepsF
        let sigmaMax: Float = 1.0
        let sigmasLinear: MLXArray
        if steps == 1 {
            sigmasLinear = MLXArray([sigmaMax]).asType(.float32)
        } else {
            sigmasLinear = linspace(sigmaMax, sigmaMin, count: steps).asType(.float32)
        }

        // Timesteps passed to the model are the *unshifted* timesteps (scaled back to training range).
        // The shifted sigmas are used only for Euler step sizing.
        let timestepsLinear = sigmasLinear * numTrainTimestepsF

        // Compute mu (time shift factor).
        let mu: Float
        if useDynamicShifting, let seqLen = imageSeqLen {
            let m = (maxShift - baseShift) / Float(maxImageSeqLen - baseImageSeqLen)
            let b = baseShift - m * Float(baseImageSeqLen)
            mu = max(baseShift, min(maxShift, m * Float(seqLen) + b))
        } else {
            mu = shift
        }

        // Apply time shift (default: exponential).
        // Exponential shift: shifted_sigma = exp(mu) / (exp(mu) + (1/sigma - 1))
        let expMu = MLXArray(exp(mu))
        let one = MLXArray(Float(1.0))
        let eps = MLXArray(Float(1e-8))
        let sigmasShifted = expMu / (expMu + (one / maximum(sigmasLinear, eps)) - one)

        // Append terminal 0.0 sigma.
        let zero = MLXArray([Float(0.0)]).asType(.float32)
        let sigmasWithZero = MLX.concatenated([sigmasShifted, zero], axis: 0)

        self.sigmas = sigmasWithZero
        self.timesteps = timestepsLinear
    }

    /// Initialize from a scheduler config
    public init(
        config: QwenImageEditSchedulerConfig,
        numInferenceSteps: Int,
        imageSeqLen: Int? = nil
    ) {
        self.numInferenceSteps = max(numInferenceSteps, 1)

        let steps = self.numInferenceSteps
        let numTrainTimesteps = config.numTrainTimesteps
        let numTrainTimestepsF = Float(numTrainTimesteps)

        let sigmaMin = 1.0 / numTrainTimestepsF
        let sigmaMax: Float = 1.0
        let sigmasLinear: MLXArray
        if steps == 1 {
            sigmasLinear = MLXArray([sigmaMax]).asType(.float32)
        } else {
            sigmasLinear = linspace(sigmaMax, sigmaMin, count: steps).asType(.float32)
        }

        let timestepsLinear = sigmasLinear * numTrainTimestepsF

        let one = MLXArray(Float(1.0))

        // Compute mu / shift parameter.
        let mu: Float
        if config.useDynamicShifting, let seqLen = imageSeqLen {
            let baseShift = config.baseShift ?? 0.5
            let maxShift = config.maxShift ?? 1.15
            let baseSeqLen = config.baseImageSeqLen ?? 256
            let maxSeqLen = config.maxImageSeqLen ?? 4096

            let m = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
            let b = baseShift - m * Float(baseSeqLen)
            mu = max(baseShift, min(maxShift, m * Float(seqLen) + b))
        } else {
            mu = config.shift
        }

        // Apply time shift.
        let timeShiftType = (config.timeShiftType ?? "exponential").lowercased()
        var sigmasShifted: MLXArray
        switch timeShiftType {
        case "linear":
            // Linear shift: shift * sigma / (1 + (shift - 1) * sigma)
            let shiftArr = MLXArray(mu)
            sigmasShifted = shiftArr * sigmasLinear / (one + (shiftArr - one) * sigmasLinear)
        default:
            // Exponential shift: exp(mu) / (exp(mu) + (1/sigma - 1))
            let expMu = MLXArray(exp(mu))
            let eps = MLXArray(Float(1e-8))
            sigmasShifted = expMu / (expMu + (one / maximum(sigmasLinear, eps)) - one)
        }

        // Optional terminal stretching.
        if let shiftTerminal = config.shiftTerminal {
            let shiftTerminalClamped = min(max(shiftTerminal, 0.0), 1.0)
            let oneMinus = one - sigmasShifted
            let lastIndex = max(0, oneMinus.shape[0] - 1)
            let lastOneMinus = oneMinus[lastIndex]
            let denom = one - MLXArray(shiftTerminalClamped)
            let scaleFactor = lastOneMinus / denom
            sigmasShifted = one - (oneMinus / scaleFactor)
        }

        if config.invertSigmas ?? false {
            sigmasShifted = one - sigmasShifted
        }

        let zero = MLXArray([Float(0.0)]).asType(.float32)
        let sigmasWithZero = MLX.concatenated([sigmasShifted, zero], axis: 0)

        self.sigmas = sigmasWithZero
        self.timesteps = timestepsLinear
    }

    /// Perform one Euler step
    /// - Parameters:
    ///   - modelOutput: The noise prediction from the model (v-prediction or epsilon)
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
        // For flow matching, model_output is the velocity v
        return sample + modelOutput * dt
    }

    /// Scale initial noise based on sigma
    /// - Parameters:
    ///   - noise: Random noise tensor
    ///   - timestepIndex: Starting timestep index
    /// - Returns: Scaled noise for initialization
    public func scaleNoise(_ noise: MLXArray, timestepIndex: Int = 0) -> MLXArray {
        let sigma = sigmas[timestepIndex]
        return noise * sigma
    }

    /// Get the sigma value at a given step
    public func sigma(at index: Int) -> MLXArray {
        sigmas[index]
    }

    /// Get the timestep value at a given step
    public func timestep(at index: Int) -> MLXArray {
        timesteps[index]
    }
}
