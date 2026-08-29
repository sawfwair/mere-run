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

        // Qwen Image supplies an explicit N-step schedule from 1.0 through
        // 1/N before the scheduler applies its shift.
        let sigmaMin = 1.0 / Float(steps)
        let sigmaMax: Float = 1.0
        let sigmasLinear: MLXArray
        if steps == 1 {
            sigmasLinear = MLXArray([sigmaMax]).asType(.float32)
        } else {
            sigmasLinear = linspace(sigmaMax, sigmaMin, count: steps).asType(.float32)
        }

        // Compute mu (time shift factor).
        let mu: Float
        if useDynamicShifting, let seqLen = imageSeqLen {
            let m = (maxShift - baseShift) / Float(maxImageSeqLen - baseImageSeqLen)
            let b = baseShift - m * Float(baseImageSeqLen)
            mu = m * Float(seqLen) + b
        } else {
            mu = shift
        }

        let one = MLXArray(Float(1.0))
        let sigmasShifted: MLXArray
        if useDynamicShifting {
            let expMu = MLXArray(exp(mu))
            sigmasShifted = expMu / (expMu + (one / sigmasLinear) - one)
        } else {
            let shiftArray = MLXArray(shift)
            sigmasShifted = shiftArray * sigmasLinear / (one + (shiftArray - one) * sigmasLinear)
        }

        // Append terminal 0.0 sigma.
        let zero = MLXArray([Float(0.0)]).asType(.float32)
        let sigmasWithZero = MLX.concatenated([sigmasShifted, zero], axis: 0)

        self.sigmas = sigmasWithZero
        self.timesteps = sigmasShifted * numTrainTimestepsF
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

        let sigmaMin = 1.0 / Float(steps)
        let sigmaMax: Float = 1.0
        let sigmasLinear: MLXArray
        if steps == 1 {
            sigmasLinear = MLXArray([sigmaMax]).asType(.float32)
        } else {
            sigmasLinear = linspace(sigmaMax, sigmaMin, count: steps).asType(.float32)
        }

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
            mu = m * Float(seqLen) + b
        } else {
            mu = config.shift
        }

        var sigmasShifted: MLXArray
        if config.useDynamicShifting {
            let timeShiftType = (config.timeShiftType ?? "exponential").lowercased()
            switch timeShiftType {
            case "linear":
                sigmasShifted = MLXArray(mu) / (MLXArray(mu) + (one / sigmasLinear) - one)
            default:
                let expMu = MLXArray(exp(mu))
                sigmasShifted = expMu / (expMu + (one / sigmasLinear) - one)
            }
        } else {
            let shiftArray = MLXArray(config.shift)
            sigmasShifted = shiftArray * sigmasLinear / (one + (shiftArray - one) * sigmasLinear)
        }

        // Optional terminal stretching.
        if let shiftTerminal = config.shiftTerminal, shiftTerminal > 0 {
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

        let timestepsShifted = sigmasShifted * numTrainTimestepsF

        let zero = MLXArray([Float(0.0)]).asType(.float32)
        let sigmasWithZero = MLX.concatenated([sigmasShifted, zero], axis: 0)

        self.sigmas = sigmasWithZero
        self.timesteps = timestepsShifted
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

    /// Qwen's transformer consumes the shifted sigma in the normalized [0, 1]
    /// domain. Diffusers exposes the corresponding scheduler timestep scaled by
    /// `num_train_timesteps`, then divides by 1,000 before the model call.
    public func modelTimestep(at index: Int) -> MLXArray {
        sigmas[index]
    }
}
