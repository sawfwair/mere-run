import MLX

public struct ZImageTurboLinearScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray

    public init(
        config: ZImageTurboInferenceConfig,
        requiresSigmaShift: Bool = true,
        sigmaShift: Float? = nil
    ) {
        let steps = max(config.numInferenceSteps, 1)

        let baseSigmas = linspace(
            Float(1.0),
            Float(1.0) / Float(steps),
            count: steps
        ).asType(.float32)

        let sigmasWithZero = concatenated([baseSigmas, zeros([1], dtype: .float32)], axis: 0)

        if let sigmaShift {
            self.sigmas = Self.applyScalarSigmaShift(sigmas: sigmasWithZero, shift: sigmaShift)
        } else if requiresSigmaShift {
            self.sigmas = Self.applySigmaShift(
                sigmas: sigmasWithZero,
                width: config.width,
                height: config.height,
                numInferenceSteps: steps
            )
        } else {
            self.sigmas = sigmasWithZero
        }

        self.timesteps = self.sigmas[0 ..< steps] * MLXArray(Float(1000.0))
    }

    public func step(
        noise: MLXArray,
        timestep: Int,
        latents: MLXArray
    ) -> MLXArray {
        let dt = sigmas[timestep + 1] - sigmas[timestep]
        return latents + noise * dt
    }

    private static func applySigmaShift(
        sigmas: MLXArray,
        width: Int,
        height: Int,
        numInferenceSteps: Int
    ) -> MLXArray {
        // Ported from mflux `LinearScheduler._get_sigmas` for Z-Image Turbo.
        let y1: Float = 0.5
        let x1: Float = 256
        let m = (Float(1.16) - y1) / (Float(4096) - x1)
        let b = y1 - m * x1

        let mu = m * Float(width * height) / Float(256) + b
        let muArr = MLXArray(mu)
        let expMu = muArr.exp()
        let one = MLXArray(Float(1.0))
        let shifted = expMu / (expMu + (one / sigmas - one))

        // Force terminal sigma to 0 (avoid inf propagation from the pre-shifted appended zero).
        let prefix = shifted[0 ..< numInferenceSteps]
        return concatenated([prefix, zeros([1], dtype: .float32)], axis: 0)
    }

    private static func applyScalarSigmaShift(sigmas: MLXArray, shift: Float) -> MLXArray {
        // Matches DiffSynth `FlowMatchScheduler.set_timesteps_z_image`:
        //   sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        let shiftArr = MLXArray(shift)
        let one = MLXArray(Float(1.0))
        return (shiftArr * sigmas) / (one + (shiftArr - one) * sigmas)
    }
}
