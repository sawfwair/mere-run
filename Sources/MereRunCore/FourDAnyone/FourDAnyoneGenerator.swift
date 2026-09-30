import MLX

/// Immutable prepared conditioning. Dense generation requires four re-encoded
/// proposal references in addition to the source. This type does not recover motion.
public struct FourDAnyonePreparedConditioning {
    public let sources: MLXArray
    public let poseFeatures: MLXArray
    public let nullPoseFeatures: MLXArray
    public let promptContext: MLXArray

    public init(
        sources: MLXArray,
        poseFeatures: MLXArray,
        nullPoseFeatures: MLXArray,
        promptContext: MLXArray
    ) {
        self.sources = sources
        self.poseFeatures = poseFeatures
        self.nullPoseFeatures = nullPoseFeatures
        self.promptContext = promptContext
    }
}

public struct FourDAnyoneProgress: Equatable, Sendable {
    public let step: Int
    public let stepCount: Int
    public let completedGroups: Int
    public let groupCount: Int
    public let cameraIDs: [Int]
}

/// Runs the Base denoiser from explicit initial noise and prepared conditioning.
/// One caller owns the runtime; MLX arrays are not shared across concurrent runs.
public final class FourDAnyoneGenerator {
    public let transformer: FourDAnyoneTransformerModel

    public init(transformer: FourDAnyoneTransformerModel) {
        self.transformer = transformer
    }

    public func generate(
        initialLatents: MLXArray,
        conditioning: FourDAnyonePreparedConditioning,
        plan: FourDAnyoneViewPlan,
        steps: Int = 24,
        maximumQueryTokens: Int = 512,
        checkCancellation: () throws -> Void = { try Task.checkCancellation() },
        progress: ((FourDAnyoneProgress) -> Void)? = nil
    ) throws -> MLXArray {
        let schedule = try FourDAnyoneSchedule(steps: steps)
        let validated = FourDAnyoneTransformerInput(
            latents: initialLatents, sources: conditioning.sources,
            poseFeatures: conditioning.poseFeatures, nullPoseFeatures: conditioning.nullPoseFeatures,
            promptContext: conditioning.promptContext, timestep: schedule.timesteps[0]
        )
        _ = try validated.validate(configuration: transformer.configuration)
        guard initialLatents.dim(0) == plan.viewCount,
              conditioning.sources.dim(0) == (plan.referencePacking ? 5 : 1) else {
            throw FourDAnyoneError.invalidInput("The prepared views and reference inputs must match the view plan.")
        }
        try checkCancellation()
        let dtype = transformer.computePrecision.dtype(for: transformer.patchEmbedding.weight.dtype)
        var latents = (0..<plan.viewCount).map { initialLatents[$0].asType(dtype) }
        eval(latents)
        for (step, timestep) in schedule.timesteps.enumerated() {
            let groups = plan.groups(step: step)
            for (groupIndex, cameraIDs) in groups.enumerated() {
                try checkCancellation()
                let local = MLX.stacked(cameraIDs.map { latents[$0] }, axis: 0)
                let poses = MLX.stacked(cameraIDs.map { conditioning.poseFeatures[$0] }, axis: 0)
                let prediction = try transformer(
                    FourDAnyoneTransformerInput(
                        latents: local, sources: conditioning.sources,
                        poseFeatures: poses, nullPoseFeatures: conditioning.nullPoseFeatures,
                        promptContext: conditioning.promptContext, timestep: timestep
                    ),
                    maximumQueryTokens: maximumQueryTokens,
                    checkCancellation: checkCancellation
                )
                let updated = schedule.step(prediction: prediction, sample: local, index: step)
                eval(updated)
                for (index, cameraID) in cameraIDs.enumerated() {
                    latents[cameraID] = updated[index]
                }
                progress?(FourDAnyoneProgress(
                    step: step + 1, stepCount: schedule.stepCount,
                    completedGroups: groupIndex + 1, groupCount: groups.count, cameraIDs: cameraIDs
                ))
            }
        }
        let result = MLX.stacked(latents, axis: 0)
        eval(result)
        return result
    }

    /// Decodes one canonical view at a time to [frames, height, width, RGB] in [0, 1].
    /// The consumer can write each video without retaining all decoded views.
    public static func decodeViews(
        _ latents: MLXArray,
        using vae: Wan2VAEModel,
        checkCancellation: () throws -> Void = { try Task.checkCancellation() },
        consume: (Int, MLXArray) throws -> Void
    ) throws {
        guard latents.ndim == 5, latents.shape.allSatisfy({ $0 > 0 }), latents.dim(1) == 48,
              vae.latentChannels == 48 else {
            throw FourDAnyoneError.invalidInput("Video decoding requires Wan2.2 48-channel latents and VAE.")
        }
        for index in 0..<latents.dim(0) {
            try checkCancellation()
            let latent = latents[index].transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
            let frames = (vae.decode(latent)[0].asType(.float32) + 1) / 2
            eval(frames)
            try consume(index, frames)
        }
    }
}
