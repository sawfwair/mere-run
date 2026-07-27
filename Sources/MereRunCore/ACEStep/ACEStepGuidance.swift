import MLX

enum ACEStepGuidance {
    static func cfg(
        conditional: MLXArray,
        unconditional: MLXArray,
        scale: Float
    ) -> MLXArray {
        unconditional + MLXArray(scale).asType(conditional.dtype)
            * (conditional - unconditional)
    }

    static func apg(
        conditional: MLXArray,
        unconditional: MLXArray,
        scale: Float,
        runningAverage: inout MLXArray?,
        momentum: Float = -0.75,
        normThreshold: Float = 2.5
    ) -> MLXArray {
        var difference = conditional - unconditional
        if let runningAverage {
            difference = difference
                + MLXArray(momentum).asType(difference.dtype) * runningAverage
        }
        runningAverage = difference

        if normThreshold > 0 {
            let norm = MLX.sqrt(
                (difference * difference).sum(axis: 1, keepDims: true)
            )
            let scaleFactor = MLX.minimum(
                MLXArray.ones(norm.shape, dtype: norm.dtype),
                MLXArray(normThreshold).asType(norm.dtype)
                    / (norm + MLXArray(Float(1e-8)).asType(norm.dtype))
            )
            difference = difference * scaleFactor
        }

        let conditionalNorm = MLX.sqrt(
            (conditional * conditional).sum(axis: 1, keepDims: true)
        )
        let direction = conditional
            / (conditionalNorm + MLXArray(Float(1e-8)).asType(conditional.dtype))
        let parallel = (difference * direction).sum(axis: 1, keepDims: true)
            * direction
        let orthogonal = difference - parallel
        return conditional
            + MLXArray(scale - 1).asType(conditional.dtype) * orthogonal
    }

    static func adg(
        latents: MLXArray,
        conditional: MLXArray,
        unconditional: MLXArray,
        sigma: Float,
        scale: Float,
        angleClip: Float = Float.pi / 6
    ) -> MLXArray {
        guard sigma > 0 else {
            return cfg(
                conditional: conditional,
                unconditional: unconditional,
                scale: scale
            )
        }
        precondition(latents.shape == conditional.shape)
        precondition(latents.shape == unconditional.shape)

        let dtype = conditional.dtype
        let sigmaArray = MLXArray(sigma).asType(dtype)
        let weight = max(scale - 1, 0) + 0.001
        let conditionalClean = latents - sigmaArray * conditional
        let unconditionalClean = latents - sigmaArray * unconditional
        let difference = conditionalClean - unconditionalClean

        let epsilon = MLXArray(Float(1e-8)).asType(dtype)
        let conditionalNorm = MLX.sqrt(
            (conditionalClean * conditionalClean).sum(axis: 2, keepDims: true)
        )
        let unconditionalNorm = MLX.sqrt(
            (unconditionalClean * unconditionalClean).sum(axis: 2, keepDims: true)
        )
        let cosine = MLX.clip(
            (conditionalClean * unconditionalClean).sum(axis: 2, keepDims: true)
                / (conditionalNorm * unconditionalNorm + epsilon),
            min: -1 + 1e-6,
            max: 1 - 1e-6
        )
        let theta = MLX.acos(cosine)
        let guidedTheta = MLX.clip(
            theta * MLXArray(weight).asType(dtype),
            min: -angleClip,
            max: angleClip
        )

        let normSquared = (unconditionalClean * unconditionalClean)
            .sum(axis: 2, keepDims: true)
        let projection = (difference * unconditionalClean)
            .sum(axis: 2, keepDims: true)
            / (normSquared + epsilon)
            * unconditionalClean
        let perpendicular = difference - projection
        let sinTheta = MLX.sin(theta)
        let scaledPerpendicular = MLX.where(
            sinTheta .> MLXArray(Float(1e-3)).asType(dtype),
            perpendicular * MLX.sin(guidedTheta) / (sinTheta + epsilon),
            perpendicular * MLXArray(weight).asType(dtype)
        )
        let guidedClean = MLX.cos(guidedTheta) * conditionalClean
            + scaledPerpendicular
        return (latents - guidedClean) / sigmaArray
    }
}
